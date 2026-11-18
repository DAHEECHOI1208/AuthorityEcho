"""Native API adapters. One HTTP request per attempt, no SDK retries or tool loop."""
import base64
import copy
import json
import re
import ssl
import urllib.error
import urllib.request
from urllib.parse import quote
from settings import REFERENCE_TOOL

def dumps(v):
    return json.dumps(v,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def endpoint_for(cfg):
    p=cfg['provider']
    if p=='openai': return 'https://api.openai.com/v1/chat/completions'
    if p=='anthropic': return 'https://api.anthropic.com/v1/messages'
    if p=='gemini':
        model=cfg['model'].removeprefix('models/')
        return 'https://generativelanguage.googleapis.com/v1beta/models/'+quote(model,safe='')+':generateContent'
    return cfg['base_url'].rstrip('/')+'/chat/completions'

def headers_for(cfg,key,client_id):
    h={'Content-Type':'application/json','Accept':'application/json',
       'X-Client-Request-Id':client_id,'User-Agent':'LLM-Evaluation/1.4.2'}
    if cfg['provider']=='anthropic':
        h.update({'x-api-key':key,'anthropic-version':'2023-06-01'})
    elif cfg['provider']=='gemini': h['x-goog-api-key']=key
    elif key: h['Authorization']='Bearer '+key
    return h

def build_payload(cfg,context):
    doc=context['document']; user=context['user_prompt']
    reference={'documents':[{'id':'doc_01','text':doc}] if doc else []}
    call_id='call_reference_'+context['question_id']
    tool=REFERENCE_TOOL['function']; p=cfg['provider']
    if p in {'openai','openai_compatible'}:
        payload={'model':cfg['model'],'messages':[
            {'role':'system','content':cfg['system_instruction']},
            {'role':'user','content':user},
            {'role':'assistant','content':None,'tool_calls':[{'id':call_id,'type':'function','function':{'name':tool['name'],'arguments':'{}'}}]},
            {'role':'tool','tool_call_id':call_id,'content':dumps(reference)}],
            'tools':[copy.deepcopy(REFERENCE_TOOL)],'tool_choice':'none','parallel_tool_calls':False,
            'reasoning_effort':cfg['reasoning_effort'],'temperature':cfg['temperature'],
            'max_completion_tokens':cfg['max_completion_tokens'],'n':1,'stream':False,'store':False}
        if p == 'openai' and cfg['model'] == 'gpt-4.1-2025-04-14':
            payload.pop('reasoning_effort')  # The non-reasoning GPT-4.1 snapshot has no reasoning control.
        if p=='openai_compatible':
            # Explicit server profile; never retry with dropped/changed parameters.
            c=cfg['compatible']; payload.pop('store'); payload.pop('reasoning_effort'); payload.pop('max_completion_tokens')
            payload[c['max_tokens_parameter']]=cfg['max_completion_tokens']
            if c['reasoning_control']!={'non_reasoning_model':True}:
                payload.update(copy.deepcopy(c['reasoning_control']))
        return payload
    if p=='anthropic':
        return {'model':cfg['model'],'system':cfg['system_instruction'],'messages':[
            {'role':'user','content':user},
            {'role':'assistant','content':[{'type':'tool_use','id':call_id,'name':tool['name'],'input':{}}]},
            {'role':'user','content':[{'type':'tool_result','tool_use_id':call_id,'content':dumps(reference)}]}],
            'tools':[{'name':tool['name'],'description':tool['description'],
                      'input_schema':copy.deepcopy(tool['parameters']),'strict':tool['strict']}],
            'tool_choice':{'type':'none'},'thinking':{'type':'disabled'},
            'temperature':cfg['temperature'],'max_tokens':cfg['max_completion_tokens'],'stream':False}
    if p=='gemini':
        return {'systemInstruction':{'parts':[{'text':cfg['system_instruction']}]},'contents':[
            {'role':'user','parts':[{'text':user}]},
            # Official literal sentinel for client-constructed tool history, not a
            # model-generated signature. No preliminary model call is performed.
            # https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures#faqs
            {'role':'model','parts':[{'functionCall':{'id':call_id,'name':tool['name'],'args':{}},
                                     'thoughtSignature':cfg['gemini']['tool_history_signature']}]},
            {'role':'user','parts':[{'functionResponse':{'id':call_id,'name':tool['name'],'response':reference}}]}],
            'tools':[{'functionDeclarations':[{'name':tool['name'],'description':tool['description'],
                                              'parametersJsonSchema':copy.deepcopy(tool['parameters'])}]}],
            'toolConfig':{'functionCallingConfig':{'mode':'NONE'}},
            'generationConfig':{'temperature':cfg['temperature'],'maxOutputTokens':cfg['max_completion_tokens'],
                                'candidateCount':1,'thinkingConfig':{'thinkingLevel':cfg['gemini']['thinking_level']}}}
    raise ValueError('Unknown provider')

def describe_settings(cfg):
    """Audit actual serialized controls separately from the common baseline.

    This describes what the client sends, not verified server-side behavior.
    """
    payload=build_payload(cfg,{'question_id':'settings','document':'','user_prompt':''})
    provider=cfg['provider']
    result={'provider':provider,'requested_model':cfg['model'],
            'baseline_reasoning_effort':cfg['reasoning_effort'],
            'tool_history_origin':'client_constructed; no preliminary model call',
            'deviations_from_common_settings':[],
            'validation_scope':'request construction only; live API acceptance not verified'}
    if provider=='gemini':
        result.update(generation_settings=payload['generationConfig'],
                      tool_call_policy=payload['toolConfig'],strict_field_sent=False,
                      tool_history_signature=cfg['gemini']['tool_history_signature'])
        result['deviations_from_common_settings']=[
            {'item':'reasoning','baseline':'none','actual':'minimal; thinking is not guaranteed off'},
            {'item':'output_budget','baseline':'4096 with reasoning disabled',
             'actual':'4096 shared by thinking and final answer; not 4096 answer tokens guaranteed'},
            {'item':'strict','baseline':'tool strict=true',
             'actual':'same parameter schema; no equivalent per-tool strict flag under mode NONE'},
            {'item':'tool_history','baseline':'client-constructed tool call and result',
             'actual':'functionCall/functionResponse with matching id/name plus official literal thoughtSignature sentinel'}]
    elif provider=='anthropic':
        result.update(generation_settings={k:payload[k] for k in ('thinking','temperature','max_tokens','stream')},
                      tool_call_policy=payload['tool_choice'],strict_field_sent=payload['tools'][0]['strict'])
    else:
        result.update(generation_settings={k:v for k,v in payload.items()
                                          if k not in {'model','messages','tools','tool_choice','parallel_tool_calls'}},
                      tool_call_policy=payload['tool_choice'],
                      strict_field_sent=payload['tools'][0]['function']['strict'])
        if provider=='openai_compatible':result['server_profile']=copy.deepcopy(cfg['compatible'])
    return result

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl): return None

def transport_for(cfg):
    endpoint=endpoint_for(cfg)
    def send(payload_bytes,headers,timeout):
        req=urllib.request.Request(endpoint,data=payload_bytes,headers=headers,method='POST')
        opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        try:
            with opener.open(req,timeout=timeout) as r: return r.status,dict(r.headers.items()),r.read()
        except urllib.error.HTTPError as e:
            return e.code,dict(e.headers.items()),e.read()
    return send

def normalize_receipt(receipt):
    """Normalize a COPY for status parsing; durable receipts/raw bytes stay native."""
    out=copy.deepcopy(receipt); native={}
    h=out.get('headers',{})
    if 'x-request-id' not in h:
        for name in ['request-id','x-goog-request-id']:
            if h.get(name): h['x-request-id']=h[name];break
    if 'body_base64' not in out or not 200 <= (out.get('http_status') or 0)<300:
        return out,native
    try:
        body=json.loads(base64.b64decode(out['body_base64']).decode('utf-8'))
        if not isinstance(body,dict): return out,native
        p=receipt.get('provider','openai')
        if p in {'openai','openai_compatible'}: return out,native
        if p=='anthropic':
            blocks=body['content']; usage=body.get('usage') or {}; reason=body['stop_reason']
            if not isinstance(blocks,list) or not all(isinstance(x,dict) for x in blocks): raise ValueError('Invalid content blocks')
            text=''.join(x['text'] for x in blocks if x.get('type')=='text')
            finish={'end_turn':'stop','stop_sequence':'stop','max_tokens':'length','refusal':'stop'}.get(reason,reason)
            tokens_in=usage.get('input_tokens'); tokens_out=usage.get('output_tokens')
            cache_read=usage.get('cache_read_input_tokens',0); cache_create=usage.get('cache_creation_input_tokens',0)
            if isinstance(tokens_in,int): tokens_in+=cache_read+cache_create
            norm={'id':body.get('id'),'model':body.get('model'),
                  'choices':[{'finish_reason':finish,'message':{'content':text,'refusal':text if reason=='refusal' else None}}],
                  'usage':{'prompt_tokens':tokens_in,'completion_tokens':tokens_out,
                           'total_tokens':tokens_in+tokens_out if isinstance(tokens_in,int) and isinstance(tokens_out,int) else None,
                           'prompt_tokens_details':{'cached_tokens':cache_read}}}
            native={'finish_reason':reason,'native_usage':usage}
        elif p=='gemini':
            usage=body.get('usageMetadata') or {}; feedback=body.get('promptFeedback') or {}
            candidates=body.get('candidates') or []
            if not candidates:
                if not feedback.get('blockReason'): return out,native
                reason=feedback['blockReason']; text=''; finish='content_filter'
            else:
                if len(candidates)!=1: raise ValueError('Expected exactly one Gemini candidate')
                c=candidates[0];reason=c['finishReason'];parts=(c.get('content') or {}).get('parts') or []
                text=''.join(x['text'] for x in parts if 'text' in x and not x.get('thought'))
                if any('functionCall' in x for x in parts): finish='tool_calls'
                else: finish={'STOP':'stop','MAX_TOKENS':'length','SAFETY':'content_filter','RECITATION':'content_filter','BLOCKLIST':'content_filter','PROHIBITED_CONTENT':'content_filter','SPII':'content_filter'}.get(reason,reason)
            norm={'id':body.get('responseId'),'model':body.get('modelVersion'),
                  'choices':[{'finish_reason':finish,'message':{'content':text}}],
                  'usage':{'prompt_tokens':usage.get('promptTokenCount'),'completion_tokens':usage.get('candidatesTokenCount'),
                           'total_tokens':usage.get('totalTokenCount'),'prompt_tokens_details':{'cached_tokens':usage.get('cachedContentTokenCount')}}}
            native={'finish_reason':reason,'native_usage':usage,'prompt_feedback':feedback}
        else: return out,native
        out['body_base64']=base64.b64encode(dumps(norm).encode()).decode()
        return out,native
    except (ValueError,TypeError,KeyError,UnicodeError):
        # Native response is retained; common parser records PROTOCOL_ERROR.
        return out,{}

def apply_error_policy(result,receipt):
    if result['status']=='API_ERROR':
        message=' '.join(str(result.get(k) or '') for k in ['error_type','error_code','error_message']).lower()
        shared=bool(re.search(r'temperature|thinking|reasoning_effort|max_tokens|maxoutputtokens|max_completion_tokens|tool_choice|toolconfig|functiondeclarations|input_schema|parametersjsonschema|chat.template|thought.signature|parallel_tool_calls|candidatecount|tool_use|tool_result|tool_calls|functioncall|functionresponse|systeminstruction|(?<![a-z])strict(?![a-z])|unsupported.*parameter|unknown.*parameter|unknown.*tools|invalid.*tools',message))
        quota=any(x in message for x in ['credit balance','insufficient credit','billing','insufficient_quota','daily quota','quota per day','perday','payment required'])
        model=bool(re.search(r'model.*(not found|not available|not supported|does not exist|no access)',message))
        status=receipt.get('http_status') or 0
        if quota or model or status==402 or 300<=status<400 or (status in {400,422} and shared):
            result.update(stop_run=True,retryable=False)
    return result
