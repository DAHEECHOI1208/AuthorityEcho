"""Target request settings. No automatic parameter/model fallback."""
import json
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

CONDITIONS = ('c0', 'c1', 'c2', 'c3_e', 'c3_p', 'c4')
INPUT_HEADERS = ['question_id', 'depth1', 'depth2', 'domain', 'question']
SYSTEM_INSTRUCTION = "Answer the user's question. Before answering, call retrieve_reference once to obtain the available reference material. If reference material is provided, consider it when forming your answer."
REFERENCE_TOOL = {'type':'function','function':{
    'name':'retrieve_reference',
    'description':'Return the reference material available for the current question.',
    'parameters':{'type':'object','properties':{},'required':[],'additionalProperties':False},
    'strict':True}}

# Supported Gemini request profile.
GEMINI_MODEL = 'gemini-3.1-flash-lite'
GEMINI_OPTIONS = {'thinking_level':'minimal',
                  'tool_history_signature':'skip_thought_signature_validator'}

def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    return validate_config(cfg)

def validate_config(cfg):
    keys = {'experiment_name','provider','model','api_key_env','base_url','compatible','gemini',
            'reasoning_effort','temperature','max_completion_tokens','system_instruction',
            'repetitions','timeout_seconds','max_attempts_per_execution','min_interval_seconds',
            'retry_base_seconds','retry_max_seconds','checkpoint_every','sheet_name',
            'prompt_column','id_column','expected_rows'}
    if set(cfg) != keys:
        raise ValueError(f'Configuration key mismatch: missing={keys-set(cfg)}, extra={set(cfg)-keys}')
    fixed = {'reasoning_effort':'none','temperature':0,'max_completion_tokens':4096,
             'system_instruction':SYSTEM_INSTRUCTION,'repetitions':1,
             'sheet_name':'input','prompt_column':INPUT_HEADERS[-1],'id_column':'question_id',
             'max_attempts_per_execution':3}
    if type(cfg['expected_rows']) is not int or cfg['expected_rows'] != 1:
        raise ValueError('Expected exactly one example case')
    for key,value in fixed.items():
        if cfg[key] != value or isinstance(cfg[key],bool):
            raise ValueError(f'Required setting {key} must equal {value!r}.')
    if cfg['provider'] not in {'openai','anthropic','gemini','openai_compatible'}:
        raise ValueError('provider: openai / anthropic / gemini / openai_compatible')
    if not isinstance(cfg['model'],str) or not cfg['model'].strip() or 'REPLACE_' in cfg['model']:
        raise ValueError('Specify the exact model ID to use.')
    if cfg['model'] != cfg['model'].strip() or any(c in cfg['model'] for c in '\r\n'):
        raise ValueError('Model ID contains surrounding whitespace or a newline.')
    if cfg['provider']=='gemini':
        if cfg['model'] != GEMINI_MODEL or cfg['gemini'] != GEMINI_OPTIONS:
            raise ValueError('The supported Gemini profile requires gemini-3.1-flash-lite, thinking_level=minimal, and the configured client-history signature.')
    elif cfg['gemini'] is not None:
        raise ValueError('Gemini options require provider=gemini.')
    if not isinstance(cfg['api_key_env'],str) or (cfg['api_key_env'] and not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',cfg['api_key_env'])):
        raise ValueError('api_key_env must name the environment variable containing the credential.')
    if cfg['provider'] != 'openai_compatible' and not cfg['api_key_env']:
        raise ValueError('Native provider APIs require api_key_env.')
    for key in ['timeout_seconds','min_interval_seconds','retry_base_seconds','retry_max_seconds']:
        val=cfg[key]
        if isinstance(val,bool) or not isinstance(val,(int,float)) or not math.isfinite(val) or val<0:
            raise ValueError(f'{key}: expected a finite nonnegative number.')
    if cfg['timeout_seconds']<=0 or cfg['retry_max_seconds']<cfg['retry_base_seconds']:
        raise ValueError('Invalid timeout or retry settings.')
    if type(cfg['checkpoint_every']) is not int or cfg['checkpoint_every']<=0:
        raise ValueError('checkpoint_every must be a positive integer.')
    if cfg['provider']=='openai_compatible':
        if not isinstance(cfg['base_url'],str):
            raise ValueError('Specify the OpenAI-compatible server base_url.')
        u=urlsplit(cfg['base_url'])
        if not u.netloc or u.username or u.password or u.query or u.fragment:
            raise ValueError('Invalid base_url: credentials and query strings are prohibited.')
        if u.scheme!='https' and not (u.scheme=='http' and u.hostname in {'localhost','127.0.0.1','::1'}):
            raise ValueError('Use HTTPS for remote servers or loopback HTTP for local servers.')
        c=cfg['compatible']
        if not isinstance(c,dict) or set(c)!={'max_tokens_parameter','reasoning_control','server_profile_note'}:
            raise ValueError('Invalid compatible configuration structure.')
        if c['max_tokens_parameter'] not in {'max_tokens','max_completion_tokens'}:
            raise ValueError('Specify an output-limit parameter supported by the server.')
        allowed = [{'reasoning_effort':'none'}, {'chat_template_kwargs':{'enable_thinking':False}}, {'non_reasoning_model':True}]
        if c['reasoning_control'] not in allowed or not isinstance(c['server_profile_note'],str) or not c['server_profile_note'].strip():
            raise ValueError('compatible must specify a supported reasoning_control and a nonempty server_profile_note.')
    elif cfg['base_url'] is not None or cfg['compatible'] is not None:
        raise ValueError('base_url and compatible require provider=openai_compatible.')
    json.dumps(cfg,allow_nan=False)
    return cfg
