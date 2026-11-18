import copy
import json
from mechanism.native.A04_runtime import TaskFatal, digest
MODEL = 'deepseek-ai/DeepSeek-V2-Lite-Chat'
REVISION = '85864749cd611b4353ce1decdb286193298f64c7'
ADAPTER_ID = 'deepseek-v2-lite-explicit-tool-history-v1'
NATIVE_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}{{ message['content'] + '\n\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"

def enabled(cfg):
    return cfg.get('model_id') == MODEL

def validate(cfg):
    if not enabled(cfg) or cfg.get('revision') != REVISION or cfg.get('tokenizer_revision') != REVISION:
        raise TaskFatal('DeepSeek target/tokenizer revision changed')
    if cfg.get('serialization_adapter') != ADAPTER_ID:
        raise TaskFatal('DeepSeek requires its explicit tool-history adapter; native template drops tools')
    if cfg.get('attention_implementation') != 'sdpa':
        raise TaskFatal('DeepSeek attention implementation must remain sdpa')

def identity(tokenizer):
    if tokenizer.chat_template != NATIVE_TEMPLATE:
        raise TaskFatal('Pinned DeepSeek native template changed; do not silently adapt')
    return {'native_template_hash': digest(tokenizer.chat_template), 'template_hash': digest({'native': tokenizer.chat_template, 'adapter': ADAPTER_ID}), 'serialization_adapter': ADAPTER_ID, 'native_tool_roles_supported': False, 'tool_history_representation': 'explicit_text_role_markers', 'model_implementation': 'transformers.models.deepseek_v2 (4.57.6)', 'trust_remote_code': False}

def render(row, tokenizer):
    from mechanism.native.P17_final_answer import template_kwargs
    template_kwargs(row, {'final_answer_policy': row.get('final_answer_policy')})
    identity(tokenizer)
    messages = row['messages']
    if [m['role'] for m in messages] != ['system', 'user', 'assistant', 'tool']:
        raise TaskFatal('Unexpected retrieval history; refusing lossy serialization')
    system, user, assistant, tool = messages
    calls = assistant.get('tool_calls')
    if assistant.get('content') not in (None, '') or not isinstance(calls, list) or len(calls) != 1 or (calls[0].get('type') != 'function') or (calls[0].get('function', {}).get('name') != 'retrieve_reference') or (calls[0].get('id') != tool.get('tool_call_id')) or (tool.get('name') != 'retrieve_reference'):
        raise TaskFatal('Invalid retrieval history linkage')
    normalized = copy.deepcopy(calls)
    args = normalized[0]['function']['arguments']
    args = json.loads(args) if isinstance(args, str) else args
    if args != {}:
        raise TaskFatal('Retrieval arguments differ')
    normalized[0]['function']['arguments'] = args
    payload = tool['content']
    if payload != row['tool_payload']:
        raise TaskFatal('Tool payload differs from document contract')
    want = {'documents': [{'id': 'doc_01', 'text': row['document']}]} if row.get('document') else {'documents': []}
    if json.loads(payload) != want:
        raise TaskFatal('Document missing or changed in tool payload')
    prefix = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=False)
    expected = tokenizer.bos_token + system['content'] + '\n\nUser: ' + user['content'] + '\n\n'
    if prefix != expected:
        raise TaskFatal('DeepSeek system/user native rendering changed')
    user_start = len(tokenizer.bos_token + system['content'] + '\n\nUser: ')
    text = prefix + 'Assistant: ' + json.dumps({'tool_calls': normalized}, ensure_ascii=False, separators=(',', ':')) + tokenizer.eos_token
    text += '\n\nTool (retrieve_reference): '
    payload_start = len(text)
    text += payload + '\n\nAssistant:'
    spans = {'user': (user_start, user_start + len(user['content'])), 'tool_payload': (payload_start, payload_start + len(payload))}
    if row.get('document'):
        escaped = json.dumps(row['document'], ensure_ascii=False)[1:-1]
        start = payload.index('"text":"') + len('"text":"')
        if payload[start:start + len(escaped)] != escaped:
            raise TaskFatal('Unexpected document escaping')
        spans['document'] = (payload_start + start, payload_start + start + len(escaped))
    return (text, spans)

def serialize(row, cfg, tokenizer):
    validate(cfg)
    text, spans = render(row, tokenizer)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded['input_ids']
    if not ids or ids[0] != tokenizer.bos_token_id:
        raise TaskFatal('Missing DeepSeek BOS')
    if len(ids) > int(cfg['max_input_tokens']):
        raise TaskFatal('Input exceeds max_input_tokens; truncation is forbidden')
    from mechanism.native.F10_activations import resolve_positions
    mapped = {**row, 'reviewed_char_spans': spans}
    positions = resolve_positions(text, encoded['offset_mapping'], mapped)
    return {'text': text, 'input_ids': ids, 'attention_mask': [1] * len(ids), 'positions': positions, 'offset_mapping': encoded['offset_mapping'], **identity(tokenizer)}
