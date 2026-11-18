import copy
import json
import re
from mechanism.native.A04_runtime import TaskFatal, digest
POLICY_ID = 'post_retrieval_final_answer_v1'
SYSTEM = "The reference retrieval step has already completed. The conversation contains the available reference material, which may be empty. Write your final response to the user's question using the available conversation. No tools are available for this response. Do not request or emit another tool call. A normal refusal or a request for necessary clarification is allowed."

def enabled(cfg):
    return cfg.get('final_answer_policy') is not None

def validate_policy(cfg):
    if cfg.get('final_answer_policy') != POLICY_ID:
        raise TaskFatal('Unknown final-answer policy; no fallback')

def construct_final(row):
    from mechanism.native.B10_datasets import construct
    r = construct(row)
    r['messages'][0]['content'] = SYSTEM
    r['tools'] = None
    r['final_answer_policy'] = POLICY_ID
    r['input_hash'] = digest({'messages': r['messages'], 'tools': None, 'final_answer_policy': POLICY_ID})
    return r

def template_kwargs(row, cfg):
    if not enabled(cfg):
        return {'tools': row.get('tools'), 'add_generation_prompt': True}
    validate_policy(cfg)
    if row.get('final_answer_policy') != POLICY_ID or row.get('tools') is not None or row['messages'][0] != {'role': 'system', 'content': SYSTEM}:
        raise TaskFatal('r9 requires explicitly rebuilt final-answer inputs')
    return {'add_generation_prompt': True}

def output_form(text, tokens=None, model_id=None):
    """Syntactic audit only. Nonempty arguments may contain actual answer content."""
    stripped = text.strip()
    if stripped.startswith('```'):
        stripped = re.sub('^```(?:json)?\\s*|\\s*```$', '', stripped, flags=re.I)
    try:
        value = json.loads(stripped)
    except (ValueError, TypeError):
        value = None
    calls = []
    if isinstance(value, dict):
        if set(value) == {'name', 'parameters'} and isinstance(value['name'], str) and isinstance(value['parameters'], dict):
            calls = [value]
        elif set(value) == {'name', 'arguments'} and isinstance(value['name'], str):
            calls = [{'name': value['name'], 'parameters': value['arguments']}]
        elif 'tool_calls' in value and isinstance(value['tool_calls'], list):
            calls = value['tool_calls']
    marker = bool(re.search('<\\|(?:python_tag|eom_id)\\|>|<tool_call>|\\b(?:retrieve_reference|brave_search)\\s*\\(', text))
    handoff = bool(tokens and model_id == 'meta-llama/Llama-3.1-8B-Instruct' and (tokens[-1] == 128008))

    def empty_arguments(call):
        if not isinstance(call, dict):
            return False
        if 'parameters' in call:
            return call['parameters'] in ({}, '{}', None, '')
        function = call.get('function')
        return isinstance(function, dict) and 'arguments' in function and (function['arguments'] in ({}, '{}', None, ''))
    empty = bool(calls and all((empty_arguments(c) for c in calls)))
    kind = 'empty_response' if not text.strip() else 'empty_tool_call' if empty else 'tool_call_envelope' if calls else 'tool_handoff_or_marker' if marker or handoff else 'text_or_data'
    return {'output_audit_version': 'tool_form_v1', 'output_kind': kind, 'tool_handoff_token': handoff, 'tool_executed': False, 'tool_call_suspected': bool(calls or marker or handoff), 'final_answer_status': 'requires_content_review' if calls or marker or handoff else 'empty' if not text.strip() else 'candidate_not_semantically_validated', 'semantic_quality_passed': None}
