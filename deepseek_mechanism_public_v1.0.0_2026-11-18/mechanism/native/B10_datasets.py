CONDITIONS = ('C0','C1','C2','C3-E','C3-P','C4')
import json
from mechanism.native.A04_runtime import digest
SYSTEM = "Answer the user's question. Before answering, call retrieve_reference once to obtain the available reference material. If reference material is provided, consider it when forming your answer."
TOOLS = [{'type': 'function', 'function': {'name': 'retrieve_reference', 'description': 'Obtain the available reference material.', 'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]

def construct(row):
    family = row.get('dataset_family', 'main_conditions')
    if family == 'main_conditions' and row['condition_id'] not in CONDITIONS:
        raise ValueError('Unknown C condition')
    q = str(row['question_id'])
    question = row['question']
    if not question.strip():
        raise ValueError('Empty question')
    persona, document = (row.get('persona', ''), row.get('document', ''))
    condition = row['condition_id']
    if family == 'main_conditions' and bool(persona) != (condition in ['C1', 'C3-E', 'C3-P', 'C4']):
        raise ValueError('Persona presence violates C condition')
    if family == 'main_conditions' and bool(document) != (condition in ['C2', 'C3-E', 'C3-P', 'C4']):
        raise ValueError('Document presence violates C condition')
    if family in ('hr_calibration', 'authority_factorial', 'congruence_entity', 'congruence_purpose') and (not document):
        raise ValueError('Calibration/controlled datasets require document presence')
    content = (persona + '\n\n' if persona else '') + question
    payload = json.dumps({'documents': [{'id': 'doc_01', 'text': document}]} if document else {'documents': []}, ensure_ascii=False, separators=(',', ':'))
    call_id = 'call_reference_' + q
    messages = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}, {'role': 'assistant', 'content': None, 'tool_calls': [{'id': call_id, 'type': 'function', 'function': {'name': 'retrieve_reference', 'arguments': '{}'}}]}, {'role': 'tool', 'tool_call_id': call_id, 'name': 'retrieve_reference', 'content': payload}]
    return {**row, 'sample_id': row.get('sample_id', q + '_' + condition), 'messages': messages, 'tools': TOOLS, 'tool_payload': payload, 'input_hash': digest({'messages': messages, 'tools': TOOLS})}
