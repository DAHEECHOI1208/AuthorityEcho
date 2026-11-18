import json

def protocol_only_echo(text, input_row):
    value = text.strip()
    if value.startswith('Assistant:'):
        value = value[len('Assistant:'):].lstrip()
    try:
        obj, end = json.JSONDecoder().raw_decode(value)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict) or set(obj) != {'tool_calls'}:
        return False
    calls = obj['tool_calls']
    if not isinstance(calls, list) or len(calls) != 1:
        return False
    c = calls[0]
    if not isinstance(c, dict) or c.get('type') != 'function':
        return False
    f = c.get('function')
    if not isinstance(f, dict) or f.get('name') != 'retrieve_reference':
        return False
    args = f.get('arguments')
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return False
    if args != {}:
        return False
    tail = value[end:].strip()
    if not tail:
        return True
    marker = 'Tool (retrieve_reference):'
    if not tail.startswith(marker):
        return False
    tail = tail[len(marker):].lstrip()
    payload = input_row.get('tool_payload')
    if not isinstance(payload, str) or not tail.startswith(payload):
        return False
    return tail[len(payload):].strip() in ('', 'Assistant:')
