import json

def unique_span(text, fragment):
    if not fragment:
        return None
    start = text.find(fragment)
    if start < 0 or text.find(fragment, start + 1) >= 0:
        raise ValueError('Boundary text absent/ambiguous; supply reviewed character spans')
    return (start, start + len(fragment))

def resolve_positions(text, offsets, row, serialized_fragments=None):
    supplied = row.get('reviewed_char_spans', {})
    user = [m['content'] for m in row['messages'] if m['role'] == 'user'][-1]
    payload = row.get('tool_payload')

    def span(name, fragment):
        s = supplied.get(name)
        if s is not None:
            if len(s) != 2 or not 0 <= s[0] < s[1] <= len(text):
                raise ValueError('Invalid reviewed character span')
            if fragment and text[s[0]:s[1]] != fragment:
                raise ValueError('Reviewed span text mismatch')
            return s
        return unique_span(text, fragment)

    def tokens(s):
        if s is None:
            return None
        indices = [i for i, (a, b) in enumerate(offsets) if b > a and a < s[1] and (b > s[0])]
        if not indices:
            raise ValueError('No token overlaps boundary')
        return indices
    fragments = serialized_fragments or {}
    user = fragments.get('user', user)
    payload = fragments.get('tool_payload', payload)
    u = tokens(span('user', user))
    c = tokens(span('tool_payload', payload)) if payload is not None else None
    doc = row.get('document', '')
    escaped = json.dumps(doc, ensure_ascii=False)[1:-1] if doc else None
    escaped = fragments.get('document', escaped)
    d = tokens(span('document', escaped)) if escaped else None
    if not offsets:
        raise ValueError('Empty encoded input')
    return {'t_U': [u[-1]], 't_D': [d[-1]] if d else None, 't_C': [c[-1]] if c else None, 't_A': [len(offsets) - 1], 'document_mean': d, 'tool_payload_mean': c}
