"""Normalize existing workbooks and per-condition outputs without changing text."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import json
import shutil

from .common import CONDITIONS, blank, condition, digest, read_json, atomic_json

ALIASES = {
    'question_id': ['question_id'],
    'question': ['question', 'original_question'],
    'answer': ['answer', 'llm_answer', 'assistant_response', 'llm_response'],
    'condition': ['condition'],
    'target_model': ['target_model', 'requested_model', 'model'],
    'repeat_index': ['repeat_index', 'repeat'],
    'target_status': ['target_status', 'status'],
    'target_finish_reason': ['target_finish_reason', 'finish_reason'],
    'target_error': ['target_error', 'error', 'error_message'],
    'full_answer_path': ['full_answer_path'],
    'excel_display_modified': ['excel_display_modified'],
}


def target_state(answer, status=None, finish=None, error=None):
    """Only structured metadata is evidence of a block/truncation/error."""
    s, f = str(status or '').strip().upper(), str(finish or '').strip().lower()
    empty = blank(answer)
    state = 'TARGET_COMPLETED'
    eligible, warning = True, None
    if f in {'length', 'max_tokens', 'max_output_tokens'} or s in {'TRUNCATED', 'INCOMPLETE', 'GENERATION_INTERRUPTED'}:
        state, eligible = 'TARGET_INCOMPLETE', False
    elif s in {'TIMEOUT', 'TARGET_TIMEOUT', 'NETWORK_ERROR', 'HTTP_ERROR', 'API_ERROR', 'ERROR', 'FAILED', 'CANCELLED'} or s.startswith(('HTTP_', 'NETWORK_', 'TIMEOUT_')):
        state, eligible = 'TARGET_GENERATION_ERROR', False
    elif f in {'content_filter', 'safety', 'blocked'} or s in {'BLOCKED', 'POLICY_BLOCK', 'CONTENT_FILTER', 'SAFETY_BLOCK'}:
        state, eligible = ('TARGET_BLOCKED' if empty else 'TARGET_BLOCKED_WITH_TEXT'), False
    elif not blank(error) and s not in {'OK', 'SUCCESS', 'COMPLETED', 'DONE', 'REFUSAL_FIELD', 'REFUSAL'}:
        state, eligible = 'TARGET_GENERATION_ERROR', False
    elif s == 'PENDING':
        state, eligible = 'TARGET_PENDING', False
        warning = 'Target generation pending; complete the target run and prepare a new input snapshot.'
    elif empty:
        state, eligible = 'INPUT_EMPTY_RESPONSE', False
        warning = 'Empty response; cause unknown unless provided in target metadata.'
    elif s == 'REFUSAL_FIELD' or (f == 'refusal' and s in {'OK', 'SUCCESS', 'COMPLETED', 'DONE', 'REFUSAL'}):
        state = 'TARGET_REFUSAL_TEXT'  # Evaluate the actual text; do not assign a label here.
    elif f and f not in {'stop', 'eos', 'eos_token', 'end_turn', 'stop_sequence'}:
        state, eligible = 'TARGET_INCOMPLETE', False
    elif s and s not in {'OK', 'SUCCESS', 'COMPLETED', 'DONE', 'REFUSAL'}:
        state, eligible = 'TARGET_STATUS_UNKNOWN', False
        warning = 'Unrecognized explicit target status; review source metadata.'
    elif not f and s not in {'COMPLETED', 'DONE'}:
        state = 'TARGET_COMPLETION_UNKNOWN'
        warning = 'Nonempty response evaluated; completion not independently established.'
    return {'target_state': state, 'target_eligible': eligible, 'target_warning': warning,
            'target_status': status, 'target_finish_reason': finish, 'target_error': error}


def read_table(path, sheet=None):
    path = Path(path)
    if path.suffix.lower() == '.xlsx':
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=False)
        try:
            name = sheet or ('repetition_details' if 'repetition_details' in wb.sheetnames else wb.sheetnames[0])
            ws = wb[name]
            iterator = ws.iter_rows()
            first = next(iterator)
            headers = [str(c.value).strip() if c.value is not None else '' for c in first]
            nonempty = [h for h in headers if h]
            if len(set(nonempty)) != len(nonempty):
                raise ValueError(f'Duplicate headers: {path.name}/{name}')
            records = []
            for index, row in enumerate(iterator, 2):
                if all(blank(c.value) for c in row):
                    continue
                data = {h: c.value for h, c in zip(headers, row) if h}
                formulas = {h for h, c in zip(headers, row) if h and c.data_type == 'f'}
                records.append((index, data, formulas))
            return name, headers, records
        finally:
            wb.close()
    if path.suffix.lower() in {'.csv', '.tsv'}:
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = csv.DictReader(f, delimiter='\t' if path.suffix.lower() == '.tsv' else ',')
            headers = rows.fieldnames or []
            if len(set(headers)) != len(headers):
                raise ValueError('Duplicate CSV/TSV headers')
            return None, headers, [(i, dict(r), set()) for i, r in enumerate(rows, 2)]
    if path.suffix.lower() == '.jsonl':
        rows = [(i, json.loads(line), set()) for i, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1) if line.strip()]
        if any(not isinstance(r, dict) for _, r, _ in rows):
            raise ValueError('JSONL rows must be objects')
        return None, list(dict.fromkeys(k for _, r, _ in rows for k in r)), rows
    raise ValueError('Supported input formats: .xlsx, .csv, .tsv, .jsonl')


def column(headers, field, custom, required=False):
    if field in custom:
        name = custom[field]
        if name not in headers:
            raise ValueError(f'Missing configured column {name!r}')
        return name
    found = [name for name in ALIASES.get(field, [field]) if name in headers]
    if len(found) > 1:
        raise ValueError(f'Ambiguous {field} columns {found}; set columns.{field} explicitly')
    if not found and required:
        raise ValueError(f'Missing {field} column; configure columns explicitly')
    return found[0] if found else None


def inspect_file(path, sheet=None):
    actual_sheet, headers, rows = read_table(path, sheet)
    return {'path': str(Path(path).resolve()), 'sheet': actual_sheet, 'headers': headers, 'rows': len(rows)}


def load_inputs(manifest_path, snapshot_dir=None):
    manifest_path = Path(manifest_path).resolve()
    cfg = read_json(manifest_path)
    strata_columns = cfg.get('strata_columns', [])
    if (not isinstance(strata_columns, list)
            or any(not isinstance(field, str) or not field.strip() for field in strata_columns)
            or len(set(strata_columns)) != len(strata_columns)):
        raise ValueError('strata_columns must be a list of unique nonempty column names')
    required = tuple(condition(c) for c in cfg.get('conditions', CONDITIONS))
    if len(set(required)) != len(required) or not required:
        raise ValueError('conditions must be nonempty and unique')
    records, provenance = [], []
    for file_index, spec in enumerate(cfg['files']):
        path = (manifest_path.parent / spec['path']).resolve()
        filehash = digest(path.read_bytes())
        sheet, headers, rows = read_table(path, spec.get('sheet'))
        missing_strata = [field for field in strata_columns if field not in headers]
        if missing_strata:
            raise ValueError(f'{path.name}: missing configured strata columns: {missing_strata}')
        custom = spec.get('columns', {})
        wide = {condition(c): h for c, h in spec.get('response_columns', {}).items()}
        if wide and any(h not in headers for h in wide.values()):
            raise ValueError('A configured response column is missing')
        columns = {f: column(headers, f, custom, f in {'question_id', 'question'} or f == 'answer' and not wide) for f in ALIASES}
        if not wide and not columns['condition'] and 'condition' not in spec:
            raise ValueError('Each long-format file needs a condition column or file condition')
        snap = None
        if snapshot_dir:
            snap = Path(snapshot_dir) / f'{file_index:03d}_{path.name}'
            snap.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snap)
        provenance.append({'file_index': file_index, 'name': path.name, 'sheet': sheet, 'sha256': filehash,
                           'snapshot': snap.name if snap else None, 'row_count': len(rows)})
        for rownum, row, formulas in rows:
            # JSONL headers are the union of keys; validate each row as well.
            # An explicitly blank cell/null remains allowed.
            missing_strata = [field for field in strata_columns if field not in row]
            if missing_strata:
                raise ValueError(f'{path.name}:{rownum}: missing configured strata fields: {missing_strata}')
            def get(field):
                return row.get(columns[field]) if columns.get(field) else None
            identity_fields = set(columns.values())
            identity_fields.update(wide.values())
            identity_fields.update(strata_columns)
            identity_fields.update(h for mapping in spec.get('condition_metadata', {}).values() for h in mapping.values())
            if formulas & identity_fields:
                raise ValueError(f'{path.name}:{rownum}: formulas in evaluated/identity fields are not allowed')
            question, qid = get('question'), get('question_id')
            if blank(qid) or not isinstance(question, str) or blank(question):
                raise ValueError(f'{path.name}:{rownum}: question_id/question required')
            if not isinstance(qid, (str, int)) or isinstance(qid, bool):
                raise ValueError('question_id must be text or an integer')
            model = spec.get('target_model', cfg.get('target_model')) or get('target_model')
            if blank(model):
                raise ValueError('target_model must be supplied in input manifest or source column')
            if not blank(get('target_model')) and str(model) != str(get('target_model')):
                raise ValueError(f'{path.name}:{rownum}: manifest target_model disagrees with source model column')
            repeat = get('repeat_index')
            repeat = spec.get('repeat_index', cfg.get('repeat_index', 1)) if blank(repeat) else repeat
            if isinstance(repeat, bool) or str(repeat) not in {str(i) for i in range(1, 10001)}:
                raise ValueError('repeat_index must be a positive integer <=10000')
            group = [str(model), str(spec.get('run_id', cfg.get('run_id', 'default'))), int(repeat), str(qid)]
            if not wide and spec.get('condition') and not blank(get('condition')) and condition(spec['condition']) != condition(get('condition')):
                raise ValueError(f'{path.name}:{rownum}: manifest condition disagrees with source condition column')
            entries = [(c, row.get(h)) for c, h in wide.items()] if wide else [(condition(spec.get('condition') or get('condition')), get('answer'))]
            for cond, answer in entries:
                if cond not in required:
                    raise ValueError(f'Unexpected condition {cond}; check manifest conditions')
                if answer is None:
                    answer = ''
                if not isinstance(answer, str):
                    raise ValueError(f'{path.name}:{rownum}: response must be text')
                # Per-condition metadata for wide files must be explicitly mapped.
                meta_map = {condition(c): m for c, m in spec.get('condition_metadata', {}).items()}.get(cond, {}) if wide else {}
                def meta(field):
                    if wide:
                        name = meta_map.get(field)
                        if name and name not in headers:
                            raise ValueError(f'Missing metadata column {name}')
                        return row.get(name) if name else None
                    return get(field)
                full_path, modified = meta('full_answer_path'), meta('excel_display_modified')
                display_flag = str(modified).strip().lower()
                explicitly_unmodified = modified is False or display_flag in {'false', '0', 'no', 'n'}
                if not blank(modified) and display_flag not in {'true', '1', 'yes', 'y', 'false', '0', 'no', 'n'}:
                    raise ValueError(f'Unknown excel_display_modified flag: {modified!r}')
                modified = modified is True or display_flag in {'true', '1', 'yes', 'y'}
                sidecar = None
                if full_path:
                    source = (path.parent / str(full_path)).resolve()
                    if not source.is_relative_to(path.parent):
                        raise ValueError('full_answer_path must stay inside the input file directory')
                    if source.is_file():
                        raw = source.read_bytes()
                        answer = raw.decode('utf-8-sig')
                        sidecar = {'name': str(source.relative_to(path.parent)), 'sha256': digest(raw), 'available': True}
                        if snapshot_dir:
                            dest = Path(snapshot_dir) / 'sidecars' / (digest(raw) + '.txt')
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(raw)
                            sidecar['snapshot'] = str(dest.relative_to(snapshot_dir))
                    elif explicitly_unmodified and len(answer) < 32767:
                        sidecar = {'name': str(full_path), 'available': False,
                                   'warning': 'Source marks display unmodified; using the original cell text. Referenced sidecar was not supplied.'}
                    else:
                        raise ValueError(f'{path.name}:{rownum}: full answer sidecar missing and original cell completeness is not established')
                elif modified or path.suffix.lower() == '.xlsx' and len(answer) >= 32767:
                    raise ValueError(f'{path.name}:{rownum}: displayed response may be shortened; full_answer_path required')
                state = target_state(answer, meta('target_status'), meta('target_finish_reason'), meta('target_error'))
                record = {'task_id': digest([*group, cond]), 'group_id': digest(group), 'target_model': group[0],
                          'target_run': group[1], 'repeat_index': group[2], 'question_id': group[3], 'condition': cond,
                          'question': question, 'answer': answer, 'question_sha256': digest(question), 'answer_sha256': digest(answer),
                          **state, 'source_file_index': file_index, 'source_file': path.name, 'source_sheet': sheet,
                          'source_row': rownum, 'source_sidecar': sidecar,
                          'strata': {field: row[field] for field in strata_columns}}
                # Imported target metadata is for display. It must not split the original judge cohorts.
                if cfg.get("target_metadata") is True:
                    record["target_metadata"] = {field: row[field] for field in
                                                 ("depth1", "depth2", "domain", "context_group", "target_attempt_id")}
                records.append(record)
    by_group, seen = defaultdict(dict), set()
    for row in records:
        if row['task_id'] in seen:
            raise ValueError(f"Duplicate target/run/repeat/question/condition: {row['question_id']}/{row['condition']}")
        seen.add(row['task_id'])
        by_group[row['group_id']][row['condition']] = row
    for rows in by_group.values():
        if set(rows) != set(required):
            raise ValueError(f"Condition ID alignment failed for {next(iter(rows.values()))['question_id']}: {sorted(rows)}; expected {required}")
        if len({r['question'] for r in rows.values()}) != 1:
            raise ValueError('Original question differs across conditions for the same identity')
        if len({json.dumps(r['strata'], sort_keys=True, default=str) for r in rows.values()}) != 1:
            raise ValueError('Stratum values differ across conditions for the same identity')
    if not records:
        raise ValueError('No input rows')
    positions = {group_id: i for i, group_id in enumerate(by_group)}
    records.sort(key=lambda r: (positions[r['group_id']], required.index(r['condition'])))
    return records, {'manifest': cfg, 'source_files': provenance, 'conditions': list(required),
                     'group_count': len(by_group), 'task_count': len(records)}
