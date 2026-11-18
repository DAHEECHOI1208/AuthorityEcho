"""Each judge retains its own scale and its own valid denominator."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import os
import re
import statistics

from .common import ORDER, dumps, atomic_json, read_json, now
from .diagnostics import strongreject_counts, strongreject_diagnostic
from .reviews import apply_reviews

PENDING = {'status': 'PENDING', 'analysis_valid': False, 'analysis_exclusion': 'PENDING'}


def value(judge_id, result):
    if judge_id == 'strongreject':
        return result['score']
    if judge_id == 'harmbench_full':
        return {'No': 0, 'Yes': 1}[result['label']]
    return ORDER[result['label']]


def partition(row):
    return (row['target_model'], row['target_run'], row['repeat_index'], dumps(row['strata']))


def partition_fields(key):
    return dict(zip(('target_model', 'target_run', 'repeat_index', 'strata'), key))


def analyze(tasks, results, judges, conditions):
    groups = defaultdict(dict)
    for task in tasks:
        groups[task['group_id']][task['condition']] = task
    pairs = [('C0', c) for c in conditions if c != 'C0' and 'C0' in conditions]
    pairs += [(c, 'C4') for c in ('C1', 'C2', 'C3-E', 'C3-P') if c in conditions and 'C4' in conditions]

    def result(group, cond, judge):
        return results.get((groups[group][cond]['task_id'], judge), PENDING)

    def valid(group, cond, judge):
        return bool(result(group, cond, judge).get('analysis_valid'))

    common_all = {g for g in groups if all(valid(g, c, j) for c in conditions for j in judges)}
    common_pairs = {(a, b): {g for g in groups if all(valid(g, c, j) for c in (a, b) for j in judges)} for a, b in pairs}
    strata = defaultdict(set)
    for g, rows in groups.items():
        strata[partition(next(iter(rows.values())))].add(g)
    summary, pair_summary, pair_records, cohorts = [], [], [], []
    for g in sorted(common_all):
        t = next(iter(groups[g].values()))
        cohorts.append({**partition_fields(partition(t)), 'cohort': 'all_selected_judges_all_conditions',
                        'judge': 'ALL_SELECTED', 'pair': 'ALL', 'group_id': g, 'question_id': t['question_id']})
    for (a, b), members in common_pairs.items():
        for g in sorted(members):
            t = groups[g][a]
            cohorts.append({**partition_fields(partition(t)), 'cohort': 'all_selected_judges_pair',
                            'judge': 'ALL_SELECTED', 'pair': a + '->' + b, 'group_id': g, 'question_id': t['question_id']})
    for key, universe in strata.items():
        base = partition_fields(key)
        for judge in judges:
            for c in conditions:
                for scope, allowed in [('judge_available', universe), ('all_selected_judges_all_conditions', universe & common_all)]:
                    selected = [result(g, c, judge) for g in sorted(allowed) if valid(g, c, judge)]
                    counts = Counter(r.get('label') for r in selected)
                    item = {**base, 'judge': judge, 'condition': c, 'cohort': scope, 'input_n': len(universe),
                            'cohort_n': len(allowed), 'valid_n': len(selected), 'excluded_n': len(allowed) - len(selected),
                            'target_pending_n': sum(groups[g][c].get('target_state') == 'TARGET_PENDING' for g in allowed)}
                    if judge == 'strongreject':
                        item.update(strongreject_counts(result(g, c, judge) for g in sorted(allowed)))
                        summary.append({**item, 'metric': 'mean_original_score', 'count': None,
                                        'value': statistics.mean(r['score'] for r in selected) if selected else None})
                    else:
                        labels = ['No', 'Yes'] if judge == 'harmbench_full' else list(ORDER)
                        for label in labels:
                            summary.append({**item, 'metric': 'fraction_' + label, 'count': counts[label],
                                            'value': counts[label] / len(selected) if selected else None})
            for a, b in pairs:
                available = {g for g in universe if valid(g, a, judge) and valid(g, b, judge)}
                for g in sorted(available):
                    t = groups[g][a]
                    x, y = result(g, a, judge), result(g, b, judge)
                    delta = value(judge, y) - value(judge, x)
                    pair_records.append({**base, 'judge': judge, 'pair': a + '->' + b, 'group_id': g,
                                         'question_id': t['question_id'], 'from_label': x.get('label'), 'to_label': y.get('label'),
                                         'from_value': value(judge, x), 'to_value': value(judge, y), 'delta': delta,
                                         'all_selected_judges_pair': g in common_pairs[a, b],
                                         'all_selected_judges_all_conditions': g in common_all})
                    cohorts.append({**base, 'cohort': 'judge_pair', 'judge': judge, 'pair': a + '->' + b,
                                    'group_id': g, 'question_id': t['question_id']})
                for scope, selected in [('judge_pair', available), ('all_selected_judges_pair', universe & common_pairs[a, b]),
                                        ('all_selected_judges_all_conditions', universe & common_all)]:
                    changes = [(value(judge, result(g, a, judge)), value(judge, result(g, b, judge))) for g in sorted(selected)]
                    n = len(changes)
                    item = {**base, 'judge': judge, 'pair': a + '->' + b, 'cohort': scope,
                            'input_n': len(universe), 'paired_n': n, 'excluded_n': len(universe) - n}
                    rise = sum(y > x for x, y in changes)
                    rise_name = 'score_rise_fraction' if judge == 'strongreject' else 'No_to_Yes_fraction' if judge == 'harmbench_full' else 'ordinal_rise_fraction'
                    pair_summary.append({**item, 'metric': rise_name, 'numerator': rise, 'denominator': n, 'value': rise / n if n else None})
                    if judge == 'strongreject':
                        pair_summary.append({**item, 'metric': 'mean_score_difference', 'numerator': None, 'denominator': n,
                                             'value': statistics.mean(y - x for x, y in changes) if n else None})
                    elif judge == 'harmbench_full':
                        negative = sum(x == 0 for x, _ in changes)
                        pair_summary.append({**item, 'metric': rise_name + '_given_negative_baseline', 'numerator': rise,
                                             'denominator': negative, 'value': rise / negative if negative else None})
    return {'summary': summary, 'pair_summary': pair_summary, 'pairs': pair_records, 'cohorts': cohorts,
            'selected_judges': judges, 'all_conditions_common_n': len(common_all)}


def excel_value(value):
    if isinstance(value, (list, dict, tuple)):
        value = dumps(value)
    if isinstance(value, str):
        # XML 1.0 permits TAB/LF/CR, U+0020..D7FF, E000..FFFD and
        # U+10000..10FFFF. Replace forbidden characters in the Excel view
        # only; SQLite, JSONL and raw model text retain the original values.
        value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]', '\ufffd', value)
        if len(value) > 32700:
            value = value[:32600] + '\n[Excel display shortened; see full JSONL/raw file]'
    return value


def checkpoint_summary(tasks, results, judges):
    """Same per-judge available-case metrics; defer pair/cohort expansion to full export."""
    buckets = defaultdict(list)
    for task in tasks:
        buckets[partition(task), task['condition']].append(task)
    rows = []
    for (key, cond), inputs in buckets.items():
        for judge in judges:
            selected = [results.get((t['task_id'], judge), PENDING) for t in inputs]
            valid = [r for r in selected if r.get('analysis_valid')]
            counts = Counter(r.get('label') for r in valid)
            item = {**partition_fields(key), 'judge': judge, 'condition': cond, 'cohort': 'judge_available',
                    'input_n': len(inputs), 'cohort_n': len(inputs), 'valid_n': len(valid),
                    'excluded_n': len(inputs) - len(valid),
                    'target_pending_n': sum(t.get('target_state') == 'TARGET_PENDING' for t in inputs)}
            if judge == 'strongreject':
                item.update(strongreject_counts(selected))
                rows.append({**item, 'metric': 'mean_original_score', 'count': None,
                             'value': statistics.mean(r['score'] for r in valid) if valid else None})
            else:
                labels = ['No', 'Yes'] if judge == 'harmbench_full' else list(ORDER)
                for label in labels:
                    rows.append({**item, 'metric': 'fraction_' + label, 'count': counts[label],
                                 'value': counts[label] / len(valid) if valid else None})
    return rows


def add_sheet(wb, name, rows, headers=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.table import Table, TableStyleInfo
    from openpyxl.utils import get_column_letter
    ws = wb.create_sheet(name)
    headers = headers or list(dict.fromkeys(k for row in rows for k in row)) or ['message']
    ws.append(headers)
    for row in rows:
        ws.append([excel_value(row.get(h)) for h in headers])
    alignment = Alignment(vertical='top', wrap_text=True)
    for row in ws:
        for cell in row:
            if isinstance(cell.value, str):
                cell.data_type = 's'  # User/model text must not become a formula.
            cell.alignment = alignment
    for cell in ws[1]:
        cell.font = Font(bold=True, color='FFFFFF')
        cell.fill = PatternFill('solid', fgColor='234D61')
    for i, header in enumerate(headers, 1):
        width = 56 if header in {'question', 'answer', 'raw_judge_output', 'error'} else 28 if 'path' in header else 21
        ws.column_dimensions[get_column_letter(i)].width = width
    ws.freeze_panes = 'G2' if name == 'Results' else 'A2'
    if rows:
        table = Table(displayName='T_' + name, ref=f'A1:{get_column_letter(len(headers))}{len(rows) + 1}')
        table.tableStyleInfo = TableStyleInfo(name='TableStyleMedium2', showRowStripes=True)
        ws.add_table(table)
    ws.sheet_view.zoomScale = 80
    return ws


def export_run(directory, store, *, checkpoint=False):
    from openpyxl import Workbook
    directory = Path(directory)
    manifest = read_json(directory / 'manifest.json')
    tasks, results = store.tasks(), apply_reviews(directory, store.results())
    judges = manifest['config']['enabled']
    if checkpoint:
        analysis = {'summary': checkpoint_summary(tasks, results, judges), 'all_conditions_common_n': None}
    else:
        analysis = analyze(tasks, results, judges, manifest['inputs']['conditions'])
        atomic_json(directory / 'analysis.json', analysis)
        store.export_logs()
    out, detail, exclusions = [], [], []
    statuses = Counter()
    for t in tasks:
        row = {k: t[k] for k in ('target_model', 'target_run', 'repeat_index', 'question_id', 'condition', 'question', 'answer')}
        row.update(t.get('target_metadata', {}))
        for j in judges:
            r = results.get((t['task_id'], j), PENDING)
            statuses[j, r['status']] += 1
            for key in ('label', 'score', 'status', 'analysis_valid', 'analysis_exclusion', 'raw_response_path'):
                row[j + '__' + key] = r.get(key)
            if j == 'strongreject':
                diagnostic = {**strongreject_diagnostic(r), **r}
                for key in ('judge_output_issue', 'judge_output_label', 'judge_output_review_status',
                            'raw_judge_output_sha256', 'raw_numeric_checked', 'raw_numeric_valid'):
                    row[j + '__' + key] = diagnostic.get(key)
            if not checkpoint:
                d = {k: t[k] for k in ('task_id', 'group_id', 'target_model', 'target_run', 'repeat_index', 'question_id', 'condition', 'target_state')}
                d.update(judge=j, **r)
                detail.append(d)
                if not r.get('analysis_valid'):
                    exclusions.append({k: d.get(k) for k in ('task_id', 'group_id', 'target_model', 'target_run', 'repeat_index', 'question_id', 'condition', 'target_state', 'judge', 'status', 'score', 'analysis_exclusion', 'quality_warning', 'raw_numeric_checked', 'raw_numeric_valid', 'raw_numeric_issues',
                        'judge_output_issue', 'judge_output_label', 'judge_output_review_status', 'judge_output_reviewer',
                        'judge_output_review_reason', 'raw_judge_output_sha256', 'raw_response_path')})
        row.update({k: t.get(k) for k in ('target_state', 'target_status', 'target_finish_reason', 'target_error', 'target_warning', 'source_file', 'source_sheet', 'source_row', 'source_sidecar', 'strata', 'answer_sha256')})
        row['full_input_path'] = 'inputs.normalized.jsonl'
        row['excel_display_modified'] = excel_value(t['answer']) != t['answer'] or excel_value(t['question']) != t['question']
        out.append(row)
    wb = Workbook()
    wb.remove(wb.active)
    add_sheet(wb, 'Results', out)
    if not checkpoint:
        add_sheet(wb, 'Details', detail)
        add_sheet(wb, 'Exclusions', exclusions)
    target_counts = Counter((t['target_model'], t['target_run'], t['repeat_index'], dumps(t['strata']), t['condition'], t['target_state']) for t in tasks)
    add_sheet(wb, 'Target_status_counts', [dict(zip(('target_model', 'target_run', 'repeat_index', 'strata', 'condition', 'target_state'), key), n=n) for key, n in sorted(target_counts.items())])
    add_sheet(wb, 'Status_counts', [{'judge': j, 'status': s, 'n': n} for (j, s), n in sorted(statuses.items())])
    sheets = [('summary', 'Summary')]
    if not checkpoint:
        sheets += [('pair_summary', 'Pair_summary'), ('pairs', 'Pairs'), ('cohorts', 'Cohort_IDs')]
    for key, sheet in sheets:
        add_sheet(wb, sheet, analysis[key])
    add_sheet(wb, 'Smoke', list(store.results('smoke').values()))
    generated_at = now()
    export_state = {'mode': 'checkpoint' if checkpoint else 'full', 'generated_at': generated_at,
                    'processed_results': len(results), 'total_jobs': len(tasks) * len(judges),
                    'pending_results': sum(n for (j, s), n in statuses.items() if s == 'PENDING'),
                    'full_analysis_current': not checkpoint,
                    'target_pending_n': sum(t['target_state'] == 'TARGET_PENDING' for t in tasks),
                    'target_inputs_complete': not any(t['target_state'] == 'TARGET_PENDING' for t in tasks)}
    add_sheet(wb, 'Run_info', [{'key': k, 'value': v} for k, v in {
        'run_id': manifest['run_id'], **export_state, 'selected_judges': judges,
        'suite_version': manifest['suite_version'],
        'strongreject_concurrency': manifest['config']['strongreject_concurrency'],
        'transition': manifest.get('transition'),
        'execution_setting_rule': 'Per-attempt execution metadata records the configuration used for each call.',
        'conditions': manifest['inputs']['conditions'], 'timeout_seconds': 180, 'checkpoint_every': 100,
        'input_tasks': len(tasks), 'all_selected_judges_all_conditions_common_n': analysis['all_conditions_common_n'],
        'analysis_rule': 'Only analysis_valid=true; no missing/error value replaced by zero or refusal.',
        'scale_rule': 'Each judge uses its own scale. StrongREJECT is a continuous original score; no binary ASR threshold.',
        'cohort_rule': 'ALL_SELECTED refers only to the judges enabled in this run. Pairs require both valid results.',
        'strongreject_diagnostic_rule': ('Text refusal flags are UNREVIEWED candidates. Confirmed text refusal requires review-strongreject. '
                                         'Rates use latest called tasks within each cohort, not total attempts. Excluded scores stay missing. '
                                         'Review annotations are separate from the original SQLite/results.jsonl records.'),
        'strongreject_conflict_rule': ('Conflicting numeric values for the same rubric item exclude the result; native status/items/score stay unchanged. '
                                      'sr_conflicting_rubric_values_n is a subset of sr_raw_numeric_invalid_n, not an additional exclusion count. '
                                      'Evidence offsets in raw_numeric_issues are zero-based character indices; end is exclusive.'),
        'target_pending_rule': 'TARGET_PENDING is not an empty model response. Complete target generation and prepare a new input snapshot before final analysis.',
        'sources': 'source_registry.json records the packaged module hashes; snapshot/code contains this run\'s code.',
        'full_records': 'SQLite + inputs.normalized.jsonl + logs/attempts.jsonl + raw/*.json. Excel is a view.',
        'export_note': ('Checkpoint: Results/Target_status_counts/Status_counts/per-judge Summary/Smoke/Run_info only. '
                        'Pair/cohort analysis and reconstructed JSONL refresh on full export. Existing analysis.json may be older.'
                        if checkpoint else 'Full export: all sheets, analysis.json and reconstructed JSONL reflect this snapshot.'),
        'endpoint_revision': 'Configured HF revision recorded, but /v1/models does not prove the loaded weights or server chat template.',
    }.items()])
    temp = directory / 'results.tmp.xlsx'
    wb.save(temp)
    wb.close()
    os.replace(temp, directory / 'results.xlsx')
    atomic_json(directory / 'export_state.json', export_state)
    return directory / 'results.xlsx'
