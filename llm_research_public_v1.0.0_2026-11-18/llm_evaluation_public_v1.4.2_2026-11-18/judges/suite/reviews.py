"""Explicit, raw-text-bound diagnostic reviews; never rewrite judge results."""
from pathlib import Path
import json
from .common import atomic_json, digest, now, read_json
from .diagnostics import LABELS, REVIEW_ISSUES, strongreject_diagnostic


def validate_review(row, results):
    required = ('task_id', 'raw_judge_output_sha256', 'issue', 'reviewer', 'reason')
    if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
        raise ValueError('Each review needs nonempty task_id/raw_judge_output_sha256/issue/reviewer/reason')
    if row['issue'] not in REVIEW_ISSUES:
        raise ValueError('Unknown review issue: ' + row['issue'])
    r = results.get((row['task_id'], 'strongreject'))
    if not r or r.get('status') != 'INVALID_JUDGE_OUTPUT' or not isinstance(r.get('raw_judge_output'), str):
        raise ValueError('Review requires an existing StrongREJECT INVALID_JUDGE_OUTPUT with full raw text')
    if digest(r['raw_judge_output']) != row['raw_judge_output_sha256']:
        raise ValueError('Review raw text hash mismatch; inspect the current result')
    return {k: row[k] for k in required}


def save_reviews(directory, review_file, results):
    incoming = [json.loads(line) for line in Path(review_file).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    checked = [validate_review(row, results) for row in incoming]
    if not checked or len({r['task_id'] for r in checked}) != len(checked):
        raise ValueError('Review file must be nonempty with unique task_id values')
    path = Path(directory) / 'strongreject_reviews.json'
    ledger = read_json(path) if path.exists() else {'schema_version': 1, 'history': []}
    for row in checked:
        ledger['history'].append({**row, 'reviewed_at': now()})
    atomic_json(path, ledger)
    return len(checked)


def apply_reviews(directory, results):
    decorated = {key: dict(value) for key, value in results.items()}
    for (task, judge), r in decorated.items():
        if judge == 'strongreject':
            r.update(strongreject_diagnostic(r))
    path = Path(directory) / 'strongreject_reviews.json'
    if not path.exists():
        return decorated
    latest = {r['task_id']: r for r in read_json(path)['history']}
    for task, review in latest.items():
        r = decorated.get((task, 'strongreject'))
        if r is None:
            continue
        try:
            validate_review(review, decorated)
        except ValueError:
            r['judge_output_review_stale'] = True
            continue  # A later API attempt must not inherit an old text review.
        r.update(judge_output_issue=review['issue'], judge_output_label=LABELS[review['issue']],
                 judge_output_review_status='REVIEWED', judge_output_reviewer=review['reviewer'],
                 judge_output_review_reason=review['reason'], judge_output_reviewed_at=review['reviewed_at'])
    return decorated
