"""Non-scoring diagnostics. Automatic text flags are candidates, not reviews."""
import re
from .common import digest
from .validation import CONFLICT

REVIEW_ISSUES = {'EVALUATION_REFUSAL', 'CONTENT_REFUSAL', 'RUBRIC_FORMAT_ERROR', 'OTHER_UNCLEAR'}
LABELS = {
    'NONE': 'Valid judgment', 'NOT_EVALUATED': 'Not evaluated or input excluded',
    'API_REFUSAL': 'Judge refusal indicated by the API',
    'EVALUATION_REFUSAL_CANDIDATE': 'Possible evaluation refusal; unreviewed',
    'CONTENT_REFUSAL_CANDIDATE': 'Possible content refusal; unreviewed',
    'PARSE_FAILURE_UNREVIEWED': 'Parse failure; raw text review required',
    'EVALUATION_REFUSAL': 'Judge declined to evaluate',
    'CONTENT_REFUSAL': 'Judge declined to provide content',
    'RUBRIC_FORMAT_ERROR': 'Rubric format error; reviewed',
    'OTHER_UNCLEAR': 'Other or unclear; reviewed',
    'RAW_NUMERIC_INVALID': 'Invalid numeric values in raw output', 'OTHER_FAILURE': 'Other judgment failure',
    CONFLICT: 'Conflicting values for a repeated rubric item',
}


def strongreject_diagnostic(result):
    status = result.get('status')
    raw = result.get('raw_judge_output')
    issue, review, evidence = 'OTHER_FAILURE', 'NOT_REQUIRED', None
    if status == 'OK':
        issue = 'NONE' if result.get('analysis_valid') else 'RAW_NUMERIC_INVALID'
        if result.get('analysis_exclusion') == CONFLICT:
            issue = CONFLICT
    elif status == 'JUDGE_REFUSAL':
        issue = 'API_REFUSAL'
    elif status == 'INVALID_JUDGE_OUTPUT':
        issue, review = 'PARSE_FAILURE_UNREVIEWED', 'UNREVIEWED'
        # Opening first-person refusal only; a valid rubric never enters here.
        text = (raw or '').strip()
        match = re.match(r"(?:I(?: am|'m) sorry[,.:!]?\s*)?I\s+(?:cannot|can't|can’t|am unable to|will not|won't)\s+([^\n.!?]+)", text, re.I)
        if match:
            issue = ('EVALUATION_REFUSAL_CANDIDATE' if re.search(r'\bevaluat\w*\b|\bscore\b|\brate\b|\bgrading\b', match.group(1), re.I)
                     else 'CONTENT_REFUSAL_CANDIDATE')
            evidence = match.group(0)
    elif status == 'PENDING' or result.get('api_called') is False:
        issue = 'NOT_EVALUATED'
    return {'judge_output_issue': issue, 'judge_output_label': LABELS[issue],
            'judge_output_review_status': review, 'judge_output_evidence': evidence,
            'raw_judge_output_sha256': digest(raw) if isinstance(raw, str) else None}


def strongreject_counts(results):
    """Latest task results within the caller's cohort, not summed attempts."""
    rows = list(results)
    called = [r for r in rows if r.get('attempt_id') or r.get('raw_judge_output') is not None
              or r.get('http_status') is not None or r.get('api_called') is True]
    confirmed = [r for r in rows if r.get('judge_output_issue') in {'API_REFUSAL', 'EVALUATION_REFUSAL', 'CONTENT_REFUSAL'}]
    candidates = [r for r in rows if r.get('judge_output_issue', '').endswith('_CANDIDATE')]
    n = len(called)
    return {'sr_judge_called_n': n, 'sr_judge_refusal_confirmed_n': len(confirmed),
            'sr_judge_refusal_confirmed_rate': len(confirmed) / n if n else None,
            'sr_judge_refusal_candidate_n': len(candidates),
            'sr_judge_refusal_candidate_rate': len(candidates) / n if n else None,
            'sr_evaluation_refusal_n': sum(r.get('judge_output_issue') == 'EVALUATION_REFUSAL' for r in rows),
            'sr_content_refusal_n': sum(r.get('judge_output_issue') == 'CONTENT_REFUSAL' for r in rows),
            'sr_api_refusal_n': sum(r.get('judge_output_issue') == 'API_REFUSAL' for r in rows),
            'sr_parse_failure_n': sum(r.get('status') == 'INVALID_JUDGE_OUTPUT' for r in rows),
            'sr_unreviewed_n': sum(r.get('judge_output_review_status') == 'UNREVIEWED' for r in rows),
            'sr_reviewed_n': sum(r.get('judge_output_review_status') == 'REVIEWED' for r in rows),
            'sr_rubric_format_error_n': sum(r.get('judge_output_issue') == 'RUBRIC_FORMAT_ERROR' for r in rows),
            'sr_other_unclear_n': sum(r.get('judge_output_issue') == 'OTHER_UNCLEAR' for r in rows),
            'sr_raw_numeric_invalid_n': sum(r.get('raw_numeric_valid') is False for r in rows),
            'sr_conflicting_rubric_values_n': sum(any(i.get('reason') == CONFLICT for i in r.get('raw_numeric_issues', [])) for r in rows)}
