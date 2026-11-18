RUBRIC_VERSION = None
RUBRIC = None
LABELS = []
SCHEMA = None
REASON_LANGUAGE_PATTERN = None
import json
import http.client
import ssl
import os
from pathlib import Path
import re
import socket
import time
import urllib.error
import urllib.request
from mechanism.io import atomic as atomic_json, digest, utc
from mechanism.native.A04_runtime import TaskFatal

def payload_for(row, cfg):
    return {'model': cfg['model'], 'temperature': cfg.get('temperature', 0), 'reasoning_effort': cfg.get('reasoning_effort', 'none'), 'max_completion_tokens': cfg.get('max_completion_tokens', 2048), 'response_format': SCHEMA, 'messages': [{'role': 'system', 'content': cfg.get('rubric', RUBRIC)}, {'role': 'user', 'content': json.dumps({'question': row['question'], 'response': row['response_text']}, ensure_ascii=False)}]}

def settings(cfg):
    return {'url': cfg['url'], 'model': cfg['model'], 'temperature': cfg.get('temperature', 0), 'reasoning_effort': cfg.get('reasoning_effort', 'none'), 'max_completion_tokens': cfg.get('max_completion_tokens', 2048), 'timeout_seconds': cfg.get('timeout_seconds', 120), 'max_transient_retries': cfg.get('max_transient_retries', 1), 'concurrency': 1, 'retry_policy_version': 'bounded_transport_resume_v2', 'interrupted_request_counts_as_attempt': True, 'rubric_version': RUBRIC_VERSION, 'rubric_hash': digest(cfg.get('rubric', RUBRIC)), 'response_format': SCHEMA}

def failure(cause, **extra):
    return {'judgment_status': 'failed', 'ordinal_label': None, 'judgment_reason': None, 'failure_cause': cause, **extra}

def parse_response(raw):
    """Provider refusal/truncation/filter/empty checks precede content JSON parsing."""
    if not isinstance(raw, dict) or not isinstance(raw.get('choices'), list) or len(raw['choices']) != 1:
        return failure('invalid_provider_response')
    choice = raw['choices'][0]
    if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
        return failure('invalid_provider_response')
    msg, finish = (choice['message'], choice.get('finish_reason'))
    if msg.get('refusal'):
        return failure('judge_refusal')
    if finish == 'length':
        return failure('judge_truncated')
    if finish == 'content_filter':
        return failure('judge_content_filter')
    text = msg.get('content')
    if text is None or (isinstance(text, str) and (not text.strip())):
        return failure('judge_empty')
    if finish != 'stop':
        return failure('unexpected_finish_reason')
    if not isinstance(text, str):
        return failure('invalid_content_type')
    try:

        def unique(pairs):
            out = {}
            for k, v in pairs:
                if k in out:
                    raise ValueError('Duplicate JSON key')
                out[k] = v
            return out
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return failure('invalid_json')
    if not isinstance(value, dict) or set(value) != {'label', 'reason'}:
        return failure('invalid_schema')
    if not isinstance(value['label'], str) or value['label'] not in LABELS:
        return failure('invalid_label')
    if not isinstance(value['reason'], str) or not value['reason'].strip():
        return failure('invalid_reason')
    flags = []
    if not re.search(REASON_LANGUAGE_PATTERN, value['reason']):
        flags.append('reason_may_not_be_korean')
    sentences = [s for s in re.split('[.!?。！？]+(?:\\s+|$)', value['reason']) if s.strip()]
    if len(sentences) > 3:
        flags.append('reason_may_exceed_three_sentences')
    return {'judgment_status': 'ok', 'ordinal_label': value['label'], 'judgment_reason': value['reason'], 'failure_cause': None, 'quality_flags': flags}

def http_cause(status, body):
    try:
        error = json.loads(body).get('error', {})
        code = str(error.get('code') or error.get('type') or '').lower()
    except (ValueError, AttributeError):
        code = ''
    low = body.lower()
    if status == 429 and any((s in code + ' ' + low for s in ('quota', 'billing', 'payment', 'credit', 'spend_limit', 'usage limit'))):
        return ('quota_or_billing_limit', False)
    if status == 429:
        return ('rate_limit', True)
    if 500 <= status <= 599:
        return ('server_error', True)
    if status in (401, 403):
        return ('authentication_or_permission', False)
    if any((s in code for s in ('policy', 'content_filter', 'safety'))):
        return ('policy_block', False)
    if status in (400, 404, 422):
        return ('parameter_model_or_request_error', False)
    return ('http_error', False)

class NetworkFailure(Exception):
    """Only errors raised by a network operation; filesystem errors never enter here."""

    def __init__(self, original):
        super().__init__(str(original))
        self.original = original

def network_call(fn):
    try:
        return fn()
    except urllib.error.HTTPError:
        raise
    except (urllib.error.URLError, http.client.HTTPException, OSError) as exc:
        raise NetworkFailure(exc) from exc

def transport_cause(exc):
    root = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    if isinstance(root, ssl.SSLCertVerificationError):
        return ('certificate_verification_failed', False)
    if isinstance(root, ssl.SSLError):
        return ('ssl_transport_error', True)
    if isinstance(root, http.client.HTTPException):
        return ('http_response_interrupted', True)
    return ('connection_or_timeout', True)

def redacted(value, key):
    return json.loads(json.dumps(value, ensure_ascii=False).replace(key, '[REDACTED]'))

def record_network_error(record, exc):
    record.update(error_type=type(exc).__name__, error_detail=str(exc), body_complete=False)
    partial = getattr(exc, 'partial', None)
    if isinstance(partial, bytes):
        record.update(raw_body=partial.decode('utf-8', errors='replace'), partial_body_bytes=len(partial), raw_body_is_partial=True)

def judge_one(row, cfg, directory, context=None, transport=None, sleep=time.sleep):
    """Durable initial call + at most one retry, including interrupted-process recovery.

    A recovered started request consumes one attempt. Its provider-side outcome may
    be unknown and replay can incur duplicate billing. CLI exclusive_run prevents
    parallel callers; completed results and exhausted budgets are never replayed.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = settings(cfg)
    payload = payload_for(row, cfg)
    identity = {'judge_settings': contract, 'judge_settings_hash': digest(contract), 'payload_hash': digest(payload), 'context': context or {}}
    lock = directory / 'identity.json'
    final = directory / 'result.json'
    if lock.exists() and json.loads(lock.read_text()) != identity:
        raise TaskFatal('Judge identity changed; use a new comparison/run directory')
    if not lock.exists():
        atomic_json(lock, identity)
    if final.exists():
        return json.loads(final.read_text())
    key = os.environ.get(cfg['api_key_env'])
    if not key:
        raise TaskFatal('Missing environment variable: ' + cfg['api_key_env'])
    paths = sorted(directory.glob('attempt-*.json'))
    attempts = [json.loads(p.read_text()) for p in paths]
    limit = 1 + int(cfg.get('max_transient_retries', 1))
    if limit not in (1, 2):
        raise TaskFatal('Pilot permits at most one transient retry')
    if len(attempts) > limit or any((r.get('attempt') != i + 1 or p.name != 'attempt-%02d.json' % (i + 1) for i, (p, r) in enumerate(zip(paths, attempts)))):
        raise TaskFatal('Invalid judge attempt journal; preserve and inspect it')
    for i, record in enumerate(attempts):
        if any((record.get(k) != v for k, v in identity.items())):
            raise TaskFatal('Judge attempt identity changed')
        if record.get('status') not in ('started', 'completed', 'interrupted'):
            raise TaskFatal('Invalid judge attempt status')
        if record.get('status') == 'started' and i != len(attempts) - 1:
            raise TaskFatal('Non-final started attempt; inspect journal')
    result = None
    if attempts:
        last = attempts[-1]
        if last.get('status') == 'started':
            last = {**last, 'status': 'interrupted', 'recovered_at': utc(), 'previous_status': 'started', 'finished_at': None, 'outcome_unknown': True, 'duplicate_processing_or_billing_possible': True, 'retryable': True, 'result': failure('interrupted_request_outcome_unknown')}
            atomic_json(paths[-1], redacted(last, key))
            attempts[-1] = last
        if not last.get('retryable') or len(attempts) >= limit:
            result = last['result']
    while result is None:
        number = len(attempts) + 1
        if attempts:
            sleep(2)
        record = {'attempt': number, 'started_at': utc(), 'finished_at': None, 'status': 'started', 'request_payload': payload, 'requested_model': cfg['model'], 'http_status': None, 'provider_request_id': None, 'raw_body': None, 'response_model': None, 'system_fingerprint': None, 'usage': None, 'finish_reason': None, 'body_complete': False, **identity}
        path = directory / ('attempt-%02d.json' % number)
        atomic_json(path, redacted(record, key))
        req = urllib.request.Request(cfg['url'], json.dumps(payload, ensure_ascii=False).encode(), {'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
        retryable = False
        response = None
        try:
            response = network_call(lambda: (transport or urllib.request.urlopen)(req, timeout=cfg.get('timeout_seconds', 120)))
            record.update(http_status=getattr(response, 'status', 200), provider_request_id=response.headers.get('x-request-id'))
            atomic_json(path, redacted(record, key))
            body = network_call(response.read).decode('utf-8', errors='replace')
            record.update(raw_body=body, body_complete=True)
            try:
                raw = json.loads(body)
            except ValueError:
                outcome = failure('invalid_provider_json')
                raw = None
            else:
                outcome = parse_response(raw)
            if isinstance(raw, dict):
                record.update(response_model=raw.get('model'), system_fingerprint=raw.get('system_fingerprint'), usage=raw.get('usage'), raw_response=raw)
                choices = raw.get('choices')
                choice = choices[0] if isinstance(choices, list) and choices else None
                record['finish_reason'] = choice.get('finish_reason') if isinstance(choice, dict) else None
        except urllib.error.HTTPError as exc:
            response = exc
            record.update(http_status=exc.code, provider_request_id=exc.headers.get('x-request-id') if exc.headers else None)
            atomic_json(path, redacted(record, key))
            try:
                body = network_call(exc.read).decode('utf-8', errors='replace')
            except NetworkFailure as failure_exc:
                record_network_error(record, failure_exc.original)
                cause, retryable = http_cause(exc.code, record.get('raw_body') or '')
                if retryable:
                    transport_reason, transport_retry = transport_cause(failure_exc.original)
                    if not transport_retry:
                        cause, retryable = (transport_reason, False)
                record['http_error_body_interrupted'] = True
            else:
                record.update(raw_body=body, body_complete=True)
                cause, retryable = http_cause(exc.code, body)
            outcome = failure(cause)
        except NetworkFailure as failure_exc:
            exc = failure_exc.original
            record_network_error(record, exc)
            cause, retryable = transport_cause(exc)
            outcome = failure(cause)
        finally:
            if response is not None:
                try:
                    network_call(response.close)
                except NetworkFailure as close_exc:
                    record['close_error'] = {'type': type(close_exc.original).__name__, 'message': str(close_exc.original)}
        if retryable and (not record['body_complete']):
            record.update(outcome_unknown=True, duplicate_processing_or_billing_possible=True)
        record.update(finished_at=utc(), status='completed', retryable=retryable, result=outcome)
        record = redacted(record, key)
        atomic_json(path, record)
        attempts.append(record)
        if not retryable or number >= limit:
            result = outcome
    last = attempts[-1] if attempts else {}
    result = {**result, 'judge_id': cfg.get('judge_id', 'primary_judge'), 'judge_model': cfg['model'], 'judge_settings_hash': identity['judge_settings_hash'], 'rubric_version': RUBRIC_VERSION, 'rubric_hash': contract['rubric_hash'], 'attempt_count': len(attempts), 'outcome_unknown_attempt_n': sum((bool(r.get('outcome_unknown')) for r in attempts)), 'attempts_path': str(directory), 'started_at': attempts[0]['started_at'] if attempts else None, 'finished_at': last.get('finished_at'), 'response_model': last.get('response_model'), 'system_fingerprint': last.get('system_fingerprint'), 'provider_request_id': last.get('provider_request_id'), 'finish_reason': last.get('finish_reason'), 'usage': last.get('usage'), 'context': context or {}}
    result = redacted(result, key)
    atomic_json(final, result)
    return result

def call_summary(root):
    attempts = [{**json.loads(p.read_text()), '_journal_directory': str(p.parent)} for p in Path(root).rglob('attempt-*.json')]
    models = sorted({r['response_model'] for r in attempts if r.get('response_model')})
    fps = sorted({r['system_fingerprint'] for r in attempts if r.get('system_fingerprint')})
    return {'api_attempt_n': len(attempts), 'retry_attempt_n': sum((r['attempt'] > 1 for r in attempts)), 'outcome_unknown_attempt_n': sum((bool(r.get('outcome_unknown')) for r in attempts)), 'benign_probe_attempt_n': sum((r.get('context', {}).get('phase') == 'benign_probe' for r in attempts)), 'unique_experiment_responses_attempted': len({r['_journal_directory'] for r in attempts if r.get('context', {}).get('phase') != 'benign_probe'}), 'call_started_at': min((r['started_at'] for r in attempts), default=None), 'call_finished_at': max((r['finished_at'] for r in attempts if r.get('finished_at')), default=None), 'response_models': models, 'system_fingerprints': fps, 'model_or_fingerprint_change_observed': len(models) > 1 or len(fps) > 1, 'fingerprint_missing_n': sum((r.get('system_fingerprint') is None for r in attempts)), 'snapshot_pinned': False, 'reproducibility_limit': 'Alias and call metadata do not provide a fixed snapshot.'}
