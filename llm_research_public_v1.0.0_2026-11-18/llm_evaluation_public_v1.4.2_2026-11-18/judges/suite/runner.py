from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from urllib.parse import urlsplit
import base64
import json
import os
import shutil
import socket
import time
import urllib.error
import urllib.request
import uuid

from . import VERSION
from .adapters import Adapter, adapters_for, definitions
from .common import ROOT, now, digest, dumps, read_json, atomic_json, code_hashes, verify_sources, run_lock, environment_identity
from .inputs import load_inputs
from .report import export_run
from .store import Store


def validate_user_agent(value):
    if (not isinstance(value, str) or not 1 <= len(value) <= 256
            or value != value.strip() or any(not 32 <= ord(c) <= 126 for c in value)):
        raise ValueError('user_agent must be 1..256 printable ASCII characters without outer whitespace')
    return value


def validate_config(config):
    allowed = {'enabled', 'timeout_seconds', 'checkpoint_every', 'pause_seconds', 'automatic_retries',
               'strongreject_concurrency', 'strongreject_stop_after_errors', 'connections'}
    if set(config) - allowed:
        raise ValueError('Unknown configuration keys: ' + str(sorted(set(config) - allowed)))
    ids = config['enabled']
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(definitions()):
        raise ValueError('enabled must contain unique supported judge IDs')
    if 'gpt56_luna_research' in ids and ids[0] != 'gpt56_luna_research':
        raise ValueError('gpt56_luna_research must be first in enabled when present')
    for field, expected in [('timeout_seconds', 180), ('checkpoint_every', 100), ('automatic_retries', 0)]:
        if config.get(field) != expected:
            raise ValueError(f'Confirmed setting {field} must be {expected}')
    if not isinstance(config.get('strongreject_concurrency'), int) or not 1 <= config['strongreject_concurrency'] <= 64:
        raise ValueError('strongreject_concurrency must be 1..64; default is sequential (1)')
    if config.get('strongreject_stop_after_errors') != 20:
        raise ValueError('StrongREJECT runner uses a 20 consecutive HTTP/network error stop')
    if not isinstance(config.get('pause_seconds'), (int, float)) or not 0 <= config['pause_seconds'] <= 60:
        raise ValueError('pause_seconds must be 0..60')
    for judge in ids:
        conn = config['connections'][judge]
        if set(conn) - {'url', 'url_env', 'key_env', 'served_model_id', 'user_agent'}:
            raise ValueError('Connection config supports URL/env names/served model ID/user_agent only; no API key values')
        if bool(conn.get('url')) == bool(conn.get('url_env')) or not conn.get('key_env'):
            raise ValueError('Each connection requires exactly one URL or URL environment name and key_env')
        if conn.get('url'):
            base_url(conn['url'])
        if 'user_agent' in conn:
            validate_user_agent(conn['user_agent'])
    return config


def prepare(input_manifest, config_path, directory):
    verify_sources()
    config = validate_config(read_json(config_path))
    directory = Path(directory).resolve()
    if directory.exists():
        raise ValueError('Run directory already exists; choose a new directory or use run to resume')
    # Validate once before creating a run, so input mistakes leave no half-prepared output.
    load_inputs(input_manifest)
    temporary = directory.with_name(directory.name + '.preparing-' + uuid.uuid4().hex[:8])
    temporary.mkdir(parents=True)
    try:
        tasks, inputs = load_inputs(input_manifest, temporary / 'snapshot/inputs')
        normalized = ''.join(dumps(t) + '\n' for t in tasks)
        (temporary / 'inputs.normalized.jsonl').write_text(normalized, encoding='utf-8', newline='\n')
        hashes = code_hashes()
        for relative in hashes:
            dest = temporary / 'snapshot/code' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, dest)
        atomic_json(temporary / 'snapshot/input_manifest.json', inputs['manifest'])
        atomic_json(temporary / 'snapshot/config.json', config)
        manifest = {'schema_version': 1, 'suite_version': VERSION, 'run_id': uuid.uuid4().hex, 'prepared_at': now(),
                    'config': config, 'inputs': inputs, 'normalized_sha256': digest(normalized),
                    'code_sha256': hashes, 'environment': environment_identity(config['enabled']),
                    'models': {j: {k: v for k, v in definitions()[j].items() if k not in {'config', 'model_spec'}} for j in config['enabled']}}
        atomic_json(temporary / 'manifest.json', manifest)
        (temporary / 'manifest.sha256').write_text(digest((temporary / 'manifest.json').read_bytes()) + '\n', encoding='ascii')
        store = Store(temporary)
        try:
            store.seed(tasks)
            store.event('prepared', run_id=manifest['run_id'], tasks=len(tasks))
            export_run(temporary, store, checkpoint=True)
        finally:
            store.close()
        temporary.rename(directory)
    except BaseException:
        # Preserve failed preparation for diagnosis; it cannot be used as a completed run.
        raise
    return directory


def verify_run(directory, *, accepted_source_hashes=None, allow_transition_stage=False):
    directory = Path(directory)
    if '.transitioning-' in directory.name and not allow_transition_stage:
        raise ValueError('Incomplete transition staging directory cannot be used as a run; keep it for diagnosis and repeat transition from the original source')
    if digest((directory / 'manifest.json').read_bytes()) != (directory / 'manifest.sha256').read_text().strip():
        raise ValueError('Frozen manifest changed; use a new run')
    manifest = read_json(directory / 'manifest.json')
    validate_config(manifest['config'])
    if digest((directory / 'inputs.normalized.jsonl').read_bytes()) != manifest['normalized_sha256']:
        raise ValueError('Normalized input snapshot changed')
    expected_code = code_hashes() if accepted_source_hashes is None else accepted_source_hashes
    if expected_code != manifest['code_sha256']:
        raise ValueError('Running code/prompts differ from this run snapshot; restore original code or prepare a new run')
    if environment_identity(manifest['config']['enabled']) != manifest['environment']:
        raise ValueError('Python/dependency environment changed; restore the recorded versions or prepare a new run')
    for relative, expected in manifest['code_sha256'].items():
        if digest((directory / 'snapshot/code' / relative).read_bytes()) != expected:
            raise ValueError('Frozen code snapshot changed: ' + relative)
    if read_json(directory / 'snapshot/config.json') != manifest['config'] or read_json(directory / 'snapshot/input_manifest.json') != manifest['inputs']['manifest']:
        raise ValueError('Frozen configuration snapshot changed')
    for source in manifest['inputs']['source_files']:
        if digest((directory / 'snapshot/inputs' / source['snapshot']).read_bytes()) != source['sha256']:
            raise ValueError('Frozen source input changed: ' + source['name'])
    for line in (directory / 'inputs.normalized.jsonl').read_text(encoding='utf-8').splitlines():
        sidecar = json.loads(line).get('source_sidecar') or {}
        if sidecar.get('snapshot') and digest((directory / 'snapshot/inputs' / sidecar['snapshot']).read_bytes()) != sidecar['sha256']:
            raise ValueError('Frozen full-answer sidecar changed')
    if manifest.get('transition'):
        raise ValueError("Use a fresh public-package run directory")
    return manifest


def base_url(value):
    value = value.strip().rstrip('/')
    p = urlsplit(value)
    if p.username or p.password or p.query or p.fragment or not p.hostname:
        raise ValueError('Endpoint must not contain credentials, query parameters, or fragments')
    if p.scheme != 'https' and not (p.scheme == 'http' and p.hostname in {'localhost', '127.0.0.1', '::1'}):
        raise ValueError('Use HTTPS for a remote endpoint')
    if p.path.rstrip('/') not in {'', '/v1'}:
        raise ValueError('Endpoint URL should end at the host or /v1, not /chat/completions')
    return value if value.endswith('/v1') else value + '/v1'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def transport_once(url, payload, api_key, timeout, request_id, method='POST', *, user_agent=None):
    """Exactly one HTTP attempt. No SDK retry and no redirects."""
    started = time.monotonic()
    headers = {'Accept': 'application/json', 'X-Client-Request-Id': request_id}
    if user_agent is not None:
        headers['User-Agent'] = validate_user_agent(user_agent)
    if api_key:
        headers['Authorization'] = 'Bearer ' + api_key
    body = None if payload is None else dumps(payload).encode('utf-8')
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(NoRedirect())
    try:
        try:
            response = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw, status = response.read(), response.code
            clean_headers = {k: ('[REDACTED]' if k.lower() in {'authorization', 'proxy-authorization', 'cookie', 'set-cookie', 'x-api-key'} else v) for k, v in response.headers.items()}
        if api_key:
            raw = raw.replace(api_key.encode('utf-8'), b'[REDACTED]')
            clean_headers = {k: v.replace(api_key, '[REDACTED]') for k, v in clean_headers.items()}
        lower = {k.lower(): v for k, v in clean_headers.items()}
        result = {'http_status': status, 'headers': clean_headers, 'raw_body_text': raw.decode('utf-8', errors='replace'),
                  'raw_body_base64': base64.b64encode(raw).decode('ascii'),
                  'server_request_id': next((lower[k] for k in ('x-request-id', 'x-amzn-requestid', 'request-id') if k in lower), None)}
    except Exception as exc:
        is_timeout = isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(exc, urllib.error.URLError) and isinstance(exc.reason, (TimeoutError, socket.timeout))
        message = f'{type(exc).__name__}: {exc}'
        result = {'http_status': None, 'network_error': message.replace(api_key, '[REDACTED]') if api_key else message,
                  'error_status': 'TIMEOUT' if is_timeout else 'NETWORK_ERROR'}
    result.update(received_at=now(), latency_ms=round((time.monotonic() - started) * 1000, 3))
    return result


def connections(config, adapters, directory, transport=transport_once):
    resolved, discoveries = {}, {}
    directory = Path(directory)
    for judge, adapter in adapters.items():
        spec = config['connections'][judge]
        url = spec.get('url') or os.environ.get(spec.get('url_env', ''))
        if not url:
            raise ValueError('Set endpoint environment variable: ' + spec.get('url_env', judge))
        if 'YOUR-' in url:
            raise ValueError('Replace placeholder endpoint URL: ' + spec.get('url_env', judge))
        base = base_url(url)
        transport_options = {'user_agent': spec['user_agent']} if 'user_agent' in spec else {}
        key = os.environ.get(spec['key_env'], '')
        if 'YOUR-' in key:
            raise ValueError('Replace placeholder credential environment variable: ' + spec['key_env'])
        if not key and urlsplit(base).hostname not in {'localhost', '127.0.0.1', '::1'}:
            raise ValueError('Missing credential environment variable: ' + spec['key_env'])
        served = adapter.model
        if adapter.definition['provider'] == 'hf':
            cache_key = (base, spec['key_env'])
            if cache_key not in discoveries:
                request_id = uuid.uuid4().hex
                response = transport(base + '/models', None, key, config['timeout_seconds'], request_id,
                                     method='GET', **transport_options)
                atomic_json(directory / 'discovery' / (request_id + '.json'), {'url': base + '/models', 'response': response})
                if response.get('http_status') != 200 or response.get('network_error'):
                    raise RuntimeError('Model discovery failed; inspect discovery/' + request_id + '.json')
                data = json.loads(response['raw_body_text'])
                discoveries[cache_key] = [row['id'] for row in data['data']]
            ids = discoveries[cache_key]
            served = spec.get('served_model_id')
            if served:
                if served not in ids:
                    raise ValueError('Configured served_model_id is not exposed by /v1/models')
            elif len(ids) == 1:
                served = ids[0]
            else:
                raise ValueError('Multiple/no served models; specify served_model_id in config and prepare a new run')
        public = {'base_url': base, 'served_model_id': served, 'configured_model': adapter.model,
                  'configured_revision': adapter.definition['revision'], 'endpoint_revision_verified_by_api': False,
                  'server_chat_template_verified_by_api': False}
        public.update(transport_options)
        path = directory / 'runtime' / (judge + '.json')
        if path.exists() and read_json(path) != public:
            raise ValueError('Endpoint/model identity changed; prepare a new run instead of mixing results')
        atomic_json(path, public)
        resolved[judge] = {**public, 'key': key}
    return resolved


def should_call(prior, retry_errors=False, retry_unknown=False):
    if prior is None:
        return True
    status = prior['status']
    if status == 'OK' or status.startswith(('INPUT_', 'TARGET_')):
        return False
    if status == 'UNKNOWN_DELIVERY':
        return retry_unknown
    return retry_errors


def smoke_state(store, selected):
    results = store.results('smoke')
    passed, failed, pending = [], {}, []
    for judge in selected:
        result = results.get(('SMOKE_ONLY', judge))
        if result is None:
            pending.append(judge)
        elif result.get('status') == 'OK' and result.get('analysis_valid') is True:
            passed.append(judge)
        else:
            failed[judge] = result.get('analysis_exclusion') or result.get('status', 'UNKNOWN')
    return {'smoke_ok': not failed and not pending, 'smoke_passed': passed,
            'smoke_failed': failed, 'smoke_pending': pending}


def smoke_should_call(prior, retry_errors=False, retry_unknown=False):
    # A preserved original status=OK can still fail the independent raw check.
    # Allow explicit smoke retry, without changing main-data retry semantics.
    if prior and prior.get('status') == 'OK' and prior.get('analysis_valid') is not True:
        return retry_errors
    return should_call(prior, retry_errors, retry_unknown)


def safe_export(directory, store, *, checkpoint=False):
    try:
        export_run(directory, store, checkpoint=checkpoint)
    except PermissionError as exc:
        store.event('excel_export_failed', error=str(exc))
        print('Excel save failed (file may be open). SQLite is saved; close Excel and run export.', flush=True)


def execute(directory, selected=None, retry_errors=False, retry_unknown=False, max_calls=None,
            phase='judge', transport=transport_once, tokenizers=None, resolved_for_test=None):
    """The only API execution entry point; interrupted attempts are never silently resent."""
    directory = Path(directory).resolve()
    with run_lock(directory):
        manifest = verify_run(directory)
        cfg = manifest['config']
        selected = selected or cfg['enabled']
        if not selected or len(set(selected)) != len(selected) or set(selected) - set(cfg['enabled']):
            raise ValueError('Selected judges must be a unique subset of the frozen enabled list')
        # Prioritize Luna even when --judges is explicitly supplied in another order.
        # Preserve the relative order of all other selected profiles.
        if 'gpt56_luna_research' in selected:
            selected = ['gpt56_luna_research', *[j for j in selected if j != 'gpt56_luna_research']]
        store = Store(directory)
        counts = {'calls': 0, 'processed': 0, 'recovered': 0, 'stopped': False}
        try:
            tasks = store.tasks()
            canonical = ''.join(dumps(t) + '\n' for t in tasks)
            if digest(canonical) != manifest['normalized_sha256']:
                raise ValueError('SQLite input rows differ from frozen normalized input')
            plain = {j: Adapter(j) for j in cfg['enabled']}
            counts['recovered'] = store.recover(plain)
            if phase == 'judge':
                state = smoke_state(store, selected)
                if state['smoke_failed']:
                    store.event('judge_blocked_by_smoke', selected=selected, **state)
                    raise ValueError('SMOKE_FAILED: ' + dumps(state['smoke_failed']) +
                                     '; resolve and retry smoke explicitly before run/resume. No judge request sent.')
            current = store.results(phase)
            if phase == 'smoke':
                tasks = [{'task_id': 'SMOKE_ONLY', 'question_id': 'SMOKE_ONLY', 'question': 'What is 2 + 2?', 'answer': '4',
                          'condition': 'SMOKE', 'target_eligible': True, 'target_state': 'TARGET_COMPLETED'}]
            call_check = smoke_should_call if phase == 'smoke' else should_call
            candidates = {j: [t for t in tasks if call_check(current.get((t['task_id'], j)), retry_errors, retry_unknown)] for j in selected}
            # Input skips require no endpoint, credentials, tokenizer download, or API call.
            for j, rows in candidates.items():
                for t in rows:
                    if not t['target_eligible']:
                        store.record(phase, t['task_id'], j, {'status': t['target_state'], 'analysis_valid': False,
                                     'analysis_exclusion': t['target_state'], 'label': None, 'api_called': False})
                        counts['processed'] += 1
                        if counts['processed'] % cfg['checkpoint_every'] == 0:
                            safe_export(directory, store, checkpoint=True)
                candidates[j] = [t for t in rows if t['target_eligible']]
            needed = [j for j in selected if candidates[j]]
            if not needed:
                if phase == 'smoke':
                    counts.update(smoke_state(store, selected))
                return counts
            adapters = adapters_for(needed, directory, tokenizers)
            conns = resolved_for_test or connections(cfg, adapters, directory, transport)
            store.event('execution_started', phase=phase, selected=selected, retry_errors=retry_errors,
                        retry_unknown=retry_unknown, max_calls=max_calls, strongreject_concurrency=cfg['strongreject_concurrency'],
                        runner_version=VERSION, transition_id=manifest.get('transition', {}).get('id'))
            for j in needed:
                adapter, conn = adapters[j], conns[j]
                concurrency = cfg['strongreject_concurrency'] if j == 'strongreject' and phase == 'judge' else 1
                failures = 0
                iterator = iter(candidates[j])
                exhausted = False
                pending = {}
                pool = ThreadPoolExecutor(max_workers=concurrency)
                try:
                    while pending or not exhausted:
                        while not exhausted and len(pending) < concurrency:
                            if max_calls is not None and counts['calls'] >= max_calls:
                                exhausted = True
                                break
                            try:
                                t = next(iterator)
                            except StopIteration:
                                exhausted = True
                                break
                            q, answer = t['question'], t['answer']
                            if phase == 'smoke' and j in {'harmbench_full', 'strongreject'}:
                                q = read_json(ROOT.parent / 'input.json')['question']
                                answer = 'I cannot provide that invitation.'
                            route, payload, audit = adapter.build(q, answer, conn['served_model_id'])
                            if audit.get('preflight_status', 'READY') != 'READY':
                                status = audit['preflight_status']
                                store.record(phase, t['task_id'], j, {'status': status, 'analysis_valid': False,
                                             'analysis_exclusion': status, 'label': None, 'api_called': False, 'request_audit': audit})
                                counts['processed'] += 1
                                if counts['processed'] % cfg['checkpoint_every'] == 0:
                                    safe_export(directory, store, checkpoint=True)
                                continue
                            if counts['calls'] and concurrency == 1 and cfg['pause_seconds']:
                                time.sleep(cfg['pause_seconds'])
                            request = {'url': conn['base_url'] + route, 'payload': payload, 'audit': audit,
                                       'configured_model': adapter.model, 'configured_revision': adapter.definition['revision'],
                                       'served_model_id': conn['served_model_id'], 'timeout_seconds': cfg['timeout_seconds']}
                            request['execution'] = {'run_id': manifest['run_id'], 'runner_version': VERSION,
                                                    'effective_concurrency': concurrency,
                                                    'transition_id': manifest.get('transition', {}).get('id')}
                            transport_options = {'user_agent': conn['user_agent']} if 'user_agent' in conn else {}
                            if transport_options:
                                request['headers'] = {'User-Agent': conn['user_agent']}
                            meta = store.begin(phase, t, j, request)
                            # Persistence happens before submit. Even a submit/crash ambiguity is explicit UNKNOWN_DELIVERY.
                            future = pool.submit(transport, request['url'], payload, conn['key'], cfg['timeout_seconds'],
                                                 meta['attempt_id'], **transport_options)
                            pending[future] = (meta, audit)
                            counts['calls'] += 1
                        if not pending:
                            continue
                        done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        for future in done:
                            meta, audit = pending.pop(future)
                            response = future.result()
                            store.received(meta, response)
                            result = adapter.interpret(response, phase)
                            result.update(api_called=True, request_audit=audit, latency_ms=response.get('latency_ms'),
                                          received_at=response.get('received_at'), http_status=response.get('http_status'))
                            store.record(phase, meta['task_id'], j, result, meta)
                            counts['processed'] += 1
                            failures = failures + 1 if result['status'] in {'HTTP_ERROR', 'NETWORK_ERROR', 'TIMEOUT'} else 0
                            if j == 'strongreject' and failures >= cfg['strongreject_stop_after_errors']:
                                exhausted = True
                                counts['stopped'] = True
                                store.event('strongreject_stopped_after_consecutive_errors', n=failures, in_flight=len(pending))
                            if counts['processed'] % cfg['checkpoint_every'] == 0:
                                safe_export(directory, store, checkpoint=True)
                            print(f"[{phase}] {j} {counts['processed']} records / {counts['calls']} calls: {result['status']}", flush=True)
                except BaseException:
                    # Collect already finished responses; running requests become UNKNOWN on the next run.
                    for future, (meta, audit) in pending.items():
                        if future.done() and not future.cancelled():
                            response = future.result()
                            store.received(meta, response)
                            result = adapter.interpret(response, phase)
                            result['request_audit'] = audit
                            store.record(phase, meta['task_id'], j, result, meta)
                        else:
                            future.cancel()
                    raise
                finally:
                    pool.shutdown(wait=False, cancel_futures=True)
                if max_calls is not None and counts['calls'] >= max_calls:
                    break
            if phase == 'smoke':
                counts.update(smoke_state(store, selected))
            store.event('execution_finished', phase=phase, **counts)
            return counts
        except BaseException as exc:
            store.event('execution_interrupted', exception_type=type(exc).__name__)
            # Mark possible deliveries without issuing a replacement request.
            store.recover({j: Adapter(j) for j in cfg['enabled']})
            raise
        finally:
            try:
                safe_export(directory, store)
            finally:
                store.close()


def export_only(directory):
    directory = Path(directory).resolve()
    with run_lock(directory):
        verify_run(directory)
        store = Store(directory)
        try:
            store.recover({j: Adapter(j) for j in read_json(directory / 'manifest.json')['config']['enabled']})
            return export_run(directory, store)
        finally:
            store.close()


def review_strongreject(directory, review_file):
    from .reviews import save_reviews
    directory = Path(directory).resolve()
    with run_lock(directory):
        verify_run(directory)
        store = Store(directory)
        try:
            n = save_reviews(directory, review_file, store.results())
            store.event('strongreject_diagnostic_reviews_added', count=n)
            export_run(directory, store)
            return {'reviewed_n': n, 'api_calls': 0, 'results': str(directory / 'results.xlsx')}
        finally:
            store.close()
