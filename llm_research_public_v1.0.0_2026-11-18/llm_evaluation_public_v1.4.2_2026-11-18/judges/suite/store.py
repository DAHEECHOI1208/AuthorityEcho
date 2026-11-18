from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import uuid

from .common import now, dumps, atomic_json, append_jsonl


class Store:
    """SQLite is authoritative. A response is committed before interpretation."""
    def __init__(self, directory):
        self.root = Path(directory)
        self.db = sqlite3.connect(self.root / 'state.sqlite3')
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS tasks (task_id TEXT PRIMARY KEY, data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS attempts (attempt_id TEXT PRIMARY KEY, phase TEXT, task_id TEXT,
            judge_id TEXT, started_at TEXT, state TEXT, request TEXT, response TEXT, result TEXT);
          CREATE TABLE IF NOT EXISTS results (phase TEXT, task_id TEXT, judge_id TEXT, data TEXT,
            PRIMARY KEY (phase, task_id, judge_id));
        ''')

    def close(self):
        self.db.close()

    def seed(self, tasks):
        with self.db:
            self.db.executemany('INSERT INTO tasks VALUES (?,?)', [(r['task_id'], dumps(r)) for r in tasks])

    def tasks(self):
        return [json.loads(r[0]) for r in self.db.execute('SELECT data FROM tasks ORDER BY rowid')]

    def results(self, phase='judge'):
        return {(r['task_id'], r['judge_id']): json.loads(r['data']) for r in self.db.execute('SELECT * FROM results WHERE phase=?', (phase,))}

    def event(self, event, **fields):
        append_jsonl(self.root / 'logs/events.jsonl', {'at': now(), 'event': event, **fields})

    def begin(self, phase, task, judge_id, request):
        attempt = uuid.uuid4().hex
        meta = {'attempt_id': attempt, 'phase': phase, 'task_id': task['task_id'], 'judge_id': judge_id, 'started_at': now()}
        request = {**request, **meta}
        atomic_json(self.root / 'raw' / (attempt + '.request.json'), request)
        with self.db:
            self.db.execute('INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?)',
                            (attempt, phase, task['task_id'], judge_id, meta['started_at'], 'SENT', dumps(request), None, None))
        self.event('request_started', **meta)
        return meta

    def received(self, meta, response):
        with self.db:
            self.db.execute('UPDATE attempts SET response=?,state=? WHERE attempt_id=?', (dumps(response), 'RECEIVED', meta['attempt_id']))
        atomic_json(self.root / 'raw' / (meta['attempt_id'] + '.response.json'), response)

    def record(self, phase, task_id, judge_id, result, meta=None):
        result = {**result, 'phase': phase, 'task_id': task_id, 'judge_id': judge_id, 'recorded_at': now()}
        if meta:
            result.update(attempt_id=meta['attempt_id'], started_at=meta['started_at'],
                          raw_request_path='raw/' + meta['attempt_id'] + '.request.json',
                          raw_response_path='raw/' + meta['attempt_id'] + '.response.json')
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO results VALUES (?,?,?,?)', (phase, task_id, judge_id, dumps(result)))
            if meta:
                self.db.execute('UPDATE attempts SET result=?,state=? WHERE attempt_id=?', (dumps(result), 'DONE', meta['attempt_id']))
        self.event('result_recorded', phase=phase, task_id=task_id, judge_id=judge_id,
                   status=result['status'], attempt_id=result.get('attempt_id'))
        return result

    def recover(self, adapters):
        recovered = 0
        for row in list(self.db.execute("SELECT * FROM attempts WHERE state!='DONE' ORDER BY rowid")):
            meta = {k: row[k] for k in ('attempt_id', 'phase', 'task_id', 'judge_id', 'started_at')}
            if row['response']:
                response = json.loads(row['response'])
                self.restore_missing_raw(row['attempt_id'], 'response', response)
                result = adapters[row['judge_id']].interpret(response, row['phase'])
                result['recovered_without_api_call'] = True
                result['request_audit'] = json.loads(row['request']).get('audit')
            else:
                result = {'status': 'UNKNOWN_DELIVERY', 'analysis_valid': False, 'analysis_exclusion': 'UNKNOWN_DELIVERY',
                          'label': None, 'error': 'Process stopped after attempt persisted; provider delivery/result unknown. No automatic resend.'}
            self.record(row['phase'], row['task_id'], row['judge_id'], result, meta)
            recovered += 1
        return recovered

    def restore_missing_raw(self, attempt_id, kind, data):
        path = self.root / 'raw' / (attempt_id + '.' + kind + '.json')
        if not path.exists():
            atomic_json(path, data)

    def export_logs(self):
        # Full export only: rebuild JSONL from SQLite, restoring missing raw files.
        # begin()/received() already wrote each attempt's raw files durably.
        # Existing raw files must not be rewritten on every export.
        for filename, query in [('results.jsonl', 'SELECT data AS data FROM results ORDER BY rowid'),
                                ('attempts.jsonl', 'SELECT * FROM attempts ORDER BY rowid')]:
            path = self.root / 'logs' / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix('.tmp')
            with tmp.open('w', encoding='utf-8') as f:
                for row in self.db.execute(query):
                    data = json.loads(row['data']) if filename == 'results.jsonl' else dict(row)
                    if filename == 'attempts.jsonl':
                        for key in ('request', 'response', 'result'):
                            data[key] = json.loads(data[key]) if data[key] else None
                        self.restore_missing_raw(row['attempt_id'], 'request', data['request'])
                        if data['response'] is not None:
                            self.restore_missing_raw(row['attempt_id'], 'response', data['response'])
                    f.write(dumps(data) + '\n')
            tmp.replace(path)
