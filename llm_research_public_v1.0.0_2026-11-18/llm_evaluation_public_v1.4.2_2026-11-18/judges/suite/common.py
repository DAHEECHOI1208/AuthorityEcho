from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import sys
import importlib.metadata

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('C0', 'C1', 'C2', 'C3-E', 'C3-P', 'C4')
ORDER = {'refusal': 0, 'clarification_request': 1, 'undetermined': 2, 'partially_compliant': 3, 'compliant': 4}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    if not isinstance(value, bytes):
        value = (value if isinstance(value, str) else dumps(value)).encode('utf-8')
    return hashlib.sha256(value).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(dumps(value) + '\n')
        f.flush()
        os.fsync(f.fileno())


def condition(value):
    text = str(value).upper().strip().replace('_', '-').replace(' ', '')
    text = {'C3E': 'C3-E', 'C3P': 'C3-P'}.get(text, text)
    if text not in CONDITIONS:
        raise ValueError(f'Unknown condition: {value}')
    return text


def blank(value):
    return value is None or isinstance(value, str) and not value.strip()


def load_env(path):
    if not path or not Path(path).exists():
        return
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.split('=', 1)
        key = key.strip()
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', key):
            raise ValueError('Invalid environment variable name')
        os.environ.setdefault(key, value.strip().strip('\"\''))


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    f = (directory / '.run.lock').open('a+b')
    acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if f.seek(0, 2) == 0:
                f.write(b'0'); f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        yield
    finally:
        if acquired:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        f.close()


def code_hashes():
    paths = [ROOT / 'judge.py', ROOT / 'source_registry.json', ROOT / 'requirements.txt', ROOT / 'configs/gpt56_luna.json', *sorted((ROOT / 'suite').glob('*.py'))]
    registry = read_json(ROOT / 'source_registry.json')
    for component in registry.values():
        paths.extend(ROOT / p for p in component['files'])
    return {str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in paths}


def verify_sources():
    for item in read_json(ROOT / 'source_registry.json').values():
        for name, expected in item['files'].items():
            if digest((ROOT / name).read_bytes()) != expected:
                raise ValueError(f'Packaged source changed: {name}')


def environment_identity(judges):
    packages = ['openpyxl', 'requests']
    if 'harmbench_full' in judges:
        packages += ['transformers', 'tokenizers', 'sentencepiece', 'protobuf', 'Jinja2', 'huggingface-hub']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python_major_minor': list(sys.version_info[:2]), 'packages': versions}
