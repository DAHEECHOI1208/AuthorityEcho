"""Identity-bound, atomic storage. Prompt JSON preserves insertion order."""
import contextlib
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def require(ok, message):
    if not ok: raise ValueError(message)

def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',',':'), default=str)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def file_hash(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def utc(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_bytes())

def atomic_bytes(path, data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.pending-')
    try:
        with os.fdopen(fd,'wb') as f: f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def atomic(path,value):
    atomic_bytes(path,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())

def jsonl(path,rows):
    atomic_bytes(path,b''.join((json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n').encode() for r in rows))

def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def csv_rows(path,rows):
    import io
    if not rows: atomic_bytes(path,b'');return
    keys=list(dict.fromkeys(k for r in rows for k in r));s=io.StringIO(newline='')
    w=csv.DictWriter(s,fieldnames=keys);w.writeheader()
    for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    atomic_bytes(path,s.getvalue().encode())

def code_hash():
    files=[p for folder in ('mechanism','configs') for p in (ROOT/folder).rglob('*')
           if p.is_file() and p.suffix in ('.py','.json') and '__pycache__' not in p.parts]
    files += [ROOT/'run.py',ROOT/'run_stage.sh',ROOT/'requirements-gpu.txt']
    return digest({str(p.relative_to(ROOT)):file_hash(p) for p in sorted(files)})

@contextlib.contextmanager
def lock(directory):
    import fcntl
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.lock').open('a+') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another process is writing this run')
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def load_run(path):
    path=Path(path);m=read(path/'manifest.json')
    require(m['run_identity']==digest({k:v for k,v in m.items() if k!='run_identity'}),'Manifest identity changed')
    require(m['code_hash']==code_hash(),'Code/configuration changed since prepare; use the original package or a new run')
    for rel,h in m['files'].items():require(file_hash(path/rel)==h,'Prepared input/configuration changed: '+rel)
    return m,read_jsonl(path/'inputs.jsonl')

def npz_save(path,**arrays):
    import io,numpy as np
    b=io.BytesIO();np.savez_compressed(b,**arrays);atomic_bytes(path,b.getvalue())

def result_id(row,arm='baseline',layer=None,alpha=None,repeat=0):
    return digest([row['sample_id'],row['input_hash'],arm,layer,alpha,repeat])

def committed_record(run,tid):
    p=Path(run)/'outputs'/f'{tid}.json'
    if not p.exists():return None
    r=read(p)
    require(r['task_id']==tid,'Result/task linkage changed')
    m=read(Path(run)/'manifest.json');require(r.get('run_identity')==m['run_identity'],'Result belongs to another run')
    require(result_id(r,r['arm'],r.get('layer'),r.get('alpha'))==tid,'Result/task fields differ')
    expected=f"{r['set_id']}:{r['variant']}:{r['question_id']}:{r['condition_id']}"
    require(r['sample_id']==expected,'Question, variant, set or condition linkage differs')
    from .native.P17_final_answer import construct_final
    require(construct_final(r)['input_hash']==r['input_hash'],'Result input fields changed')
    if r.get('activation_file'):
        require(file_hash(Path(run)/r['activation_file'])==r['activation_sha256'],'Activation file changed')
    return r

def private_artifact(path,receipt,kind):
    require(path is not None and receipt is not None,'Provide both '+kind+' and its identity receipt')
    r=read(receipt);require(r.get('sha256')==file_hash(path),kind+' hash mismatch')
    require(r.get('kind')==kind,'Artifact kind differs')
    return r
