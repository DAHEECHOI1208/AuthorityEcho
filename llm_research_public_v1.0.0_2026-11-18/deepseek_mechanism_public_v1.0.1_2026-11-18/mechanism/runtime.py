"""Pinned native inference, capture, environment verification, and safe resume."""
import contextlib
import importlib.metadata
import platform
import time
from pathlib import Path
import numpy as np
from .io import ROOT,require,read,digest,file_hash,atomic,utc,load_run,result_id,committed_record,npz_save
from .hooks import Capture
from .output_audit import protocol_only_echo

def environment_check():
    expected=read(ROOT/'configs/environment.json')
    actual={k:importlib.metadata.version(k) for k in expected}
    require(actual==expected,'Installed libraries differ from pinned environment; inspect configs/environment.json')
    require(platform.python_version()=='3.12.14','Pinned runtime requires Python 3.12.14')
    import torch
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA GPU with BF16 is required')
    return {'versions':actual,'python':platform.python_version(),'cuda':torch.version.cuda,
            'device':torch.cuda.get_device_name(0),'tf32':bool(torch.backends.cuda.matmul.allow_tf32),
            'bf16_reduction':bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            'flash_sdp':torch.backends.cuda.flash_sdp_enabled(),'mem_efficient_sdp':torch.backends.cuda.mem_efficient_sdp_enabled(),
            'math_sdp':torch.backends.cuda.math_sdp_enabled()}

def verify_snapshot():
    from huggingface_hub import snapshot_download
    cfg=read(ROOT/'configs/target.json')
    path=Path(snapshot_download(cfg['model_id'],revision=cfg['revision'],local_files_only=True))
    require(path.name==cfg['revision'],'Model snapshot revision mismatch')
    pinned=read(ROOT/'configs/tokenizer_files.json')['files']
    for name,h in pinned.items():require((path/name).is_file() and file_hash(path/name)==h,'Pinned file missing or changed: '+name)
    weights=sorted(path.glob('*.safetensors'));require(weights,'No model weight shards')
    return {p.name:file_hash(p) for p in weights}

@contextlib.contextmanager
def adapter_for(run):
    from .native.C10_model import HFModel
    m,_=load_run(run);env=environment_check();weights=verify_snapshot()
    info={'environment':env,'weights':weights};p=Path(run)/'runtime.json'
    if p.exists():require(read(p)['runtime_hash']==digest(info),'Runtime or model weights changed')
    else:atomic(p,{'runtime_hash':digest(info),**info})
    cfg=read(Path(run)/'target.json');model=HFModel(cfg)
    require(len(model.layers)==27 and model.model.config.hidden_size==2048,'Model architecture differs')
    try:yield model,digest(info)
    finally:model.close()

def generate_one(adapter,run,row,identity,runtime_hash,arm='baseline',layer=None,alpha=None,patch=None):
    tid=result_id(row,arm,layer,alpha)
    old=committed_record(run,tid)
    if old:
        require(old['run_identity']==identity and old['runtime_hash']==runtime_hash,'Stored output identity differs')
        require(old['technical_status']=='ok','Recorded target failure requires explicit retry-target')
        return old
    serial=adapter.serialize(row);trace=Capture(adapter,serial,patch)
    previous=adapter.session;start=time.monotonic()
    def session(s,*args,**kwargs):
        require(s['input_ids']==serial['input_ids'],'Tokenization changed inside generation');return trace
    adapter.session=session
    try:
        adapter.torch.cuda.synchronize();r=adapter.generate(row);adapter.torch.cuda.synchronize()
    except Exception as ex:
        atomic(Path(run)/'outputs'/f'{tid}.json',{**row,'task_id':tid,'run_identity':identity,'runtime_hash':runtime_hash,
               'technical_status':'failed','error_type':type(ex).__name__,'failure_message':str(ex),
               'condition_id':row['condition_id'],'sample_id':row['sample_id'],'arm':arm,'layer':layer,'alpha':alpha,
               'response_text':'','created_at':utc()})
        raise
    finally:adapter.session=previous
    filename=f'activations/{tid}.npz';npz_save(Path(run)/filename,**trace.arrays())
    r.update(task_id=tid,run_identity=identity,runtime_hash=runtime_hash,arm=arm,layer=layer,alpha=alpha,
             activation_file=filename,activation_sha256=file_hash(Path(run)/filename),
             capture_phase='first_prefill',capture_use_cache=True,capture_dtype='float32',
             seconds=time.monotonic()-start,created_at=utc(),donor_provenance={k:patch[k] for k in ('donor_task_id','donor_activation_sha256') if k in patch} if patch else None,diagnostics={'effects':trace.effects,'forward_calls':trace.calls},
             protocol_nonanswer_confirmed=protocol_only_echo(r['response_text'],row))
    atomic(Path(run)/'outputs'/f'{tid}.json',r)
    print('generated',row['sample_id'],arm,layer,r['technical_status'],flush=True)
    return r

def baseline(run,limit=None,cross=False):
    m,rows=load_run(run)
    if cross:
        from .inputs import cross_rows
        rows=cross_rows(rows,read(Path(run)/'cross.json'))
    # Preserve condition-major source order for standard baseline generation.
    from .inputs import CONDITIONS
    if not cross:rows=sorted(enumerate(rows),key=lambda z:(CONDITIONS.index(z[1]['condition_id']),z[0]));rows=[r for _,r in rows]
    pending=[r for r in rows if committed_record(run,result_id(r)) is None]
    failed=[r for r in rows if committed_record(run,result_id(r)) is not None and committed_record(run,result_id(r))['technical_status']!='ok']
    require(not failed,'Recorded target failure: inspect status and use retry-target explicitly')
    if limit is not None:pending=pending[:limit]
    if not pending:return {'new_tasks':0,'planned':len(rows)}
    with adapter_for(run) as (adapter,runtime_hash):
        for r in pending:generate_one(adapter,run,r,m['run_identity'],runtime_hash)
    return {'new_tasks':len(pending),'planned':len(rows)}

def extract(run):
    m,rows=load_run(run);n=0
    with adapter_for(run) as (adapter,runtime_hash):
        for row in rows:
            tid=result_id(row);dest=Path(run)/'extracted'/f'{tid}.npz';receipt=dest.with_suffix('.json')
            if receipt.exists():
                old=read(receipt);require(old['sha256']==file_hash(dest) and old['runtime_hash']==runtime_hash,'Extraction identity changed');continue
            a,info=adapter.extract(row,list(range(27)),['t_U','t_D','t_C','t_A'])
            npz_save(dest,activations=a,valid_positions=np.isfinite(a).all(axis=-1),layers=np.arange(27),positions=np.asarray(['t_U','t_D','t_C','t_A']))
            atomic(receipt,{'task_id':tid,'sha256':file_hash(dest),'runtime_hash':runtime_hash,'input_hash':row['input_hash'],
                           'run_identity':m['run_identity'],'use_cache':False,'metadata':info});n+=1
    return {'new_extractions':n,'scope':'separate_native_prefill_use_cache_false'}

def smoke(run):
    m,rows=load_run(run);result=[]
    with adapter_for(run) as (adapter,runtime_hash):
        for row in rows[:6]:
            with Capture(adapter,adapter.serialize(row)) as trace:
                with adapter.torch.inference_mode():adapter.model(**adapter.tensors(adapter.serialize(row)),use_cache=False)
            old,info=adapter.extract(row,list(range(27)),['t_U','t_D','t_C','t_A'])
            a=trace.arrays()['activations'];mask=np.isfinite(a)&np.isfinite(old)
            require(np.array_equal(np.isnan(a),np.isnan(old)),'Missing-position masks differ')
            require(np.array_equal(a[mask],old[mask]),'Ported capture differs from native extraction')
            generated=generate_one(adapter,run,row,m['run_identity'],runtime_hash)
            raw=adapter.generate(row)
            equal=raw['response_token_ids']==generated['response_token_ids']
            result.append({'sample_id':row['sample_id'],'capture_bitwise_equal':True,'unhooked_generation_token_equal':equal})
            require(equal,'Unhooked generation replay differs; inspect numerical environment')
    atomic(Path(run)/'smoke.json',{'passed':True,'checks':result,'runtime_hash':runtime_hash,'run_identity':m['run_identity']})
    return {'passed':True,'checked':len(result)}
