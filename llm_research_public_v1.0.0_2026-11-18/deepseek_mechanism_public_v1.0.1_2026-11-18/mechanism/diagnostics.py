"""Optional diagnostic-only repeat, prefix, layer-zero, and projection checks."""
from pathlib import Path
import numpy as np
from .io import require,atomic,npz_save,file_hash,load_run,result_id,committed_record
from .scoring import load_tensor,POSITIONS


def compare(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape:return {'status':'shape_mismatch'}
    if not np.isfinite(a).all() or not np.isfinite(b).all():return {'status':'nonfinite_or_missing'}
    an=np.linalg.norm(a);bn=np.linalg.norm(b)
    if min(an,bn)==0:return {'status':'zero_norm'}
    l2=float(np.linalg.norm(b-a)/an);cos=float(np.dot(a.ravel(),b.ravel())/(an*bn))
    return {'status':'ok','bitwise_equal':bool(np.array_equal(a,b)),'relative_l2':l2,'cosine':cos,'max_abs':float(np.max(abs(a-b))),
      'left_norm':float(an),'right_norm':float(bn),'norm_ratio':float(bn/an),'reference_exceedance':l2>.04 or cos<.9994}


def prefix_audit(run,out):
    run=Path(run);m,rows=load_run(run);index={(r['set_id'],r['variant'],r['question_id'],r['condition_id']):r for r in rows};report=[]
    for row in rows:
        if row['condition_id']!='C4':continue
        c4=committed_record(run,result_id(row))
        if not c4:continue
        A=load_tensor(run/c4['activation_file'])
        for condition in ('C3-E','C3-P'):
            other=index[row['set_id'],row['variant'],row['question_id'],condition];r=committed_record(run,result_id(other))
            if not r:continue
            ia=c4['serialized']['positions']['t_U'][0];ib=r['serialized']['positions']['t_U'][0]
            prefix_equal=ia==ib and c4['serialized']['input_ids'][:ia+1]==r['serialized']['input_ids'][:ib+1]
            B=load_tensor(run/r['activation_file'])
            for layer in range(27):report.append({'sample_id':row['sample_id'],'contrast':condition+'->C4','layer':layer,'position':'t_U','prefix_token_ids_equal':prefix_equal,**compare(B[layer,0],A[layer,0])})
    atomic(out,{'run_identity':m['run_identity'],'rows':report,'status':'diagnostic_only_no_threshold_relaxation',
       'reference_relative_l2_max':.04,'reference_cosine_min':.9994,'reference_policy_is_deepseek_validated':False,
       'passed':None,'same_prefix_does_not_imply_same_shape_computation':True})
    return {'checks':len(report),'exceedances':sum(r.get('reference_exceedance',False) for r in report),'acceptance_gate':False}


def gpu_diagnostics(run,out,mode='repeat',sample_id=None,repeats=3):
    from .runtime import adapter_for
    from .hooks import Capture
    run=Path(run);out=Path(out);require(not out.exists(),'New diagnostic output directory required');m,rows=load_run(run)
    row=next((r for r in rows if r['sample_id']==sample_id),None) if sample_id else next(r for r in rows if r['condition_id']=='C4')
    require(row is not None,'Requested sample does not exist');out.mkdir(parents=True);report=[]
    with adapter_for(run) as (adapter,runtime):
        serial=adapter.serialize(row);torch=adapter.torch
        if mode=='repeat':
            require(2<=repeats<=10,'Specify 2 to 10 diagnostic repeats');first=None
            for i in range(repeats):
                a,meta=adapter.extract(row,list(range(27)),POSITIONS);npz_save(out/f'repeat_{i}.npz',activations=a)
                if first is None:first=a
                for layer in range(27):
                    for p,pos in enumerate(POSITIONS):report.append({'repeat':i,'layer':layer,'position':pos,**compare(first[layer,p],a[layer,p])})
        elif mode in ('layer0','length','o-proj'):
            modules=dict(adapter.model.named_modules());required=['model.layers.0.input_layernorm','model.layers.0.self_attn','model.layers.0.self_attn.o_proj','model.layers.0.post_attention_layernorm','model.layers.0.mlp']
            require(all(n in modules for n in required),'Expected layer-zero modules unavailable')
            captured={};handles=[];target=serial['positions']['t_U'][0]
            def make_hook(name):
                def hook(mod,args,outval):
                    h=outval[0] if isinstance(outval,(tuple,list)) else outval
                    if hasattr(h,'shape') and h.ndim==3:captured[name]=h.detach().float().cpu().numpy()
                    if name.endswith('o_proj'):captured['projection_input_tensor']=args[0].detach().clone()
                return hook
            try:
                for name in required:handles.append(modules[name].register_forward_hook(make_hook(name)))
                with torch.inference_mode():adapter.model(**adapter.tensors(serial),use_cache=False)
            finally:
                for h in handles:h.remove()
            arrays={k.replace('.','_'):v[0,target].copy() for k,v in captured.items() if isinstance(v,np.ndarray)}
            npz_save(out/'layer0_tU.npz',**arrays)
            if mode in ('length','o-proj'):
                x=captured['projection_input_tensor'];proj=modules['model.layers.0.self_attn.o_proj'];full_len=x.shape[1]
                with torch.inference_mode():
                    full=proj(x)[0,target].float().cpu().numpy()
                    for n in sorted({target+1,min(full_len,target+8),full_len}):
                        shortened=proj(x[:,:n])[0,target].float().cpu().numpy()
                        report.append({'module':'model.layers.0.self_attn.o_proj','same_prefix_values':True,'full_input_tokens':int(full_len),'test_input_tokens':int(n),'token_index':target,**compare(full,shortened)})
            else:report=[{'module':k,'captured_token':target,'shape':list(v.shape)} for k,v in arrays.items()]
        else:raise ValueError('Unknown diagnostic mode')
    atomic(out/'report.json',{'run_identity':m['run_identity'],'runtime_hash':runtime,'mode':mode,'sample_id':row['sample_id'],'rows':report,
      'diagnostic_only':True,'thresholds_changed':False,'universal_determinism_proven':False,'generation_or_judge_reused':False})
    return {'diagnostic_rows':len(report),'model_calls_executed':True}
