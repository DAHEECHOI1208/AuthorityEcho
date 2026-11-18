"""Condition-balanced centers, all-layer/position scores, and frozen probes."""
import collections
from pathlib import Path
import numpy as np
from .io import require,read,read_jsonl,file_hash,npz_save,atomic,jsonl,digest,load_run
from .judging import metric,status,verdict
from .validation_base import weighted_auc

POSITIONS=['t_U','t_D','t_C','t_A']
MAIN=['C2','C3-E','C3-P','C4']

def load_tensor(path):
    path=Path(path)
    if path.suffix=='.npy':a=np.load(path,allow_pickle=False)
    else:
        with np.load(path,allow_pickle=False) as z:
            if 'activations' in z:a=z['activations'].copy()
            elif all(f'layer_{l}' in z for l in range(27)):
                a=np.full((27,4,2048),np.nan,dtype=np.float32)
                for l in range(27):a[l,3]=z[f'layer_{l}']
            else:raise ValueError('Unknown activation schema; import with explicit axes')
    require(a.shape==(27,4,2048) and a.dtype==np.float32,'Expected [27,4,2048] FP32 activations')
    require(not np.isinf(a).any(),'Infinite activation')
    for v in a.reshape(-1,2048):require(np.isfinite(v).all() or np.isnan(v).all(),'Partially missing activation vector')
    return a

def export_ledger(run,out,batch='main',prefer_extraction=True):
    run=Path(run);m,rows=load_run(run);result=[]
    for p in sorted((run/'outputs').glob('*.json')):
        r=read(p)
        if not r.get('activation_file'):continue
        v=verdict(run,r['task_id'],batch);s=status(r,v);valid=metric(r,v,'strict_hcr')
        group=('H' if r['variant']=='harmful' else 'B')+('A' if s in ('compliant','partially_compliant') else 'R') if valid is not None and s!='clarification_request' else None
        a=run/r['activation_file'];mode='generation_prefill_use_cache_true'
        xp=run/'extracted'/f'{r["task_id"]}.npz'
        if prefer_extraction and r['arm']=='baseline' and xp.exists():a=xp;mode='separate_prefill_use_cache_false'
        result.append({k:r[k] for k in ['sample_id','task_id','question_id','parent_question_id','set_id','variant',
                       'condition_id','dev_role','shared_component_id','arm','layer','alpha']}|
                      {'group':group,'strict_valid':valid is not None,'canonical_label':s,
                       'activation_path':str(a.resolve()),'activation_sha256':file_hash(a),'capture_mode':mode,
                       'response_record_sha256':file_hash(p),'run_identity':m['run_identity'],
                       'input_tokens':len(r['serialized']['input_ids']),'question_chars':len(r['question']),'document_chars':len(r.get('document','')),'document':r.get('document',''),'depth1':r['depth1'],'domain':r['domain']})
    require(len({r['task_id'] for r in result})==len(result),'Duplicate ledger task')
    jsonl(out,result)
    atomic(str(out)+'.frame.json',{'run_identity':m['run_identity'],'rows':[{k:r[k] for k in ('set_id','dev_role','shared_component_id','parent_question_id')} for r in rows]})
    return result

def verified_ledger(path):
    rows=read_jsonl(path);seen=set();base=Path(path).parent
    for r in rows:
        require(r['task_id'] not in seen,'Duplicate ledger task');seen.add(r['task_id'])
        p=Path(r['activation_path']);p=p if p.is_absolute() else base/p
        require(file_hash(p)==r['activation_sha256'],'Activation ledger hash mismatch')
        r['activation_path']=str(p)
    # Components, rather than individual condition rows, must remain in one role.
    roles=collections.defaultdict(set)
    for r in rows:roles[r['set_id'],r['shared_component_id']].add(r['dev_role'])
    require(all(len(v)==1 for v in roles.values()),'Shared component crosses partitions')
    return rows

def fit_centers(ledger,out,scope='fit250',conditions=None):
    conditions=conditions or MAIN;rows=verified_ledger(ledger)
    require(len({r['set_id'] for r in rows})==1,'Fit each set separately; overlapping sets cannot be pooled')
    rows=[r for r in rows if r['arm']=='baseline' and r.get('group') in ('HR','HA','BA')
          and r['condition_id'] in conditions and (scope=='full500' or r['dev_role']=='calibration_fit')]
    require(rows,'No eligible center-fit rows')
    buckets=collections.defaultdict(list)
    for r in rows:buckets[r['group'],r['condition_id'],r['parent_question_id']].append(load_tensor(r['activation_path']).astype(float))
    mu=np.full((27,4,3,2048),np.nan);counts={};missing=[]
    for gi,g in enumerate(('HR','HA','BA')):
        condition_means=[]
        for c in conditions:
            parents=[np.mean(v,axis=0) for (gg,cc,parent),v in buckets.items() if (gg,cc)==(g,c)]
            counts[g+'_'+c]=len(parents)
            if not parents:missing.append(g+'_'+c)
            else:condition_means.append(np.mean(parents,axis=0))
        if len(condition_means)==len(conditions):mu[:,:,gi]=np.mean(condition_means,axis=0)
    require(not missing,'Missing center group/condition cells: '+', '.join(missing))
    valid=np.isfinite(mu).all(axis=(2,3))
    require(valid[:,3].all() and np.all(np.linalg.norm(mu[:,3],axis=-1)>0),'Invalid t_A centers')
    npz_save(out,centers=mu,groups=np.asarray(['HR','HA','BA']),layers=np.arange(27),positions=np.asarray(POSITIONS))
    receipt={'kind':'centers','scope':scope,'sha256':file_hash(out),'conditions':conditions,'center_parent_counts':counts,'valid_layer_positions':valid.tolist(),
             'method':'raw parent means within group-condition; equal condition means; no per-vector normalization before fitting',
             'ledger_sha256':file_hash(ledger),'fit_parent_ids':sorted({r['parent_question_id'] for r in rows}),
             'fit_component_ids':sorted({r['shared_component_id'] for r in rows}),
             'not_independent_validation':scope=='full500'}
    atomic(str(out)+'.json',receipt);return receipt

def load_center_bank(path):
    with np.load(path,allow_pickle=False) as z:
        require(z['groups'].tolist()==['HR','HA','BA'],'Invalid center group order')
        mu=z['centers'].copy()
        if mu.shape==(27,3,2048):
            require(z['position'].item()=='t_A','Invalid center position')
            bank=np.full((27,4,3,2048),np.nan);bank[:,3]=mu;mu=bank
        require(mu.shape==(27,4,3,2048),'Invalid center shape')
        require(z['layers'].tolist()==list(range(27)),'Center layer indexing differs')
        if 'positions' in z:require(z['positions'].tolist()==POSITIONS,'Center position order differs')
        require(np.isfinite(mu[:,3]).all() and np.all(np.linalg.norm(mu[:,3],axis=-1)>0),'Invalid t_A centers')
    return mu

def score_ledger(ledger,center_file,out,direction_file=None):
    rows=verified_ledger(ledger);mu=load_center_bank(center_file)
    norm=np.linalg.norm(mu,axis=-1,keepdims=True);unit=mu/norm
    dirs=None;shared=None
    if direction_file:
        from .directions import load_directions
        dirs,shared=load_directions(direction_file)
    result=[]
    for r in rows:
        a=load_tensor(r['activation_path']).astype(float);an=np.linalg.norm(a,axis=-1,keepdims=True)
        cos=np.einsum('lpd,lpgd->lpg',a/an,unit)
        for l in range(27):
            for p,pos in enumerate(POSITIONS):
                c=cos[l,p];valid=np.isfinite(c).all()
                s={**r,'layer_index':l,'position':pos,'norm':float(an[l,p,0]) if np.isfinite(an[l,p,0]) else None,
                   'f':float(c[0]-c[2]) if valid else None,'g':float(c[1]-c[0]) if valid else None,
                   'a':float(c[1]-c[2]) if valid else None,'cos_HR':float(c[0]) if valid else None,
                   'cos_HA':float(c[1]) if valid else None,'cos_BA':float(c[2]) if valid else None,
                   'center_sha256':file_hash(center_file)}
                if dirs is not None:
                    for j,key in enumerate(('entity_s','purpose_s')):
                        val=float(a[l,p]@dirs[j,l,p]);s[key]=val if np.isfinite(val) else None
                if shared is not None:
                    val=float(a[l,p]@shared[l,p]);s['shared_s']=val if np.isfinite(val) else None
                result.append(s)
    jsonl(out,result);return {'score_rows':len(result),'center_sha256':file_hash(center_file)}

def fit_probes(ledger,out):
    rows=verified_ledger(ledger);index={}
    require(len({r['set_id'] for r in rows})==1,'Fit probes separately by set')
    for r in rows:
        if r['arm']=='baseline' and r['dev_role']=='calibration_fit' and r['variant']=='harmful':
            k=(r['set_id'],r['variant'],r['parent_question_id'],r['condition_id'])
            require(k not in index,'Duplicate parent/condition for probe fit');index[k]=r
    dirs=[];counts={}
    for c in ('C3-E','C3-P'):
        delta=[]
        for k,r in index.items():
            if k[-1]!='C4':continue
            other=index.get((*k[:-1],c))
            if other:delta.append(load_tensor(r['activation_path']).astype(float)-load_tensor(other['activation_path']).astype(float))
        require(delta,'No paired fit inputs for alignment contrast '+c)
        d=np.mean(delta,axis=0);norm=np.linalg.norm(d,axis=-1,keepdims=True)
        require(np.isfinite(d).all(),'Nonfinite alignment fit')
        dirs.append(np.divide(d,norm,out=np.zeros_like(d),where=norm>0));counts[c]=len(delta)
    npz_save(out,directions=np.asarray(dirs),contrasts=np.asarray(['C3-E','C3-P']),layers=np.arange(27),positions=np.asarray(POSITIONS))
    receipt={'kind':'directions','scope':'fit250','sha256':file_hash(out),'fit_pair_counts':counts,
             'direction':'normalize(mean(h_C4 - h_C3)) on fit partition; behavior labels unused',
             'ledger_sha256':file_hash(ledger)}
    atomic(str(out)+'.json',receipt);return receipt
