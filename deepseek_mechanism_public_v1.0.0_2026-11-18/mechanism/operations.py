"""Read-only audits, explicit recoveries, human/temporal review, and full backups."""
import collections
import json
import os
import tarfile
from pathlib import Path
import numpy as np
from .io import ROOT,require,read,atomic,utc,load_run,file_hash,digest,csv_rows,jsonl,read_jsonl,result_id,committed_record

def status(run):
    run=Path(run);m,rows=load_run(run);outputs=[read(p) for p in (run/'outputs').glob('*.json')]
    counts=collections.Counter((r.get('arm'),r.get('technical_status')) for r in outputs)
    judges={b.name:dict(collections.Counter(read(p).get('judgment_status') for p in (b/'links').glob('*.json'))) for b in (run/'judges').glob('*') if b.is_dir()}
    stages=read(run/'stages.json') if (run/'stages.json').exists() else {}
    result={'run_identity':m['run_identity'],'planned_baselines':len(rows),'output_records':len(outputs),
      'output_status_counts':{'|'.join(map(str,k)):v for k,v in counts.items()},'judges':judges,'stages':stages,
      'fresh_RA_frozen':(run/'fresh_RA.json').exists(),'counts_do_not_prove_success':True}
    if (run/'fresh_RA.json').exists():
        from .interventions import load_cohort,plans
        c=load_cohort(run);p=plans(rows,c,m['study']);pending=[x for x in p if committed_record(run,result_id(x[0],*x[1:])) is None]
        result.update(fresh_RA=len(c['members']),planned_interventions=len(p),unrecorded_interventions=len(pending))
        eta=0;missing=0
        for _,arm,layer,alpha in pending:
            times=[r['seconds'] for r in outputs if r.get('arm')==arm and r.get('layer')==layer and r.get('alpha')==alpha and r.get('technical_status')=='ok' and r.get('seconds')]
            if len(times)>=3:eta+=float(np.median(times[-20:]))
            else:missing+=1
        result['generation_eta_seconds_excludes_judging_loading']=eta if not missing else None
        result['eta_tasks_without_timing_sample']=missing
    return result

def run_stage(run,name,fn):
    run=Path(run);p=run/'stages.json';st=read(p) if p.exists() else {};st[name]={'status':'running','started_at':utc(),'pid':os.getpid()};atomic(p,st)
    try:r=fn()
    except BaseException as e:
        st[name].update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',ended_at=utc(),error_type=type(e).__name__)
        atomic(p,st);raise
    st[name].update(status='completed',ended_at=utc());atomic(p,st);return r

def retry_target(run,task_id):
    run=Path(run);m,_=load_run(run);r=committed_record(run,task_id)
    require(r is not None and r['technical_status']!='ok','Only recorded unsuccessful target work can be retried')
    require(not any((b/'links'/f'{task_id}.json').exists() for b in (run/'judges').glob('*')),'Judged output cannot be replaced by target retry')
    require(not (run/'fresh_RA.json').exists(),'Frozen run cannot change baseline output; use a separate run')
    dest=run/'recovery'/utc().replace(':','-');dest.mkdir(parents=True)
    (run/'outputs'/f'{task_id}.json').rename(dest/f'{task_id}.json')
    return {'archived_failed_record':str(dest),'retry_occurs_only_on_next_run':True}

def retry_judge(run,batch,task_id):
    run=Path(run);load_run(run);p=run/'judges'/batch/'links'/f'{task_id}.json';v=read(p)
    require(v.get('judgment_status')=='failed','Only a failed judgment may be explicitly reset')
    if (run/'fresh_RA.json').exists():
        require(f'judges/{batch}/links/{task_id}.json' not in read(run/'fresh_RA.json')['baseline_bindings'],'Cannot change frozen baseline selection evidence')
    key=v['payload_sha256'];base=run/'judges'/batch
    linked=[x for x in (base/'links').glob('*.json') if read(x).get('payload_sha256')==key]
    require(all(read(x).get('judgment_status')=='failed' for x in linked),'A successful result is linked to this payload')
    dest=run/'recovery'/utc().replace(':','-');dest.mkdir(parents=True)
    for x in linked:x.rename(dest/x.name)
    cache=base/'cache'/key
    if cache.exists():cache.rename(dest/key)
    atomic(dest/'recovery.json',{'action':'explicit_failed_judge_retry','batch':batch,'payload':key,'at':utc(),'prior_attempts_preserved':True})
    return {'archived_failure':str(dest),'no_api_called':True}

def audit_npz(path,out):
    path=Path(path);files=sorted(path.rglob('*.npz')) if path.is_dir() else [path];results=[]
    for f in files:
        row={'path':str(f),'sha256':file_hash(f),'arrays':{},'valid':True}
        try:
            with np.load(f,allow_pickle=False) as z:
                for k in z.files:
                    a=z[k];numeric=a.dtype.kind in 'fiu';nans=int(np.isnan(a).sum()) if a.dtype.kind=='f' else 0
                    row['arrays'][k]={'shape':list(a.shape),'dtype':str(a.dtype),'nan_count':nans,'infinite_count':int(np.isinf(a).sum()) if numeric else 0}
                    if numeric and np.isinf(a).any():row['valid']=False
                if 'activations' in z:
                    from .scoring import load_tensor
                    load_tensor(f)
        except Exception as e:row.update(valid=False,error_type=type(e).__name__,error=str(e))
        results.append(row)
    atomic(out,{'files':results,'count':len(files),'all_structural_checks_passed':bool(files) and all(r['valid'] for r in results),
                'missing_positions_are_not_numeric_failures':True,'no_threshold_acceptance_claim':True})
    return {'files':len(files),'invalid':sum(not r['valid'] for r in results)}

def backup(run,out):
    run=Path(run).resolve();out=Path(out).resolve();load_run(run)
    require(not out.exists() and not out.is_relative_to(run) and not out.is_relative_to(ROOT),'Backup must be new and outside source and run')
    entries=[p for root in (run,ROOT) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='.lock']
    before={str(p):file_hash(p) for p in entries};out.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(out,'x:gz') as tar:
        for p in entries:
            root=run if p.is_relative_to(run) else ROOT
            tar.add(p,arcname=('run/' if root==run else 'source/')+str(p.relative_to(root)),recursive=False)
    require(all(file_hash(p)==h for p,h in before.items()),'Files changed during backup; archive is not a completed snapshot')
    atomic(str(out)+'.inventory.json',{'created_at':utc(),'archive_sha256':file_hash(out),'file_hashes':before,'includes_activations':True,'contains_private_data':True})
    Path(str(out)+'.sha256').write_text(file_hash(out)+'  '+out.name+'\n')
    return {'archive':str(out),'sha256':file_hash(out),'files':len(entries),'private_backup_do_not_publish':True}

def temporal_report(run,first,second,out):
    run=Path(run);load_run(run);rows=[]
    a=run/'judges'/first/'links';b=run/'judges'/second/'links'
    for p in sorted(a.glob('*.json')):
        q=b/p.name
        if not q.exists():continue
        x,y=read(p),read(q);require(x['response_record_sha256']==y['response_record_sha256'],'Temporal comparison response changed')
        valid=x['judgment_status']==y['judgment_status']=='ok'
        rows.append({'task_id':p.stem,'first_label':x.get('canonical_label'),'second_label':y.get('canonical_label'),
                     'comparable':valid,'agreement':x.get('canonical_label')==y.get('canonical_label') if valid else None,
                     'same_text_hash':x['response_record_sha256'],'first_fingerprint':x.get('system_fingerprint'),'second_fingerprint':y.get('system_fingerprint')})
    good=[r for r in rows if r['comparable']]
    atomic(out,{'first_batch':first,'second_batch':second,'rows':rows,'comparable_n':len(good),
       'agreement':sum(r['agreement'] for r in good)/len(good) if good else None,'fingerprint_null_is_match':False,
       'agreement_is_not_accuracy':True,'historical_labels_changed':False})
    return {'comparable':len(good),'changed':sum(not r['agreement'] for r in good)}

def human_review(run,out,review_file=None,batch='main'):
    from .judging import verdict,status,CODES
    run=Path(run);load_run(run);records={p.stem:read(p) for p in (run/'outputs').glob('*.json')};rows=[]
    if review_file is None:
        for tid,r in records.items():
            rows.append({'task_id':tid,'response_sha256':digest(r.get('response_text','')),'question':r.get('question'),
              'response':r.get('response_text'),'human_label':None,'human_reason':None})
        jsonl(out,rows);return {'exported':len(rows),'no_sampling_or_quality_claim':True}
    seen=set()
    for h in read_jsonl(review_file):
        tid=h['task_id'];require(tid not in seen and tid in records,'Duplicate or unknown review task');seen.add(tid);r=records[tid]
        require(h['response_sha256']==digest(r.get('response_text','')),'Human review text changed')
        require(h['human_label'] in CODES+('technical_error',),'Invalid human label')
        if h['human_label']=='technical_error':require(r.get('technical_status')!='ok' or not r.get('response_text','').strip(),'Technical error cannot replace content disagreement')
        v=verdict(run,tid,batch);s=status(r,v);comparable=s in CODES and h['human_label'] in CODES
        rows.append({**h,'judge_label':s,'comparable':comparable,'agreement':s==h['human_label'] if comparable else None})
    atomic(out,{'rows':rows,'judge_quality_validated':False,'historical_labels_overwritten':False,'human_is_not_automatic_ground_truth':True})
    return {'reviewed':len(rows),'comparable':sum(r['comparable'] for r in rows)}
