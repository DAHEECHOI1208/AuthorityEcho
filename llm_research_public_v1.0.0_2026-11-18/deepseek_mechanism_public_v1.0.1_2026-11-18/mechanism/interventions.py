"""Freeze a pre-intervention cohort and execute independently reset treatments."""
from pathlib import Path
import numpy as np
from .io import require,read,atomic,digest,file_hash,load_run,result_id,committed_record,csv_rows
from .judging import verdict,metric,status
from .runtime import adapter_for,generate_one

def item_key(r):return (r['set_id'],r['variant'],r['question_id'])

def freeze(run):
    run=Path(run);m,rows=load_run(run);index={(*item_key(r),r['condition_id']):r for r in rows}
    chosen=[];audit=[];bindings={}
    for r in [r for r in rows if r['condition_id']=='C0']:
        c4=index[(*item_key(r),'C4')];t0=result_id(r);t4=result_id(c4)
        a,b=committed_record(run,t0),committed_record(run,t4);va,vb=verdict(run,t0),verdict(run,t4)
        require(a is not None and b is not None and va is not None and vb is not None,'Complete C0/C4 generation and judging before freezing')
        for tid in (t0,t4):
            for rel in (f'outputs/{tid}.json',f'judges/main/links/{tid}.json'):bindings[rel]=file_hash(run/rel)
        s0,s4=status(a,va),status(b,vb);r0=metric(a,va,'refusal');a4=metric(b,vb,'strict_hcr')
        is_ra=r['variant']=='harmful' and r0==1 and a4==1
        state='fresh_RA' if is_ra else 'fresh_RR' if r0==1 and metric(b,vb,'refusal')==1 else 'other'
        reasons=[]
        if r['variant']!='harmful':reasons.append('benign_not_in_harmful_RA')
        if r0!=1:reasons.append('fresh_C0_refusal_not_confirmed')
        if a4!=1:reasons.append('fresh_C4_compliance_not_confirmed')
        if r.get('dev_role')=='calibration_fit':
            is_ra=False;reasons.append('center_fit_partition_not_intervention_evaluation')
        if is_ra:chosen.append(c4['sample_id'])
        audit.append({'sample_id':r['sample_id'],'set_id':r['set_id'],'variant':r['variant'],
                      'parent_question_id':r['parent_question_id'],'shared_component_id':r['shared_component_id'],
                      'C0_status':s0,'C4_status':s4,'fresh_C0_refusal':r0,'fresh_C4_compliant':a4,
                      'baseline_transition':state,'fresh_RA':is_ra,'exclusion_reasons':reasons})
    generation_members=list(chosen)
    if read(run/'protocol.json').get('cohort_mode')=='historical':
        harmful={(r['set_id'],r['parent_question_id']) for r in rows if r['variant']=='harmful' and r.get('historical_C0_label')=='refusal' and r['dev_role']!='calibration_fit'}
        generation_members=[r['sample_id'] for r in rows if r['condition_id']=='C4' and (r['set_id'],r['parent_question_id']) in harmful and r['dev_role']!='calibration_fit']
    result={'generation_members':generation_members,'run_identity':m['run_identity'],'selection_rule':'valid fresh C0 explicit refusal and valid fresh C4 full/partial compliance',
            'members':chosen,'audit':audit,'baseline_bindings':bindings,'intervention_outcomes_used':False}
    result['cohort_hash']=digest(result)
    if (run/'fresh_RA.json').exists():require(read(run/'fresh_RA.json')==result,'Frozen baseline selection changed')
    else:atomic(run/'fresh_RA.json',result)
    csv_rows(run/'fresh_C0_audit.csv',audit)
    return {'fresh_RA':len(chosen),'eligible_items':len(audit),'cohort_hash':result['cohort_hash']}

def load_cohort(run):
    run=Path(run);c=read(run/'fresh_RA.json')
    require(c['cohort_hash']==digest({k:v for k,v in c.items() if k!='cohort_hash'}),'Cohort record changed')
    for rel,h in c['baseline_bindings'].items():require(file_hash(run/rel)==h,'Frozen baseline evidence changed')
    return c

def centers(run):
    run=Path(run);receipt=read(run/'centers_receipt.json')
    require(receipt['scope']=='fit250' and receipt['sha256']==file_hash(run/'centers.npz'),'Invalid fixed centers')
    with np.load(run/'centers.npz',allow_pickle=False) as z:
        require(z['groups'].tolist()==['HR','HA','BA'],'Center group order differs')
        a=z['centers'].copy()
        if a.shape==(27,3,2048):
            require(z['layers'].tolist()==list(range(27)) and z['position'].item()=='t_A','Center axes differ')
            return a
        require(a.shape==(27,4,3,2048) and z['positions'].tolist()==['t_U','t_D','t_C','t_A'],'Center axes differ')
        return a[:,3]

def vector_from(run,row,layer):
    rec=committed_record(run,result_id(row));require(rec is not None and rec['technical_status']=='ok','Donor/baseline result missing')
    require(rec['input_hash']==row['input_hash'] and item_key(rec)==item_key(row),'Donor linkage mismatch')
    with np.load(Path(run)/rec['activation_file'],allow_pickle=False) as z:
        require(z['positions'].tolist()==['t_U','t_D','t_C','t_A'] and z['layers'].tolist()==list(range(27)),'Activation axes changed')
        v=z['activations'][layer,3].copy()
    require(v.shape==(2048,) and v.dtype==np.float32 and np.isfinite(v).all(),'Invalid donor vector')
    return v,rec

def plans(rows,cohort,study):
    selected=[r for r in rows if r['sample_id'] in cohort.get('generation_members',cohort['members'])]
    out=[]
    for r in selected:
        if study in ('restore','all','strength'):
            for l in (16,20):
                for arm in ('sham','restore','random'):out.append((r,arm,l,0. if arm=='sham' else 1.))
        if study in ('strength','all'):
            for a in (1.5,2.):
                for arm in ('restore','random'):out.append((r,arm,16,a))
            out.append((r,'hrmean',16,1.))
        if study in ('full-vector','all'):
            for l in (16,20):out.append((r,'full_replace',l,1.))
    return out

def run_interventions(run,study=None,limit=None):
    run=Path(run);m,rows=load_run(run);study=study or m['study'];cohort=load_cohort(run)
    planned=plans(rows,cohort,study)
    pending=[p for p in planned if committed_record(run,result_id(p[0],*p[1:])) is None]
    require(all((committed_record(run,result_id(p[0],*p[1:])) or {'technical_status':'ok'})['technical_status']=='ok' for p in planned),'Recorded intervention failure requires explicit retry-target')
    if limit is not None:pending=pending[:limit]
    if not pending:return {'planned':len(planned),'new_tasks':0,'fresh_RA':len(cohort['members'])}
    mu=centers(run);index={(*item_key(r),r['condition_id']):r for r in rows}
    with adapter_for(run) as (adapter,env):
        for row,arm,layer,alpha in pending:
            c0=index[(*item_key(row),'C0')];donor,dr=vector_from(run,c0,layer);before,br=vector_from(run,row,layer)
            require(dr['runtime_hash']==br['runtime_hash']==env and dr['run_identity']==m['run_identity'],'Donor environment or execution identity mismatch')
            norm=None
            if arm=='random':
                restored=committed_record(run,result_id(row,'restore',layer,alpha))
                require(restored is not None and restored['technical_status']=='ok','Corresponding restore result required before random control')
                norm=restored['diagnostics']['effects'][0]['actual_delta_norm']
            patch={'arm':arm,'layer':layer,'alpha':alpha,'donor':donor,'expected_before':before,'mu':mu[layer],
                   'identity':':'.join(item_key(row))+'|'+str(layer),'random_target':norm,'donor_task_id':dr['task_id'],'donor_activation_sha256':dr['activation_sha256']}
            generate_one(adapter,run,row,m['run_identity'],env,arm,layer,alpha,patch)
    return {'planned':len(planned),'new_tasks':len(pending),'fresh_RA':len(cohort['members'])}

def alignment_restore(run,axis='both',limit=None):
    run=Path(run);m,rows=load_run(run)
    receipt=read(run/'directions_receipt.json');require(receipt['sha256']==file_hash(run/'directions.npz'),'Direction identity changed')
    require(receipt['scope']=='fit250','Alignment directions must be frozen on fit250')
    from .directions import load_directions
    direct,shared=load_directions(run/'directions.npz')
    if (run/'cross.json').exists():
        from .inputs import cross_rows
        rows=rows+cross_rows(rows,read(run/'cross.json'))
    index={(*item_key(r),r['condition_id']):r for r in rows}
    mu=centers(run) if (run/'centers.npz').exists() else None
    pending=[]
    axes=[('entity',0,'C3-E'),('purpose',1,'C3-P')]
    for label,j,condition in axes:
        if axis not in ('both',label):continue
        targets=[(r,condition,'A') for r in rows if r['condition_id']=='C4' and r['dev_role']!='calibration_fit']
        targets += [(r,label+'_BA','B') for r in rows if r['condition_id']==label+'_BB' and r['dev_role']!='calibration_fit']
        for r,donor_condition,persona in targets:
            for arm in ('sham','restore','random','full_swap'):
                name=label+'_'+arm
                old=committed_record(run,result_id(r,name,12,1.))
                require(old is None or old['technical_status']=='ok','Prior alignment failure must be resolved explicitly')
                if old is None:pending.append((r,label,j,donor_condition,arm,name,persona))
    if limit is not None:pending=pending[:limit]
    if not pending:return {'new_tasks':0}
    with adapter_for(run) as (adapter,env):
        for r,label,j,condition,arm,name,persona in pending:
            drow=index[(*item_key(r),condition)];donor,dr=vector_from(run,drow,12);before,br=vector_from(run,r,12)
            require(dr['runtime_hash']==br['runtime_hash']==env,'Alignment donor environment mismatch')
            target=None
            if arm=='random':
                prev=committed_record(run,result_id(r,label+'_restore',12,1.));require(prev is not None,'Alignment restoration missing')
                target=prev['diagnostics']['effects'][0]['actual_delta_norm']
            patch={'layer':12,'arm':arm,'donor':donor,'expected_before':before,'w':shared[12,3],
                   'mu':mu[12] if mu is not None else None,'seed':42,'identity':':'.join(item_key(r))+'|'+label+'|'+persona,
                   'random_target':target,'donor_task_id':dr['task_id'],'donor_activation_sha256':dr['activation_sha256']}
            generate_one(adapter,run,r,m['run_identity'],env,name,12,1.,patch)
    return {'new_tasks':len(pending),'entity_extension':'new implementation; no historical execution claimed'}
