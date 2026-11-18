"""Bridge unified saved outputs to the original purpose A/B analysis functions."""
from pathlib import Path
import collections
import numpy as np
from .io import require,read_jsonl,read,atomic,csv_rows,file_hash
from . import alignment_statistics as stats


def alignment_report(run,scores,out):
    run=Path(run);out=Path(out);require(not out.exists(),'New alignment report directory required')
    rows=read_jsonl(scores);index={};table=[];missing=[]
    for r in rows:
        if r['position']!='t_A' or r['layer_index'] not in stats.LAYERS:continue
        k=tuple(r[x] for x in ('set_id','variant','parent_question_id','condition_id','arm','layer_index'))
        require(k not in index,'Duplicate alignment readout');index[k]=r
    def state(r):
        group=r.get('group');return group[-1] if group else 'U'
    for sid,variant,parent in sorted({k[:3] for k in index}):
      for axis in ('entity','purpose'):
       for user,target,donor in [('A','C4','C3-E' if axis=='entity' else 'C3-P'),('B',axis+'_BB',axis+'_BA')]:
        b=index.get((sid,variant,parent,target,'baseline',12));d=index.get((sid,variant,parent,donor,'baseline',12))
        if b is None or d is None:continue
        modes={mode:index.get((sid,variant,parent,target,axis+'_'+mode,12)) for mode in stats.CONDITIONS}
        if not all(modes.values()):
            missing.append({'set_id':sid,'variant':variant,'parent':parent,'axis':axis,'user':user,'reason':'missing_alignment_arm'});continue
        r={'set_id':sid,'variant':variant,'axis':axis,'q':parent,'user':user,'split':b['dev_role'],'cluster':b['shared_component_id'],
          'RA':state(d)=='R' and state(b)=='A','donor_baseline':state(d),'target_baseline':state(b)}
        for role,condition in [('donor',donor),('target',target)]:
            for l in stats.LAYERS:
                score=index.get((sid,variant,parent,condition,'baseline',l));require(score is not None and score.get('shared_s') is not None,'Required shared readout missing')
                r[f'{role}_s{l}']=score['shared_s']
        for mode,m in modes.items():
            r[mode+'_state']=state(m)
            for l in stats.LAYERS:
                score=index[sid,variant,parent,target,axis+'_'+mode,l];require(score.get('shared_s') is not None,'Missing shared follow-up score');r[mode+'_s'+str(l)]=score['shared_s']
                if l==16:r[mode+'_f16']=score['f'];r[mode+'_g16']=score['g']
            r[mode+'_donor_residual']=r[mode+'_s12']-r['donor_s12']
            if mode=='random':r['random_relative_norm_error']=read(run/'outputs'/f'{m["task_id"]}.json')['diagnostics']['effects'][0]['relative_norm_error']
        table.append(r)
    summaries={};layer_rows=[]
    for sid,variant,axis in sorted({(r['set_id'],r['variant'],r['axis']) for r in table}):
      for panel,role in [('all_evaluation',None),('check','readout_check'),('select','readout_select')]:
        sub=[r for r in table if (r['set_id'],r['variant'],r['axis'])==(sid,variant,axis) and (role is None or r['split']==role)]
        for user in ('A','B'):
          own=[r for r in sub if r['user']==user]
          for cohort,members in [('all',own),('baseline_RA',[r for r in own if r['RA']])]:
            key='/'.join((sid,variant,axis,panel,user,cohort));summaries[key]=stats.summarize(members)
            layer_rows.extend({'set_id':sid,'variant':variant,'axis':axis,'panel':panel,'user':user,'cohort':cohort,**r} for r in stats.layer_table(members))
        if sub and {r['user'] for r in sub}=={'A','B'}:
            a,b=stats.match_users(sub);keys=[i for i in range(len(a)) if a[i]['RA'] and b[i]['RA']]
            for cohort,aa,bb in [('all',a,b),('RA_intersection',[a[i] for i in keys],[b[i] for i in keys])]:
                summaries['/'.join((sid,variant,axis,panel,'AB_balanced',cohort))]=stats.balanced(aa,bb)
                layer_rows.extend({'set_id':sid,'variant':variant,'axis':axis,'panel':panel,'user':'AB_balanced','cohort':cohort,**r} for r in stats.layer_table(aa,bb))
    out.mkdir(parents=True);atomic(out/'result.json',{'summaries':summaries,'missing':missing,'new_entity_extension':True,
       'direction':'same frozen shared congruence direction for entity and purpose','statistics':'exact cluster resampling for <=4 clusters; otherwise 4000 draws seed 20261108; pointwise intervals',
       'cross_axis_multiple_comparison_claim':False,'scores_sha256':file_hash(scores)})
    csv_rows(out/'item_ledger.csv',table);csv_rows(out/'layer_recovery.csv',layer_rows)
    return {'items':len(table),'panels':len(summaries),'missing':len(missing)}
