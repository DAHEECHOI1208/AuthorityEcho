"""All-layer/position readouts, common cohorts, cross contrasts and alerts."""
import collections
from pathlib import Path
import numpy as np
from .io import require,read_jsonl,atomic,csv_rows,jsonl,file_hash
from .validation_base import weighted_auc
from .representation import cluster_mean
from .alert_base import calibrate_threshold,confusion


def auc_interval(scores,y,clusters,seed=20261102,repeats=2000):
    s=np.asarray(scores,float);y=np.asarray(y,bool);g=np.asarray(clusters);levels=sorted(set(g.tolist()))
    point=weighted_auc(s,y,np.ones(len(s)));samples=[]
    if len(levels)>=2 and point is not None:
        counts=np.random.default_rng(seed).multinomial(len(levels),[1/len(levels)]*len(levels),size=repeats)
        ix=np.array([levels.index(c) for c in g])
        for weights in counts[:,ix]:
            v=weighted_auc(s,y,weights)
            if v is not None:samples.append(v)
    return {'auc':point,'n':len(s),'components':len(levels),'positive_n':int(y.sum()),
      'ci95':np.quantile(samples,[.025,.975]).tolist() if len(samples)>=.95*repeats else None,
      'valid_bootstrap':len(samples),'bootstrap_total':repeats,
      'reason':None if len(samples)>=.95*repeats else 'missing_class_or_insufficient_components'}


def auc_curves(scores,out):
    rows=read_jsonl(scores);buckets=collections.defaultdict(list);result=[]
    for r in rows:
        if r['arm']=='baseline' and r.get('group'):
            buckets[r['set_id'],r['dev_role'],r['condition_id'],r['layer_index'],r['position']].append(r)
    for key,rs in buckets.items():
        for metric,positive,negative,comparison in [('f','HA','BA','H_A'),('f','HR','BR','H_R'),('f','HR','HA','R_H'),('f','BR','BA','R_B'),
          ('a','HA','BA','HA_BA'),('g','HA','HR','HA_HR')]:
            use=[r for r in rs if r.get(metric) is not None and r['group'] in (positive,negative)]
            result.append(dict(zip(['set_id','role','condition','layer','position'],key))|
              {'metric':metric,'positive_group':positive,'negative_group':negative,'comparison':comparison,
               **auc_interval([r[metric] for r in use],[r['group']==positive for r in use],[r['shared_component_id'] for r in use])})
    csv_rows(out,result);return {'rows':len(result),'fixed_centers':True,'pointwise_intervals_not_layer_search_corrected':True}


def common_contrasts(scores,out):
    rows=[r for r in read_jsonl(scores) if r['arm']=='baseline'];index={};buckets=collections.defaultdict(list);summary=[];ledger=[]
    for r in rows:
        k=tuple(r[x] for x in ('set_id','variant','dev_role','parent_question_id','layer_index','position'))
        require((k,r['condition_id']) not in index,'Duplicate score condition');index[k,r['condition_id']]=r
    for k in sorted({k for k,c in index}):
        group=[index.get((k,c)) for c in ('C0','C3-E','C3-P','C4')]
        for name in ('f','g','a','entity_s','purpose_s','shared_s'):
            valid=all(r is not None and r.get(name) is not None and r.get('group') is not None for r in group)
            row=dict(zip(('set_id','variant','role','parent_question_id','layer','position'),k))|{'metric':name,'eligible':valid}
            if valid:
                v0,ve,vp,v4=[r[name] for r in group];row.update(component=group[0]['shared_component_id'],C0_C4=v4-v0,C0_C3E=ve-v0,C0_C3P=vp-v0,C3E_C4=v4-ve,C3P_C4=v4-vp,
                      entity_additivity_residual=(v4-v0)-((ve-v0)+(v4-ve)),purpose_additivity_residual=(v4-v0)-((vp-v0)+(v4-vp)))
                buckets[*(row[x] for x in ('set_id','variant','role','layer','position','metric'))].append(row)
            ledger.append(row)
    for k,rs in buckets.items():
        for contrast in ('C0_C4','C0_C3E','C0_C3P','C3E_C4','C3P_C4'):
            summary.append(dict(zip(('set_id','variant','role','layer','position','metric'),k))|
                           {'contrast':contrast,**cluster_mean([r[contrast] for r in rs],[r['component'] for r in rs])})
    out=Path(out);require(not out.exists(),'New output directory required');out.mkdir(parents=True)
    csv_rows(out/'common_cohort_changes.csv',summary);jsonl(out/'common_cohort_audit.jsonl',ledger)
    return {'summary_rows':len(summary),'cohort_rule':'All four conditions strictly eligible on identical parents; additive identity is not causal mediation'}


def cross_report(scores,out):
    rows=read_jsonl(scores);index={};result=[];ledger=[]
    for r in rows:
        if r['arm']!='baseline' or not any(r['condition_id'].startswith(a+'_') for a in ('entity','purpose')):continue
        key=tuple(r[x] for x in ('set_id','variant','parent_question_id','layer_index','position','dev_role'))
        require((key,r['condition_id']) not in index,'Duplicate cross score');index[key,r['condition_id']]=r
    buckets=collections.defaultdict(list)
    for key in sorted({k for k,c in index}):
      for axis in ('entity','purpose'):
        rs=[index.get((key,axis+'_'+c)) for c in ('AA','AB','BA','BB')]
        for metric in ('f','g','a',axis+'_s','shared_s','behavior'):
            vals=[None if r is None else (int(r['group'][-1]=='A') if r.get('group') else None) if metric=='behavior' else r.get(metric) for r in rs]
            valid=all(v is not None for v in vals)
            if not valid:continue
            aa,ab,ba,bb=vals
            row=dict(zip(('set_id','variant','parent_question_id','layer','position','role'),key))|{'axis':axis,'metric':metric,'component':rs[0]['shared_component_id'],
              'AA_minus_AB':aa-ab,'BB_minus_BA':bb-ba,'mean_aligned_minus_misaligned':(aa+bb-ab-ba)/2,'difference_in_differences':(aa-ab)-(ba-bb)}
            ledger.append(row);buckets[*(row[x] for x in ('set_id','variant','layer','position','role','axis','metric'))].append(row)
    for k,rs in buckets.items():
      for c in ('AA_minus_AB','BB_minus_BA','mean_aligned_minus_misaligned','difference_in_differences'):
        from .alignment_statistics import ClusterSample
        sample=ClusterSample([{'cluster':r['component']} for r in rs])
        stat=sample.mean([r[c] for r in rs],np.ones(len(rs),bool))
        result.append(dict(zip(('set_id','variant','layer','position','role','axis','metric'),k))|{'contrast':c,**stat,'cluster_interval_method':sample.method})
    out=Path(out);require(not out.exists(),'New output directory required');out.mkdir(parents=True)
    csv_rows(out/'cross_contrasts.csv',result);jsonl(out/'cross_pairs.jsonl',ledger)
    return {'rows':len(result),'center_refit':False,'no_causal_interpretation_from_observation_alone':True}


def alert_analysis(scores,out,layer=16,position='t_A',metric='g',mode='c4',budget=.05):
    require(mode in ('c4','delta'),'Alert mode must be c4 or delta')
    rows=[r for r in read_jsonl(scores) if r['arm']=='baseline' and r['layer_index']==layer and r['position']==position]
    index={};samples=[]
    for r in rows:
        k=(r['set_id'],r['variant'],r['parent_question_id'],r['condition_id']);require(k not in index,'Duplicate alert observation');index[k]=r
    for k,r in index.items():
        if k[-1]!='C4':continue
        c0=index.get((*k[:-1],'C0'));value=r.get(metric)
        if mode=='delta':value=value-c0[metric] if value is not None and c0 and c0.get(metric) is not None else None
        if value is None or not r.get('group'):continue
        samples.append({'set_id':r['set_id'],'parent':r['parent_question_id'],'variant':r['variant'],'role':r['dev_role'],
            'component':r['shared_component_id'],'score':value,'y':int(r['group']=='HA'),'group':r['group']})
    results=[]
    for sid in sorted({r['set_id'] for r in samples}):
        rs=[r for r in samples if r['set_id']==sid];fit=[r for r in rs if r['role']=='readout_select'];ev=[r for r in rs if r['role']=='readout_check']
        require(not {r['component'] for r in fit}&{r['component'] for r in ev},'Alert calibration/evaluation component overlap')
        negatives=[r['score'] for r in fit if r['variant']=='harmful' and r['y']==0]
        if not negatives or not ev:
            results.append({'set_id':sid,'status':'not_estimable','reason':'missing_select_negatives_or_check_records'});continue
        tau=calibrate_threshold(negatives,budget)
        for variant in ('harmful','benign'):
            use=[r for r in ev if r['variant']==variant];g=[r['component'] for r in use];s=np.array([r['score'] for r in use]);y=np.array([r['y'] for r in use])
            conf=confusion(y,s,tau) if use else {};rates=cluster_mean([float(r['score']>tau) for r in use],g)
            results.append({'set_id':sid,'variant':variant,'threshold':tau,'calibration_negative_n':len(negatives),
              'budget':budget,'layer':layer,'position':position,'metric':metric,'mode':mode,'confusion':conf,'alert_rate':rates,
              'auc':auc_interval(s,y,g),'threshold_recalibrated_on_evaluation':False})
    atomic(out,{'results':results,'scores_sha256':file_hash(scores),'threshold_rule':'strict score > tau; ties conservative',
      'fixed_threshold_uncertainty':True,'exploratory_unless_candidate_precommitted':True,'benign_alert_is_not_benign_refusal':True})
    return {'rows':len(results)}
