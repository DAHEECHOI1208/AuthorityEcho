"""Original adjusted common/benign logistic models on explicit paired score data."""
import collections
from pathlib import Path
import numpy as np
from . import regression_base as rb
from .io import require,read_jsonl,atomic,csv_rows,jsonl,file_hash


def clean(value):
    if isinstance(value,np.ndarray):return clean(value.tolist())
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(x) for x in value]
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    return value


def build_features(scores):
    rows=[r for r in read_jsonl(scores) if r['arm']=='baseline' and r['position']=='t_A'];index={};output=[];excluded=[]
    for r in rows:
        k=(r['set_id'],r['parent_question_id'],r['variant'],r['condition_id'],r['layer_index'])
        require(k not in index,'Duplicate paired regression score');index[k]=r
    for sid,parent in sorted({k[:2] for k in index}):
      for layer in range(27):
        rs=[index.get((sid,parent,v,c,layer)) for v in ('harmful','benign') for c in ('C0','C4')]
        if any(r is None for r in rs):excluded.append({'set_id':sid,'parent':parent,'layer':layer,'reason':'missing_matched_H_B_C0_C4'});continue
        h0,h4,b0,b4=rs
        if h0.get('group')!='HR' or h4.get('group') not in ('HA','HR') or any(r.get('g') is None or r.get('f') is None for r in rs):
            excluded.append({'set_id':sid,'parent':parent,'layer':layer,'reason':'C0_refusal_C4_RA_RR_or_numeric_eligibility'});continue
        require(len({r['dev_role'] for r in rs})==1 and len({r['shared_component_id'] for r in rs})==1,'Paired roles/components differ')
        rec={'set_id':sid,'parent_id':parent,'layer':layer,'component':h0['shared_component_id'],'role':h0['dev_role'],'depth1':h0['depth1'],'y':int(h4['group']=='HA'),
             'benign_transition':str(b0.get('group'))+'->'+str(b4.get('group'))}
        for prefix,a,b in [('H',h0,h4),('B',b0,b4)]:
            required=('question_chars','input_tokens','document_chars')
            require(all(k in a and k in b for k in required),'Regression covariates absent; export ledger from original inputs instead of guessing lengths')
            rec[prefix+'_question_chars']=a['question_chars'];rec[prefix+'_tokens0']=a['input_tokens'];rec[prefix+'_added_tokens']=b['input_tokens']-a['input_tokens'];rec[prefix+'_document_chars4']=b['document_chars']
            for name in ('g','f'):
                rec[prefix+'_'+name+'0']=a[name];rec[prefix+'_'+name+'4']=b[name];rec[prefix+'_d'+name]=b[name]-a[name]
            ma=np.mean([a['cos_HR'],a['cos_HA'],a['cos_BA']]);mb=np.mean([b['cos_HR'],b['cos_HA'],b['cos_BA']]);rec[prefix+'_dm']=float(mb-ma)
        output.append(rec)
    return output,excluded


def regression(scores,out):
    features,excluded=build_features(scores);out=Path(out);require(not out.exists(),'New regression output directory required');out.mkdir(parents=True)
    csv_rows(out/'feature_audit.csv',features);csv_rows(out/'excluded.csv',excluded)
    metrics=[];increments=[];folds_out=[];coefficients=[];predictions=[];not_estimable=[]
    for sid in sorted({r['set_id'] for r in features}):
      # Keep a common parent cohort across all layers; selection never uses model performance.
      per=collections.Counter(r['parent_id'] for r in features if r['set_id']==sid)
      common={k for k,n in per.items() if n==27};curves={};points={}
      for scheme in ('LOGO_all','select_to_check','check_LOGO'):
       for layer in range(27):
        if scheme=='check_LOGO' and layer not in (11,16):continue
        rs=sorted([r for r in features if r['set_id']==sid and r['layer']==layer and r['parent_id'] in common],key=lambda r:r['parent_id'])
        y=np.array([r['y'] for r in rs]);g=np.array([r['component'] for r in rs]);roles=np.array([r['role'] for r in rs]);allids=np.arange(len(rs))
        if scheme=='LOGO_all':ids=allids;folds=[(c,allids[g!=c],allids[g==c]) for c in sorted(set(g))]
        elif scheme=='select_to_check':ids=allids[roles=='readout_check'];folds=[('select_to_check',allids[roles=='readout_select'],ids)]
        else:ids=allids[roles=='readout_check'];folds=[(c,ids[g[ids]!=c],ids[g[ids]==c]) for c in sorted(set(g[ids]))]
        if not len(ids) or len(set(g[ids]))<2 or any(not len(tr) or not len(te) or len(set(y[tr]))<2 or len(set(y[te]))<2 for _,tr,te in folds):
            not_estimable.append({'set_id':sid,'scheme':scheme,'layer':layer,'reason':'missing_class_or_insufficient_independent_folds'});continue
        for _,tr,te in folds:require(not set(g[tr])&set(g[te]),'Regression fold component leakage')
        F={k:np.array([r[k] for r in rs]) for k in set(rb.X_NAMES+rb.B_NAMES+sum(rb.EXTRA.values(),[]))}
        W,draws,levels=rb.bootweights(g[ids]);boot={};point={}
        models=['A','B','C'] if scheme=='check_LOGO' else rb.MODELS
        for model in models:
            p=np.full(len(rs),np.nan)
            for fold,tr,te in folds:
                pp,fr,co=rb.fitone(rs,F,y,tr,te,layer,model,scheme,fold);p[te]=pp;folds_out.append({'set_id':sid,**fr});coefficients.append({'set_id':sid,**co})
            require(np.isfinite(p[ids]).all(),'Incomplete out-of-fold prediction')
            point[model]=rb.metric(y[ids],p[ids]);boot[model]=rb.metricmany(y[ids],p[ids],W)
            row={'set_id':sid,'scheme':scheme,'layer':layer,'model':model,'n':len(ids),'components':len(levels)}
            for metric,val in point[model].items():
                lo,hi,n=rb.interval(boot[model][metric]);row.update({metric:val,metric+'_ci95':[lo,hi],metric+'_valid_bootstrap':n})
            metrics.append(row)
            for i in ids:predictions.append({'set_id':sid,'scheme':scheme,'layer':layer,'model':model,'parent':rs[i]['parent_id'],'component':g[i],'label':int(y[i]),'probability':p[i]})
        for a,b in rb.COMPS:
            if a not in models or b not in models:continue
            row={'set_id':sid,'scheme':scheme,'layer':layer,'comparison':a+'-'+b}
            for metric in ('auc','log_loss','brier'):
                diff=point[a][metric]-point[b][metric];bs=boot[a][metric]-boot[b][metric];lo,hi,n=rb.interval(bs)
                row.update({metric+'_gain':diff,metric+'_ci95':[lo,hi],metric+'_valid_bootstrap':n})
                curves[scheme,a+'-'+b,metric,layer]=bs;points[scheme,a+'-'+b,metric,layer]=diff
            increments.append(row)
       if scheme!='check_LOGO':
        for a,b in rb.COMPS:
         for metric in ('auc','log_loss','brier'):
            keys=[(scheme,a+'-'+b,metric,l) for l in range(27)]
            if not all(k in curves for k in keys):continue
            lo,hi,crit=rb.simultaneous(np.array([points[k] for k in keys]),np.stack([curves[k] for k in keys],1))
            for r in increments:
                if (r['set_id'],r['scheme'],r['comparison'])==(sid,scheme,a+'-'+b):r[metric+'_sim27']=[lo[r['layer']],hi[r['layer']]]
        keys=[(scheme,c,'auc',l) for c in ('C-B','MG-M','NG-N') for l in range(27)]
        if all(k in curves for k in keys):
            lo,hi,crit=rb.simultaneous(np.array([points[k] for k in keys]),np.stack([curves[k] for k in keys],1))
            for j,(_,c,_,l) in enumerate(keys):
                for r in increments:
                    if (r['set_id'],r['scheme'],r['comparison'],r['layer'])==(sid,scheme,c,l):r['auc_sim81']=[lo[j],hi[j]]
    csv_rows(out/'regression_metrics.csv',clean(metrics));csv_rows(out/'incremental_metrics.csv',clean(increments));csv_rows(out/'fold_metrics.csv',clean(folds_out));csv_rows(out/'out_of_fold_predictions.csv',clean(predictions))
    atomic(out/'fitted_fold_models.json',clean(coefficients));atomic(out/'not_estimable.json',not_estimable)
    atomic(out/'provenance.json',{'scores_sha256':file_hash(scores),'estimator':'Original logistic C=1; lbfgs; train-fold standardization and topic encoding',
      'bootstrap':'2000 shared-component draws on fixed out-of-fold predictions; fit uncertainty excluded','set_results_pooled':False,'causal_claim':False})
    return {'models':len(metrics),'not_estimable':len(not_estimable)}
