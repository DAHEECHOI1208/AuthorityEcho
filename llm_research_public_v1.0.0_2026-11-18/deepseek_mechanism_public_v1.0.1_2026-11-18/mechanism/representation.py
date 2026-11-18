from .joint_bootstrap import joint_bootstrap
"""Saved-score linkage, clustered uncertainty, and original split-based f validation."""
import collections
from pathlib import Path
import numpy as np
from .io import ROOT,require,read,read_jsonl,atomic,csv_rows,file_hash,digest
from .analysis import CONTRASTS
from . import validation_base as vb
from .scoring import verified_ledger,load_tensor,POSITIONS

def cluster_mean(values,clusters,seed=20261102,repeats=2000,family=1):
    x=np.asarray(values,dtype=float);g=np.asarray(clusters);keep=np.isfinite(x);x=x[keep];g=g[keep]
    ids=sorted(set(g.tolist()));k=len(ids)
    result={'n':len(x),'components':k,'mean':float(x.mean()) if len(x) else None,'ci':None,
            'ci_level':1-.05/family,'valid_replicates':0,'total_replicates':repeats,'reason':None}
    if k<2:result['reason']='fewer_than_two_components';return result
    draws=np.random.Generator(np.random.PCG64(seed)).multinomial(k,[1/k]*k,size=repeats)
    sums=np.asarray([x[g==c].sum() for c in ids]);counts=np.asarray([(g==c).sum() for c in ids])
    den=draws@counts;good=den>0;v=draws[good]@sums/den[good];result['valid_replicates']=len(v)
    if len(v)>=.95*repeats:result['ci']=np.quantile(v,[.025/family,1-.025/family]).tolist()
    else:result['reason']='bootstrap_valid_fraction_below_minimum'
    return result

def linkage(scores,out):
    rows=read_jsonl(scores);rows=[r for r in rows if r['arm']=='baseline']
    key=lambda r:(r['set_id'],r['variant'],r['parent_question_id'],r['layer_index'],r['position'])
    index={}
    for r in rows:
        k=(*key(r),r['condition_id']);require(k not in index,'Duplicate parent/condition score');index[k]=r
    pairs=[];summary=[];differences=[]
    for a,b in CONTRASTS:
      for r in rows:
        if r['condition_id']!=a:continue
        other=index.get((*key(r),b))
        if not other:continue
        ga,gb=r.get('group'),other.get('group');transition=None
        if ga and gb:transition=ga[-1]+'_to_'+gb[-1]
        for metric in ('f','g','a','entity_s','purpose_s','shared_s'):
          x,y=r.get(metric),other.get(metric)
          pairs.append({'set_id':r['set_id'],'variant':r['variant'],'parent_question_id':r['parent_question_id'],
              'component':r['shared_component_id'],'dev_role':r['dev_role'],'layer':r['layer_index'],
              'position':r['position'],'contrast':a+'->'+b,'metric':metric,'transition':transition,
              'left':x,'right':y,'delta':y-x if x is not None and y is not None else None})
    buckets=collections.defaultdict(list)
    for r in pairs:
        for transition in ('ALL',r['transition']):
            if transition is not None:buckets[*(r[k] for k in ('set_id','variant','dev_role','layer','position','contrast','metric')),transition].append(r)
    for k,rs in buckets.items():
        valid=[r for r in rs if r['delta'] is not None]
        summary.append(dict(zip(('set_id','variant','dev_role','layer','position','contrast','metric','transition'),k))|
                       {'assigned':len(rs),'excluded':len(rs)-len(valid),**cluster_mean([r['delta'] for r in valid],[r['component'] for r in valid])})
    # Compare transitions within a shared baseline behavior, using the same cluster draw.
    for k in {k[:-1] for k in buckets}:
      for first,second in [('R_to_A','R_to_R'),('A_to_R','A_to_A')]:
        a=[r for r in buckets.get((*k,first),[]) if r['delta'] is not None]
        b=[r for r in buckets.get((*k,second),[]) if r['delta'] is not None]
        point=np.mean([r['delta'] for r in a])-np.mean([r['delta'] for r in b]) if a and b else None
        ids=sorted({r['component'] for r in a+b});ci=None;valid=0;reason=None
        if min(len({r['component'] for r in a}),len({r['component'] for r in b}))<2:reason='insufficient_transition_components'
        else:
          w=np.random.default_rng(20261102).multinomial(len(ids),[1/len(ids)]*len(ids),size=2000)
          means=[]
          for group in (a,b):
            counts=np.array([sum(r['component']==c for r in group) for c in ids]);sums=np.array([sum(r['delta'] for r in group if r['component']==c) for c in ids])
            den=w@counts;means.append(np.divide(w@sums,den,out=np.full(2000,np.nan),where=den>0))
          d=means[0]-means[1];good=np.isfinite(d);valid=int(good.sum())
          if valid>=1900:ci=np.quantile(d[good],[.025,.975]).tolist()
          else:reason='bootstrap_valid_fraction_below_minimum'
        differences.append(dict(zip(('set_id','variant','dev_role','layer','position','contrast','metric'),k))|
                           {'difference':first+' minus '+second,'n_first':len(a),'n_second':len(b),'mean_difference':float(point) if point is not None else None,'ci95':ci,'valid_bootstrap':valid,'reason':reason})
    out=Path(out);require(not out.exists(),'Use a new output directory');out.mkdir(parents=True)
    csv_rows(out/'score_changes.csv',summary);csv_rows(out/'transition_differences.csv',differences)
    from .io import jsonl
    jsonl(out/'pair_rows.jsonl',pairs)
    atomic(out/'provenance.json',{'scores_sha256':file_hash(scores),'centers':'fixed','exploratory':True,
          'causal_effect_or_independent_axis_claim':False,'uncertainty_excludes_center_and_selection_refitting':True})
    return {'pairs':len(pairs),'summary_rows':len(summary)}

def validate_f(ledger,out,positions=('t_C','t_A'),scope='split'):
    protocol=read(ROOT/'configs/f_protocol.json')
    allrows=verified_ledger(ledger);require(len({r['set_id'] for r in allrows})==1,'Validate one set at a time; do not pool overlapping sets')
    framepath=Path(str(ledger)+'.frame.json');framerows=read(framepath)['rows'] if framepath.exists() else allrows
    rows=[r for r in allrows if r['arm']=='baseline' and r.get('group') in ('HR','HA','BR','BA') and r['condition_id'] in vb.MAIN]
    fit=[r for r in rows if scope=='full500' or r['dev_role']=='calibration_fit']
    select=[r for r in rows if scope=='full500' or r['dev_role']=='readout_select']
    check=[r for r in rows if scope=='full500' or r['dev_role']=='readout_check']
    require(fit and select and check,'Fit/select/check records are required; one case cannot fit centers or select a layer')
    cache=Path(str(out)+'.cache.npy');receipt=Path(str(out)+'.cache.json')
    identity=digest({'ledger':file_hash(ledger),'tasks':[r['task_id'] for r in rows]})
    if receipt.exists():
        info=read(receipt);require(info['identity']==identity and info['sha256']==file_hash(cache),'Validation cache identity differs')
    else:
        require(not cache.exists(),'Incomplete cache exists; preserve and use a new output path')
        cache.parent.mkdir(parents=True,exist_ok=True)
        mm=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(len(rows),27,4,2048))
        for i,r in enumerate(rows):mm[i]=load_tensor(r['activation_path'])
        mm.flush();del mm;atomic(receipt,{'identity':identity,'sha256':file_hash(cache)})
    M=np.load(cache,mmap_mode='r',allow_pickle=False);ix={r['task_id']:i for i,r in enumerate(rows)}
    fi=np.array([ix[r['task_id']] for r in fit]);si=np.array([ix[r['task_id']] for r in select]);ci=np.array([ix[r['task_id']] for r in check])
    fc=sorted({r['shared_component_id'] for r in framerows if scope=='full500' or r['dev_role']=='calibration_fit'});sc=sorted({r['shared_component_id'] for r in framerows if scope=='full500' or r['dev_role']=='readout_select'});cc=sorted({r['shared_component_id'] for r in framerows if scope=='full500' or r['dev_role']=='readout_check'})
    if scope=='split':require(not(set(fc)&set(sc) or set(fc)&set(cc) or set(sc)&set(cc)),'Cluster leakage across partitions')
    candidates=[]
    for l in range(27):
      for pos in positions:
        p=POSITIONS.index(pos);x=M[fi,l,p].astype(float);y=M[si,l,p].astype(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():continue
        mu=vb.centers(fit,x)
        if mu is None:continue
        score=y@ (mu[0]/np.linalg.norm(mu[0])-mu[1]/np.linalg.norm(mu[1]))/np.linalg.norm(y,axis=1)
        metric=vb.metrics(select,score);boot=(joint_bootstrap(vb,np,fit,x,fc,{'bootstrap_repeats':2000,'bootstrap_seed':20261023,'minimum_valid_bootstrap_fraction':.95})[0] if scope=='full500' else vb.bootstrap(fit,x,select,y,2000,20261023,fc,sc,minimum_valid_fraction=.95))
        candidates.append({'layer':l,'position':pos,'comparisons':{k:{**metric[k],**boot[k]} for k in metric}})
    primary=[c for c in candidates if c['position'] in ('t_C','t_A')]
    selection={k:vb.choose(primary,k,protocol=protocol) for k in ('H','R')}
    conditional={}
    for comparison in ('H_A','R_H'):
        eligible=[c for c in primary if not vb.ci_issues(c['comparisons'][comparison],protocol)]
        if eligible:
            best=max(c['comparisons'][comparison]['ci95'][0] for c in eligible)
            ties=[c for c in eligible if best-c['comparisons'][comparison]['ci95'][0]<=.005+1e-12]
            chosen=min(ties,key=lambda c:(0 if c['position']=='t_C' else 1,c['layer']))
            l=chosen['layer'];p=POSITIONS.index(chosen['position']);mu=vb.centers(fit,M[fi,l,p].astype(float))
            y=M[ci,l,p].astype(float);score=y@(mu[0]/np.linalg.norm(mu[0])-mu[1]/np.linalg.norm(mu[1]))/np.linalg.norm(y,axis=1)
            boot=(joint_bootstrap(vb,np,fit,M[fi,l,p].astype(float),fc,{'bootstrap_repeats':2000,'bootstrap_seed':20261023,'minimum_valid_bootstrap_fraction':.95})[0] if scope=='full500' else vb.bootstrap(fit,M[fi,l,p].astype(float),check,y,2000,20261023,fc,cc,minimum_valid_fraction=.95))
            conditional[comparison]={'candidate':{'layer':l,'position':chosen['position']},'check':{**vb.metrics(check,score)[comparison],**boot[comparison]},'status':'exploratory_conditional_only'}
        else:conditional[comparison]={'candidate':None,'status':'not_estimable'}
    original_check={};full_only={}
    for kind,sel in selection.items():
        chosen=sel.get('candidate')
        if chosen is None:
            original_check[kind]={'status':'inconclusive','reason':'no_eligible_select_candidate'};continue
        l=chosen['layer'];p=POSITIONS.index(chosen['position'])
        if scope=='full500':
            original_check[kind]={'status':'same_sample_descriptive','candidate':chosen,'independent_validation':False};continue
        result_check,mu=vb.evaluate_candidate(fit,M[fi,l,p].astype(float),check,M[ci,l,p].astype(float),protocol,fc,cc)
        original_check[kind]={'candidate':chosen,**result_check,'interpretation':vb.verdict_details(kind,result_check,rows,protocol)}
        ffull=np.array([i for i,r in enumerate(fit) if r.get('canonical_label')!='partially_compliant']);cfull=np.array([i for i,r in enumerate(check) if r.get('canonical_label')!='partially_compliant'])
        if len(ffull) and len(cfull):
            full_only[kind]=vb.evaluate_candidate([fit[i] for i in ffull],M[fi[ffull],l,p].astype(float),[check[i] for i in cfull],M[ci[cfull],l,p].astype(float),protocol,fc,cc)[0]
        else:full_only[kind]={'status':'not_estimable'}
    result={'scope':scope,'bootstrap_frame_source':'assigned_frame' if framepath.exists() else 'available_ledger_components','candidate_count':len(candidates),'candidates':candidates,'original_selection':selection,'original_check':original_check,'full_only_sensitivity_no_reselection':full_only,'support':vb.support(rows,protocol),'conditional':conditional,
            'full_HR_disentanglement_validated':False,'original_HR_status':'inconclusive' if any(v['candidate'] is None for v in selection.values()) else 'selection_only_check_required',
            'new_independent_validation':False,'ledger_sha256':file_hash(ledger)}
    require(not Path(out).exists(),'Use a new result path');atomic(out,result);return {'candidates':len(candidates),'original_HR_status':result['original_HR_status']}
