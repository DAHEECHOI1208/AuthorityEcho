"""Frozen paired C3/C4 probes and surface controls using the original design."""
import itertools
from pathlib import Path
import numpy as np
from .io import require,atomic,csv_rows,npz_save,file_hash
from .scoring import verified_ledger,load_tensor,load_center_bank,POSITIONS


def unit(v):
    n=np.linalg.norm(v);return v/n if n else np.zeros_like(v)


def auc(a,b):
    a=np.sort(np.asarray(a));b=np.asarray(b)
    return float(np.mean(np.searchsorted(a,b,side='left')+np.searchsorted(a,b,side='right'))/(2*len(a))) if len(a) and len(b) else None


def clustered(a,b,groups):
    levels,ids=np.unique(groups,return_inverse=True);k=len(levels)
    if k<2:return {'ci95':None,'reason':'fewer_than_two_components','components':k}
    M=np.eye(k)[ids]
    draws=np.array(list(itertools.product(range(k),repeat=k))) if k==4 else np.random.default_rng(1707).integers(0,k,size=(2000,k))
    w=np.eye(k)[draws].sum(1);den=(w@M.sum(0))**2
    delta=b[:,None]-a[None,:];U=M.T@((delta>0)+.5*(delta==0))@M
    values=((w@U)*w).sum(1)/den
    diag=np.diag(U)/M.sum(0)**2
    return {'ci95':np.quantile(values,[.025,.975]).tolist(),'components':k,'draws':len(draws),'enumerated':k==4,'group_auc_min':float(min(diag)),'group_auc_max':float(max(diag))}


def probe_analysis(ledger,centers,out):
    rows=verified_ledger(ledger);rows=[r for r in rows if r['arm']=='baseline' and r['variant']=='harmful'];mu=load_center_bank(centers)
    index={};result=[];controls=[];direction_banks={};surface=[]
    for r in rows:
        k=(r['set_id'],r['variant'],r['parent_question_id'],r['condition_id']);require(k not in index,'Duplicate probe parent/condition');index[k]=r
    for sid,variant in sorted({k[:2] for k in index}):
        parents=sorted({k[2] for k in index if k[:2]==(sid,variant) and all((sid,variant,k[2],c) in index for c in ('C3-E','C3-P','C4'))})
        metas=[[index[sid,variant,p,c] for c in ('C3-E','C3-P','C4')] for p in parents]
        if not metas:continue
        roles=np.array([r[0]['dev_role'] for r in metas]);groups=np.array([r[0]['shared_component_id'] for r in metas]);fit=np.flatnonzero(roles=='calibration_fit')
        require(len(fit)>0,'Probe fitting requires a separate calibration_fit partition')
        X=np.asarray([[load_tensor(r['activation_path']) for r in rr] for rr in metas]);bank=np.zeros((2,27,4,2048))
        for l,j in itertools.product(range(27),range(4)):
            H=X[:,:,l,j,:].astype(float)
            if not np.isfinite(H).all():continue
            normalized=H/np.linalg.norm(H,axis=-1,keepdims=True);axes=[unit((H[:,2]-H[:,c])[fit].mean(0)) for c in (0,1)];bank[:,l,j]=axes
            center=mu[l,j];fg=None
            if np.isfinite(center).all():
                hr,ha,ba=center/np.linalg.norm(center,axis=-1,keepdims=True);fg=np.stack([hr-ba,ha-hr],axis=1);Q=np.linalg.qr(fg)[0]
            for c in (0,1):
                d=axes[c];common=unit(axes[0]+axes[1]);methods={'direct':H@d,'other_contrast':H@axes[1-c],'shared':H@common}
                if fg is not None:methods.update(orthogonal_fg=H@unit(d-Q@(Q.T@d)),shared_orthogonal_fg=H@unit(common-Q@(Q.T@common)),minus_f=normalized@(-fg[:,0]),g=normalized@fg[:,1])
                for role in ('calibration_fit','readout_select','readout_check'):
                    ii=np.flatnonzero(roles==role)
                    if not len(ii):continue
                    dif=H[ii,2]-H[ii,c];norm=np.linalg.norm(dif,axis=1)
                    controls.append({'set_id':sid,'variant':variant,'contrast':('C3-E','C3-P')[c],'layer':l,'position':POSITIONS[j],'role':role,'n':len(ii),'exact_equal':int((norm==0).sum()),'mean_l2':float(norm.mean()),'mean_relative_l2':float(np.mean(norm/np.linalg.norm(H[ii,2],axis=1)))})
                    if role=='calibration_fit':continue
                    for name,S in methods.items():
                        a,b=S[ii,c],S[ii,2];r={'set_id':sid,'variant':variant,'contrast':('C3-E','C3-P')[c],'layer':l,'position':POSITIONS[j],'role':role,'method':name,'auc':auc(a,b),'paired_up':int((b>a).sum()),'paired_ties':int((b==a).sum()),'n':len(ii),'mean_step':float((b-a).mean())}
                        if name in ('direct','shared'):r.update(clustered(a,b,groups[ii]))
                        pre=np.array([metas[i][c].get('group')=='HR' for i in ii]);post=np.array([metas[i][2].get('group') for i in ii]);delta=b-a
                        r['RA_RR_auc']=auc(delta[pre&(post=='HR')],delta[pre&(post=='HA')]);result.append(r)
        direction_banks[sid+'_'+variant]=bank
        # Surface-only controls remain distinguishable from latent probes.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        for c in (0,1):
            text_available=all('document' in r for rr in metas for r in rr)
            tf=clf=None
            if text_available:
                texts=[metas[i][c]['document'] for i in fit]+[metas[i][2]['document'] for i in fit]
                try:
                    tf=TfidfVectorizer(analyzer='char',ngram_range=(2,4),sublinear_tf=True);clf=LogisticRegression(C=1,max_iter=1000).fit(tf.fit_transform(texts),[0]*len(fit)+[1]*len(fit))
                except ValueError:tf=clf=None
            for role in ('readout_select','readout_check'):
                ii=np.flatnonzero(roles==role)
                if not len(ii):continue
                for feature in ('document_chars','input_tokens'):
                    if not all(feature in r for rr in metas for r in rr):continue
                    V=np.array([[r[feature] for r in rr] for rr in metas]);orient=1 if auc(V[fit,c],V[fit,2])>=.5 else -1
                    surface.append({'set_id':sid,'variant':variant,'contrast':('C3-E','C3-P')[c],'role':role,'feature':feature,'orientation_fit_only':orient,'auc':auc(V[ii,c]*orient,V[ii,2]*orient)})
                if clf is not None:
                    a=clf.predict_proba(tf.transform([metas[i][c]['document'] for i in ii]))[:,1];b=clf.predict_proba(tf.transform([metas[i][2]['document'] for i in ii]))[:,1]
                    surface.append({'set_id':sid,'variant':variant,'contrast':('C3-E','C3-P')[c],'role':role,'feature':'character_tfidf_2_to_4','auc':auc(a,b)})
    out=Path(out);require(not out.exists(),'New output directory required');out.mkdir(parents=True)
    csv_rows(out/'probe_results.csv',result);csv_rows(out/'vector_diagnostics.csv',controls);csv_rows(out/'surface_controls.csv',surface)
    for name,bank in direction_banks.items():
        path=out/(name+'_directions.npz');npz_save(path,directions=bank,contrasts=np.asarray(['C3-E','C3-P']),layers=np.arange(27),positions=np.asarray(POSITIONS))
        atomic(str(path)+'.json',{'kind':'directions','scope':'fit250','sha256':file_hash(path),'ledger_sha256':file_hash(ledger),'zero_directions':'retained for diagnostic equality; cannot drive an intervention'})
    atomic(out/'provenance.json',{'ledger_sha256':file_hash(ledger),'fit_behavior_filter':False,'bootstrap_rule':'All 4^4 cluster draws when four components; otherwise 2000 draws with seed 1707',
      'unit_of_resampling':'shared component','fixed_direction_no_bootstrap_refit':True})
    return {'probe_rows':len(result),'surface_rows':len(surface)}
