"""One-component, full-vector, and norm-matched control interventions."""
import numpy as np
from .geometry_base import unit,vector,direction,scores,bf16,matched_random
from .io import require

def edit(h,donor,arm,alpha=1.0,mu=None,w=None,seed=20261102,identity='',repeat=0,random_target=None):
    h=vector(h);donor=vector(donor)
    require(h.shape==donor.shape,'Donor shape differs')
    require(np.array_equal(bf16(h),h),'Input activation is not actual BF16')
    require(alpha in (0.,1.,1.5,2.),'Unsupported frozen intervention strength')
    u=unit(w) if w is not None else direction(mu) if mu is not None else None
    if arm in ('full_replace','full_swap'):
        intended=donor-h;y=bf16(donor);reference=donor
    elif arm=='sham':
        intended=np.zeros_like(h);y=h.astype(np.float32);reference=donor
    else:
        require(u is not None,'Projection intervention requires a fixed direction')
        reference=np.asarray(mu[0],dtype=float) if arm=='hrmean' else donor
        intended=alpha*float((reference-h)@u)*u
        y=bf16(h+intended)
        if arm=='random':
            require(random_target is not None,'Random control requires saved actual restoration norm')
            y,intended,extra=matched_random(h,u,random_target,seed,identity,repeat,48)
            y=y.astype(np.float32)
        else:
            require(arm in ('restore','hrmean'),'Unknown intervention arm');extra={}
    actual=y.astype(float)-h
    info={'arm':arm,'alpha':alpha,'actual_delta_norm':float(np.linalg.norm(actual)),
          'intended_delta_norm':float(np.linalg.norm(intended)),
          'relative_delta_norm':float(np.linalg.norm(actual)/np.linalg.norm(h)),
          'rounding_error_norm':float(np.linalg.norm(actual-intended)),
          'cosine_to_before':float(unit(y)@unit(h)),
          'full_donor_bf16_equal':bool(np.array_equal(y,bf16(donor))) if arm in ('full_replace','full_swap') else None}
    if u is not None:
        info.update(projection_before=float(h@u),projection_after=float(y@u),projection_reference=float(reference@u),
                    intended_orthogonal_delta_norm=float(np.linalg.norm(intended-(intended@u)*u)),
                    actual_orthogonal_delta_norm=float(np.linalg.norm(actual-(actual@u)*u)))
    if mu is not None:
        info.update(before=scores(h,mu),after=scores(y,mu))
        info.update(delta_f=info['after']['f']-info['before']['f'],delta_g=info['after']['g']-info['before']['g'])
    if arm=='random':info.update(extra)
    arrays={'before':h.astype(np.float32),'donor':donor.astype(np.float32),'after':y,
            'intended_delta':intended,'actual_delta':actual}
    if u is not None:arrays['direction']=u
    return y,info,arrays
