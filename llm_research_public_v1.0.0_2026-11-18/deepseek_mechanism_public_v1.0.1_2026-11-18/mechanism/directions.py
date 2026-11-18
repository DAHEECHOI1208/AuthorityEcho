"""Frozen direct probes and shared congruence directions; never silently refit."""
from pathlib import Path
import numpy as np
from .io import require


def load_directions(path):
    direct=None;shared=np.full((27,4,2048),np.nan)
    with np.load(path,allow_pickle=False) as z:
        a=z['directions'].astype(float)
        if a.shape==(2,27,4,2048):
            require(z['contrasts'].tolist()==['C3-E','C3-P'],'Direct probe order differs')
            direct=a;total=a[0]+a[1];norm=np.linalg.norm(total,axis=-1,keepdims=True)
            shared=np.divide(total,norm,out=np.zeros_like(total),where=norm>0)
        elif a.shape==(5,2048):
            require(z['layers'].tolist()==[12,13,14,15,16] and np.array_equal(a[0],z['w']),'Original shared readout axes differ')
            require(np.allclose(np.linalg.norm(a,axis=1),1,atol=1e-10),'Original shared readouts are not unit vectors')
            shared[12:17,3]=a
        else:raise ValueError('Unknown direction artifact shape')
    require(np.isfinite(shared[12,3]).all() and np.linalg.norm(shared[12,3])>1e-12,'Shared layer12 direction missing or degenerate')
    return direct,shared
