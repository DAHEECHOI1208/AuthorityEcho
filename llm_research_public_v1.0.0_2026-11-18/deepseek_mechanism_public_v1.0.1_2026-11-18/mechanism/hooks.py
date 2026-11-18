"""Capture actual decoder block outputs; patch one token once in first prefill."""
import numpy as np
from .io import require
from .geometry import edit

class Capture:
    def __init__(self,adapter,serialized,patch=None):
        self.adapter=adapter;self.serialized=serialized;self.patch=patch
        self.ids=serialized['input_ids'];self.n=len(self.ids)
        self.positions=['t_U','t_D','t_C','t_A'];self.handles=[];self.calls=[]
        self.seen=set();self.forward_count=0;self.applications=0;self.effects=[];self.edit_arrays={}
        self.values={};self.before={}
        require(serialized['positions']['t_A']==[self.n-1] and self.n>1,'Invalid t_A')

    def root(self,module,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        require(ids is not None and ids.ndim==2 and ids.shape[0]==1,'Expected batch size one')
        cache=kwargs.get('past_key_values')
        require(cache is None or hasattr(cache,'get_seq_length'),'Unknown cache type')
        past=int(cache.get_seq_length()) if cache is not None else 0
        self.forward_count+=1
        if self.forward_count==1:require(past==0 and ids[0].tolist()==self.ids,'Expected independent empty-cache prefill')
        else:require(ids.shape[1]==1 and past>=self.n,'Unexpected decode cache state')
        self.calls.append({'phase':'prefill' if self.forward_count==1 else 'decode','tokens':int(ids.shape[1]),'past':past})

    def block(self,layer,output):
        require(self.forward_count>0,'Block called outside root forward')
        if self.forward_count>1:return output
        h=output[0] if isinstance(output,(tuple,list)) else output
        require(h.ndim==3 and h.shape[0]==1 and h.shape[1]==self.n,'Unexpected block tensor')
        require(str(h.dtype)=='torch.bfloat16','Expected BF16 block output')
        require(layer not in self.seen,'Duplicate prefill layer hook');self.seen.add(layer)
        updated=h
        if self.patch is not None and layer==self.patch['layer']:
            import torch
            p=self.patch;before=h[0,self.n-1].detach().float().cpu().numpy().copy()
            require(self.applications==0,'Repeated intervention')
            if p.get('expected_before') is not None:
                if p['arm'] in ('full_replace','full_swap'):require(np.array_equal(before,p['expected_before']),'Live pre-edit state differs from the stored baseline')
            y,rec,arrays=edit(before,p['donor'],p['arm'],p.get('alpha',1.),p.get('mu'),p.get('w'),
                               p.get('seed',20261102),p.get('identity',''),p.get('repeat',0),p.get('random_target'))
            value=torch.from_numpy(y).to(device=h.device,dtype=h.dtype)
            require(np.array_equal(value.detach().float().cpu().numpy(),y),'Torch BF16 differs from CPU oracle')
            updated=h.clone();updated[0,self.n-1]=value
            require(torch.equal(updated[:,:self.n-1],h[:,:self.n-1]),'Non-target tokens changed')
            self.applications+=1;self.edit_arrays=arrays
            rec['pre_edit_drift_l2']=float(np.linalg.norm(before.astype(float)-p['expected_before'])) if p.get('expected_before') is not None else None
            rec.update(layer=layer,position='t_A',token_index=self.n-1,phase='prefill',applications=1,
                       module_path=f'model.layers.{layer}',other_tokens_bitwise_equal=True,decode_edits=0)
            self.effects.append(rec)
        for pos in self.positions:
            indices=self.serialized['positions'].get(pos)
            if indices:
                self.values[layer,pos]=updated[0,indices].detach().float().mean(dim=0).cpu().numpy().copy()
                self.before[layer,pos]=h[0,indices].detach().float().mean(dim=0).cpu().numpy().copy()
        if updated is h:return output
        if isinstance(output,tuple):return (updated,)+output[1:]
        if isinstance(output,list):return [updated]+output[1:]
        return updated

    def __enter__(self):
        try:
            self.handles.append(self.adapter.model.register_forward_pre_hook(self.root,with_kwargs=True))
            for i,b in enumerate(self.adapter.layers):
                self.handles.append(b.register_forward_hook(lambda m,a,o,i=i:self.block(i,o)))
            return self
        except BaseException:self.remove();raise

    def remove(self):
        for h in self.handles:h.remove()
        self.handles=[]

    def __exit__(self,typ,value,tb):
        self.remove()
        if typ is None:
            require(self.seen==set(range(len(self.adapter.layers))),'Missing decoder block capture')
            require(self.applications==(1 if self.patch else 0),'Incorrect intervention count')
        return False

    def arrays(self):
        dim=int(self.adapter.model.config.hidden_size)
        a=np.full((len(self.adapter.layers),4,dim),np.nan,dtype=np.float32)
        mask=np.zeros(a.shape[:2],dtype=bool)
        for (l,p),v in self.values.items():
            require(np.isfinite(v).all(),'Nonfinite captured state');a[l,self.positions.index(p)]=v;mask[l,self.positions.index(p)]=True
        return {'activations':a,'valid_positions':mask,'layers':np.arange(len(self.adapter.layers)),
                'positions':np.asarray(self.positions),**{'edit_'+k:v for k,v in self.edit_arrays.items()}}
