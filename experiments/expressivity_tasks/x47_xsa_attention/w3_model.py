"""Batched causal byte LM: fixed native address plan, learned written values and rebuilt 4-slot views."""
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .workbench_run import sha
from .workbench_witness import require

ANCHOR=Path('/home/erikg/c14_dense_bytes/transformer_dense.pt')
ANCHOR_SHA='9b0da60f0fed0d2343afebfd0e2b3f0afbc8bdfc3ea4853e8c870f7f6b8c41a9'
ARMS=('full','blind','fixed','direct')


def base_model():
    from ..x43_fork_resolved.run import Control
    require(sha(ANCHOR)==ANCHOR_SHA,'wrong natural LM checkpoint')
    state=torch.load(ANCHOR,map_location='cpu',weights_only=True); prefix='head.base.'
    model=Control(2000324); model.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)})
    return model.eval().requires_grad_(False)


@torch.no_grad()
def backbone(model, x):
    b,t=x.shape; require(t%128==0,'natural LM reset geometry'); ids=x.reshape(-1,128)
    with torch.autocast(device_type=x.device.type,enabled=x.is_cuda,dtype=torch.bfloat16):
        h=model.embed(ids)+model.position(torch.arange(128,device=x.device)); mask=torch.ones(128,128,device=x.device,dtype=torch.bool).triu(1)
        for block in model.blocks: h=block(h,src_mask=mask,is_causal=True)
        h=model.norm(h); logits=model.head(h).float()
    return h.float().reshape(b,t,256), logits.reshape(b,t,256).softmax(-1)


class Head(nn.Module):
    def __init__(self,d=24):
        super().__init__(); self.d=d
        self.writer=nn.Linear(256,256); self.mem=nn.Linear(256,d); self.query=nn.Linear(256,d)
        self.support=nn.Linear(9,d); self.code=nn.Embedding(256,d); self.hist=nn.Linear(256,d,bias=False)
        self.types=nn.Parameter(torch.randn(4,d)*.02); self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
        self.ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.source=nn.Linear(d,5); self.delta=nn.Linear(d,1)
        with torch.no_grad():
            for m in (self.writer,self.source,self.delta): m.weight.zero_(); m.bias.zero_()
            self.source.bias.copy_(torch.tensor([0.,0.,0.,-2.,-2.]))

    def write(self,h,x):
        # This value is a function exclusively of the old causal state and the byte
        # already observed there. Gather the actual tensor later, not a detached label.
        tag=F.one_hot(x,256).to(h.dtype)
        raw=(tag+.1*self.writer(h)).clamp_min(0); s=raw.sum(-1,keepdim=True)
        return self.mem(h).tanh(),torch.where(s>0,raw/s.clamp_min(1e-12),tag),tag

    def forward(self,h,x,features,ptr,codes,prior,children,on,mode):
        """No y or target probability is an input. Returns normalized mixture policy."""
        if mode not in (*ARMS,'erase'): raise ValueError(mode)
        rows=torch.arange(x.shape[0],device=x.device)[:,None]; hit=ptr>=0; idx=ptr.clamp_min(0)
        payload,computed,tag=self.write(h,x); cue=(tag if mode=='fixed' else computed)[rows,idx]
        saved=payload[rows,idx]*hit[...,None]; first=x[rows,(idx-1).clamp_min(0)]
        if mode=='erase': cue=cue*0; saved=saved*0
        if mode in ('blind','direct'): children=children.mean(-2,keepdim=True).expand_as(children)
        p=torch.stack((prior,1-prior),-1)
        # Typed view: current, saved computed state (+ cheaper previous-byte copy),
        # and the two bound archive edges. Shape identical for all read modes.
        edges=self.code(codes)+self.hist(children)
        z=torch.cat(((self.query(h)+self.support(features)).unsqueeze(-2),
                     (saved+self.code(first)*hit[...,None]).unsqueeze(-2),edges),-2)+self.types
        shape=z.shape; z=z.reshape(-1,4,self.d)
        q,k,v=self.qkv(z).chunk(3,-1)
        z=z+self.proj(F.scaled_dot_product_attention(q[:,None],k[:,None],v[:,None])[:,0]); z=z+self.ff(z)
        z=z.reshape(shape); corr=self.delta(z[...,2:,:]).squeeze(-1)
        binding=(children*cue.unsqueeze(-2)).sum(-1)
        ep=(p.clamp_min(1e-12).log()+torch.log1p(32*binding)+(0 if mode=='fixed' else 1)*corr).softmax(-1)
        ok=torch.stack((torch.ones_like(hit),features[...,1]>0,features[...,4]>0,hit,on & (mode!='direct')),-1)
        weights=self.source(z[...,0,:]).masked_fill(~ok,-torch.inf).softmax(-1)
        return weights,ep,first


def likelihood(parts,neural_y,other_y,codes,y):
    weights,edges,first=parts
    experts=torch.cat((neural_y[...,None],other_y,(first==y)[...,None],((codes==y[...,None])*edges).sum(-1,keepdim=True)),-1)
    return (experts*weights).sum(-1).clamp_min(1e-30)
