"""Same tile/dot algorithm using IEEE FP32 operands; experimental comparator.

Not the production reduction tree or a new certified reference.
"""
from contextlib import contextmanager
from unittest.mock import patch
import torch
import triton
import triton.language as tl
from ..x35_tape_cell_v3 import causal_read


@triton.jit
def tiles(Q,Keys,Values,Versions,Strength,Max,Denom,Partial,
          C:tl.constexpr,S:tl.constexpr,M:tl.constexpr,K:tl.constexpr,V:tl.constexpr,R:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    n=tl.program_id(0);t=tl.program_id(1);tile=tl.program_id(2)
    s=tile*32+tl.arange(0,32);rows=tl.arange(0,16);kd=tl.arange(0,BK);vd=tl.arange(0,BV)
    version=tl.load(Versions+(n*C+t)*S+s,s<S,-1)
    tl.device_assert(tl.sum(((version < -1)|(version>=M)).to(tl.int32),0)==0,'invalid version')
    live=(version>=0)&(version<M);header=(n*C+t)*R+tile
    if tl.sum(live.to(tl.int32),0)>0:
        query=tl.load(Q+(n*C+t)*K+kd[None,:]+tl.zeros((16,1),tl.int32),kd[None,:]<K,0.).to(tl.float32)
        keys=tl.load(Keys+(n*M+version[None,:])*K+kd[:,None],live[None,:]&(kd[:,None]<K),0.).to(tl.float32)
        scores=tl.dot(query,keys,input_precision="ieee").to(tl.float32)
        score=tl.sum(tl.where(rows[:,None]==0,scores,0.),0)*(2./tl.sqrt(float(K)))
        score=tl.where(live,score,-1.e30);maximum=tl.max(score,0)
        numer=tl.where(live,tl.exp(score-maximum),0.)
        strength=tl.load(Strength+(n*C+t)*S+s,live,0.)
        weight=tl.broadcast_to((numer*strength)[None,:],(16,32)).to(tl.float32)
        value=tl.load(Values+(n*M+version[:,None])*V+vd[None,:],live[:,None]&(vd[None,:]<V),0.).to(tl.float32)
        products=tl.dot(weight,value,input_precision="ieee").to(tl.float32)
        partial=tl.sum(tl.where(rows[:,None]==0,products,0.),0)
        tl.store(Max+header,maximum);tl.store(Denom+header,tl.sum(numer,0));tl.store(Partial+header*V+vd,partial,vd<V)
    else:
        tl.store(Max+header,-1.e30);tl.store(Denom+header,0.)


def ieee_read(q,keys,values,versions,strength):
    n,c,k,m,v,s=causal_read._validate(q,keys,values,versions,strength)
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,keys,values,strength)):
        raise ValueError('component forward only; recorded model uses explicitly surrogate FP32 replay')
    r=triton.cdiv(s,32)
    maximum=torch.empty((n,c,r),device=q.device);denom=torch.empty_like(maximum)
    partial=torch.empty((n,c,r,v),device=q.device);out=torch.empty((n,c,v),device=q.device)
    tiles[(n,c,r)](q,keys,values,versions,strength,maximum,denom,partial,c,s,m,k,v,r,triton.next_power_of_2(max(32,k)),triton.next_power_of_2(v),num_warps=4,debug=True,enable_fp_fusion=False)
    causal_read._reduce[(n,c,triton.cdiv(v,8))](maximum,denom,partial,out,c,v,r,triton.next_power_of_2(r),num_warps=4,enable_fp_fusion=False)
    return out
