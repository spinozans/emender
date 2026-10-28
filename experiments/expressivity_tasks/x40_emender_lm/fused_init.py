"""One launch initializes the seven fresh-origin native/history owners."""
import torch
import triton
import triton.language as tl
from ..x30_tape_cell.naive_reference import TapeState


@triton.jit
def _initialize(Keys,Values,Strength,Live,Age,Usage,History,
                S:tl.constexpr,K:tl.constexpr,V:tl.constexpr,Block:tl.constexpr):
    n=tl.program_id(0).to(tl.int64)
    x=tl.program_id(1)*Block+tl.arange(0,Block)
    tl.store(Keys+n*S*K+x,0.,x<S*K)
    tl.store(Values+n*S*V+x,0.,x<S*V)
    tl.store(Strength+n*S+x,0.,x<S)
    tl.store(Live+n*S+x,False,x<S)
    tl.store(Age+n*S+x,0,x<S)
    tl.store(Usage+n*S+x,0.,x<S)
    tl.store(History+n*128+x,0,x<128)


def fresh_state(batch,cfg,device):
    b,h,s,k,v=batch,cfg.n_heads,cfg.slots,cfg.key_dim,cfg.value_dim
    state=TapeState(torch.empty((b,h,s,k),device=device),torch.empty((b,h,s,v),device=device),
        torch.empty((b,h,s),device=device),torch.empty((b,h,s),device=device,dtype=torch.bool),
        torch.empty((b,h,s),device=device,dtype=torch.int64),torch.empty((b,h,s),device=device))
    history=torch.empty((b*h,128),device=device,dtype=torch.int32)
    _initialize[(b*h,triton.cdiv(max(s*k,s*v,128),256))](state.keys,state.values,state.strength,state.live,state.age,state.usage,history,s,k,v,256)
    return state,history
