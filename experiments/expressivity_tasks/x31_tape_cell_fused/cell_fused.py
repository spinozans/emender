"""Experimental inference-only fused scan. Training fails closed.

This baseline exposes the native functional state and routing trace to make
comparison straightforward. It is not the requested sparse training cell:
dense metadata/trace and entry-state copies remain explicit.
"""
from dataclasses import dataclass
import torch
import triton
from ..x30_tape_cell.naive_reference import TapeHeadConfig, TapeState, init_state
from ..x30_tape_cell.routing_trace import RoutingTrace
from .kernels import scan_kernel


@dataclass
class FusedState:
    native: TapeState
    history: torch.Tensor
    tokens: int = 0


def tape_scan(k, q, v, write_strength, pop_strength, append_logit, state,
              cfg: TapeHeadConfig, append_err_coef=None):
    """One scan launch per chunk; no Python token loop or scalar routing.

    Inputs [batch,time,heads,dim]. No silent fallback to X30 autograd.
    Usage EMA instrumentation is excluded from this functional baseline.
    """
    tensors = (k, q, v, write_strength, pop_strength, append_logit)
    if torch.is_grad_enabled() and any(x.requires_grad for x in tensors):
        raise NotImplementedError('X31 fused reverse replay is not implemented')
    if append_err_coef is not None and torch.is_grad_enabled() and append_err_coef.requires_grad:
        raise NotImplementedError('X31 fused reverse replay is not implemented')
    if not (0 < cfg.decay_rate <= 1 and 0 <= cfg.free_threshold < 1 and cfg.slots > 0):
        raise ValueError("invalid decay, threshold or slot budget")
    b, t, h, dk = k.shape
    if (t < 1 or dk != cfg.key_dim or h != cfg.n_heads or q.shape != k.shape
        or v.shape != (b,t,h,cfg.value_dim)
        or any(x.shape != (b,t,h) for x in tensors[3:])):
        raise ValueError('input shape/config mismatch')
    if any(x.device != k.device or x.dtype != torch.float32 for x in tensors):
        raise ValueError('inputs must share device and use fp32')
    if state is None:
        state = FusedState(init_state(b,cfg,k.device),
                           torch.zeros(b*h,128,device=k.device,dtype=torch.int32))
    elif isinstance(state, TapeState):
        state = FusedState(state,torch.zeros(b*h,128,device=k.device,dtype=torch.int32))
    native = state.native
    if (native.live.shape != (b,h,cfg.slots)
        or native.keys.shape != (b,h,cfg.slots,dk)
        or native.values.shape != (b,h,cfg.slots,cfg.value_dim)
        or native.strength.shape != native.live.shape or native.age.shape != native.live.shape):
        raise ValueError('state/config mismatch')
    if any(getattr(native,name).device != k.device for name in ("keys","values","strength","live","age")):
        raise ValueError("state/input device mismatch")
    if append_err_coef is not None and (append_err_coef.shape != (h,) or append_err_coef.device != k.device):
        raise ValueError("append coefficient shape/device mismatch")
    if torch.is_grad_enabled() and any(getattr(native,name).requires_grad
                                     for name in ('keys','values','strength')):
        raise NotImplementedError('X31 state adjoints are not implemented')
    def flat(x):
        return x.transpose(1,2).contiguous()
    args = list(map(flat,tensors))
    coef = (torch.zeros(h,device=k.device) if append_err_coef is None
            else append_err_coef).repeat(b).contiguous()
    keys, values, strength, live, age = [getattr(native,name).clone().contiguous()
                                        for name in ('keys','values','strength','live','age')]
    output = torch.empty(b,h,t,cfg.value_dim,device=k.device)
    case, matched, alloc, pops = [torch.empty(b,h,t,device=k.device,dtype=torch.int64)
                                 for _ in range(4)]
    freed = torch.empty(b,h,t,cfg.slots,device=k.device,dtype=torch.int8)
    pressure = torch.empty(b,h,t,device=k.device)
    history = state.history.clone()
    scan_kernel[(b*h,)](*args,coef,keys,values,strength,live,age,
        output,case,matched,alloc,pops,freed,pressure,history,state.tokens,
        t,cfg.slots,dk,cfg.value_dim,triton.next_power_of_2(cfg.slots),
        triton.next_power_of_2(cfg.value_dim),cfg.decay_rate,cfg.free_threshold,
        cfg.write_enable_threshold,cfg.pop_threshold,cfg.eps_residual,
        enable_fp_fusion=False)
    def unflat(x):
        return x.transpose(1,2).contiguous()
    trace = RoutingTrace(*map(unflat,(case,matched,pops>=0,pops,alloc,
                                     args[3]>cfg.write_enable_threshold,freed)))
    final = TapeState(keys,values,strength,live,age,native.usage)
    return unflat(output),FusedState(final,history,state.tokens+t),trace,unflat(pressure)


class TapeCell(torch.nn.Module):
    """F12g projection wrapper for inference; training is explicitly unsupported."""
    def __init__(self, d_model, cfg=TapeHeadConfig()):
        super().__init__()
        self.cfg = cfg
        h,k,v = cfg.n_heads,cfg.key_dim,cfg.value_dim
        self.W_q = torch.nn.Linear(d_model,h*k)
        self.W_k = torch.nn.Linear(d_model,h*k)
        self.W_v = torch.nn.Linear(d_model,h*v)
        self.W_write = torch.nn.Linear(d_model,h)
        self.W_pop = torch.nn.Linear(d_model,h)
        self.W_append = torch.nn.Linear(d_model,h)
        self.append_err_coef = torch.nn.Parameter(torch.zeros(h))
        with torch.no_grad():
            self.W_q.load_state_dict(self.W_k.state_dict())
            self.W_write.bias.fill_(2.)
            self.W_pop.bias.fill_(-4.)
        self.W_k.requires_grad_(False)
        self.W_write.requires_grad_(False)

    def forward(self,x,token_ids,state=None):
        b,t = x.shape[:2]
        h,cfg = self.cfg.n_heads,self.cfg
        k = self.W_k(x).reshape(b,t,h,cfg.key_dim).detach()
        q = self.W_q(x).reshape_as(k)
        v = self.W_v(x).reshape(b,t,h,cfg.value_dim)
        beta = ((self.W_write(x).sigmoid().detach()>cfg.write_enable_threshold)
                & (token_ids!=0).unsqueeze(-1)).float()
        return tape_scan(k,q,v,beta,self.W_pop(x).sigmoid(),self.W_append(x),
                         state,cfg,self.append_err_coef)
