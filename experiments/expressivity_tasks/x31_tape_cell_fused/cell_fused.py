"""Experimental fused forward and first-order reverse replay.

Inference continuation reuses owned buffers. Autograd retains private replay
copies and a conservatively reserved typed undo log; see REPORT.md for gates.
"""
from dataclasses import dataclass
import struct
import torch
import triton
from ..x30_tape_cell.naive_reference import TapeHeadConfig, TapeState, init_state
from ..x30_tape_cell.routing_trace import RoutingTrace
from .kernels import scan_kernel
from .birth_index import initialize as initialize_birth_index
from .write_journal import WriteJournal
from .all_field_journal import AllFieldJournal


@dataclass
class FusedState:
    """Owned continuation buffer. Passing it to scan consumes this handle.

    Native TapeState imports remain immutable. Use snapshot() to branch a
    continuation explicitly; ordinary continuation reuses device allocations.
    """
    native: TapeState
    history: torch.Tensor
    tokens: int = 0
    metadata: tuple | None = None
    configuration: tuple | None = None
    consumed: bool = False
    write_journal: WriteJournal | None = None
    all_field_journal: AllFieldJournal | None = None

    def snapshot(self):
        if self.consumed:
            raise ValueError('FusedState handle was already consumed')
        native = TapeState(*(getattr(self.native,n).clone() for n in
                             ('keys','values','strength','live','age','usage')))
        meta = None if self.metadata is None else tuple(x.clone() for x in self.metadata)
        history=self.history.clone()
        all_journal=(None if self.all_field_journal is None else
                     self.all_field_journal.snapshot((native.keys,native.values,native.strength,
                         native.live,native.age,history,*meta)))
        return FusedState(native,history,self.tokens,meta,self.configuration,
                          all_field_journal=all_journal,
                          write_journal=(None if self.write_journal is None
                                         else self.write_journal.snapshot()))


def tape_scan(k, q, v, write_strength, pop_strength, append_logit, state,
              cfg: TapeHeadConfig, append_err_coef=None, *, record_payload=False, record_all=False):
    """One scan launch per chunk; no Python token loop or scalar routing.

    Inputs [batch,time,heads,dim]. Native TapeState imports are copied. Returned
    FusedState buffers are consumed in place on continuation; snapshot first
    when an independent branch or immutable historical state is needed.
    Gradients use the X31 custom Function and recorded-decision reverse kernel.
    Usage EMA instrumentation is excluded from this functional baseline.
    record_payload captures only key/value before-images; record_all captures
    all owned mutable storage with exact consuming rollback. Recording is
    allocation-free when both options and gradients are disabled.
    """
    tensors = (k, q, v, write_strength, pop_strength, append_logit)
    native_input = state.native if isinstance(state,FusedState) else state
    needs_grad = any(x.requires_grad for x in tensors)
    needs_grad |= append_err_coef is not None and append_err_coef.requires_grad
    if native_input is not None:
        needs_grad |= any(getattr(native_input,name).requires_grad
                          for name in ('keys','values','strength'))
    if torch.is_grad_enabled() and needs_grad:
        if record_payload or record_all:
            raise ValueError('training manages its private journal automatically')
        from .backward import autograd_scan
        return autograd_scan(tensors,state,cfg,append_err_coef)
    if not (0 < cfg.decay_rate <= 1 and 0 <= cfg.free_threshold < 1 and 0 < cfg.slots <= 262144):
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
    if state.consumed:
        raise ValueError('FusedState handle was already consumed; snapshot before branching')
    configuration = (cfg.slots,cfg.key_dim,cfg.value_dim,cfg.n_heads,
                     cfg.decay_rate,cfg.free_threshold)
    if state.metadata is not None and state.configuration != configuration:
        raise ValueError('persistent metadata/config mismatch')
    if (native.live.shape != (b,h,cfg.slots)
        or native.keys.shape != (b,h,cfg.slots,dk)
        or native.values.shape != (b,h,cfg.slots,cfg.value_dim)
        or native.strength.shape != native.live.shape or native.age.shape != native.live.shape):
        raise ValueError('state/config mismatch')
    if any(getattr(native,name).device != k.device for name in ("keys","values","strength","live","age")):
        raise ValueError("state/input device mismatch")
    if append_err_coef is not None and (append_err_coef.shape != (h,) or append_err_coef.device != k.device):
        raise ValueError("append coefficient shape/device mismatch")
    def flat(x):
        return x.transpose(1,2).contiguous()
    args = list(map(flat,tensors))
    coef = (torch.zeros(h,device=k.device) if append_err_coef is None
            else append_err_coef).repeat(b).contiguous()
    fresh = state.metadata is None
    keys, values, strength, live, age = [
        getattr(native,name).clone().contiguous() if fresh else getattr(native,name)
        for name in ('keys','values','strength','live','age')]
    output = torch.empty(b,h,t,cfg.value_dim,device=k.device)
    case, matched, alloc, pops = [torch.empty(b,h,t,device=k.device,dtype=torch.int64)
                                 for _ in range(4)]
    freed = torch.empty(b,h,t,cfg.slots,device=k.device,dtype=torch.int8)
    pressure = torch.empty(b,h,t,device=k.device)
    history = state.history.clone() if fresh else state.history
    if fresh:
        heap = torch.empty((b*h,cfg.slots),device=k.device,dtype=torch.int32)
        position = torch.empty_like(heap)
        due = torch.empty((b*h,cfg.slots),device=k.device,dtype=torch.int64)
        terminal = torch.empty((b*h,cfg.slots),device=k.device,dtype=torch.float32)
        births = torch.empty_like(due)
        touches = torch.empty_like(due)
        level0 = torch.zeros((b*h,triton.cdiv(cfg.slots,64)),device=k.device,dtype=torch.uint64)
        level1 = torch.zeros((b*h,triton.cdiv(triton.cdiv(cfg.slots,64),64)),device=k.device,dtype=torch.uint64)
        root = torch.zeros(b*h,device=k.device,dtype=torch.uint64)
        birth_next, birth_prev, birth_head = initialize_birth_index(live, age)
        sizes = torch.empty(b*h,device=k.device,dtype=torch.int32)
        references = torch.empty_like(strength)
        scores = torch.empty_like(strength)
        metadata = (heap,position,due,terminal,births,touches,level0,level1,root,
                    birth_next,birth_prev,birth_head,sizes,references,scores)
    else:
        metadata = state.metadata
        (heap,position,due,terminal,births,touches,level0,level1,root,
         birth_next,birth_prev,birth_head,sizes,references,scores) = metadata
    journal = (WriteJournal.allocate(b*h,t,dk,cfg.value_dim,k.device)
               if record_payload else None)
    journal_args = ((journal.slots,journal.keys,journal.values) if journal is not None
                    else (case,keys,values))  # unused pointers in the disabled specialization
    all_journal = (AllFieldJournal.allocate(b*h,t,cfg.slots,dk,cfg.value_dim,
                   (keys,values,strength,live,age,history,*metadata),state.tokens,fresh)
                   if record_all else None)
    if all_journal is not None:
        all_journal.parent=state.all_field_journal
    all_args = ((all_journal.addresses,all_journal.before,all_journal.widths,
                 all_journal.counts,all_journal.boundaries,all_journal.gates,
                 all_journal.errors,all_journal.capacity,True)
                if record_all else (case,case,case,case,case,case,case,0,False))
    scan_kernel[(b*h,)](*args,coef,keys,values,strength,live,age,
        output,case,matched,alloc,pops,freed,pressure,history,heap,position,due,terminal,
        births,touches,level0,level1,root,birth_next,birth_prev,birth_head,sizes,references,scores,
        *journal_args,record_payload,triton.next_power_of_2(max(dk,cfg.value_dim)),
        *all_args,fresh,state.tokens,
        t,cfg.slots,dk,cfg.value_dim,triton.next_power_of_2(cfg.slots),
        triton.next_power_of_2(cfg.value_dim),triton.next_power_of_2(max(32,dk)),
        struct.unpack('<f',struct.pack('<f',cfg.decay_rate))[0],cfg.free_threshold,
        cfg.write_enable_threshold,cfg.pop_threshold,cfg.eps_residual,
        enable_fp_fusion=False,debug=True,
        num_warps=4)
    def unflat(x):
        return x.transpose(1,2).contiguous()
    trace = RoutingTrace(*map(unflat,(case,matched,pops>=0,pops,alloc,
                                     args[3]>cfg.write_enable_threshold,freed)))
    final = TapeState(keys,values,strength,live,age,native.usage)
    state.consumed = True
    return unflat(output),FusedState(final,history,state.tokens+t,metadata,configuration,
                                    write_journal=journal,all_field_journal=all_journal),trace,unflat(pressure)


class TapeCell(torch.nn.Module):
    """F12g projections with first-order reverse replay; see acceptance receipts."""
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
