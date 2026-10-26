"""Parallel mutation prototype for an externally certified append-only phase.

Not a replacement for scan: excludes reads, existing live-row decay, routing,
corrections, POP and deaths. Reports overflow without silently writing slot 0.
"""
import torch
import triton
from .kernels import free_rank_kernel, append_phase_kernel


def allocate_free_ranks(live, events):
    if live.ndim != 2 or events < 1:
        raise ValueError('expected [stores,slots] and positive event count')
    n,s = live.shape
    slots = torch.full((n,events),-1,device=live.device,dtype=torch.int64)
    overflow = torch.empty(n,device=live.device,dtype=torch.bool)
    free_rank_kernel[(n,)](live.contiguous(),slots,overflow,s,events,triton.next_power_of_2(s))
    return slots,overflow


def append_phase_(keys, values, strength, live, age, input_keys, content, gates,
                  slots, exact_curve):
    """Two-dimensional event/store grid; caller certifies slots and no deaths.

    Free-rank overflow must be handled before invoking this mutation. Inputs
    are detached; output buffers are explicitly mutable, with no checkpoint.
    """
    n,s,dk = keys.shape
    t = content.shape[1]
    dv = values.shape[-1]
    arrays = (keys,values,strength,live,age,input_keys,content,gates,slots,exact_curve)
    if any(x.requires_grad for x in arrays):
        raise ValueError('append mutation prototype does not implement autograd')
    if not all(x.is_contiguous() and x.device == keys.device for x in arrays):
        raise ValueError('all arrays must be contiguous on one device')
    if (input_keys.shape != (n,t,dk) or content.shape != (n,t,dv)
        or slots.shape != (n,t) or gates.shape != (n,t)
        or any(x.shape != (n,s) for x in (strength,live,age))
        or exact_curve.numel() < t+1):
        raise ValueError('append phase shape mismatch')
    if bool(((slots < 0) | (slots >= s)).any()):
        raise ValueError("append destinations must be valid; handle overflow first")
    if t > 1 and bool((slots.sort(dim=1).values[:,1:] == slots.sort(dim=1).values[:,:-1]).any()):
        raise ValueError("append destinations must be distinct per store")
    append_phase_kernel[(t,n)](content,gates,slots,input_keys,keys,values,
        strength,live,age,t,s,dk,dv,exact_curve,triton.next_power_of_2(max(dk,dv)),
        enable_fp_fusion=False)
