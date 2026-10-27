"""Chunk-wide read primitive over recorded pre-token payload versions.

This component does not decide routes or construct versions. The caller must
supply one version index per logical slot at every pre-token read boundary;
-1 means dead. Version rows can be shared until a key/value mutation. Strengths
must already contain X31's rounded, time-indexed decay (not analytic powers).
Two launches read the complete chunk with X31's slot tiling/reduction tree.
The differentiable reference is for component validation, not replay integration.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _tiles(Q, Keys, Values, Versions, Strength, Max, Denom, Partial,
           C: tl.constexpr, S: tl.constexpr, M: tl.constexpr,
           K: tl.constexpr, V: tl.constexpr, R: tl.constexpr,
           BK: tl.constexpr, BV: tl.constexpr):
    n = tl.program_id(0)
    t = tl.program_id(1)
    tile = tl.program_id(2)
    s = tile * 32 + tl.arange(0, 32)
    version = tl.load(Versions + (n*C+t)*S+s, s<S, -1)
    tl.device_assert(tl.sum(((version < -1) | (version >= M)).to(tl.int32), 0) == 0, "invalid payload version")
    live = (version >= 0) & (version < M)
    header = (n*C+t)*R+tile
    if tl.sum(live.to(tl.int32), 0) > 0:
        kd = tl.arange(0, BK)
        query = tl.load(Q+(n*C+t)*K+kd, kd<K, 0.)
        key = tl.load(Keys+(n*M+version[:,None])*K+kd[None,:],
                      live[:,None] & (kd[None,:]<K), 0.)
        score = tl.sum(key*query[None,:], 1)*(2.0/tl.sqrt(float(K)))
        score = tl.where(live, score, -1.0e30)
        maximum = tl.max(score, 0)
        numer = tl.where(live, tl.exp(score-maximum), 0.)
        strength = tl.load(Strength+(n*C+t)*S+s, live, 0.)
        d = tl.arange(0, BV)
        value = tl.load(Values+(n*M+version[:,None])*V+d[None,:],
                        live[:,None] & (d[None,:]<V), 0.)
        partial = tl.sum((numer*strength)[:,None]*value, 0)
        tl.store(Max+header, maximum)
        tl.store(Denom+header, tl.sum(numer, 0))
        tl.store(Partial+header*V+d, partial, d<V)
    else:
        tl.store(Max+header, -1.0e30)
        tl.store(Denom+header, 0.)


@triton.jit
def _reduce(Max, Denom, Partial, Out, C: tl.constexpr, V: tl.constexpr,
            R: tl.constexpr, BR: tl.constexpr):
    n = tl.program_id(0)
    t = tl.program_id(1)
    d = tl.program_id(2)*8+tl.arange(0, 8)
    r = tl.arange(0, BR)
    base = (n*C+t)*R
    maximum = tl.load(Max+base+r, r<R, -1.0e30)
    denom = tl.load(Denom+base+r, r<R, 0.)
    scale = tl.exp(maximum-tl.max(maximum, 0))
    total = tl.sum(denom*scale, 0)
    payload = tl.load(Partial+(base+r[:,None])*V+d[None,:],
                      (r[:,None]<R) & (denom[:,None]>0) & (d[None,:]<V), 0.)
    result = tl.sum(payload*scale[:,None], 0)/tl.maximum(total, 1.0e-30)
    tl.store(Out+(n*C+t)*V+d, result, d<V)


def _validate(q, keys, values, versions, strength):
    if q.ndim != 3 or keys.ndim != 3 or values.ndim != 3 or versions.ndim != 3:
        raise ValueError('expected rank-three chunk/store tensors')
    n, c, k = q.shape
    m, v = keys.shape[1], values.shape[-1]
    s = versions.shape[-1]
    if min(n, c, k, m, v, s) < 1 or keys.shape != (n,m,k) or values.shape != (n,m,v) or versions.shape[:2] != (n,c) or strength.shape != versions.shape:
        raise ValueError('inconsistent chunk/version dimensions')
    if versions.dtype != torch.int64 or any(x.dtype != torch.float32 for x in (q,keys,values,strength)):
        raise ValueError('fp32 payloads/strength and int64 version IDs required')
    if any(x.device != q.device or not x.is_contiguous() for x in (q,keys,values,versions,strength)):
        raise ValueError('matching devices and contiguous tensors required')
    # Values of version IDs are a caller invariant; validate on device in the
    # gather reference or explicitly before a trusted kernel invocation.
    return n, c, k, m, v, s


def reference_read(q, keys, values, versions, strength):
    """Differentiable specification; intentionally materializes gathered views."""
    n,c,k,m,v,s = _validate(q,keys,values,versions,strength)
    if ((versions < -1) | (versions >= m)).any():
        raise ValueError('version ID outside [-1, version count)')
    live = versions >= 0
    index = versions.clamp_min(0)
    batch = torch.arange(n,device=q.device)[:,None,None]
    ks = keys[batch,index].masked_fill(~live[:,:,:,None],0.)
    vs = values[batch,index].masked_fill(~live[:,:,:,None],0.)
    strength = strength.masked_fill(~live,0.)
    scores = (ks*q[:,:,None,:]).sum(-1)*(2.0/k**0.5)
    scores = scores.masked_fill(~live,-1e30)
    numer = torch.exp(scores-scores.max(-1,keepdim=True).values)*live
    weights = numer/numer.sum(-1,keepdim=True).clamp_min(1e-30)
    return (weights[:,:,:,None]*strength[:,:,:,None]*vs).sum(-2)


def chunk_read(q, keys, values, versions, strength):
    """Inference component. Version IDs are checked by a device assertion.

    Differentiable callers must currently use reference_read: no G10 replay
    composition is claimed by this primitive. This fails closed on grad inputs.
    """
    n,c,k,m,v,s = _validate(q,keys,values,versions,strength)
    if not q.is_cuda:
        raise ValueError('chunk_read requires CUDA')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,keys,values,strength)):
        raise ValueError('recorded-decision read adjoints are not integrated')
    r = triton.cdiv(s,32)
    maximum = torch.empty((n,c,r),device=q.device)
    denom = torch.empty_like(maximum)
    partial = torch.empty((n,c,r,v),device=q.device)
    out = torch.empty((n,c,v),device=q.device)
    _tiles[(n,c,r)](q,keys,values,versions,strength,maximum,denom,partial,
                    c,s,m,k,v,r,triton.next_power_of_2(max(32,k)),
                    triton.next_power_of_2(v),enable_fp_fusion=False,num_warps=4,debug=True)
    _reduce[(n,c,triton.cdiv(v,8))](maximum,denom,partial,out,c,v,r,
                                 triton.next_power_of_2(r),enable_fp_fusion=False,num_warps=4)
    return out
