"""Triton read over bounded native address pages. No per-byte Python work here.

One program per (query,head,source), operating on an F x D tile. No expanded
[R,F,D] K/V tensors. The adjoint recomputes weights; aliased K/V slots use
FP32 atomic sums (not a bit-deterministic backward). No Triton search or
assumption that corpus K/V fit on device. See bank_read.BoundPage.
"""
import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .bank_read import check_layout


@triton.jit
def _load_read(Q, K, V, I, B, r, f, d, F: tl.constexpr, D: tl.constexpr, N,
               SCALE: tl.constexpr):
    slot = tl.load(I + r*F + f, mask=f < F, other=-1)
    valid = (slot >= 0) & (slot < N)
    # A fictitious padded -1 need not even form an address before the table.
    offset = tl.maximum(slot, 0).to(tl.int64)[:, None]*D + d[None, :]
    take = valid[:, None] & (d[None, :] < D)
    k = tl.load(K + offset, mask=take, other=0.).to(tl.float32)
    v = tl.load(V + offset, mask=take, other=0.).to(tl.float32)
    q = tl.load(Q + r*D + d, mask=d < D, other=0.).to(tl.float32)
    b = tl.load(B + r*F + f, mask=valid, other=0.).to(tl.float32)
    score = tl.where(valid, b + tl.sum(q[None, :]*k, 1)*SCALE, -float('inf'))
    return valid, offset, take, q, k, v, score


@triton.jit
def _forward(Q, K, V, I, B, Y, Stats, N, F: tl.constexpr, D: tl.constexpr,
             BF: tl.constexpr, BD: tl.constexpr, SCALE: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    f, d = tl.arange(0, BF), tl.arange(0, BD)
    valid, offset, take, q, k, v, score = _load_read(Q,K,V,I,B,r,f,d,F,D,N,SCALE)
    # The empty row's max is -inf. Use finite stats so no inf-inf division
    # or NaN participates in either the read or its adjoint.
    m = tl.max(score, 0)
    m = tl.where(m == -float('inf'), 0., m)
    e = tl.exp(score-m)
    inv = 1./tl.maximum(tl.sum(e, 0), 1.)
    w = e*inv
    tl.store(Y + r*D + d, tl.sum(w[:,None]*v, 0), mask=d < D)
    tl.store(Stats + 2*r, m)
    tl.store(Stats + 2*r+1, inv)


@triton.jit
def _backward(Q, K, V, I, B, Y, Stats, DY, DQ, DK, DV, DB, N,
              F: tl.constexpr, D: tl.constexpr, BF: tl.constexpr,
              BD: tl.constexpr, SCALE: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    f, d = tl.arange(0, BF), tl.arange(0, BD)
    valid, offset, take, q, k, v, score = _load_read(Q,K,V,I,B,r,f,d,F,D,N,SCALE)
    m, inv = tl.load(Stats + 2*r), tl.load(Stats + 2*r+1)
    w = tl.exp(score-m)*inv
    dy = tl.load(DY + r*D + d, mask=d < D, other=0.).to(tl.float32)
    y = tl.load(Y + r*D + d, mask=d < D, other=0.)
    ds = w*tl.sum((v-y[None,:])*dy[None,:], 1)
    tl.store(DB + r*F + f, ds, mask=f < F)
    tl.store(DQ + r*D + d, SCALE*tl.sum(ds[:,None]*k, 0), mask=d < D)
    tl.atomic_add(DK + offset, (ds*SCALE)[:,None]*q[None,:], mask=take, sem='relaxed')
    tl.atomic_add(DV + offset, w[:,None]*dy[None,:], mask=take, sem='relaxed')


class _Read(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ids, bias):
        r, d = q.shape
        n, f = k.shape[0], ids.shape[1]
        y = torch.empty((r,d), device=q.device, dtype=torch.float32)
        stats = torch.empty((r,2), device=q.device, dtype=torch.float32)
        if n:
            with torch.cuda.device(q.device):
                _forward[(r,)](q,k,v,ids,bias,y,stats,n,f,d,triton.next_power_of_2(f),
                               triton.next_power_of_2(d),d**-.5,num_warps=4)
        else:
            y.zero_()
        ctx.save_for_backward(q,k,v,ids,bias,y,stats)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        q,k,v,ids,bias,y,stats = ctx.saved_tensors
        r,d = q.shape
        n,f = k.shape[0],ids.shape[1]
        if not n:
            return torch.zeros_like(q),torch.zeros_like(k),torch.zeros_like(v),None,torch.zeros_like(bias)
        dq = torch.empty_like(q, dtype=torch.float32)
        dk = torch.zeros_like(k, dtype=torch.float32)
        dv = torch.zeros_like(v, dtype=torch.float32)
        db = torch.empty_like(bias)
        with torch.cuda.device(q.device):
            _backward[(r,)](q,k,v,ids,bias,y,stats,dy.contiguous(),dq,dk,dv,db,n,f,d,
                            triton.next_power_of_2(f),triton.next_power_of_2(d),d**-.5,num_warps=4)
        return dq.to(q.dtype),dk.to(k.dtype),dv.to(v.dtype),None,db


def triton_read(q, k, v, page, log_prior):
    """Input buffers already resident. Validation of IDs belongs at page binding.

    First-order VJP supported. Device timing must include gradient buffer zeroing
    and casting when benchmarking full backward. Corpus values may instead be
    frozen/on-demand; this initial implementation doesn't optimize that case.
    """
    check_layout(q,k,v,page,log_prior)
    if not q.is_cuda or q.dtype not in (torch.float32, torch.bfloat16) or log_prior.dtype != torch.float32:
        raise ValueError('expected CUDA fp32/bf16 QKV, fp32 log prior; no CPU fallback')
    return _Read.apply(q,k,v,page.ids,log_prior)
