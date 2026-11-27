"""Disjoint binary log-mass/value merges for upper levels of the learned bank."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .triton_prefix import supported


@triton.jit
def _parent(
    Z,
    V,
    Y,
    U,
    P,
    N,
    R: tl.constexpr,
    W: tl.constexpr,
    BF: tl.constexpr,
    BV: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    f = tl.program_id(1) * BF + tl.arange(0, BF)
    j = tl.program_id(2) * BV + tl.arange(0, BV)
    a = 2 * n * R + f
    b = a + R
    dst = n * R + f
    second = 2 * n + 1 < N
    fm = f < R
    za = tl.load(Z + a, fm, 0.0)
    zb = tl.load(Z + b, fm & second, -float("inf"))
    pk = tl.maximum(za, zb)
    ea = tl.exp(za - pk)
    eb = tl.exp(zb - pk)
    den = ea + eb
    pa = ea / den
    pb = eb / den
    mask = fm[:, None] & (j < W)[None, :]
    va = tl.load(V + a[:, None] * W + j[None, :], mask, 0.0)
    vb = tl.load(V + b[:, None] * W + j[None, :], mask & second, 0.0)
    tl.store(
        U + dst[:, None] * W + j[None, :], pa[:, None] * va + pb[:, None] * vb, mask
    )
    if tl.program_id(2) == 0:
        tl.store(Y + dst, pk + tl.log(den), fm)
        tl.store(P + a, pa, fm)
        tl.store(P + b, pb, fm & second)


@triton.jit
def _unparent(
    V,
    U,
    P,
    GY,
    GU,
    DZ,
    DV,
    R: tl.constexpr,
    W: tl.constexpr,
    BF: tl.constexpr,
    BV: tl.constexpr,
):
    n = tl.program_id(0).to(tl.int64)
    f = tl.program_id(1) * BF + tl.arange(0, BF)
    fm = f < R
    a = n * R + f
    b = (n // 2) * R + f
    prob = tl.load(P + a, fm, 0.0)
    acc = tl.load(GY + b, fm, 0.0)
    for chunk in range(tl.cdiv(W, BV)):
        j = chunk * BV + tl.arange(0, BV)
        m = fm[:, None] & (j < W)[None, :]
        av = tl.load(V + a[:, None] * W + j[None, :], m, 0.0)
        bv = tl.load(U + b[:, None] * W + j[None, :], m, 0.0)
        up = tl.load(GU + b[:, None] * W + j[None, :], m, 0.0)
        acc += tl.sum((av - bv) * up, 1)
        tl.store(DV + a[:, None] * W + j[None, :], prob[:, None] * up, m)
    tl.store(DZ + a, prob * acc, fm)


class _Parent(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, v):
        n, r, w = v.shape
        m = (n + 1) // 2
        bf = min(8, triton.next_power_of_2(r))
        bv = min(64, triton.next_power_of_2(w))
        y = z.new_empty((m, r))
        u = v.new_empty((m, r, w))
        p = torch.empty_like(z)
        _parent[(m, triton.cdiv(r, bf), triton.cdiv(w, bv))](
            z, v, y, u, p, n, r, w, bf, bv
        )
        ctx.save_for_backward(v, u, p)
        ctx.args = (r, w, bf, bv)
        return y, u

    @staticmethod
    @once_differentiable
    def backward(ctx, gy, gu):
        v, u, p = ctx.saved_tensors
        dz = torch.empty_like(p)
        dv = torch.empty_like(v)
        r, w, bf, bv = ctx.args
        _unparent[(len(v), triton.cdiv(r, bf))](
            v, u, p, gy.contiguous(), gu.contiguous(), dz, dv, r, w, bf, bv
        )
        return dz, dv


def parents(z, v):
    if (
        not supported(z)
        or z.ndim != 2
        or v.ndim != 3
        or z.shape != v.shape[:2]
        or v.device != z.device
        or v.dtype != z.dtype
        or min(v.shape) < 1
    ):
        raise ValueError("invalid FP32 parent inputs")
    return _Parent.apply(z.contiguous(), v.contiguous())
