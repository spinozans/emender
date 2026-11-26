"""Current-weight log-feature mass and weighted value means for one level of leaves.

No [rows,rank,value] tensor or per-row normalized-feature tape. Output VJPs are
uniquely owned (no atomics). FP32/IEEE throughout; no implicit TF32 approximation.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .triton_prefix import supported


@triton.jit
def _scores(P, B, row, rr, live, R: tl.constexpr):
    ok = live[:, None] & (rr < R)[None, :]
    s = (
        tl.load(P + row[:, None] * R + rr[None, :], ok, 0.0)
        + tl.load(B + row, live, 0.0)[:, None]
    )
    return tl.where(ok, s, -float("inf"))


@triton.jit
def _leaf(
    P,
    B,
    V,
    Z,
    Mu,
    Peak,
    Inv,
    N,
    R: tl.constexpr,
    W: tl.constexpr,
    L: tl.constexpr,
    BR: tl.constexpr,
    BF: tl.constexpr,
    BV: tl.constexpr,
):
    a = tl.program_id(0).to(tl.int64)
    fi = tl.program_id(1)
    vi = tl.program_id(2)
    i = tl.arange(0, BR)
    row = a * L + i
    live = (i < L) & (row < N)
    rr = fi * BF + tl.arange(0, BF)
    j = vi * BV + tl.arange(0, BV)
    s = _scores(P, B, row, rr, live, R)
    mx = tl.where(rr < R, tl.max(s, 0), 0.0)
    e = tl.exp(s - mx[None, :])
    den = tl.where(rr < R, tl.sum(e, 0), 1.0)
    inv = 1.0 / den
    prob = e * inv[None, :]
    v = tl.load(
        V + row[:, None] * W + j[None, :], live[:, None] & (j < W)[None, :], 0.0
    )
    out = tl.dot(tl.trans(prob), v, input_precision="ieee")
    nr = a * R + rr
    tl.store(
        Mu + nr[:, None] * W + j[None, :], out, (rr < R)[:, None] & (j < W)[None, :]
    )
    if vi == 0:
        tl.store(Z + nr, mx + tl.log(den), rr < R)
        tl.store(Peak + nr, mx, rr < R)
        tl.store(Inv + nr, inv, rr < R)


@triton.jit
def _probs(P, B, Peak, Inv, a, row, rr, live, R: tl.constexpr):
    s = _scores(P, B, row, rr, live, R)
    nr = a * R + rr
    mx = tl.load(Peak + nr, rr < R, 0.0)
    inv = tl.load(Inv + nr, rr < R, 0.0)
    return tl.exp(s - mx[None, :]) * inv[None, :]


@triton.jit
def _bscore(
    P,
    B,
    V,
    Peak,
    Inv,
    Mu,
    Gz,
    Gu,
    Ds,
    N,
    R: tl.constexpr,
    W: tl.constexpr,
    L: tl.constexpr,
    BR: tl.constexpr,
    BF: tl.constexpr,
    BV: tl.constexpr,
):
    a = tl.program_id(0).to(tl.int64)
    fi = tl.program_id(1)
    i = tl.arange(0, BR)
    row = a * L + i
    live = (i < L) & (row < N)
    rr = fi * BF + tl.arange(0, BF)
    nr = a * R + rr
    pr = _probs(P, B, Peak, Inv, a, row, rr, live, R)
    proj = tl.full((BR, BF), 0, tl.float32)
    base = tl.load(Gz + nr, rr < R, 0.0)
    for vi in range(tl.cdiv(W, BV)):
        j = vi * BV + tl.arange(0, BV)
        rm = (rr < R)[:, None] & (j < W)[None, :]
        up = tl.load(Gu + nr[:, None] * W + j[None, :], rm, 0.0)
        mu = tl.load(Mu + nr[:, None] * W + j[None, :], rm, 0.0)
        val = tl.load(
            V + row[:, None] * W + j[None, :], live[:, None] & (j < W)[None, :], 0.0
        )
        proj += tl.dot(val, tl.trans(up), input_precision="ieee")
        base -= tl.sum(mu * up, 1)
    tl.store(
        Ds + row[:, None] * R + rr[None, :],
        pr * (proj + base[None, :]),
        live[:, None] & (rr < R)[None, :],
    )


@triton.jit
def _bvalue(
    P,
    B,
    Peak,
    Inv,
    Gu,
    Dv,
    N,
    R: tl.constexpr,
    W: tl.constexpr,
    L: tl.constexpr,
    BR: tl.constexpr,
    BF: tl.constexpr,
    BV: tl.constexpr,
):
    a = tl.program_id(0).to(tl.int64)
    vi = tl.program_id(1)
    i = tl.arange(0, BR)
    row = a * L + i
    live = (i < L) & (row < N)
    j = vi * BV + tl.arange(0, BV)
    grad = tl.full((BR, BV), 0, tl.float32)
    for fi in range(tl.cdiv(R, BF)):
        rr = fi * BF + tl.arange(0, BF)
        pr = _probs(P, B, Peak, Inv, a, row, rr, live, R)
        nr = a * R + rr
        up = tl.load(
            Gu + nr[:, None] * W + j[None, :], (rr < R)[:, None] & (j < W)[None, :], 0.0
        )
        grad += tl.dot(pr, up, input_precision="ieee")
    tl.store(Dv + row[:, None] * W + j[None, :], grad, live[:, None] & (j < W)[None, :])


class _Leaves(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, b, v, leaf):
        n, r = p.shape
        w = v.shape[1]
        nl = triton.cdiv(n, leaf)
        bf = max(16, min(32, triton.next_power_of_2(r)))
        bv = max(16, min(32, triton.next_power_of_2(w)))
        args = (n, r, w, leaf, max(16, triton.next_power_of_2(leaf)), bf, bv)
        grid = (nl, triton.cdiv(r, bf), triton.cdiv(w, bv))
        z = p.new_empty((nl, r))
        mu = p.new_empty((nl, r, w))
        pk = torch.empty_like(z)
        iv = torch.empty_like(z)
        _leaf[grid](p, b, v, z, mu, pk, iv, *args)
        ctx.save_for_backward(p, b, v, pk, iv, mu)
        ctx.args = args
        ctx.grid = grid
        return z, mu

    @staticmethod
    @once_differentiable
    def backward(ctx, gz, gu):
        p, b, v, pk, iv, mu = ctx.saved_tensors
        nl, f, w = ctx.grid
        up = gu.contiguous()
        ds = dv = None
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ds = torch.empty_like(p)
            _bscore[(nl, f)](p, b, v, pk, iv, mu, gz.contiguous(), up, ds, *ctx.args)
        if ctx.needs_input_grad[2]:
            dv = torch.empty_like(v)
            _bvalue[(nl, w)](p, b, pk, iv, up, dv, *ctx.args)
        return (
            ds if ctx.needs_input_grad[0] else None,
            ds.sum(1) if ctx.needs_input_grad[1] else None,
            dv,
            None,
        )


def leaves(p, b, v, leaf):
    if (
        not supported(p)
        or any(t.device != p.device or t.dtype != p.dtype for t in (b, v))
        or p.ndim != 2
        or not len(p)
        or b.shape != (len(p),)
        or v.ndim != 2
        or len(v) != len(p)
        or min(p.shape[1], v.shape[1]) < 1
        or not 1 <= leaf <= 256
    ):
        raise ValueError("Triton leaf needs finite FP32 rows and a leaf size in 1..256")
    return _Leaves.apply(p.contiguous(), b.contiguous(), v.contiguous(), leaf)
