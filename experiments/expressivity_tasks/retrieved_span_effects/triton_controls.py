"""Fused projection *epilogue* for the small co-trained readers, not a checkpoint converter."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .triton_read import _inputs, pot


@triton.jit
def _run(
    X,
    Y,
    Up,
    Dx,
    N: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BR: tl.constexpr,
    E97: tl.constexpr,
    BWD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    kx, vx = tl.arange(0, BK), tl.arange(0, BV)
    mk = (row < N)[:, None] & (kx < K)[None, :]
    mv = (row < N)[:, None] & (vx < V)[None, :]
    xoff = row * (3 * K + 2 * V)
    off = row * (3 * K + V)
    k = tl.load(X + xoff[:, None] + kx[None, :], mk, 0.0).to(tl.float32)
    kn = tl.sqrt(tl.sum(k * k, 1))
    kinv = 1.0 / tl.maximum(kn, 1.0e-12)
    key = k * kinv[:, None]
    if E97:
        # This standalone E97-style head uses one learned decay per token.
        dr = tl.load(X + xoff + K, row < N, 0.0).to(tl.float32)
        d = tl.sigmoid(dr[:, None] + 2.0 + 0.0 * kx[None, :])
    else:
        d = tl.sigmoid(
            2.0 + tl.load(X + xoff[:, None] + K + kx[None, :], mk, 0.0).to(tl.float32)
        )
    e = tl.sigmoid(
        tl.load(X + xoff[:, None] + 2 * K + kx[None, :], mk, 0.0).to(tl.float32)
    )
    w = tl.sigmoid(
        tl.load(X + xoff[:, None] + 3 * K + vx[None, :], mv, 0.0).to(tl.float32)
    )
    value = tl.load(X + xoff[:, None] + 3 * K + V + vx[None, :], mv, 0.0).to(tl.float32)
    if BWD:
        dk = tl.load(Up + off[:, None] + kx[None, :], mk, 0.0).to(tl.float32)
        dd = tl.load(Up + off[:, None] + K + kx[None, :], mk, 0.0).to(tl.float32)
        drk = tl.load(Up + off[:, None] + 2 * K + kx[None, :], mk, 0.0).to(tl.float32)
        du = tl.load(Up + off[:, None] + 3 * K + vx[None, :], mv, 0.0).to(tl.float32)
        dk = dk + drk * e
        dk = (
            dk
            - tl.where(
                (kn >= 1.0e-12)[:, None], key * tl.sum(dk * key, 1)[:, None], 0.0
            )
        ) * kinv[:, None]
        dd = dd * d * (1.0 - d)
        if E97:
            dd = tl.where(kx[None, :] == 0, tl.sum(dd, 1)[:, None], 0.0)
        tl.store(Dx + xoff[:, None] + kx[None, :], dk, mk)
        tl.store(Dx + xoff[:, None] + K + kx[None, :], dd, mk)
        tl.store(
            Dx + xoff[:, None] + 2 * K + kx[None, :], drk * key * e * (1.0 - e), mk
        )
        tl.store(
            Dx + xoff[:, None] + 3 * K + vx[None, :], du * value * w * (1.0 - w), mv
        )
        tl.store(Dx + xoff[:, None] + 3 * K + V + vx[None, :], du * w, mv)
    else:
        tl.store(Y + off[:, None] + kx[None, :], key, mk)
        tl.store(Y + off[:, None] + K + kx[None, :], d, mk)
        tl.store(Y + off[:, None] + 2 * K + kx[None, :], e * key, mk)
        tl.store(Y + off[:, None] + 3 * K + vx[None, :], w * value, mv)


class _Pack(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k, v, e97):
        y = x.new_empty((*x.shape[:-1], 3 * k + v))
        n = x.numel() // x.shape[-1]
        args = (n, k, v, pot(k), pot(v), 16, e97)
        grid = (triton.cdiv(n, 16),)
        _run[grid](x, y, x, x, *args, False)
        ctx.save_for_backward(x)
        ctx.args = args
        ctx.grid = grid
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        (x,) = ctx.saved_tensors
        dx = torch.empty_like(x)
        _run[ctx.grid](x, x, up.contiguous(), dx, *ctx.args, True)
        return dx, None, None, None


def pack(raw, k, v, *, family="gdn2"):
    (x,) = _inputs(raw)
    if (
        x.shape[-1] != 3 * k + 2 * v
        or not x.is_floating_point()
        or min(k, v) <= 0
        or family not in ("gdn2", "e97")
    ):
        raise ValueError("expected raw k, decay, erase, write gate, value")
    return _Pack.apply(x, k, v, family == "e97")
