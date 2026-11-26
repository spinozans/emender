"""FP32 shifted softmax NLL. Keep peak AND reciprocal normalizer for the VJP."""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .triton_prefix import supported


@triton.jit
def _nll(X, Y, L, Peak, Inv, V: tl.constexpr, BV: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BV)
    pk = -float("inf")
    den = 0.0
    for k in range(tl.cdiv(V, BV)):
        j = k * BV + i
        s = tl.load(X + row * V + j, j < V, -float("inf"))
        mx = tl.maximum(pk, tl.max(s, 0))
        den = den * tl.exp(pk - mx) + tl.sum(tl.exp(s - mx), 0)
        pk = mx
    y = tl.load(Y + row)
    ok = (y >= 0) & (y < V)
    v = tl.load(X + row * V + y, ok, 0.0)
    tl.store(L + row, tl.where(ok, (pk - v) + tl.log(den), float("nan")))
    tl.store(Peak + row, pk)
    tl.store(Inv + row, 1.0 / den)


@triton.jit
def _dnll(X, Y, Peak, Inv, Up, Dx, V: tl.constexpr, BV: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * BV + tl.arange(0, BV)
    y = tl.load(Y + row)
    pk = tl.load(Peak + row)
    iv = tl.load(Inv + row)
    w = tl.exp(tl.load(X + row * V + j, j < V, -float("inf")) - pk) * iv
    grad = (w - (j == y).to(tl.float32)) * tl.load(Up + row)
    tl.store(Dx + row * V + j, tl.where((y >= 0) & (y < V), grad, float("nan")), j < V)


class _NLL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        n, v = x.shape
        bv = min(4096, triton.next_power_of_2(v))
        losses = x.new_empty(n)
        pk = torch.empty_like(losses)
        iv = torch.empty_like(losses)
        ctx.args = (v, bv)
        ctx.nw = 8 if bv >= 2048 else 4
        _nll[(n,)](x, y, losses, pk, iv, *ctx.args, num_warps=ctx.nw)
        ctx.save_for_backward(x, y, pk, iv)
        return losses

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        x, y, pk, iv = ctx.saved_tensors
        v, bv = ctx.args
        dx = torch.empty_like(x)
        _dnll[(len(y), triton.cdiv(v, bv))](
            x, y, pk, iv, up.contiguous(), dx, v, bv, num_warps=ctx.nw
        )
        return dx, None


def nll(x, y):
    if (
        not supported(x)
        or x.ndim != 2
        or not min(x.shape)
        or y.shape != (len(x),)
        or y.dtype != torch.int64
        or y.device != x.device
    ):
        raise ValueError("invalid native token NLL inputs")
    return _NLL.apply(x.contiguous(), y.contiguous()).sum()
