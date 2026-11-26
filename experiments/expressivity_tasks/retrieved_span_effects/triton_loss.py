"""Exact full-vocabulary NLL from padded GEMM rows. Missing targets contribute zero."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _ce(
    X,
    Target,
    Loss,
    Z,
    Up,
    Dx,
    V: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    BWD: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, BD)
    y = tl.load(Target + row)
    live = (y >= 0) & (y < V)
    x = tl.load(X + row * D + i, i < V, -float("inf")).to(tl.float32)
    if BWD:
        z = tl.load(Z + row)
        dl = tl.load(Up + row)
        grad = dl * (tl.exp(x - z) - (i == y).to(tl.float32))
        tl.store(Dx + row * D + i, tl.where(live & (i < V), grad, 0.0), i < D)
    else:
        mx = tl.max(x, 0)
        z = mx + tl.log(tl.sum(tl.exp(x - mx), 0))
        correct = tl.load(X + row * D + y, live, 0.0).to(tl.float32)
        tl.store(Loss + row, tl.where(live, z - correct, 0.0))
        tl.store(Z + row, z)


class _CE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, v):
        loss = x.new_empty(y.shape, dtype=torch.float32)
        z = torch.empty_like(loss)
        d = x.shape[-1]
        args = (v, d, triton.next_power_of_2(d))
        nw = 16 if d >= 4096 else 4
        _ce[(y.numel(),)](x, y, loss, z, x, x, *args, False, num_warps=nw)
        ctx.save_for_backward(x, y, z)
        ctx.args = args
        ctx.nw = nw
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, dl):
        x, y, z = ctx.saved_tensors
        dx = torch.empty_like(x)
        _ce[(y.numel(),)](
            x, y, z, z, dl.contiguous(), dx, *ctx.args, True, num_warps=ctx.nw
        )
        return dx, None, None


def token_nll(x, y, v):
    if (
        x.shape[:-1] != y.shape
        or not 0 < v <= x.shape[-1]
        or y.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("invalid token loss row")
    return _CE.apply(x.contiguous(), y.contiguous(), v)
