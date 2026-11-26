"""Segmented prefix mean and its causal first VJP, packed real tokens, FP32."""

import os

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


def supported(x):
    return x.dtype == torch.float32 and (
        x.is_cuda
        or (
            x.device.type == "cpu"
            and os.getenv("CUDA_VISIBLE_DEVICES") == ""
            and os.getenv("TRITON_INTERPRET") == "1"
        )
    )


@triton.jit
def _scan(
    X,
    Spans,
    Y,
    N,
    D: tl.constexpr,
    BT: tl.constexpr,
    BF: tl.constexpr,
    REVERSE: tl.constexpr,
):
    doc = tl.program_id(0).to(tl.int64)
    start = tl.load(Spans + 2 * doc)
    length = tl.load(Spans + 2 * doc + 1)
    f = tl.program_id(1) * BF + tl.arange(0, BF)
    i = tl.arange(0, BT)
    carry = tl.full((BF,), 0, tl.float32)
    base = tl.full((), 0, tl.int64)
    while base < length:
        lane = base + i
        if REVERSE:
            t = length - 1 - lane
        else:
            t = lane
        row = start + t
        ok = (lane < length) & (row >= 0) & (row < N)
        mask = ok[:, None] & (f < D)[None, :]
        z = tl.load(X + row[:, None] * D + f[None, :], mask, 0.0)
        den = tl.maximum(t + 1, 1).to(tl.float32)
        if REVERSE:
            z = z / den[:, None]
        acc = tl.cumsum(z, 0) + carry[None, :]
        if not REVERSE:
            acc = acc / den[:, None]
        tl.store(Y + row[:, None] * D + f[None, :], acc, mask)
        carry += tl.sum(z, 0)
        base += BT


class _Prefix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, spans):
        n, d = x.shape
        bf = min(32, triton.next_power_of_2(d))
        ctx.args = (n, d, 32, bf)
        ctx.grid = (len(spans), triton.cdiv(d, bf))
        ctx.save_for_backward(spans)
        y = torch.empty_like(x)
        _scan[ctx.grid](x, spans, y, *ctx.args, False)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        (spans,) = ctx.saved_tensors
        dx = torch.empty_like(dy, memory_format=torch.contiguous_format)
        _scan[ctx.grid](dy.contiguous(), spans, dx, *ctx.args, True)
        return dx, None


def prefix(x, inputs):
    spans = inputs.spans
    if (
        not supported(x)
        or x.ndim != 2
        or not x.shape[1]
        or len(x) != len(inputs.ids)
        or spans.device != x.device
        or spans.dtype != torch.int64
        or spans.shape != (len(inputs.lengths), 2)
    ):
        raise ValueError("invalid packed Triton prefix inputs")
    if not len(x):
        return x
    return _Prefix.apply(x.contiguous(), spans.contiguous())
