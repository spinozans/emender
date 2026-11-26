"""Fused FP32 ragged block attention and first VJP. No query-by-corpus score tensor.

One program per (descriptor, value tile), not per occurrence. Descriptors have no
semantic fanout cap; the last row, key and value tiles are masked. Shared rows and
queries accumulate VJPs with FP32 atomics. CPU interpreter is for correctness only.
"""

import os

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .attention import TILE


@triton.jit
def _score(
    Q,
    K,
    Bias,
    g,
    row,
    live,
    D: tl.constexpr,
    BD: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
):
    if MODE == 2:
        s = tl.load(Q + g * D + row % D, live, 0.0)
    else:
        d = tl.arange(0, BD)
        x = tl.load(Q + g * D + d, d < D, 0.0)
        y = tl.load(
            K + row[:, None] * D + d[None, :], live[:, None] & (d < D)[None, :], 0.0
        )
        if MODE == 0:
            s = tl.sum(x[None, :] * y, 1) * SCALE
        else:
            a = tl.where((d < D)[None, :], x[None, :] + y, -float("inf"))
            mx = tl.max(a, 1)
            s = mx + tl.log(tl.sum(tl.exp(a - mx[:, None]), 1))
    return tl.where(live, s + tl.load(Bias + row, live, 0.0), -float("inf"))


@triton.jit
def _forward(
    Q,
    K,
    V,
    Bias,
    Desc,
    Peak,
    Den,
    Num,
    G,
    N,
    D: tl.constexpr,
    W: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    g0 = tl.load(Desc + tile * 3)
    g = tl.minimum(tl.maximum(g0, 0), G - 1)
    start = tl.load(Desc + tile * 3 + 1)
    size = tl.load(Desc + tile * 3 + 2)
    i = tl.arange(0, BR)
    row = start + i
    live = (i < size) & (row >= 0) & (row < N) & (g0 == g)
    s = _score(Q, K, Bias, g, row, live, D, BD, MODE, SCALE)
    pk = tl.max(s, 0)
    pk = tl.where(pk == -float("inf"), 0.0, pk)
    w = tl.exp(s - pk)
    j = chunk * BV + tl.arange(0, BV)
    valid = live[:, None] & (j < W)[None, :]
    value = tl.load(V + row[:, None] * W + j[None, :], valid, 0.0)
    u = tl.sum(w[:, None] * value, 0)
    tl.store(Num + tile * W + j, u, j < W)
    if chunk == 0:
        tl.store(Peak + tile, pk)
        tl.store(Den + tile, tl.sum(w, 0))


@triton.jit
def _backward(
    Q,
    K,
    V,
    Bias,
    Desc,
    Peak,
    Gz,
    Gu,
    Dq,
    Dk,
    Dv,
    Db,
    G,
    N,
    D: tl.constexpr,
    W: tl.constexpr,
    BR: tl.constexpr,
    BD: tl.constexpr,
    BV: tl.constexpr,
    MODE: tl.constexpr,
    SCALE: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    g0 = tl.load(Desc + tile * 3)
    g = tl.minimum(tl.maximum(g0, 0), G - 1)
    start = tl.load(Desc + tile * 3 + 1)
    size = tl.load(Desc + tile * 3 + 2)
    i = tl.arange(0, BR)
    row = start + i
    live = (i < size) & (row >= 0) & (row < N) & (g0 == g)
    s = _score(Q, K, Bias, g, row, live, D, BD, MODE, SCALE)
    w = tl.exp(s - tl.load(Peak + tile))
    j = chunk * BV + tl.arange(0, BV)
    vm = live[:, None] & (j < W)[None, :]
    value = tl.load(V + row[:, None] * W + j[None, :], vm, 0.0)
    du = tl.load(Gu + tile * W + j, j < W, 0.0)
    dz = tl.load(Gz + tile) if chunk == 0 else 0.0
    ds = w * (dz + tl.sum(value * du[None, :], 1))
    tl.atomic_add(
        Dv + row[:, None] * W + j[None, :], w[:, None] * du[None, :], vm, sem="relaxed"
    )
    tl.atomic_add(Db + row, ds, live, sem="relaxed")
    if MODE == 2:
        tl.atomic_add(Dq + g * D + row % D, ds, live, sem="relaxed")
    else:
        d = tl.arange(0, BD)
        km = live[:, None] & (d < D)[None, :]
        x = tl.load(Q + g * D + d, d < D, 0.0)
        y = tl.load(K + row[:, None] * D + d[None, :], km, 0.0)
        if MODE == 0:
            a = ds[:, None] * SCALE
            dx = tl.sum(a * y, 0)
            dy = a * x[None, :]
        else:
            f = tl.where((d < D)[None, :], x[None, :] + y, -float("inf"))
            wfeat = tl.exp(f - tl.max(f, 1)[:, None])
            dy = ds[:, None] * (wfeat / tl.sum(wfeat, 1)[:, None])
            dx = tl.sum(dy, 0)
        tl.atomic_add(Dq + g * D + d, dx, d < D, sem="relaxed")
        tl.atomic_add(Dk + row[:, None] * D + d[None, :], dy, km, sem="relaxed")


class _Tiles(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, b, desc, mode):
        m = len(desc)
        width = v.shape[1]
        dim = q.shape[1]
        bv = min(32, triton.next_power_of_2(width))
        args = (
            len(q),
            len(v),
            dim,
            width,
            TILE,
            triton.next_power_of_2(dim),
            bv,
            mode,
            dim**-0.5,
        )
        grid = (m, triton.cdiv(width, bv))
        pk = q.new_empty(m)
        z = q.new_empty(m)
        u = q.new_empty((m, width))
        _forward[grid](q, k, v, b, desc, pk, z, u, *args)
        ctx.mark_non_differentiable(pk)
        ctx.save_for_backward(q, k, v, b, desc, pk)
        ctx.grid = grid
        ctx.args = args
        ctx.mode = mode
        return pk, z, u

    @staticmethod
    @once_differentiable
    def backward(ctx, _dpeak, dz, du):
        q, k, v, b, desc, pk = ctx.saved_tensors
        dq = torch.zeros_like(q)
        dv = torch.zeros_like(v)
        db = torch.zeros_like(b)
        dk = torch.zeros_like(k) if ctx.mode != 2 else db
        _backward[ctx.grid](
            q,
            k,
            v,
            b,
            desc,
            pk,
            dz.contiguous(),
            du.contiguous(),
            dq,
            dk,
            dv,
            db,
            *ctx.args,
        )
        return dq, dk if ctx.mode != 2 else None, dv, db, None, None


def reduce_tiles(mode, q, k, v, b, desc):
    if not q.is_cuda and not (
        q.device.type == "cpu"
        and os.getenv("TRITON_INTERPRET") == "1"
        and os.getenv("CUDA_VISIBLE_DEVICES") == ""
    ):
        raise ValueError("Triton requires accelerator or explicit CPU interpreter")
    # Mode 2's implicit key is a unit basis vector for the feature channel.
    k = b if mode == 2 else k
    if (
        mode not in (0, 1, 2)
        or q.ndim != 2
        or v.ndim != 2
        or not len(desc)
        or desc.shape != (len(desc), 3)
        or desc.dtype != torch.int64
        or desc.device != q.device
        or any(x.dtype != torch.float32 or x.device != q.device for x in (q, k, v, b))
        or b.shape != (len(v),)
        or (mode != 2 and k.shape != (len(v), q.shape[1]))
    ):
        raise ValueError("invalid FP32 attention tile inputs")
    return _Tiles.apply(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        b.contiguous(),
        desc.contiguous(),
        mode,
    )
