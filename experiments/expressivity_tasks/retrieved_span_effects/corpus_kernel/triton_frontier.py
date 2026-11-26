"""Batched document-safe MEM interval discovery on a resident reverse endpoint index.

Two lower bounds per observed suffix depth. No occurrence gather, four-row cap,
read of a future token, or kernel launch per prefix token. An absent match returns
[0,0), not a claim of zero background attention.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _older(Flat, Ends, Offsets, mid, active, depth):
    p = tl.load(Ends + mid, active, 0)
    offset = tl.load(Offsets + mid, active, 0)
    return tl.load(Flat + p - depth, active & (offset >= depth), -1)


@triton.jit
def _seek(
    Flat,
    Ends,
    Offsets,
    X,
    Lengths,
    Out,
    B,
    N,
    T: tl.constexpr,
    DEPTH: tl.constexpr,
    IT: tl.constexpr,
    BQ: tl.constexpr,
    ALL: tl.constexpr,
):
    g = tl.program_id(0) * BQ + tl.arange(0, BQ)
    n = tl.load(Lengths + g, g < B, 0)
    valid = (g < B) & (n >= 0) & (n <= T)
    lo = tl.full((BQ,), 0, tl.int64)
    hi = lo + N
    for depth in range(DEPTH):
        active = valid & (depth < n) & (lo < hi)
        t = tl.load(X + g.to(tl.int64) * T + n - 1 - depth, active, 0)
        a = lo
        b = hi
        c = lo
        d = hi
        for _ in range(IT):
            m = (a + b) // 2
            live = active & (a < b)
            right = _older(Flat, Ends, Offsets, m, live, depth) < t
            a = tl.where(live & right, m + 1, a)
            b = tl.where(live & ~right, m, b)
            m = (c + d) // 2
            live = active & (c < d)
            right = _older(Flat, Ends, Offsets, m, live, depth) <= t
            c = tl.where(live & right, m + 1, c)
            d = tl.where(live & ~right, m, d)
        hit = active & (a < c) & (t >= 0)
        lo = tl.where(hit, a, 0)
        hi = tl.where(hit, c, 0)
        if ALL:
            j = (g.to(tl.int64) * DEPTH + depth) * 2
            tl.store(Out + j, lo, g < B)
            tl.store(Out + j + 1, hi, g < B)
    if not ALL:
        j = g.to(tl.int64) * 2
        tl.store(Out + j, lo, g < B)
        tl.store(Out + j + 1, hi, g < B)


class ResidentIndex:
    def __init__(self, index, device):
        self.version = index.version
        self.n = len(index)
        self.flat = torch.tensor(index._flat, device=device, dtype=torch.int64)
        self.ends = torch.tensor(index._ends, device=device, dtype=torch.int64)
        self.offsets = torch.tensor(index.row_offset, device=device, dtype=torch.int64)

    def intervals(self, observations, lengths, *, min_match=1, all_depths=False):
        x = observations
        device = self.flat.device
        if not x.is_cuda and not (
            device.type == "cpu"
            and os.getenv("TRITON_INTERPRET") == "1"
            and os.getenv("CUDA_VISIBLE_DEVICES") == ""
        ):
            raise ValueError("requires accelerator or explicit CPU interpreter")
        if (
            x.ndim != 2
            or not len(x)
            or lengths.shape != (len(x),)
            or min_match < 1
            or x.device != device
            or lengths.device != device
            or x.dtype not in (torch.int64, torch.int32)
            or lengths.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("invalid observed-prefix batch")
        b, t = x.shape
        depth = t if all_depths else min_match
        shape = (b, depth, 2) if all_depths else (b, 2)
        if not self.n or not depth or depth > t:
            return torch.zeros(shape, device=device, dtype=torch.int64)
        out = torch.empty(shape, device=device, dtype=torch.int64)
        _seek[(triton.cdiv(b, 16),)](
            self.flat,
            self.ends,
            self.offsets,
            x.contiguous(),
            lengths.contiguous(),
            out,
            b,
            self.n,
            t,
            depth,
            self.n.bit_length(),
            16,
            all_depths,
        )
        return out
