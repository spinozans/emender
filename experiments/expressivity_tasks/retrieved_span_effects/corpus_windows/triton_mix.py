"""NLL fast path: online joint normalization of every tile, across all three sources.

Tiles return a detached peak p and differentiated (Z,U) in exp(-p) units.
This computes sum(exp(p) U)/sum(exp(p) Z) without exposing source diagnostics.
Their gradients still reach every partial, including the base and local rows.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .triton_prefix import supported


def csr(owners, queries):
    """One descriptor belongs to exactly one query; cache integers, not learned data."""
    ix = owners.argsort(stable=True)
    off = torch.searchsorted(owners[ix], torch.arange(queries + 1, device=ix.device))
    return ix, off


@triton.jit
def _mix(
    P, Z, U, Ids, Off, Y, Max, Inv, W: tl.constexpr, BR: tl.constexpr, BV: tl.constexpr
):
    q = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * BV + tl.arange(0, BV)
    off = tl.load(Off + q)
    end = tl.load(Off + q + 1)
    lanes = tl.arange(0, BR)
    pk = tl.full((), -float("inf"), tl.float32)
    s = tl.full((), 0.0, tl.float32)
    v = tl.full((BV,), 0.0, tl.float32)
    while off < end:
        live = off + lanes < end
        i = tl.load(Ids + off + lanes, live, 0)
        z = tl.load(Z + i, live, 0.0)
        p = tl.load(P + i, live & (z > 0), -float("inf"))
        hi = tl.maximum(pk, tl.max(p, 0))
        origin = tl.where(hi == -float("inf"), 0.0, hi)
        a = tl.exp(pk - origin)
        w = tl.exp(p - origin)
        u = tl.load(
            U + i[:, None] * W + j[None, :], live[:, None] & (j < W)[None, :], 0.0
        )
        s = a * s + tl.sum(w * z, 0)
        v = a * v + tl.sum(w[:, None] * u, 0)
        pk = hi
        off += BR
    inv = 1.0 / tl.where(s > 0, s, 1.0)
    tl.store(Y + q * W + j, v * inv, j < W)
    if tl.program_id(1) == 0:
        tl.store(Max + q, tl.where(s > 0, pk, 0.0))
        tl.store(Inv + q, inv)


@triton.jit
def _dmix(
    P,
    Z,
    Y,
    Max,
    Inv,
    Ids,
    Off,
    DY,
    DZ,
    DU,
    W: tl.constexpr,
    BR: tl.constexpr,
    BV: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * BV + tl.arange(0, BV)
    off = tl.load(Off + q)
    end = tl.load(Off + q + 1)
    lanes = tl.arange(0, BR)
    up = tl.load(DY + q * W + j, j < W, 0.0)
    v = tl.load(Y + q * W + j, j < W, 0.0)
    base = -tl.sum(v * up, 0)
    pk = tl.load(Max + q)
    inv = tl.load(Inv + q)
    while off < end:
        live = off + lanes < end
        i = tl.load(Ids + off + lanes, live, 0)
        z = tl.load(Z + i, live, 0.0)
        p = tl.load(P + i, live & (z > 0), -float("inf"))
        s = tl.exp(p - pk) * inv
        tl.store(
            DU + i[:, None] * W + j[None, :],
            s[:, None] * up[None, :],
            live[:, None] & (j < W)[None, :],
        )
        if BV >= W:
            tl.store(DZ + i, s * base, live)
        else:
            tl.atomic_add(DZ + i, s * base, live, sem="relaxed")
        off += BR


class _Mix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, z, u, ids, off):
        q = len(off) - 1
        w = u.shape[1]
        bv = min(64, triton.next_power_of_2(w))
        grid = (q, triton.cdiv(w, bv))
        out = u.new_empty((q, w))
        mx = p.new_empty(q)
        inv = p.new_empty(q)
        _mix[grid](p, z, u, ids, off, out, mx, inv, w, 16, bv)
        ctx.save_for_backward(p, z, out, mx, inv, ids, off)
        ctx.grid = grid
        ctx.args = (w, 16, bv)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        p, z, out, mx, inv, ids, off = ctx.saved_tensors
        w, br, bv = ctx.args
        dz = torch.zeros_like(z) if w > bv else torch.empty_like(z)
        du = z.new_empty((len(z), w))
        _dmix[ctx.grid](
            p, z, out, mx, inv, ids, off, up.contiguous(), dz, du, *ctx.args
        )
        return None, dz, du, None, None


def mix(p, z, u, route):
    ids, off = route
    if (
        not supported(p)
        or p.requires_grad
        or z.shape != p.shape
        or p.ndim != 1
        or u.ndim != 2
        or len(u) != len(z)
        or min(u.shape) < 1
        or any(t.device != p.device for t in (z, u, ids, off))
        or z.dtype != p.dtype
        or u.dtype != p.dtype
        or ids.ndim != 1
        or off.ndim != 1
        or len(ids) != len(z)
        or len(off) < 2
        or ids.dtype != torch.int64
        or off.dtype != torch.int64
    ):
        raise ValueError(
            "invalid partials or route; peaks must be detached scaling constants"
        )
    return _Mix.apply(
        p.contiguous(),
        z.contiguous(),
        u.contiguous(),
        ids.contiguous(),
        off.contiguous(),
    )
