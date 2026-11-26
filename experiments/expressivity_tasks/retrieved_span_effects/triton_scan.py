"""All causal host reads + final memory, and reusable hit-weighted terminal memory."""

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable

from .triton_read import _inputs, _lengths, _matrix_b, _matrix_f, _sum_hits, pot


class _Scan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, q, s, lens, e97, save):
        g, t, _ = c.shape
        _, k, v = s.shape
        out = c.new_empty((g, t, v), dtype=torch.float32)
        final = torch.empty_like(s, dtype=torch.float32)
        tape = c.new_empty((g, t, k, v) if save else (0,), dtype=torch.float32)
        args = (1, t, k, v, pot(k), pot(v), e97, True)
        _matrix_f[(g,)](c, s, lens, final, tape, q, out, *args, save, True)
        ctx.save_for_backward(c, q, s, lens, tape)
        ctx.args = args
        return out, final

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, dh):
        c, q, s, lens, tape = ctx.saved_tensors
        dc, dq, ds = map(torch.empty_like, (c, q, s))
        _matrix_b[(s.shape[0],)](
            c,
            lens,
            tape,
            dh.contiguous(),
            dc,
            ds,
            q,
            dy.contiguous(),
            dq,
            *ctx.args,
            True,
        )
        return dc, dq, ds, None, None, None


def scan(c, q, s, lengths, *, family="gdn2"):
    c, q, s, lens = _inputs(c, q, s, lengths)
    g, t, d = c.shape
    gs, k, v = s.shape
    _lengths(lens, g, 1)
    if (
        gs != g
        or d != 3 * k + v
        or q.shape != (g, t, k)
        or min(k, v) <= 0
        or family not in ("e97", "gdn2")
        or not all(x.is_floating_point() for x in (c, q, s))
    ):
        raise ValueError("invalid scan geometry/dtype")
    return _Scan.apply(
        c,
        q,
        s,
        lens,
        family == "e97",
        torch.is_grad_enabled() and any(x.requires_grad for x in (c, q, s)),
    )


@triton.jit
def _pool(
    X,
    P,
    Y,
    Dy,
    Dx,
    Dp,
    Pn: tl.constexpr,
    D: tl.constexpr,
    BP: tl.constexpr,
    BD: tl.constexpr,
    NT: tl.constexpr,
    BWD: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    p = tl.arange(0, BP)
    d = tile * BD + tl.arange(0, BD)
    mask = (p < Pn)[:, None] & (d < D)[None, :]
    ix = (g * Pn + p[:, None]) * D + d[None, :]
    xx = tl.load(X + ix, mask, 0.0).to(tl.float32)
    w = tl.load(P + g * Pn + p, p < Pn, 0.0).to(tl.float32)
    if BWD:
        dy = tl.load(Dy + g * D + d, d < D, 0.0).to(tl.float32)
        tl.store(Dx + ix, w[:, None] * dy[None, :], mask)
        tl.store(Dp + (g * Pn + p) * NT + tile, tl.sum(xx * dy[None, :], 1), p < Pn)
    else:
        tl.store(Y + g * D + d, tl.sum(w[:, None] * xx, 0), d < D)


class _Pool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        g, n = p.shape
        d = x.numel() // (g * n)
        bd = min(128, pot(d))
        nt = triton.cdiv(d, bd)
        y = x.new_empty((g, *x.shape[2:]), dtype=torch.float32)
        args = (n, d, pot(n), bd, nt)
        _pool[(g, nt)](x, p, y, x, x, x, *args, False)
        ctx.save_for_backward(x, p)
        ctx.args = args
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        x, p = ctx.saved_tensors
        g, n = p.shape
        nt = ctx.args[-1]
        dx = torch.empty_like(x)
        dp = torch.empty_like(p)
        parts = dp if nt == 1 else x.new_empty((g, n, nt), dtype=torch.float32)
        _pool[(g, nt)](x, p, x, up.contiguous(), dx, parts, *ctx.args, True)
        if nt != 1:
            _sum_hits[(g * n, 1)](parts, dp, nt, 1, pot(nt), 1)
        return dx, dp


def pool(states, prob):
    x, p = _inputs(states, prob)
    if (
        p.ndim != 2
        or x.ndim < 3
        or x.shape[:2] != p.shape
        or min(x.shape) <= 0
        or not all(t.is_floating_point() for t in (x, p))
    ):
        raise ValueError("invalid pool geometry/dtype")
    return _Pool.apply(x, p)
