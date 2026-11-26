"""Trainable passage operations. Axes: G=(frontier,head), P=hits, T=tokens.

Independent implementation. Controls [k,d,r,u] are *post*-activation: r=erase*k,
u=write*value. LENGTH=-1 means missing; 0 means a present identity passage.
Scores are over the supplied candidates, not an occurrence locator. No host sync
or mutation of the live memory. The CPU interpreter is for correctness only.
"""

import os

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable


@triton.jit
def _route_f(
    Q,
    Keys,
    Prior,
    L,
    Pout,
    P: tl.constexpr,
    R: tl.constexpr,
    T: tl.constexpr,
    BP: tl.constexpr,
    BR: tl.constexpr,
    SC: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    p, r = tl.arange(0, BP), tl.arange(0, BR)
    length = tl.load(L + g * P + p, p < P, -1)
    valid = (length >= 0) & (length <= T)
    q = tl.load(Q + g * R + r, r < R, 0).to(tl.float32)
    k = tl.load(
        Keys + (g * P + p[:, None]) * R + r[None, :],
        valid[:, None] & (r < R)[None, :],
        0,
    ).to(tl.float32)
    score = tl.sum(k * q[None, :], 1) * SC + tl.load(Prior + g * P + p, valid, 0).to(
        tl.float32
    )
    score = tl.where(valid, score, -float("inf"))
    mx = tl.max(score, 0)
    exp = tl.exp(score - tl.where(mx == -float("inf"), 0.0, mx))
    weight = exp / tl.maximum(tl.sum(exp, 0), 1.0)
    tl.store(Pout + g * P + p, weight, p < P)


@triton.jit
def _route_b(
    Q,
    Keys,
    L,
    Prob,
    Up,
    Dq,
    Dk,
    Dprior,
    P: tl.constexpr,
    R: tl.constexpr,
    T: tl.constexpr,
    BP: tl.constexpr,
    BR: tl.constexpr,
    SC: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    p, r = tl.arange(0, BP), tl.arange(0, BR)
    length = tl.load(L + g * P + p, p < P, -1)
    valid = (length >= 0) & (length <= T)
    weight = tl.load(Prob + g * P + p, p < P, 0)
    up = tl.load(Up + g * P + p, valid, 0).to(tl.float32)
    ds = weight * (up - tl.sum(weight * up, 0))
    q = tl.load(Q + g * R + r, r < R, 0).to(tl.float32)
    ix = (g * P + p[:, None]) * R + r[None, :]
    k = tl.load(Keys + ix, valid[:, None] & (r < R)[None, :], 0).to(tl.float32)
    tl.store(
        Dk + ix, ds[:, None] * (SC * q[None, :]), (p < P)[:, None] & (r < R)[None, :]
    )
    tl.store(Dq + g * R + r, tl.sum(ds[:, None] * k, 0) * SC, r < R)
    tl.store(Dprior + g * P + p, ds, p < P)


@triton.jit
def _ctrl(C, row, t, active, ki, vi, T: tl.constexpr, K: tl.constexpr, V: tl.constexpr):
    base = C + (row * T + t) * (3 * K + V)
    mk = active[:, None] & (ki < K)[None, :]
    k = tl.load(base[:, None] + ki[None, :], mk, 0.0).to(tl.float32)
    d = tl.load(base[:, None] + K + ki[None, :], mk, 1.0).to(tl.float32)
    r = tl.load(base[:, None] + 2 * K + ki[None, :], mk, 0.0).to(tl.float32)
    u = tl.load(
        base[:, None] + 3 * K + vi[None, :], active[:, None] & (vi < V)[None, :], 0.0
    ).to(tl.float32)
    return k, d, r, u


@triton.jit
def _dual_f(
    C,
    Q,
    L,
    Edge,
    Tape,
    N: tl.constexpr,
    P: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SAVE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    ki, vi = tl.arange(0, BK), tl.arange(0, BV)
    n = tl.load(L + row, row < N, -1)
    ok = (n >= 0) & (n <= T)
    a = tl.load(
        Q + (row // P)[:, None] * K + ki[None, :], ok[:, None] & (ki < K)[None, :], 0.0
    ).to(tl.float32)
    b = tl.full((BR, BV), 0.0, tl.float32)
    for j in range(T):
        t = T - 1 - j
        active = ok & (t < n)
        k, d, r, u = _ctrl(C, row, t, active, ki, vi, T, K, V)
        if SAVE:
            tl.store(
                Tape + (row * T + t)[:, None] * K + ki[None, :],
                a,
                active[:, None] & (ki < K)[None, :],
            )
        s = tl.sum(k * a, 1)
        b += s[:, None] * u
        a = d * (a - s[:, None] * r)
    base = Edge + row * (K + V)
    tl.store(base[:, None] + ki[None, :], a, (row < N)[:, None] & (ki < K)[None, :])
    tl.store(base[:, None] + K + vi[None, :], b, (row < N)[:, None] & (vi < V)[None, :])


@triton.jit
def _dual_b(
    C,
    L,
    Up,
    Tape,
    Dc,
    Dqh,
    N: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BR: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64) * BR + tl.arange(0, BR)
    ki, vi = tl.arange(0, BK), tl.arange(0, BV)
    n = tl.load(L + row, row < N, -1)
    ok = (n >= 0) & (n <= T)
    base = Up + row * (K + V)
    da = tl.load(base[:, None] + ki[None, :], ok[:, None] & (ki < K)[None, :], 0.0).to(
        tl.float32
    )
    db = tl.load(
        base[:, None] + K + vi[None, :], ok[:, None] & (vi < V)[None, :], 0.0
    ).to(tl.float32)
    mk = (row < N)[:, None] & (ki < K)[None, :]
    mv = (row < N)[:, None] & (vi < V)[None, :]
    for t in range(T):
        active = ok & (t < n)
        k, d, r, u = _ctrl(C, row, t, active, ki, vi, T, K, V)
        a = tl.load(
            Tape + (row * T + t)[:, None] * K + ki[None, :],
            active[:, None] & (ki < K)[None, :],
            0.0,
        )
        s = tl.sum(a * k, 1)
        dm = da * d
        ds = tl.sum(db * u, 1) - tl.sum(dm * r, 1)
        dc = Dc + (row * T + t) * (3 * K + V)
        tl.store(dc[:, None] + ki[None, :], ds[:, None] * a, mk)
        tl.store(dc[:, None] + K + ki[None, :], da * (a - s[:, None] * r), mk)
        tl.store(dc[:, None] + 2 * K + ki[None, :], -s[:, None] * dm, mk)
        tl.store(dc[:, None] + 3 * K + vi[None, :], s[:, None] * db, mv)
        da = dm + ds[:, None] * k
    tl.store(Dqh + row[:, None] * K + ki[None, :], da, mk)


@triton.jit
def _merge_f(
    Edge,
    Prob,
    S,
    Y,
    Abar,
    P: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BP: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SEED: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    p, k, v = tl.arange(0, BP), tl.arange(0, BK), tl.arange(0, BV)
    prob = tl.load(Prob + g * P + p, p < P, 0.0).to(tl.float32)
    base = Edge + (g * P + p) * (K + V)
    b = tl.load(
        base[:, None] + K + v[None, :], (p < P)[:, None] & (v < V)[None, :], 0.0
    )
    y = tl.sum(b * prob[:, None], 0)
    if SEED:
        a = tl.load(
            base[:, None] + k[None, :], (p < P)[:, None] & (k < K)[None, :], 0.0
        )
        aa = tl.sum(a * prob[:, None], 0)
        st = tl.load(
            S + (g * K + k[:, None]) * V + v[None, :],
            (k < K)[:, None] & (v < V)[None, :],
            0.0,
        ).to(tl.float32)
        y += tl.sum(st * aa[:, None], 0)
        tl.store(Abar + g * K + k, aa, k < K)
    tl.store(Y + g * V + v, y, v < V)


@triton.jit
def _merge_b(
    Edge,
    Prob,
    S,
    Abar,
    Up,
    De,
    Dp,
    Ds,
    P: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BP: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SEED: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    p, k, v = tl.arange(0, BP), tl.arange(0, BK), tl.arange(0, BV)
    prob = tl.load(Prob + g * P + p, p < P, 0.0).to(tl.float32)
    dy = tl.load(Up + g * V + v, v < V, 0.0).to(tl.float32)
    ix = (g * P + p) * (K + V)
    mv = (p < P)[:, None] & (v < V)[None, :]
    mk = (p < P)[:, None] & (k < K)[None, :]
    b = tl.load(Edge + ix[:, None] + K + v[None, :], mv, 0.0)
    dp = tl.sum(b * dy[None, :], 1)
    da = tl.full((BK,), 0.0, tl.float32)
    if SEED:
        off = (g * K + k[:, None]) * V + v[None, :]
        mask = (k < K)[:, None] & (v < V)[None, :]
        st = tl.load(S + off, mask, 0.0).to(tl.float32)
        aa = tl.load(Abar + g * K + k, k < K, 0.0)
        tl.store(Ds + off, aa[:, None] * dy[None, :], mask)
        da = tl.sum(st * dy[None, :], 1)
        a = tl.load(Edge + ix[:, None] + k[None, :], mk, 0.0)
        dp += tl.sum(a * da[None, :], 1)
    tl.store(Dp + g * P + p, dp, p < P)
    tl.store(De + ix[:, None] + k[None, :], prob[:, None] * da[None, :], mk)
    tl.store(De + ix[:, None] + K + v[None, :], prob[:, None] * dy[None, :], mv)


@triton.jit
def _matrix_f(
    C,
    S,
    L,
    Y,
    Tape,
    Q,
    Reads,
    P: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    E97: tl.constexpr,
    SEED: tl.constexpr,
    SAVE: tl.constexpr,
    READ: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(L + row)
    ok = (n >= 0) & (n <= T)
    length = tl.where(ok, n, 0)
    kx, vx = tl.arange(0, BK), tl.arange(0, BV)
    ix = kx[:, None] * V + vx[None, :]
    mask = (kx < K)[:, None] & (vx < V)[None, :]
    h = tl.full((BK, BV), 0.0, tl.float32)
    if SEED:
        h = tl.load(S + (row // P) * K * V + ix, mask & ok, 0.0).to(tl.float32)
    for t in range(T):
        active = t < length
        base = C + (row * T + t) * (3 * K + V)
        k = tl.load(base + kx, active & (kx < K), 0.0).to(tl.float32)
        d = tl.load(base + K + kx, active & (kx < K), 1.0).to(tl.float32)
        r = tl.load(base + 2 * K + kx, active & (kx < K), 0.0).to(tl.float32)
        u = tl.load(base + 3 * K + vx, active & (vx < V), 0.0).to(tl.float32)
        if SAVE:
            tl.store(Tape + (row * T + t) * K * V + ix, h, mask)
        z = d[:, None] * h
        if E97:
            er = u - tl.sum(r[:, None] * h, 0)
            h = tl.where(
                active, 2.0 * tl.sigmoid(2.0 * (z + k[:, None] * er[None, :])) - 1.0, h
            )
        else:
            er = u - tl.sum(r[:, None] * z, 0)
            h = z + k[:, None] * er[None, :]
        if READ:
            q = tl.load(Q + (row * T + t) * K + kx, active & (kx < K), 0.0).to(
                tl.float32
            )
            tl.store(Reads + (row * T + t) * V + vx, tl.sum(q[:, None] * h, 0), vx < V)
    tl.store(Y + row * K * V + ix, h, mask)


@triton.jit
def _matrix_b(
    C,
    L,
    Tape,
    Up,
    Dc,
    Dseed,
    Q,
    Dreads,
    Dq,
    P: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    E97: tl.constexpr,
    SEED: tl.constexpr,
    READ: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(L + row)
    ok = (n >= 0) & (n <= T)
    kx, vx = tl.arange(0, BK), tl.arange(0, BV)
    ix = kx[:, None] * V + vx[None, :]
    mask = (kx < K)[:, None] & (vx < V)[None, :]
    dh = tl.load(Up + row * K * V + ix, mask & ok, 0.0).to(tl.float32)
    for j in range(T):
        t = T - 1 - j
        active = ok & (t < n)
        off = (row * T + t) * (3 * K + V)
        k = tl.load(C + off + kx, active & (kx < K), 0.0).to(tl.float32)
        d = tl.load(C + off + K + kx, active & (kx < K), 1.0).to(tl.float32)
        r = tl.load(C + off + 2 * K + kx, active & (kx < K), 0.0).to(tl.float32)
        u = tl.load(C + off + 3 * K + vx, active & (vx < V), 0.0).to(tl.float32)
        h = tl.load(Tape + (row * T + t) * K * V + ix, mask & active, 0.0)
        z = d[:, None] * h
        if E97:
            er = u - tl.sum(r[:, None] * h, 0)
            out = 2.0 * tl.sigmoid(2.0 * (z + k[:, None] * er[None, :])) - 1.0
        else:
            er = u - tl.sum(r[:, None] * z, 0)
            out = z + k[:, None] * er[None, :]
        if READ:
            q = tl.load(Q + (row * T + t) * K + kx, active & (kx < K), 0.0).to(
                tl.float32
            )
            dy = tl.load(Dreads + (row * T + t) * V + vx, active & (vx < V), 0.0).to(
                tl.float32
            )
            dh += q[:, None] * dy[None, :]
            tl.store(Dq + (row * T + t) * K + kx, tl.sum(out * dy[None, :], 1), kx < K)
        if E97:
            dh = dh * (1.0 - out * out)
        dk = tl.sum(dh * er[None, :], 1)
        du = tl.sum(dh * k[:, None], 0)
        if E97:
            dr = -tl.sum(h * du[None, :], 1)
            dd = tl.sum(dh * h, 1)
            dh = d[:, None] * dh - r[:, None] * du[None, :]
        else:
            dr = -tl.sum(z * du[None, :], 1)
            dz = dh - r[:, None] * du[None, :]
            dd = tl.sum(dz * h, 1)
            dh = d[:, None] * dz
        tl.store(Dc + off + kx, dk, kx < K)
        tl.store(Dc + off + K + kx, dd, kx < K)
        tl.store(Dc + off + 2 * K + kx, dr, kx < K)
        tl.store(Dc + off + 3 * K + vx, du, vx < V)
    if SEED:
        tl.store(Dseed + row * K * V + ix, dh, mask)


@triton.jit
def _sum_hits(
    X, Y, P: tl.constexpr, D: tl.constexpr, BP: tl.constexpr, BD: tl.constexpr
):
    g = tl.program_id(0).to(tl.int64)
    p = tl.arange(0, BP)
    d = tl.program_id(1) * BD + tl.arange(0, BD)
    x = tl.load(
        X + (g * P + p[:, None]) * D + d[None, :],
        (p < P)[:, None] & (d < D)[None, :],
        0.0,
    ).to(tl.float32)
    tl.store(Y + g * D + d, tl.sum(x, 0), d < D)


pot = triton.next_power_of_2


def _inputs(*ts):
    if len({t.device for t in ts}) != 1 or (
        not ts[0].is_cuda
        and not (ts[0].device.type == "cpu" and os.getenv("TRITON_INTERPRET") == "1")
    ):
        raise ValueError("expected one accelerator or explicit CPU Triton interpreter")
    if not all(
        t.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.int32)
        for t in ts
    ):
        raise ValueError("FP32/BF16/FP16 controls, signed int32 lengths required")
    return [t.contiguous() for t in ts]


def _lengths(lens, g, p):
    if lens.shape != (g, p) or lens.dtype != torch.int32 or min(g, p) <= 0:
        raise ValueError(
            "expected nonempty [groups,hit slots] int32 length layout; -1 is a missing hit"
        )


def _red(x, like):
    g, p = x.shape[:2]
    d = like.numel() // g
    if p == 1:
        return x.view_as(like).to(like.dtype)
    out = torch.empty_like(like)
    bd = min(64, pot(d))
    _sum_hits[(g, triton.cdiv(d, bd))](x, out, p, d, pot(p), bd)
    return out


class _Route(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, keys, prior, lens, t):
        g, p, r = keys.shape
        args = (p, r, t, pot(p), pot(r), r**-0.5)
        prob = torch.empty((g, p), device=q.device, dtype=torch.float32)
        _route_f[(g,)](q, keys, prior, lens, prob, *args)
        ctx.save_for_backward(q, keys, prior, lens, prob)
        ctx.args = args
        return prob

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        q, k, b, lens, p = ctx.saved_tensors
        dq, dk, db = map(torch.empty_like, (q, k, b))
        _route_b[(q.shape[0],)](q, k, lens, p, up.contiguous(), dq, dk, db, *ctx.args)
        return dq, dk, db, None, None


def weights(address, keys, prior, lengths, t):
    """Masked learned dot-product weighting plus finite prior logits; all-miss => zeros."""
    q, k, b, lens = _inputs(address, keys, prior, lengths)
    g, p, r = k.shape
    _lengths(lens, g, p)
    if (
        q.shape != (g, r)
        or b.shape != (g, p)
        or min(r, t + 1) <= 0
        or not all(x.is_floating_point() for x in (q, k, b))
    ):
        raise ValueError("invalid route geometry/dtype")
    return _Route.apply(q, k, b, lens, t)


class _Dual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, q, lens, rows, save):
        g, p, t, d = c.shape
        k = q.shape[-1]
        v = d - 3 * k
        edges = c.new_empty((g, p, k + v), dtype=torch.float32)
        tape = c.new_empty((g, p, t, k) if save else (0,), dtype=torch.float32)
        args = (g * p, t, k, v, rows, pot(k), pot(v))
        grid = (triton.cdiv(g * p, rows),)
        _dual_f[grid](c, q, lens, edges, tape, g * p, p, *args[1:], save)
        ctx.save_for_backward(c, q, lens, tape)
        ctx.args = args
        ctx.grid = grid
        return edges

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        c, q, lens, tape = ctx.saved_tensors
        dc = torch.empty_like(c)
        dq = c.new_empty((*lens.shape, q.shape[-1]), dtype=torch.float32)
        _dual_b[ctx.grid](c, lens, up.contiguous(), tape, dc, dq, *ctx.args)
        return dc, _red(dq, q), None, None, None


def dual_edges(c, q, lengths, rows=4):
    """One linear read per group: [A_i.T @ q, B_i.T @ q], no per-token K×V matrices."""
    c, q, lens = _inputs(c, q, lengths)
    g, p, t, d = c.shape
    _lengths(lens, g, p)
    if (
        q.ndim != 2
        or q.shape[0] != g
        or q.shape[1] < 1
        or d <= 3 * q.shape[1]
        or rows not in (1, 2, 4, 8, 16, 32)
        or not all(x.is_floating_point() for x in (c, q))
    ):
        raise ValueError("invalid dual geometry/dtype")
    return _Dual.apply(
        c,
        q,
        lens,
        rows,
        torch.is_grad_enabled() and (q.requires_grad or c.requires_grad),
    )


class _Merge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, e, p, s, seeded):
        g, hits = p.shape
        _, k, v = s.shape
        args = (hits, k, v, pot(hits), pot(k), pot(v), seeded)
        out = e.new_empty((g, v))
        aa = e.new_empty((g, k) if seeded else (0,))
        _merge_f[(g,)](e, p, s, out, aa, *args)
        ctx.save_for_backward(e, p, s, aa)
        ctx.args = args
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        e, p, s, aa = ctx.saved_tensors
        de, dp = map(torch.empty_like, (e, p))
        seeded = ctx.args[-1]
        ds = torch.empty_like(s) if seeded else aa
        _merge_b[(s.shape[0],)](e, p, s, aa, up.contiguous(), de, dp, ds, *ctx.args)
        return de, dp, ds if seeded else None, None


def merge(edges, prob, seed, seeded=True):
    e, p, s = _inputs(edges, prob, seed)
    g, k, v = s.shape
    if (
        e.shape != (*p.shape, k + v)
        or p.ndim != 2
        or p.shape[0] != g
        or min(g, k, v, p.shape[1]) <= 0
        or e.dtype != torch.float32
        or not all(x.is_floating_point() for x in (p, s))
    ):
        raise ValueError("invalid merge geometry/dtype")
    return _Merge.apply(e, p, s, seeded)


class _Matrix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, c, s, lens, e97, seeded, save):
        g, p, t, d = c.shape
        _, k, v = s.shape
        out = c.new_empty((g, p, k, v), dtype=torch.float32)
        tape = c.new_empty((g, p, t, k, v) if save else (0,), dtype=torch.float32)
        args = (p, t, k, v, pot(k), pot(v), e97, seeded)
        _matrix_f[(g * p,)](c, s, lens, out, tape, s, s, *args, save, False)
        ctx.save_for_backward(c, s, lens, tape)
        ctx.args = args
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        c, s, lens, tape = ctx.saved_tensors
        seeded = ctx.args[-1]
        dc = torch.empty_like(c)
        ds = (
            torch.empty_like(up, memory_format=torch.contiguous_format)
            if seeded
            else s.new_empty(0)
        )
        _matrix_b[(lens.numel(),)](
            c, lens, tape, up.contiguous(), dc, ds, s, s, s, *ctx.args, False
        )
        return dc, _red(ds, s) if seeded else None, None, None, None, None


def terminal(c, seed, lengths, *, family="gdn2", seeded=True):
    """Per-hit terminal states; exact seeded replay, incl. nonlinear E97 split-edit.

    E97 callers broadcast their scalar decay over K and supply k, erase*k and
    write*value after the named layer's activations. No per-hit output norm here.
    """
    c, s, lens = _inputs(c, seed, lengths)
    g, p, t, d = c.shape
    gs, k, v = s.shape
    _lengths(lens, g, p)
    if (
        g != gs
        or d != 3 * k + v
        or min(k, v) <= 0
        or family not in ("gdn2", "e97")
        or not all(x.is_floating_point() for x in (s, c))
    ):
        raise ValueError("invalid terminal geometry/dtype")
    save = torch.is_grad_enabled() and (c.requires_grad or (seeded and s.requires_grad))
    return _Matrix.apply(c, s, lens, family == "e97", seeded, save)
