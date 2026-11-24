"""FP32 scores for the packed controller's *legal* ragged operands, not an index/search kernel.

One lane row per actual candidate, not per hypothetical program. Only the selected feature
needs materialization for the next pointer. The ordinary first VJP recomputes feature vectors
and accumulates duplicates in FP32. Offset lists come from PackedRoundPlanner's host layout.
"""

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl

from .packed import PackedMemoryPolicy, PackedRoundPlanner


@triton.jit
def _row(T, U, I, B, J, Y, r, N, E, K, R, P, O):
    t = tl.load(T + r, r < N, other=0)
    u = tl.load(U + r, r < N, other=-1)
    i = tl.load(I + r, r < N, other=0)
    b = tl.load(B + r, r < N, other=0)
    j = tl.load(J + r, r < N, other=0)
    y = tl.load(Y + r, r < N, other=0)
    ok = (
        (r < N)
        & (t >= 0)
        & (t < E)
        & (u >= -1)
        & (u < E)
        & (i >= 0)
        & (i < K)
        & (j >= 0)
        & (j < R)
        & (b >= 0)
        & (b < P)
        & (y >= 0)
        & (y < O)
    )
    return t, u, i, b, j, y, ok


@triton.jit
def _value(W, A, Q, t, u, i, j, ok, d, D: tl.constexpr):
    keep = ok[:, None] & (d[None, :] < D)
    x = tl.load(W + t[:, None] * D + d[None, :], keep, 0.0)
    z = tl.load(W + tl.maximum(u, 0)[:, None] * D + d[None, :], keep & (u >= 0)[:, None], 0.0)
    v = x + z + tl.load(A + i[:, None] * D + d[None, :], keep, 0.0)
    q = tl.load(Q + j[:, None] * D + d[None, :], keep, 0.0)
    return v, q, keep


@triton.jit(
    do_not_specialize=["N", "E", "K", "R", "P", "O"],
    do_not_specialize_on_alignment=["T", "U", "I", "B", "J", "Y"],
)
def _score(
    W,
    A,
    Q,
    Prior,
    T,
    U,
    I,
    B,
    J,
    Y,
    Out,
    N,
    E,
    K,
    R,
    P,
    O,
    D: tl.constexpr,
    BD: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
):
    r = tl.program_id(0) * M + tl.arange(0, M)
    d = tl.arange(0, BD)
    t, u, i, b, j, y, ok = _row(T, U, I, B, J, Y, r, N, E, K, R, P, O)
    v, q, keep = _value(W, A, Q, t, u, i, j, ok, d, D)
    tl.store(Out + y, tl.sum(v * q, 1) * S + tl.load(Prior + b, ok, 0.0), ok)


@triton.jit(
    do_not_specialize=["N", "E", "K", "R", "P", "O"],
    do_not_specialize_on_alignment=["T", "U", "I", "B", "J", "Y"],
)
def _adj(
    W,
    A,
    Q,
    T,
    U,
    I,
    B,
    J,
    Y,
    Up,
    DW,
    DA,
    DQ,
    DP,
    N,
    E,
    K,
    R,
    P,
    O,
    D: tl.constexpr,
    BD: tl.constexpr,
    M: tl.constexpr,
    S: tl.constexpr,
):
    r = tl.program_id(0) * M + tl.arange(0, M)
    d = tl.arange(0, BD)
    t, u, i, b, j, y, ok = _row(T, U, I, B, J, Y, r, N, E, K, R, P, O)
    v, q, keep = _value(W, A, Q, t, u, i, j, ok, d, D)
    dy = tl.load(Up + y, ok, 0.0)
    g = (dy * S)[:, None]
    tl.atomic_add(DW + t[:, None] * D + d[None, :], g * q, keep, sem="relaxed")
    tl.atomic_add(
        DW + tl.maximum(u, 0)[:, None] * D + d[None, :], g * q, keep & (u >= 0)[:, None], sem="relaxed"
    )
    tl.atomic_add(DA + i[:, None] * D + d[None, :], g * q, keep, sem="relaxed")
    tl.atomic_add(DQ + j[:, None] * D + d[None, :], g * v, keep, sem="relaxed")
    tl.atomic_add(DP + b, dy, ok, sem="relaxed")


class _Scores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, a, q, prior, t, u, i, b, j, y, width):
        n = t.numel()
        d = q.size(1)
        m = 16
        out = q.new_full((len(q), width), -torch.inf)
        ctx.shape = (
            n,
            len(w),
            len(a),
            len(q),
            len(prior),
            out.numel(),
            d,
            triton.next_power_of_2(d),
            m,
            d**-0.5,
        )
        ctx.save_for_backward(w, a, q, prior, t, u, i, b, j, y)
        with torch.cuda.device(w.device):
            _score[(triton.cdiv(n, m),)](w, a, q, prior, t, u, i, b, j, y, out, *ctx.shape, num_warps=4)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, up):
        w, a, q, prior, t, u, i, b, j, y = ctx.saved_tensors
        grads = tuple(torch.zeros_like(x) for x in (w, a, q, prior))
        n, *_ = ctx.shape
        with torch.cuda.device(w.device):
            _adj[(triton.cdiv(n, 16),)](
                w, a, q, t, u, i, b, j, y, up.contiguous(), *grads, *ctx.shape, num_warps=4
            )
        return *grads, None, None, None, None, None, None, None


def _operand_logits(w, a, q, prior, layout, width):
    # Private. Each active owner has >=1 candidate, each destination is unique, and all
    # indices were bound against this round's tables before transfer. No device→host scan.
    if not q.is_cuda or not (
        w.ndim == a.ndim == q.ndim == 2
        and 0 < w.size(1) == a.size(1) == q.size(1) <= 128
        and prior.ndim == 1
        and width >= 1
    ):
        raise ValueError("expected bounded CUDA operand layout")
    if (
        len(layout) != 6
        or not len(layout[0])
        or any(t.ndim != 1 or len(t) != len(layout[0]) or t.dtype != torch.long for t in layout)
    ):
        raise ValueError("bad operand descriptors")
    if any(x.dtype != torch.float32 for x in (w, a, q, prior)) or any(
        not x.is_contiguous() or x.device != q.device for x in (w, a, q, prior, *layout)
    ):
        raise ValueError("expected single device, contiguous FP32 features and pointers")
    return _Scores.apply(w, a, q, prior, *layout, width)


class TritonRoundPlanner(PackedRoundPlanner):
    def operand_scores(self, q, prior, layout, width, second):
        t, u, i, *_ = layout
        logits = _operand_logits(self.p.embed.weight, self.bank, q, prior, layout, width)

        def selected(sel):
            v = self.p.embed(t[sel])
            if second:
                ix = u[sel]
                v = v + self.p.embed(ix.clamp_min(0)) * (ix >= 0)[:, None]
            return v + self.bank[i[sel]]

        return logits, selected


class TritonMemoryPolicy(PackedMemoryPolicy):
    planner_type = TritonRoundPlanner
