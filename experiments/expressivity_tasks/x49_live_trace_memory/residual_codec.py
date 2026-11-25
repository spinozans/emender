"""RB: eight fixed response axes, a one-byte residual code and write-only FP64 cell means."""

from dataclasses import dataclass
from hashlib import sha256
import numpy as np
import torch
from .decode_fit import VALUE

BITS = 8


def response_axes(head):
    w = head.mix.weight.detach()
    start = head.direct.host_dim
    if (
        head.training
        or w.dtype != torch.float32
        or w.device.type != "cpu"
        or w.shape != (64, start + 32 + 16 + 11 + 256)
    ):
        raise ValueError("wrong frozen outer response matrix")
    a = w[:, start : start + VALUE].double().contiguous().numpy()
    if not np.isfinite(a).all():
        raise ValueError("nonfinite response weights")
    gram = a.T @ a
    vals, vec = np.linalg.eigh(gram)
    vals = vals[::-1]
    dirs = vec.T[::-1][:BITS].copy()
    gaps = vals[:BITS] - vals[1 : BITS + 1]
    eps = np.finfo(np.float64).eps
    if (
        not np.isfinite(vals).all()
        or not np.isfinite(dirs).all()
        or vals[0] <= 0
        or not (gaps > 64 * eps * vals[0]).all()
    ):
        raise ValueError("ambiguous leading response axes")
    orth = float(np.linalg.norm(dirs @ dirs.T - np.eye(BITS), ord=np.inf))
    err = float(
        np.linalg.norm(dirs @ gram - vals[:BITS, None] * dirs, ord=np.inf) / vals[0]
    )
    if max(orth, err) > 64 * VALUE * eps:
        raise ValueError("invalid response eigensystem")
    piv = np.abs(dirs).argmax(1)
    dirs *= np.sign(dirs[np.arange(BITS), piv])[:, None]
    rows = np.ascontiguousarray(dirs, dtype=np.float32)
    note = dict(
        axes_sha256=sha256(rows.astype("<f4", copy=False).tobytes()).hexdigest(),
        minimum_relative_leading_gap=float(gaps.min() / vals[0]),
        gram_relative_residual_inf=err,
        orthogonality_error_inf=orth,
    )
    return torch.from_numpy(rows), note


def encode(e, axes):
    if (
        e.ndim != 2
        or e.shape[1] != VALUE
        or axes.shape != (BITS, VALUE)
        or any(
            t.device.type != "cpu" or t.dtype != torch.float32 or not t.isfinite().all()
            for t in (e, axes)
        )
    ):
        raise ValueError("invalid residual/axes")
    # Each product of two finite bounded FP32 values is exact in FP64. Do not use
    # a size-dependent GEMM reduction (especially for codes on decision boundaries).
    axes = axes.double()
    r = e.double()
    s = torch.zeros((len(e), BITS), dtype=torch.float64)
    for j in range(VALUE):
        s += r[:, j, None] * axes[:, j]
    return (
        ((s > 0).to(torch.uint8) << torch.arange(BITS, dtype=torch.uint8))
        .sum(1)
        .to(torch.uint8)
    )


@dataclass(frozen=True, slots=True)
class Book:
    means: torch.Tensor
    global_mean: torch.Tensor
    count: (
        torch.Tensor
    )  # FIT support for auditing only; inference needs just the installed means.

    def __post_init__(self):
        m, g, n = self.means, self.global_mean, self.count
        if (
            m.shape != (256, VALUE)
            or g.shape != (VALUE,)
            or n.shape != (256,)
            or any(x.device.type != "cpu" for x in (m, g, n))
            or m.dtype != torch.float32
            or g.dtype != torch.float32
            or n.dtype != torch.int64
            or n.min() < 0
            or n.sum() <= 0
            or not m.isfinite().all()
            or not g.isfinite().all()
            or m.abs().max() > 2
            or g.abs().max() > 2
            or not (m[n == 0] == g).all()
        ):
            raise ValueError("invalid residual means/fallback")


class Moments:
    def __init__(self):
        self.count = np.zeros(256, np.int64)
        self.total = np.zeros((256, VALUE), np.float64)

    def add(self, codes, e):
        # Both arguments contain only eligible writes (slot 0 already excluded).
        if (
            codes.ndim != 1
            or e.shape != (len(codes), VALUE)
            or codes.dtype != torch.uint8
            or e.dtype != torch.float32
            or any(x.device.type != "cpu" for x in (e, codes))
            or not e.isfinite().all()
            or (e.abs() > 2).any()
        ):
            raise ValueError("invalid observed residual writes")
        hist = np.bincount(codes.numpy(), minlength=256)
        self.count += hist
        if len(codes):
            order = np.argsort(codes.numpy(), kind="stable")
            idx = np.flatnonzero(hist)
            first = np.r_[0, np.cumsum(hist[idx])[:-1]]
            self.total[idx] += np.add.reduceat(
                e.numpy()[order].astype(np.float64), first, axis=0
            )
        return hist.tolist()

    def solve(self):
        n = self.count
        if (
            np.any(n < 0)
            or n.sum() <= 0
            or np.any(self.total[n == 0])
            or not np.isfinite(self.total).all()
        ):
            raise ValueError("invalid residual FIT statistics")
        g = self.total.sum(0) / n.sum()
        means = np.broadcast_to(g, self.total.shape).copy()
        seen = n > 0
        means[seen] = self.total[seen] / n[seen, None]
        return Book(
            torch.from_numpy(means.astype(np.float32)),
            torch.from_numpy(g.astype(np.float32)),
            torch.from_numpy(n.copy()),
        )
