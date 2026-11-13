"""The orthonormal key family the matrix state is indexed by.

The registered requirement (``PREREGISTRATION_MATRIX_EMENDER.md``, "Keys") is an
orthonormal family whose *active table is a prefix* ``K[0:r]`` and which is never
re-orthonormalised, so that growth at a chunk boundary only reveals further rows.
Orthonormality is the whole premise of the exactness claim: for an orthonormal
family ``k_j^T k_i = delta_ij`` for every pair of active keys, so the coefficient
``S k_j`` along direction ``j`` is the *whole* of row ``j`` and CORRECT/POP cannot
leak into another direction.

Three deterministic families are provided.  All three satisfy
``K[0:r] @ K[0:r].T == I_r`` for every ``r`` in ``ORTHONORMAL_RANKS``:

``standard``
    the one-hot family ``k_j = e_j`` (row ``j`` of the identity).  It is the only
    family whose prefixes keep the *state shape* ``r x d_v`` nested, i.e. the only
    one for which "reveal row j" and "key direction j" are the same axis.  The
    model runs on this family; the ops are written on the coefficient axis it
    induces.
``hadamard``
    the rows of the Sylvester Hadamard matrix of order ``dim`` scaled by
    ``dim**-0.5``.  Any prefix of its rows is orthonormal in ``R^dim``.
``qr``
    the rows of ``Q`` from the QR decomposition of a seeded Gaussian matrix.

``hadamard``/``qr`` are used by the exactness tests to check that the ops are
exact for a *non-trivial* orthonormal family (an arbitrary fixed rotation of the
coefficient axis), not only for the one-hot family the model happens to use.
"""
from __future__ import annotations

import math

import torch

# The precompiled shape set the registration fixes for rank growth.
ORTHONORMAL_RANKS = (64, 128, 256, 512, 1024, 2048)
MAX_RANK = ORTHONORMAL_RANKS[-1]
FAMILIES = ("standard", "hadamard", "qr")


def snap_rank(rank: int) -> int:
    """Smallest precompiled rank that can hold ``rank`` directions."""
    if rank < 1:
        raise ValueError(f"rank must be positive, got {rank}")
    for candidate in ORTHONORMAL_RANKS:
        if candidate >= rank:
            return candidate
    raise ValueError(
        f"rank {rank} exceeds the precompiled shape set {ORTHONORMAL_RANKS}")


def growth_target(rank: int, geometric: float = 2.0, stride_floor: int = 64) -> int:
    """``max(r + stride_floor, ceil(g*r))`` snapped to the precompiled set.

    Already at the largest precompiled rank, the target is that rank (no growth),
    which is how the trigger saturates instead of erroring.
    """
    if geometric <= 1.0:
        raise ValueError(f"geometric factor must exceed 1, got {geometric}")
    if stride_floor < 1:
        raise ValueError(f"stride floor must be positive, got {stride_floor}")
    wanted = max(rank + stride_floor, math.ceil(geometric * rank))
    return snap_rank(min(wanted, MAX_RANK))


class KeyFamily:
    """A deterministic orthonormal family; ``basis_rows(r)`` is its ``r``-prefix."""

    def __init__(self, kind: str = "standard", dim: int = MAX_RANK, seed: int = 3401):
        if kind not in FAMILIES:
            raise ValueError(f"unknown key family {kind!r}; expected one of {FAMILIES}")
        if not 1 <= dim <= MAX_RANK:
            raise ValueError(f"key dimension must be in [1, {MAX_RANK}], got {dim}")
        if kind == "hadamard" and dim & (dim - 1):
            raise ValueError(f"the Hadamard family needs a power-of-two dim, got {dim}")
        self.kind = kind
        self.dim = int(dim)
        self.seed = int(seed)
        self._table: torch.Tensor | None = None

    def __repr__(self) -> str:
        return f"KeyFamily(kind={self.kind!r}, dim={self.dim}, seed={self.seed})"

    def _build(self) -> torch.Tensor:
        if self.kind == "standard":
            return torch.eye(self.dim)
        if self.kind == "hadamard":
            matrix = torch.ones(1, 1)
            while matrix.shape[0] < self.dim:
                matrix = torch.cat(
                    (torch.cat((matrix, matrix), 1),
                     torch.cat((matrix, -matrix), 1)), 0)
            return matrix / math.sqrt(self.dim)
        generator = torch.Generator().manual_seed(self.seed)
        gaussian = torch.randn(self.dim, self.dim, generator=generator, dtype=torch.float64)
        q, _ = torch.linalg.qr(gaussian)
        return q.T.to(torch.float32)

    @property
    def table(self) -> torch.Tensor:
        """The full ``dim x dim`` orthonormal table, materialised on first use."""
        if self._table is None:
            self._table = self._build()
        return self._table

    def basis_rows(self, rank: int) -> torch.Tensor:
        """The active key table ``K[0:rank]`` (shape ``rank x dim``)."""
        if rank < 1 or rank > self.dim:
            raise ValueError(f"rank must be in [1, {self.dim}], got {rank}")
        return self.table[:rank]
