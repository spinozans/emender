"""The one read of the matrix store, in one place.

The four arms that ever read a store in this project are the tape Emender
(``x30``/``x40``), the X44 chunked replay, and now this one.  They do *not* all
normalise the same way, and the difference is large enough to move a held-out
number, so this module fixes exactly one definition and the tests pin its relation
to the certified reference numerically.

Definitions of the read over a store of rows ``S[0..r)`` with liveness ``live``,
row strengths ``s_j = ||S[j]||`` and query ``q_t``:

``certified`` (``x30_tape_cell/naive_reference.py``)
    ``a = softmax_j(scale * q.k_j over live)``; ``out_t = sum_j a_tj s_j S[j]``.
    No denominator: the read magnitude scales with the stored strengths.
``x35 training`` (``x35_tape_cell_v3/causal_read.py``)
    ``out_t = sum_j a_tj s_j S[j] / sum_j a_tj`` -- the denominator counts the
    softmax weights only, so strength drops out of the normalisation.  This is the
    inconsistency that made in-arm read parity unmeasurable.
``strength`` (this module, :data:`READ_NORMALISATION`)
    ``out_t = sum_j a_tj s_j S[j] / sum_j a_tj s_j``

This arm implements ``strength``.  Reasons: (i) it is the only one of the three
that is a convex combination, so the read magnitude is bounded by the stored
content and cannot be inflated by a large-strength row; (ii) it keeps ``s_j`` in
the mixture in the same place on both sides of the fraction, which is what makes
a POP (``s_j -> 0``) remove a slot from the read *and* from the normalisation in
one step -- under the ``x35`` form a dying row would keep its softmax weight and
dilute the denominator with content that is no longer there; (iii) it reduces to
the ``certified`` read exactly when ``sum_j a_tj s_j == 1``, so the parity test can
state the relation as an identity rather than as a tolerance.

The division in ``sum_j a_tj s_j`` is the read's softmax normaliser, not an
accumulated decay: the scan's decay/erase channel stays division-free (see
``scan.py``).
"""
from __future__ import annotations

import torch

READ_NORMALISATION = "strength"
CERTIFIED_NORMALISATION = "certified"
TRAINING_NORMALISATION = "x35"

EPS = 1e-6


def unit(x: torch.Tensor) -> torch.Tensor:
    """Row-wise L2 normalisation with a floor (direction only, no magnitude)."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def row_strength(rows: torch.Tensor, live: torch.Tensor) -> torch.Tensor:
    """``s_j = ||S[j]||`` for live rows, exactly 0 for dead ones."""
    return rows.norm(dim=-1) * live.to(rows.dtype)


def match_scores(query: torch.Tensor, rows: torch.Tensor, live: torch.Tensor,
                 scale: float) -> torch.Tensor:
    """``scale * cos(q_t, S[j])`` over live rows; dead rows score ``-inf``."""
    scores = scale * (unit(query) @ unit(rows).transpose(-1, -2))
    return scores.masked_fill(~live.unsqueeze(-2), float("-inf"))


def attention_probs(query, rows, live, scale) -> torch.Tensor:
    """Softmax over live rows; all-dead rows give exactly zero attention."""
    scores = match_scores(query, rows, live, scale)
    all_dead = (~live.any(-1)).unsqueeze(-1).unsqueeze(-1)   # (b, h, 1, 1)
    safe = torch.where(all_dead, torch.zeros_like(scores), scores)
    probs = torch.softmax(safe, dim=-1)
    return torch.where(all_dead, torch.zeros_like(probs), probs)


def mix(probs, rows, live):
    """``(w, sum_j a_j s_j)``: the strength-weighted mixture of the stored rows.

    ``w`` is the read-weight channel and the returned sum is exactly the
    normaliser that separates the ``strength`` read from the ``certified`` one.
    """
    weighted = probs * row_strength(rows, live).unsqueeze(-2)
    total = weighted.sum(-1)
    weights = torch.where(total.unsqueeze(-1) > 0,
                          weighted / total.clamp_min(EPS).unsqueeze(-1),
                          torch.zeros_like(weighted))
    return weights, total


def read_weights(query, rows, live, scale) -> torch.Tensor:
    """The single read definition (``strength`` normalisation, see module doc)."""
    return mix(attention_probs(query, rows, live, scale), rows, live)[0]


def read(weights: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``out_t = sum_j w_tj S[j]`` as one dense matmul (never a gather)."""
    return weights @ rows


def certified_read(weights: torch.Tensor, rows: torch.Tensor,
                   denominator: torch.Tensor) -> torch.Tensor:
    """The ``certified`` read (no strength normalisation), only for the parity test.

    ``certified_read(w, rows, total) == read(w, rows) * total`` -- the identity the
    parity test asserts against ``naive_reference``'s executed formula.
    """
    return (weights * denominator.unsqueeze(-1)) @ rows
