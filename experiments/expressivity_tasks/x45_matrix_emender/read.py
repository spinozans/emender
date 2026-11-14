"""The one read of the matrix store, in one place, matching the certified reference.

The certified read (``x30_tape_cell/naive_reference.py`` docstring, and the
production kernel ``x35_tape_cell_v3/causal_read.py``) is

    out_t = sum_j softmax_j(score_t over LIVE slots) * strength_j * S[j]

with no division by the strength mass: the read *magnitude* carries the stored
strengths, which is by design, and the softmax is normalised over the live slots
only (``numer = where(live, exp(score - max), 0)``, ``denom = sum(numer)``).

This module implements exactly that, once:

* :func:`attention_probs` -- softmax over live rows, exactly zero when nothing is live;
* :func:`row_strength` -- the strength channel.  The matrix store has no separate
  strength scalar: its strength **is** the row norm ``||S[j]||``, so a POP
  (``S[j] := 0``) removes a row from the numerator in the same step;
* :func:`read_weights` -- ``softmax_j * strength_j``, the channel the scan consumes;
* :func:`read` -- ``w @ S`` as one dense matmul.

Scores are cosines between a learned query in the value space and the stored rows
(``scale * cos(q_t, S[j])``); the tape arm's keys and the matrix store's rows are the
same object, so this is the certified ``keys . q`` score with the key normalised.
That normalisation is the only difference, and ``test_read_matches_certified_reference``
holds the two to each other on identical state.

Mask: this arm masks the read by *store liveness* (a row exists because a write put
content there and no POP removed it), which is the same predicate the two certified
kernels use -- ``x30``'s ``live`` flag and ``x35``'s ``version >= 0``.
"""
from __future__ import annotations

import torch

READ_NORMALISATION = "certified"   # sum_j softmax_j * strength_j * v_j, no renormaliser
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
    """Softmax over live rows; an all-dead read is exactly zero attention."""
    scores = match_scores(query, rows, live, scale)
    all_dead = (~live.any(-1)).unsqueeze(-1).unsqueeze(-1)   # (b, h, 1, 1)
    safe = torch.where(all_dead, torch.zeros_like(scores), scores)
    probs = torch.softmax(safe, dim=-1)
    return torch.where(all_dead, torch.zeros_like(probs), probs)


def read_weights(query, rows, live, scale) -> torch.Tensor:
    """The single read definition: ``softmax_j(live) * strength_j`` (certified)."""
    return attention_probs(query, rows, live, scale) * row_strength(rows, live).unsqueeze(-2)


def read(weights: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    """``out_t = sum_j w_tj S[j]`` as one dense matmul (never a gather)."""
    return weights @ rows
