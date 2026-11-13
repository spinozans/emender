"""Chunked prefix scan of a diagonal affine recurrence, without divisions.

The Emender memory step is affine in the numeric state once the discrete
decisions are fixed (``formal/lean/EMENDER_OPERATOR_ALGEBRA.typ``):

    s_{t+1} = D_t * s_t + b_t ,  D_t diagonal

Composing a run of steps gives the pair ``(P, I)`` with

    s_out = P * s_in + I ,  P = prod_k d_k ,  I = sum_e b_e * prod_{k>e} d_k

and pairs compose associatively (left factor first, matching
``MemoryAlgebra.AffineOperator``)::

    (P2, I2) after (P1, I1)  ==  (P2*P1, P2*I1 + I2)

Zero diagonal entries are ordinary data, not a singularity: POP death and
overwrite erasure both produce ``d = 0``.  The composition above never divides
by an accumulated decay, so it is exact in the presence of zeros; a
``cumprod``/division formulation would be wrong here.

Two entry points:

``scan_with_entry``      one chunk, ``[C, L]`` lanes, a known entry state
                         (intra-chunk prefix + renormalisation);
``chunked_prefix_scan``  all chunks, per-chunk summaries composed over the
                         chunk axis with ``prefix_compose`` (Hillis-Steele),
                         so the chunk-entry states are obtained without ever
                         forming an inverse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class AffineSummary:
    """A composed affine map ``s_out = product * s_in + injection``."""

    product: torch.Tensor    # [..., L] float
    injection: torch.Tensor  # [..., L] float

    def apply(self, state: torch.Tensor) -> torch.Tensor:
        return self.product * state + self.injection


def compose(later: AffineSummary, earlier: AffineSummary) -> AffineSummary:
    """``later`` applied AFTER ``earlier``: (P2,I2) o (P1,I1) = (P2P1, P2I1+I2)."""
    return AffineSummary(later.product * earlier.product,
                         later.product * earlier.injection + later.injection)


def step_summaries(multiplier: torch.Tensor, injection: torch.Tensor
                   ) -> Tuple[AffineSummary, torch.Tensor, torch.Tensor]:
    """Intra-chunk pass over ``[N, C, L]`` (chunk, step, lane).

    Returns ``(final, prefix_product, prefix_injection)``: ``final`` maps a
    chunk's entry state to its exit state; the two prefix tensors are the
    half-open local prefixes, so a state that starts at zero after step ``i``
    is ``prefix_injection[..., i, :]`` and the entry contribution is
    ``prefix_product[..., i, :] * entry``.
    """
    if multiplier.shape != injection.shape or multiplier.dim() != 3:
        raise ValueError("multiplier and injection must share an [N, C, L] shape")
    product = torch.ones_like(multiplier[:, 0])
    injected = torch.zeros_like(injection[:, 0])
    products, injections = [], []
    for i in range(multiplier.shape[1]):
        products.append(product)
        injections.append(injected)
        injected = multiplier[:, i] * injected + injection[:, i]
        product = multiplier[:, i] * product
    return (AffineSummary(product, injected),
            torch.stack(products, dim=1), torch.stack(injections, dim=1))


def prefix_compose(summaries: AffineSummary, *, dtype=torch.float64) -> AffineSummary:
    """Exclusive prefix of per-chunk summaries over the chunk axis.

    ``result[i]`` maps the sequence entry state to the ENTRY state of chunk
    ``i``.  Hillis-Steele inclusive scan of pairwise compositions, then a
    one-slot shift: only compositions, so no inverse (and hence no division)
    is ever formed, and a zero diagonal in any chunk stays exact.
    """
    chunks = summaries.product.shape[0]
    if chunks == 1:
        return AffineSummary(torch.ones_like(summaries.product),
                             torch.zeros_like(summaries.injection))
    product = summaries.product.to(dtype)
    injected = summaries.injection.to(dtype)
    inclusive_p, inclusive_i = product, injected
    offset = 1
    while offset < chunks:
        shifted_p = torch.cat([torch.ones_like(inclusive_p[:offset]),
                               inclusive_p[:-offset]], 0)
        shifted_i = torch.cat([torch.zeros_like(inclusive_i[:offset]),
                               inclusive_i[:-offset]], 0)
        inclusive_i = inclusive_p * shifted_i + inclusive_i
        inclusive_p = inclusive_p * shifted_p
        offset *= 2
    exclusive_p = torch.cat([torch.ones_like(inclusive_p[:1]), inclusive_p[:-1]], 0)
    exclusive_i = torch.cat([torch.zeros_like(inclusive_i[:1]), inclusive_i[:-1]], 0)
    return AffineSummary(exclusive_p.to(summaries.product.dtype),
                         exclusive_i.to(summaries.injection.dtype))


def _post_step(prefix_product, prefix_injection, final):
    """Half-open local prefixes -> the states AFTER each step."""
    product = torch.cat([prefix_product[..., 1:, :], final.product.unsqueeze(-2)], dim=-2)
    injected = torch.cat([prefix_injection[..., 1:, :],
                          final.injection.unsqueeze(-2)], dim=-2)
    return product, injected


def scan_with_entry(multiplier: torch.Tensor, injection: torch.Tensor,
                    entry: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """States of ``s_i = d_i s_{i-1} + b_i`` for one chunk.

    ``multiplier``/``injection`` are ``[C, L]``, ``entry`` is ``[L]``.
    Returns ``(states [C, L], final [L])`` with ``states[i]`` the state AFTER
    step ``i`` and ``final`` equal to ``states[-1]``.

    The entry state is carried by the recurrence itself rather than being
    re-applied as ``prefix_product * entry``: with a known entry the two are
    algebraically equal but not bitwise equal, and the certified sequential path
    uses this association.  ``chunked_prefix_scan`` keeps the renormalised form
    because there the entry comes from the composed chunk summaries.
    """
    states = []
    state = entry
    for index in range(multiplier.shape[0]):
        state = multiplier[index] * state + injection[index]
        states.append(state)
    return torch.stack(states), states[-1]


def chunked_prefix_scan(multiplier: torch.Tensor, injection: torch.Tensor, *,
                        boundary_dtype=torch.float64,
                        entry: torch.Tensor | None = None
                        ) -> Tuple[torch.Tensor, AffineSummary]:
    """Prefix states for ``[N, C, L]`` lanes, composed across chunks.

    Returns ``(states, sequence_summary)`` where ``states`` is ``[N, C, L]``
    holding the state after each step, starting from ``entry`` (``[L]``,
    default zero), and ``sequence_summary`` maps the state before chunk 0 to
    the state after the last chunk.
    """
    final, prefix_product, prefix_injection = step_summaries(multiplier, injection)
    entries = prefix_compose(final, dtype=boundary_dtype)
    post_product, post_injection = _post_step(prefix_product, prefix_injection, final)
    # The state entering each chunk, from the composed summaries alone.
    zero = torch.zeros_like(entries.injection)
    chunk_entry = entries.apply(zero if entry is None else entry)
    # Renormalise: re-multiply the chunk entry state by the local prefix product
    # instead of dividing a running state by an accumulated decay.
    states = post_product * chunk_entry.unsqueeze(1) + post_injection
    last_chunk = AffineSummary(final.product[-1], final.injection[-1])
    last_entry = AffineSummary(entries.product[-1], entries.injection[-1])
    sequence = compose(last_chunk, last_entry)
    if entry is not None:
        sequence = AffineSummary(sequence.product, sequence.apply(entry))
    return states, sequence
