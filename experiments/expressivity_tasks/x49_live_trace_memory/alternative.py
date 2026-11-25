"""X49-V causal proposal fields and 256-way answers. No targets, source reader or fitted state."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import torch
from . import surprise as u

DIMS = {"D": 2, "V": 4, "K": 4}


def _write_alternatives(logits, ids):
    # The observed arrival is known. Mark it using ONLY the preceding forecast.
    n = len(ids)
    if logits.shape != (n, 256) or not logits.isfinite().all():
        raise ValueError("expected finite aligned forecasts")
    tape = torch.zeros(n, dtype=torch.uint8)
    previous = logits[:-1].clone()
    previous[torch.arange(n - 1), ids[1:]] = -torch.inf
    tape[1:] = previous.argmax(dim=1).to(
        torch.uint8
    )  # argmax returns the smallest tied index
    return tape


@dataclass(frozen=True, slots=True)
class Proposal:
    byte: torch.Tensor
    feature: torch.Tensor
    probability: torch.Tensor


def _proposal(scores, norm, symbol):
    idx = torch.arange(len(symbol))
    lp = scores[idx, symbol] - norm
    if not lp.isfinite().all():
        raise ValueError("invalid non-copy proposal")
    code = u.quantize(-lp)
    return Proposal(
        symbol, torch.stack((torch.ones_like(lp), code.double() / 64), dim=1), lp.exp()
    )


@dataclass(frozen=True, slots=True)
class Packet:
    base: torch.Tensor  # confirmed U, not B
    query: torch.Tensor
    copy: torch.Tensor
    other: torch.Tensor  # U at hits, copy masked to -inf
    norm: torch.Tensor
    blocks: dict[str, tuple[Proposal, ...]]
    work: dict
    seconds: dict


@torch.inference_mode()
def fields(head, word, hidden, logits, u_beta):
    """Exactly one B/U path. Only the source of the additional byte varies between V and K."""
    b = u.bound_fields(head, word, hidden, logits)
    work = dict(b.work)
    secs = dict(b.seconds)
    tic = perf_counter()
    base = u.response(b, "U", u_beta)
    secs["incumbent_U_response_s"] = perf_counter() - tic
    tic = perf_counter()
    q, c, j = b.query, b.candidate, b.latest
    ix = torch.arange(len(q))
    other = base[q]
    other[ix, c] = -torch.inf
    norm = other.logsumexp(-1)
    cur = _proposal(other, norm, other.argmax(dim=1))
    secs["current_alternative_s"] = perf_counter() - tic
    tic = perf_counter()
    ids = torch.tensor(list(word), dtype=torch.long)
    tape = _write_alternatives(logits, ids)
    secs["alternative_build_s"] = perf_counter() - tic
    tic = perf_counter()
    v = _proposal(other, norm, tape[j].long())
    secs["V_read_and_feature_s"] = perf_counter() - tic
    tic = perf_counter()
    k = u._rotation(word, q, j, b.depth)
    if not torch.equal(ids[k], c):
        raise ValueError("wrong K follower")
    sham = _proposal(other, norm, tape[k].long())
    secs["K_rotation_read_and_feature_s"] = perf_counter() - tic
    work.update(
        alternative_storage_bytes=tape.nbytes,
        alternatives_encoded=len(tape) - 1,
        V_alternative_reads=len(q),
        K_alternative_reads=len(q),
        K_slots_displaced=int((k != j).sum()),
        K_codes_changed=int((sham.byte != v.byte).sum()),
        V_equals_current=int((v.byte == cur.byte).sum()),
        K_equals_current=int((sham.byte == cur.byte).sum()),
    )
    return Packet(
        base,
        q,
        c,
        other,
        norm,
        {"D": (cur,), "V": (cur, v), "K": (cur, sham)},
        work,
        secs,
    )


def checked_beta(arm, coefficients):
    v = torch.as_tensor(coefficients, dtype=torch.float64, device="cpu")
    if arm not in DIMS or v.shape != (DIMS[arm],) or not v.isfinite().all():
        raise ValueError("invalid alternative readout")
    return v


@torch.inference_mode()
def response(packet, arm, coefficients):
    """Shift only relative non-copy odds. Combine coincident nominations before normalization."""
    v = checked_beta(arm, coefficients)
    ix = torch.arange(len(packet.query))
    moved = packet.other.clone()
    for block, beta in zip(packet.blocks[arm], v.reshape(-1, 2), strict=True):
        moved[ix, block.byte] += block.feature @ beta
    moved -= (moved.logsumexp(-1) - packet.norm)[:, None]
    moved[ix, packet.copy] = packet.base[packet.query, packet.copy]
    out = packet.base.index_copy(0, packet.query, moved)
    if not out.isfinite().all():
        raise ValueError("non-finite alternative response")
    return out
