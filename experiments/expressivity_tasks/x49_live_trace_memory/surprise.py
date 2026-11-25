"""X49-U: one-byte forecast marks on B's real path. No target, supplied address or file loader."""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import torch
from .literal import Match, rotated_payload_index
from .values import LiveValueHead

HORIZON = 4096
DIMS = {"N": 3, "U": 4, "S": 4}


def quantize(s):
    """Round half up, then saturate. Never allow NaN/inf to cast to a plausible byte."""
    if s.dtype != torch.float64 or not torch.isfinite(s).all():
        raise ValueError("surprise must be finite FP64")
    return (16 * s.clamp(0.0, 16.0) + 0.5).floor().clamp_max_(255).to(torch.uint8)


def _softplus(x):
    return torch.logaddexp(x, torch.zeros((), dtype=x.dtype, device=x.device))


def _candidate_margin(scores, q, c):
    # Do not use log(1-exp(L_c)): a confident FP32 incumbent can round that argument to zero.
    other = scores[q].double()
    i = torch.arange(len(q))
    old = other[i, c]
    other[i, c] = -torch.inf
    return old - other.logsumexp(-1)


def _arrival_tape(z, ids, query, now_norms):
    # Always write each j>=1 on arrival; the forecast is row j-1, NOT j. A normalizer
    # already computed at a hit frontier can be reused for the next arrival at this weight.
    n = len(ids)
    norms = torch.empty(n - 1, dtype=torch.float64)
    needs = torch.ones(n - 1, dtype=torch.bool)
    keep = query < n - 1
    shared = query[keep]
    norms[shared] = now_norms[keep]
    needs[shared] = False
    missing = needs.nonzero().flatten()
    norms[missing] = z[missing].double().logsumexp(-1)
    tape = torch.zeros(n, dtype=torch.uint8)
    tape[1:] = quantize(norms - z[torch.arange(n - 1), ids[1:]].double())
    return tape, len(shared), len(missing)


@dataclass(frozen=True, slots=True)
class Fields:
    """Transient same-weight single-window packet. No tape/activations to carry into another window."""

    base: torch.Tensor
    query: torch.Tensor
    candidate: torch.Tensor
    margin: torch.Tensor
    phi: dict[str, torch.Tensor]
    work: dict
    seconds: dict
    latest: torch.Tensor
    depth: torch.Tensor


def fields(head, word, hidden, logits):
    """Registered N/U/S comparison, including the wrong-mark intervention."""
    return _fields_and_tape(head, word, hidden, logits, controls=True)[0]


def bound_fields(head, word, hidden, logits):
    """U only: no sham search/read and no alternative response. All addresses are B's outputs."""
    return _fields_and_tape(head, word, hidden, logits, controls=False)[0]


def _rotation(word, query, latest, depth):
    donor = torch.tensor(
        [
            rotated_payload_index(word, t, Match(d, j, ()), HORIZON)
            for t, d, j in zip(
                query.tolist(), depth.tolist(), latest.tolist(), strict=True
            )
        ],
        dtype=torch.long,
    )
    if not ((donor > 0) & (donor <= query - 16)).all():
        raise ValueError("invalid rotated donor")
    return donor


@torch.inference_mode()
def _fields_and_tape(head, word, hidden, logits, *, controls):
    """Build all required inputs before the caller gets access to the loss/fit seam."""
    if (
        type(head) is not LiveValueHead
        or head.training
        or type(word) is not bytes
        or not 1 <= len(word) <= HORIZON
        or logits.device.type != "cpu"
        or hidden.device.type != "cpu"
        or logits.dtype != torch.float32
        or logits.shape != (len(word), 256)
    ):
        raise ValueError("expected one frozen aligned B packet on CPU")
    clock = {}
    work = {}
    routes = {}
    t0 = perf_counter()
    base = head(
        word, hidden, logits, mode="B", horizon=HORIZON, work=work, hit_out=routes
    )
    q, j, depth = (routes[k] for k in ("query", "latest", "depth"))
    if (
        base.dtype != torch.float32
        or not base.isfinite().all()
        or not ((j > 0) & (j <= q - 16)).all()
    ):
        raise ValueError("invalid B scores/bindings")
    clock["base_B_s"] = perf_counter() - t0
    t0 = perf_counter()
    ids = torch.tensor(list(word), dtype=torch.long)
    c = ids[j]
    v = _candidate_margin(base, q, c)
    # N uses no tape and pays normalization only on its actually queried host rows.
    norm = logits[q].double().logsumexp(-1)
    now = quantize(norm - logits[q, c].double())
    bcode = quantize(_softplus(-v))
    phi = torch.stack(
        (torch.ones_like(v), bcode.double() / 64, now.double() / 64), dim=1
    )
    clock["shared_current_s"] = perf_counter() - t0
    t0 = perf_counter()
    tape, reused, extra = _arrival_tape(logits, ids, q, norm)
    clock["shared_US_mark_build_s"] = perf_counter() - t0
    t0 = perf_counter()
    u = tape[j]
    up = torch.cat((phi, (u.double() / 64)[:, None]), dim=1)
    clock["U_mark_read_s"] = perf_counter() - t0
    features = {"U": up}
    if controls:
        t0 = perf_counter()
        donor = _rotation(word, q, j, depth)
        s = tape[donor]
        features.update(N=phi, S=torch.cat((phi, (s.double() / 64)[:, None]), dim=1))
        clock["S_rotation_and_mark_read_s"] = perf_counter() - t0
        if not torch.equal(ids[donor], c):
            raise ValueError("unmatched forecast mark")
        work.update(
            S_mark_reads=len(s),
            S_read_code_saturated=int((s == 255).sum()),
            S_slots_displaced=int((donor != j).sum()),
            S_codes_changed=int((s != u).sum()),
        )
    if reused + extra != len(ids) - 1:
        raise ValueError("lost arrival normalization")
    work.update(
        mark_storage_bytes=tape.nbytes,
        arrivals_encoded=len(ids) - 1,
        mark_norms_from_current=reused,
        mark_norms_extra=extra,
        marks_saturated=int((tape[1:] == 255).sum()),
        B_candidate_code_saturated=int((bcode == 255).sum()),
        now_candidate_code_saturated=int((now == 255).sum()),
        U_mark_reads=len(u),
        U_read_code_saturated=int((u == 255).sum()),
    )
    return Fields(base, q, c, v, features, work, clock, j, depth), tape


def checked_beta(arm, coefficients):
    return checked_vector(DIMS, arm, coefficients)


def checked_vector(dims, arm, coefficients):
    beta = torch.as_tensor(coefficients, dtype=torch.float64, device="cpu")
    if arm not in dims or beta.shape != (dims[arm],) or not beta.isfinite().all():
        raise ValueError("invalid fitted calibration coefficients")
    return beta


@torch.inference_mode()
def response(packet, arm, coefficients):
    """One complete 256-way FP64 answer at every frontier, with B's original log mass."""
    beta = checked_beta(arm, coefficients)
    if arm not in packet.phi:
        raise ValueError("readout fields not built")
    return copy_response(packet, packet.phi[arm], beta)


@torch.inference_mode()
def copy_response(packet, features, beta):
    """Shared mass-preserving change to the candidate log odds on B's actual hit rows."""
    q = packet.query
    v = packet.margin
    dl = features @ beta
    shift = _softplus(v + dl) - _softplus(v)
    out = packet.base.double()
    out[q] -= shift[:, None]
    out[q, packet.candidate] += dl
    if not out.isfinite().all():
        raise ValueError("invalid calibrated response")
    return out
