"""ET: one same-weight J prefill; actually build and time each chosen history, separately from controls."""

from dataclasses import dataclass
from math import isfinite
from time import perf_counter
import torch
from . import (
    error_history as eh,
    error_history_index as ix,
    joint_reader as joint,
    surprise as u,
)
from .endpoint import frozen_prefill
from .panels import SEEDS, T
from .surprise_fit import _labels

BRANCHES = ("HISTORY", "ERROR")


def first(seed, rank):
    if type(seed) is not int or seed not in SEEDS or type(rank) is not int or rank < 0:
        raise ValueError("unknown ET seed or reserved rank")
    return BRANCHES[(SEEDS.index(seed) + rank) % 2]


@dataclass(frozen=True, slots=True)
class _Shared:
    observed: bytes
    z: torch.Tensor
    packet: u.Fields
    lp: torch.Tensor
    literal: torch.Tensor
    neural: torch.Tensor


@torch.inference_mode()
def _shared(head, obs, h, z, jb, writer):
    p, nt = u._fields_and_tape(head, obs, h, z, controls=False)
    tic = perf_counter()
    lt, tw = writer(obs)
    text_s = perf_counter() - tic
    tic = perf_counter()
    lp = joint._from_tape(p, lt, jb)
    resp_s = perf_counter() - tic
    return (
        _Shared(obs, z, p, lp, lt, nt),
        dict(p.work, **tw),
        dict(p.seconds, text_build_s=text_s, J_response_s=resp_s),
    )


def _features(base, tab, other, w):
    active, row, sym, x, prob = eh._columns(base.lp, tab, other, w)
    return eh.Packet(
        base.lp,
        {},
        base.packet.query,
        base.packet.candidate,
        active,
        row,
        sym,
        x,
        prob,
        w,
        {},
    )


def _tapes(base, arm):
    g = ix.guesses(base.z)
    n = base.neural.numpy().tobytes()
    lcode = base.literal.numpy().tobytes()
    if arm == "HISTORY":
        # Compare only after arrival. Keep flags, no guess identities in this index.
        tape = b"\0" + bytes(
            a != c for a, c in zip(g[1:], base.observed[1:], strict=True)
        )
        m = sum(tape)
    elif arm == "ERROR":
        tape = g
        m = sum(a != c for a, c in zip(g[1:], base.observed[1:], strict=True))
    else:
        raise ValueError("unknown chosen history")
    return (
        lcode,
        n,
        tape,
        dict(
            top1_written=len(g) - 1,
            wrong_top1=m,
            top1_bytes=len(g),
            miss_flag_bytes=len(tape) if arm == "HISTORY" else 0,
        ),
    )


@torch.inference_mode()
def _chosen(base, arm, beta):
    # No shared history index/guesses and no other coefficients, diagnostic donor or target.
    tic = perf_counter()
    lit, nt, tape, w = _tapes(base, arm)
    s = dict(prep_s=perf_counter() - tic)
    tic = perf_counter()
    a = (ix.history_counts if arm == "HISTORY" else ix.true_counts)(
        base.observed, lit, nt, tape
    )
    eh._geometry(a, base.packet)
    w.update(a.work)
    s["pool_s"] = perf_counter() - tic
    tic = perf_counter()
    p = _features(base, a, None, w)
    s["feature_s"] = perf_counter() - tic
    tic = perf_counter()
    lp = eh.response(p, arm, beta)
    s["response_s"] = perf_counter() - tic
    return lp, w, s


@torch.inference_mode()
def _controls(base, beta):
    # Rebuild true and displaced snapshots as research cost, not part of either chosen path.
    lit, nt, g, w = _tapes(base, "ERROR")
    tab = ix.true_counts(base.observed, lit, nt, g)
    eh._geometry(tab, base.packet)
    other, sw = ix.displaced_counts(base.observed, nt, g)
    w.update({"pool_" + k: v for k, v in tab.work.items()})
    w.update({"shift_" + k: v for k, v in (other.work | sw).items()})
    p = _features(base, tab, other, w)
    lp = dict(
        J=base.lp,
        B=base.packet.base.double(),
        U=u.response(base.packet, "U", beta["U"]),
    )
    for a in ("COUNT", "SHIFT", "SWAP"):
        lp[a] = eh.response(p, a, beta["ERROR" if a == "SWAP" else a])
    return lp, w


def _validate(obs, beta):
    if (
        type(obs) is not bytes
        or not 1 <= len(obs) < T
        or type(beta) is not dict
        or set(beta) != {"U", "J", *eh.DIMS}
    ):
        raise ValueError("need one short observed prefix and six frozen ET readouts")


@torch.inference_mode()
def _answers(head, obs, h, z, beta, seed, rank, writer, *, pre_s, tic):
    order = first(seed, rank)
    base, w, s = _shared(head, obs, h, z, beta["J"], writer)
    s.update(shared_J_s=perf_counter() - tic, prefill_s=pre_s)
    chosen = {}
    counts = {}
    for arm in BRANCHES if order == "HISTORY" else reversed(BRANCHES):
        tic = perf_counter()
        chosen[arm], counts[arm], clock = _chosen(base, arm, beta[arm])
        s[arm + "_branch_s"] = perf_counter() - tic
        s.update({arm + "_" + k: v for k, v in clock.items()})
    tic = perf_counter()
    lp, work = _controls(base, beta)
    lp.update(chosen)
    s["comparison_only_s"] = perf_counter() - tic
    e = counts["ERROR"]
    hwork = counts["HISTORY"]
    common_counts = (
        "hits",
        "active_rows",
        "key_admissions",
        "committed_records",
        "peak_keys",
        "older_occurrence_uses",
        "older_mistake_uses",
        "top1_written",
        "wrong_top1",
    )
    if (
        any(hwork[k] != e[k] for k in common_counts)
        or any(
            e[k.removeprefix("pool_")] != v
            for k, v in work.items()
            if k.startswith("pool_")
        )
        or work["wrong_top1"] != e["wrong_top1"]
        or hwork["peak_cells"] > e["peak_cells"]
    ):
        raise ValueError("different actual exposure across ET branch clocks")
    return (
        lp,
        base.packet,
        dict(
            targets=len(obs),
            host_prefills=1,
            first_branch=order,
            work=dict(w, **work),
            chosen_work=counts,
            seconds=s,
        ),
    )


@torch.inference_mode()
def answers(host, head, observed, beta, seed, rank, writer, *, width=256):
    _validate(observed, beta)
    first(seed, rank)
    tic = perf_counter()
    h, z, pre = frozen_prefill(host, observed, width)
    return _answers(head, observed, h, z, beta, seed, rank, writer, pre_s=pre, tic=tic)


@torch.inference_mode()
def predict(host, head, observed, jb, arm, coefficients, writer, *, width=256):
    """A single frozen alternative from raw inputs, with no comparison-only arguments."""
    if type(observed) is not bytes or not 1 <= len(observed) < T or arm not in BRANCHES:
        raise ValueError("expected a short input and one chosen history")
    h, z, _ = frozen_prefill(host, observed, width)
    base, _, _ = _shared(head, observed, h, z, jb, writer)
    return _chosen(base, arm, coefficients)[0]


@torch.inference_mode()
def document(host, head, word, beta, seed, rank, writer, *, width=256):
    if type(word) is not bytes or not 2 <= len(word) <= T:
        raise ValueError("one whole short shifted document required")
    lp, p, row = answers(host, head, word[:-1], beta, seed, rank, writer, width=width)
    ys, right = _labels(p, word[1:])
    idx = torch.arange(len(ys))
    sums = {a: float(-lp[a][idx, ys].sum()) for a in eh.ARMS}
    if not all(isfinite(v) and v >= 0 for v in sums.values()):
        raise ValueError("invalid ET loss sum")
    return dict(row, nll_sums=sums, candidate_correct=int(right.sum()))
