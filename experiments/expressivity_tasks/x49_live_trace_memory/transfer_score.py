"""Fixed JT scoring: one shared causal prefill, time just the actual mutually exclusive U/J branches."""

from __future__ import annotations
from math import isfinite
from time import perf_counter
import torch
from . import joint_reader as joint, surprise as u
from .endpoint import frozen_prefill
from .panels import SEEDS, T
from .surprise_fit import _labels


def first(seed, rank):
    if type(seed) is not int or seed not in SEEDS or type(rank) is not int or rank < 0:
        raise ValueError("unknown seed or document rank")
    return ("U", "J")[(SEEDS.index(seed) + rank) % 2]


@torch.inference_mode()
def answers(host, head, observed, beta, seed, rank, writer, *, width=256):
    if (
        type(observed) is not bytes
        or not 1 <= len(observed) < T
        or type(beta) is not dict
        or set(beta) != {"U", "H", "J"}
    ):
        raise ValueError("one short observed prefix, three fixed readouts required")
    order = first(seed, rank)
    tic = perf_counter()
    h, z, pre = frozen_prefill(host, observed, width)
    p = u.bound_fields(head, observed, h, z)
    secs = dict(shared_s=perf_counter() - tic, prefill_s=pre, **p.seconds)
    lp = {}
    work = dict(p.work)
    for arm in (order, "J" if order == "U" else "U"):
        tic = perf_counter()
        if arm == "U":
            lp[arm] = u.response(p, "U", beta["U"])
        else:
            lp[arm], w, ws = joint.branch(p, observed, beta["J"], writer)
            work.update(w)
            secs["text_build_s"] = ws
        secs[arm + "_branch_s"] = perf_counter() - tic
    tic = perf_counter()
    lp["H"] = u.response(p, "U", beta["H"])
    secs["H_control_s"] = perf_counter() - tic
    lp["B"] = p.base.double()
    return (
        lp,
        p,
        dict(
            targets=len(observed),
            host_prefills=1,
            first_branch=order,
            work=work,
            seconds=secs,
        ),
    )


@torch.inference_mode()
def document(host, head, word, beta, seed, rank, writer, *, width=256):
    if type(word) is not bytes or not 2 <= len(word) <= T:
        raise ValueError("one whole short document required")
    lp, p, row = answers(host, head, word[:-1], beta, seed, rank, writer, width=width)
    ys, correct = _labels(p, word[1:])
    idx = torch.arange(len(ys))
    loss = {a: float(-v[idx, ys].sum()) for a, v in lp.items()}
    if not all(isfinite(v) and v >= 0 for v in loss.values()):
        raise ValueError("invalid all-byte scores")
    return dict(**row, nll_sums=loss, candidate_correct=int(correct.sum()))
