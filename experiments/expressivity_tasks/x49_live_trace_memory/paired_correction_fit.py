"""PC: one FIT-only Fisher step at the fixed ERROR distribution and all-target PROBE loss."""

from math import isfinite
from time import perf_counter
import torch
from .paired_correction import ANCHORS, ARMS, DIMS, fields, response
from .surprise_fit import SparseMoments, _labels
from .endpoint import frozen_prefill


class Moments(SparseMoments):
    dims = DIMS
    common_arm = "MARG"


@torch.inference_mode()
def score(packet, beta, targets):
    if (
        type(beta) is not dict
        or set(beta) != set(DIMS)
        or set(packet.anchors) != set(ANCHORS)
    ):
        raise ValueError("need fixed ERROR anchors and the three frozen PC readouts")
    lp = dict(packet.anchors)
    clock = {}
    for a in (*DIMS, "SWAP"):
        tic = perf_counter()
        lp[a] = response(packet, a, beta["PAIR" if a == "SWAP" else a])
        clock[a + "_response_s"] = perf_counter() - tic
    # No labels or label-dependent requests until every 256-way answer is available.
    y, right = _labels(packet, targets)
    at = torch.arange(len(y))
    loss = {k: -float(lp[k][at, y].sum()) for k in ARMS}
    if not all(isfinite(v) and v >= 0 for v in loss.values()):
        raise ValueError("invalid PC PROBE loss sum")
    return int(right.sum()), loss, clock


@torch.inference_mode()
def document(
    host, head, word, *, ub, jb, hb, eb, writer, fit=None, beta=None, width=256
):
    if (
        type(word) is not bytes
        or len(word) != 4097
        or (fit is None) == (beta is None)
        or (fit is not None and type(fit) is not Moments)
    ):
        raise ValueError(
            "one fully shifted PC window and exactly one FIT or PROBE phase"
        )
    h, z, pre = frozen_prefill(host, word[:-1], width)
    p = fields(
        head,
        word[:-1],
        h,
        z,
        jb,
        eb,
        writer,
        mode="panel",
        anchors=(ub, hb) if fit is None else None,
    )
    seconds = dict(prefill_s=pre, **p.seconds)
    row = dict(targets=len(word) - 1, host_prefills=1, work=p.work, seconds=seconds)
    if fit is not None:
        row["candidate_correct"], seconds["fit_statistics_s"] = fit.add(p, word[1:])
    else:
        row["candidate_correct"], row["nll_sums"], clock = score(p, beta, word[1:])
        seconds.update(clock)
    return row
