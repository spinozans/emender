"""One multiclass Fisher step at frozen J. No update/label in the stream index or read path."""

from __future__ import annotations
from time import perf_counter
from math import isfinite
import numpy as np
import torch
from .error_history import ARMS, DIMS, fields, response
from .surprise_fit import SparseMoments, _labels
from .endpoint import frozen_prefill


class Moments(SparseMoments):
    dims = DIMS
    common_arm = "COUNT"

    def freeze(self):
        z = super().freeze()
        d = DIMS["HISTORY"]
        n = self.targets
        for k in ("ERROR", "SHIFT"):
            if not (
                np.allclose(
                    self.gram[k][:d, :d] / n,
                    self.gram["HISTORY"] / n,
                    rtol=1e-13,
                    atol=1e-13,
                )
                and np.allclose(
                    self.rhs[k][:d] / n, self.rhs["HISTORY"] / n, rtol=1e-13, atol=1e-13
                )
            ):
                raise ValueError("different HISTORY features or exposure in EH fits")
        return z


@torch.inference_mode()
def score(packet, beta, targets):
    if (
        type(beta) is not dict
        or set(beta) != set(DIMS)
        or set(packet.anchors) != set(("B", "U", "J"))
    ):
        raise ValueError("need all EH anchors and four frozen fits")
    lp = dict(packet.anchors)
    times = {}
    for a in (*DIMS, "SWAP"):
        tic = perf_counter()
        lp[a] = response(packet, a, beta["ERROR" if a == "SWAP" else a])
        times[a + "_response_s"] = perf_counter() - tic
    y, right = _labels(packet, targets)
    idx = torch.arange(len(y))
    z = {a: float(-lp[a][idx, y].sum()) for a in ARMS}
    if not all(isfinite(v) and v >= 0 for v in z.values()):
        raise ValueError("wrong EH loss sum")
    return int(right.sum()), z, times


@torch.inference_mode()
def document(host, head, word, *, ub, jb, writer, fit=None, beta=None, width=256):
    if (
        type(word) is not bytes
        or len(word) != 4097
        or (fit is None) == (beta is None)
        or (fit is not None and type(fit) is not Moments)
    ):
        raise ValueError("one shifted window and exactly one EH phase")
    h, z, sec = frozen_prefill(host, word[:-1], width)
    p = fields(head, word[:-1], h, z, jb, writer, ub=ub)
    times = dict(prefill_s=sec, **p.seconds)
    row = dict(targets=len(word) - 1, host_prefills=1, work=p.work, seconds=times)
    if fit is not None:
        row["candidate_correct"], times["fit_statistics_s"] = fit.add(p, word[1:])
    else:
        row["candidate_correct"], row["nll_sums"], clock = score(p, beta, word[1:])
        times.update(clock)
    return row
