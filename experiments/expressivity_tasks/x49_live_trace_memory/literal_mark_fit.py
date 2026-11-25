"""LS fits matched copy-odds models N/T/H/J/S. U is scored with the original coefficients only."""

from __future__ import annotations
from time import perf_counter
from math import isfinite
import numpy as np
import torch
from . import surprise_fit as old
from .endpoint import frozen_prefill
from .literal_mark import DIMS, fields, response
from .surprise import checked_beta


class Moments(old.Moments):
    dims = DIMS
    common_arm = "N"

    def add(self, packet, targets):
        if set(packet.phi) != set(self.dims):
            raise ValueError("need exactly five LS designs")
        return super().add(packet, targets)

    def freeze(self):
        state = super().freeze()
        for arm in ("J", "S"):
            if not (
                np.allclose(
                    self.gram[arm][:4, :4] / self.targets,
                    self.gram["T"] / self.targets,
                    rtol=1e-13,
                    atol=1e-13,
                )
                and np.allclose(
                    self.rhs[arm][:4] / self.targets,
                    self.rhs["T"] / self.targets,
                    rtol=1e-13,
                    atol=1e-13,
                )
            ):
                raise ValueError(
                    "joint predictor lost the correctly bound literal field"
                )
        return state


def score(packet, beta, u_beta, targets):
    if type(beta) is not dict or set(beta) != set(DIMS):
        raise ValueError("need five frozen LS readouts")
    answers = {}
    seconds = {}
    for arm in (*DIMS, "U"):
        tic = perf_counter()
        answers[arm] = response(packet, arm, u_beta if arm == "U" else beta[arm])
        seconds["response_" + arm + "_s"] = perf_counter() - tic
    y, correct = old._labels(packet, targets)
    idx = torch.arange(len(y))
    sums = {a: float(-lp[idx, y].sum()) for a, lp in answers.items()}
    if not all(isfinite(v) and v >= 0 for v in sums.values()):
        raise ValueError("invalid LS scores")
    return dict(candidate_correct=int(correct.sum()), nll_sums=sums, seconds=seconds)


@torch.inference_mode()
def document(host, head, word, *, u_beta, fit=None, beta=None, width=256):
    if (
        type(word) is not bytes
        or len(word) != 4097
        or (fit is None) == (beta is None)
        or (fit is not None and type(fit) is not Moments)
    ):
        raise ValueError("one full shifted document and exactly one LS phase required")
    u_beta = checked_beta("U", u_beta)
    inp = word[:-1]
    h, z, sec = frozen_prefill(host, inp, width)
    p = fields(head, inp, h, z)
    out = dict(
        host_prefills=1,
        targets=len(inp),
        work=p.work,
        seconds=dict(prefill_s=sec, **p.seconds),
    )
    if fit is not None:
        r = fit.add(p, word[1:])
        out["seconds"]["fit_statistics_s"] = r["fit_statistics_s"]
    else:
        r = score(p, beta, u_beta, word[1:])
        out["nll_sums"] = r["nll_sums"]
        out["seconds"].update(r["seconds"])
    out["candidate_correct"] = r["candidate_correct"]
    return out
