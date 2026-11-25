"""X49-V one-step conditional-softmax fit and frozen evaluation. Keep U's copy score untouched."""

from __future__ import annotations
from time import perf_counter
from math import isfinite
import torch
from .alternative import DIMS, fields, response
from .endpoint import frozen_prefill
from .surprise_fit import RidgeMoments


def _outcomes(packet, targets):
    if type(targets) is not bytes or len(targets) != len(packet.base):
        raise ValueError("one real target per input")
    y = torch.tensor(list(targets), dtype=torch.long)
    hit_target = y[packet.query]
    copy = hit_target == packet.copy
    counts = dict(copy_correct=int(copy.sum()), hit_noncopy=int((~copy).sum()))
    for arm, blocks in packet.blocks.items():
        counts[arm + "_alternative_correct"] = int(
            (blocks[-1].byte == hit_target).sum()
        )
    return y, hit_target, ~copy, counts


class Moments(RidgeMoments):
    dims = DIMS
    common_arm = "D"

    @torch.inference_mode()
    def add(self, packet, targets):
        start = perf_counter()
        _, target, mask, counts = _outcomes(packet, targets)
        for arm, blocks in packet.blocks.items():
            for i, x in enumerate(blocks):
                ii = slice(i * 2, (i + 1) * 2)
                residual = mask * ((target == x.byte).double() - x.probability)
                self.rhs[arm][ii] += (x.feature.T @ residual).numpy()
                for j, y in enumerate(blocks):
                    jj = slice(j * 2, (j + 1) * 2)
                    # Cov(1[b=x],1[b=y]) = P(x) * (1[x=y] - P(y)), including coincident proposals.
                    cov = (
                        mask
                        * x.probability
                        * ((x.byte == y.byte).double() - y.probability)
                    )
                    self.gram[arm][ii, jj] += (
                        x.feature.T @ (cov[:, None] * y.feature)
                    ).numpy()
        self.documents += 1
        self.targets += len(targets)
        return counts, perf_counter() - start


def score(packet, coefficients, targets):
    if type(coefficients) is not dict or set(coefficients) != set(DIMS):
        raise ValueError("missing frozen V readouts")
    preds = {"U": packet.base}
    times = {}
    for arm in DIMS:
        t = perf_counter()
        preds[arm] = response(packet, arm, coefficients[arm])
        times["response_" + arm + "_s"] = perf_counter() - t
    # Only now expose the shifted targets; there is an answer for each possible byte including misses.
    y, _, _, counts = _outcomes(packet, targets)
    ix = torch.arange(len(y))
    sums = {a: float(-p[ix, y].sum()) for a, p in preds.items()}
    if not all(isfinite(v) and v >= 0 for v in sums.values()):
        raise ValueError("bad alternative NLL")
    return counts, sums, times


@torch.inference_mode()
def document(host, head, word, *, u_beta, fit=None, beta=None, width=256):
    if (
        type(word) is not bytes
        or len(word) != 4097
        or (fit is None) == (beta is None)
        or (fit is not None and type(fit) is not Moments)
    ):
        raise ValueError("one full shifted window and exactly one V phase required")
    inputs = word[:-1]
    h, z, sec = frozen_prefill(host, inputs, width)
    packet = fields(head, inputs, h, z, u_beta)
    result = dict(
        host_prefills=1,
        targets=len(inputs),
        work=packet.work,
        seconds=dict(prefill_s=sec, **packet.seconds),
    )
    if fit is not None:
        result["outcomes"], result["seconds"]["fit_statistics_s"] = fit.add(
            packet, word[1:]
        )
    else:
        result["outcomes"], result["nll_sums"], times = score(packet, beta, word[1:])
        result["seconds"].update(times)
    return result
