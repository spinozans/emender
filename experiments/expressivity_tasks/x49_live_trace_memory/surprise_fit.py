"""Registered one-step fit and held-out response scoring for X49-U. No persistent document state."""

from __future__ import annotations
from time import perf_counter
import numpy as np
import torch
from .endpoint import frozen_prefill
from .surprise import DIMS, fields, response

RIDGE = 1 / 4096


def _labels(packet, targets):
    if type(targets) is not bytes or len(targets) != len(packet.base):
        raise ValueError("one real shifted target per frontier required")
    ys = torch.tensor(list(targets), dtype=torch.long)
    return ys, ys[packet.query] == packet.candidate


class Moments:
    """Within a seed, only 3 small Gram matrices, 3 vectors and exposure counts survive each FIT document."""

    def __init__(self):
        self.documents = 0
        self.targets = 0
        self.gram = {k: np.zeros((d, d), np.float64) for k, d in DIMS.items()}
        self.rhs = {k: np.zeros(d, np.float64) for k, d in DIMS.items()}

    def add(self, packet, targets):
        start = perf_counter()
        _, correct = _labels(packet, targets)
        v = packet.margin
        pos = torch.sigmoid(v)
        neg = torch.sigmoid(-v)
        res = torch.where(correct, neg, -pos)
        curv = pos * neg
        for k, phi in packet.phi.items():
            self.rhs[k] += (phi.T @ res).numpy()
            self.gram[k] += (phi.T @ (curv[:, None] * phi)).numpy()
        self.documents += 1
        self.targets += len(targets)
        return dict(
            candidate_correct=int(correct.sum()),
            fit_statistics_s=perf_counter() - start,
        )

    def freeze(self):
        if self.targets <= 0 or self.documents <= 0:
            raise ValueError("no fit population")
        beta = {}
        for k, d in DIMS.items():
            m = self.gram[k] / self.targets
            r = self.rhs[k] / self.targets
            if not (
                np.isfinite(m).all()
                and np.isfinite(r).all()
                and np.allclose(m, m.T, atol=1e-13, rtol=1e-13)
            ):
                raise ValueError("invalid score moments")
            if not (
                np.allclose(
                    m[:3, :3], self.gram["N"] / self.targets, atol=1e-13, rtol=1e-13
                )
                and np.allclose(
                    r[:3], self.rhs["N"] / self.targets, atol=1e-13, rtol=1e-13
                )
            ):
                raise ValueError("mismatched current features or fit exposures")
            system = (m + m.T) * 0.5 + RIDGE * np.eye(d)
            if np.linalg.eigvalsh(system)[0] <= 0:
                raise ValueError("invalid regularized system")
            b = np.linalg.solve(system, r)
            if not (
                np.isfinite(b).all()
                and np.allclose(system @ b, r, atol=1e-13, rtol=1e-12)
            ):
                raise ValueError("incorrect Fisher solve")
            beta[k] = b.tolist()
        return dict(
            documents=self.documents,
            targets=self.targets,
            ridge=RIDGE,
            gram={k: x.tolist() for k, x in self.gram.items()},
            rhs={k: x.tolist() for k, x in self.rhs.items()},
            beta=beta,
        )


def score(packet, coefficients, targets):
    if type(coefficients) is not dict or set(coefficients) != set(DIMS):
        raise ValueError("need all frozen readouts")
    preds = {"B": packet.base.double()}
    times = {}
    for k in DIMS:
        tic = perf_counter()
        preds[k] = response(packet, k, coefficients[k])
        times["response_" + k + "_s"] = perf_counter() - tic
    # Gather losses ONLY after all alternatives have returned all 256 possibilities.
    y, correct = _labels(packet, targets)
    idx = torch.arange(len(y))
    sums = {a: float(-lp[idx, y].sum()) for a, lp in preds.items()}
    if not all(np.isfinite(x) and x >= 0 for x in sums.values()):
        raise ValueError("nonfinite all-byte loss")
    return dict(candidate_correct=int(correct.sum()), nll_sums=sums, seconds=times)


@torch.inference_mode()
def document(host, head, word, *, fit=None, beta=None, width=256):
    if (fit is None) == (beta is None) or (
        fit is not None and not isinstance(fit, Moments)
    ):
        raise ValueError("choose exactly one of FIT or frozen PROBE")
    if type(word) is not bytes or len(word) != 4097:
        raise ValueError("need one fully shifted document window")
    inputs = word[:-1]
    h, z, prefill = frozen_prefill(host, inputs, width)
    p = fields(head, inputs, h, z)
    common = dict(
        host_prefills=1,
        targets=len(inputs),
        work=p.work,
        seconds=dict(prefill_s=prefill, **p.seconds),
    )
    if fit is not None:
        r = fit.add(p, word[1:])
        common["candidate_correct"] = r["candidate_correct"]
        common["seconds"]["fit_statistics_s"] = r["fit_statistics_s"]
    else:
        r = score(p, beta, word[1:])
        common.update(candidate_correct=r["candidate_correct"], nll_sums=r["nll_sums"])
        common["seconds"].update(r["seconds"])
    return common
