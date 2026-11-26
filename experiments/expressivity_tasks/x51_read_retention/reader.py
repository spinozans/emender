"""Same read/position count for KEEP and DISCARD; remove only older answer content."""

import math
import numpy as np
import torch
from torch.nn import functional as F

from experiments.expressivity_tasks.x50_read_tape.program import T, D, log_prior

ARMS = ("NONE", "DISCARD", "KEEP")


def unigram_counts(fit):
    if fit.ndim != 2 or fit.dtype != np.uint8 or fit.shape[1] != T + 1 or not len(fit):
        raise ValueError("need genuine shifted FIT windows")
    return np.bincount(fit[:, 1:].ravel(), minlength=256)


def unigram_log(counts):
    n = np.asarray(counts)
    if n.shape != (256,) or n.dtype.kind not in "iu" or n.min() < 0 or n.sum() <= 0:
        raise ValueError("invalid FIT count table")
    return torch.from_numpy(((n + 0.5) / (n.sum() + 128)).astype(np.float32)).log()


def acquire(bank, inputs, arm):
    if (
        arm not in ARMS
        or inputs.ndim != 2
        or inputs.dtype != np.uint8
        or not 1 <= inputs.shape[1] <= T
        or not len(inputs)
    ):
        raise ValueError("unknown arm or invalid inputs")
    if (bank is None) != (arm == "NONE"):
        raise ValueError(
            "NONE must not receive an archive capability; reading arms require one"
        )
    x = torch.from_numpy(inputs.astype(np.int64))
    return x, None if bank is None else torch.from_numpy(bank.scan(inputs).pdf)


def predict(model, x, pdf, arm, unigram):
    """No target input. All arms have two slots/frontier (OBS, ANSWER).

    Observation and older-row K/V projections are noncontextual: a prior answer
    cannot leak indirectly through an observation. At t the query sees history
    through OBS[t] plus its own current row; the latter is appended exactly once.
    """
    if arm not in ARMS or (pdf is None) != (arm == "NONE"):
        raise ValueError("wrong evidence for this arm")
    times = torch.arange(x.shape[1])
    obs = model.row(0, x, times)
    shell = model.row(1, x, times)
    if pdf is None:
        current = shell
        prior = unigram.expand(*x.shape, 256)
    else:
        if pdf.shape != (*x.shape, 256):
            raise ValueError("misaligned answer")
        current = shell + model.answer(pdf.sqrt())
        prior = log_prior(pdf)
    past = current if arm == "KEEP" else shell
    history = torch.stack((obs, past), dim=2).flatten(1, 2)
    q = model.q(current)
    k, v = model.k(history), model.v(history)
    scores = (q @ k.transpose(1, 2)) / math.sqrt(D)
    scores = scores.masked_fill(
        torch.arange(2 * len(times))[None, :] > 2 * times[:, None], -torch.inf
    )
    self_score = (q * model.k(current)).sum(dim=-1, keepdim=True) / math.sqrt(D)
    weights = torch.cat((scores, self_score), dim=-1).softmax(-1)
    attended = weights[:, :, :-1] @ v + weights[:, :, -1:] * model.v(current)
    return model.emit(model.norm(current + attended)) + prior, prior


def nll(logits, y):
    if logits.shape != (*y.shape, 256):
        raise ValueError("misaligned targets")
    return F.cross_entropy(logits.transpose(1, 2), y, reduction="none")
