"""Full, absent, or genuinely sparse archive reads. No gap lookup and no stale logit prior."""

from dataclasses import dataclass
import math

import numpy as np
import torch

from experiments.expressivity_tasks.x50_read_tape.kernel import DEPTH
from experiments.expressivity_tasks.x50_read_tape.program import D, T, log_prior
from experiments.expressivity_tasks.x51_read_retention.reader import (
    nll,
    unigram_counts,
    unigram_log,
)

ARMS = ("NONE", "EVERY", "DISCARD4", "KEEP4")
STRIDE = 4
__all__ = [
    "ARMS",
    "T",
    "STRIDE",
    "Packet",
    "schedule",
    "acquire",
    "predict",
    "nll",
    "unigram_counts",
    "unigram_log",
]


def schedule(arm, steps):
    if arm not in ARMS or not 1 <= steps <= T:
        raise ValueError("invalid arm or window length")
    return range(0, 0 if arm == "NONE" else steps, 1 if arm == "EVERY" else STRIDE)


@dataclass(frozen=True)
class Packet:
    x: torch.Tensor
    at: tuple[int, ...]
    pdf: (
        torch.Tensor | None
    )  # compact: only the queried positions, no placeholder distributions


def acquire(bank, inputs, arm):
    if inputs.dtype != np.uint8 or inputs.ndim != 2 or not len(inputs):
        raise ValueError("expected observed bytes [document,time]")
    at = tuple(schedule(arm, inputs.shape[1]))
    if (bank is None) != (arm == "NONE"):
        raise ValueError("wrong source capability")
    x = torch.from_numpy(inputs.astype(np.int64))
    if bank is None:
        return Packet(x, at, None)
    if arm == "EVERY":
        p = bank.scan(inputs).pdf
    else:
        # Only real suffixes ending at a refresh are sent to the native service.
        # Never run the dense prefix scanner and then discard three answers.
        keys = [
            row[max(0, t + 1 - DEPTH) : t + 1].tobytes() for row in inputs for t in at
        ]
        p = bank.read(keys).pdf.reshape(len(x), len(at), 256)
    return Packet(x, at, torch.from_numpy(p))


def predict(m, packet, arm, unigram):
    x, at, p = packet.x, packet.at, packet.pdf
    b, steps = x.shape
    if (
        at != tuple(schedule(arm, steps))
        or (p is None) != (arm == "NONE")
        or (p is not None and p.shape != (b, len(at), 256))
        or unigram.shape != (256,)
    ):
        raise ValueError("misaligned read packet or prior")
    times = torch.arange(steps)
    obs = m.row(0, x, times)
    shell = m.row(1, x, times)
    current = shell
    prior = unigram.expand(b, steps, 256)
    if p is not None:
        ix = torch.tensor(at, dtype=torch.long)
        current = shell.index_add(1, ix, m.answer(p.sqrt()))
        prior = prior.index_copy(1, ix, log_prior(p))
    past = current if arm in ("EVERY", "KEEP4") else shell
    # Noncontextual K/V: discarded earlier answers cannot survive in observation rows.
    hist = torch.stack((obs, past), dim=2).flatten(1, 2)
    q = m.q(current)
    scores = (q @ m.k(hist).transpose(1, 2)) / math.sqrt(D)
    scores = scores.masked_fill(
        torch.arange(2 * steps)[None, :] > 2 * times[:, None], -torch.inf
    )
    own = (q * m.k(current)).sum(-1, keepdim=True) / math.sqrt(D)
    w = torch.cat((scores, own), dim=-1).softmax(-1)
    context = w[:, :, :-1] @ m.v(hist) + w[:, :, -1:] * m.v(current)
    return m.emit(m.norm(current + context)) + prior, prior
