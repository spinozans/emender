"""F/C: private categorical keys address actual earlier neural writes in one live window.

There is exactly one private query per frontier and one filing per j>=1. A tag is a
one-byte string in a separate namespace, not a new observed byte. No key labels in the loss.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import log

import torch
from torch import nn
from torch.nn import functional as F
from .literal import HORIZONS, MIN_AGE, check_window
from .panels import SEEDS
from .values import LiveValueHead


def lookup(filed: bytes, queried: bytes, horizon: int) -> tuple[int | None, ...]:
    """Exact one-symbol string lookup, latest eligible record only. Slot 0 holds logical j=1."""
    check_window(queried, horizon)
    if type(filed) is not bytes or len(filed) + 1 != len(queried):
        raise ValueError("one private write per noninitial observation")
    recent = {}
    rows = []
    for t, key in enumerate(queried):
        j = t - MIN_AGE
        if j >= 1:
            recent[filed[j - 1]] = j
        hit = recent.get(key)
        rows.append(hit if hit is not None and hit >= max(1, t + 1 - horizon) else None)
    return tuple(rows)


def rotate_trace(word, t, j, h):
    width = 1 << ((t - j).bit_length() - 1)
    lo = max(1, t + 1 - h, t - 2 * width + 1)
    hi = t - max(MIN_AGE, width)
    if not lo <= j <= hi:
        raise ValueError("ineligible trace donor")
    code = word[j : j + 1]
    k = word.find(code, j + 1, hi + 1)
    return k if k >= 0 else word.index(code, lo, j + 1)


def choices(labels, hidden, correction, *, seed, document, visit, role):
    if (
        seed not in SEEDS
        or any(type(n) is not int or n < 0 for n in (document, visit))
        or role not in ("file", "query")
    ):
        raise ValueError("bad draw identity")
    # One independent, reproducible stream per (model seed, logical doc, visit, role).
    # Neither lengths nor a hash of future document CONTENTS enter the RNG seed.
    seed64 = int.from_bytes(
        hashlib.sha256(
            f"x49-private-v1:{seed}:{document}:{visit}:{role}".encode()
        ).digest()[:8],
        "big",
    )
    if labels.device.type != "cpu" or hidden.device.type != "cpu":
        raise ValueError("reference sampler is CPU resident")
    rng = torch.Generator(device="cpu").manual_seed(seed64)
    u = torch.rand((len(labels), 1), dtype=torch.float64, generator=rng)
    prior = F.one_hot(labels, 256).float() * log(1 + 9 * 256)
    lp = (prior if correction is None else prior + correction(hidden)).log_softmax(-1)
    if not torch.isfinite(lp).all():
        raise ValueError("nonfinite private key distribution")
    # Score the chosen category; the native/string index never receives a table of alternative requests.
    with torch.no_grad():
        cdf = lp.double().exp().cumsum(-1)
        cdf /= cdf[:, -1:].clone()
        idx = torch.searchsorted(cdf.contiguous(), u, right=True)
    return bytes(idx[:, 0].tolist()), lp.gather(1, idx)[:, 0]


@dataclass
class Rollout:
    lp: torch.Tensor
    file_lp: torch.Tensor
    query_lp: torch.Tensor
    filed: bytes
    queried: bytes
    source: tuple
    horizon: int
    rotated_count: int


class TraceHead(nn.Module):
    def __init__(self, literal: LiveValueHead, *, learned: bool):
        super().__init__()
        self.literal = literal
        self.learned = learned
        d = literal.direct.host_dim
        self.filing = nn.Linear(d, 256)
        self.ask = nn.Linear(d, 256)
        for mod in (self.filing, self.ask):
            nn.init.zeros_(mod.weight)
            nn.init.zeros_(mod.bias)
            mod.requires_grad_(learned)
        self.value = nn.Linear(d + 16, 32)
        self.mix = nn.Linear(d + 16 + 32, 64)
        self.delta = nn.Linear(64, 256)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(
        self,
        word,
        hidden,
        logits,
        *,
        seed,
        document,
        visit,
        horizon=4096,
        wrong_binding=False,
    ):
        # Both arms retain B's full literal answer. No gold addresses or external targets accepted.
        lp = self.literal(word, hidden, logits, mode="B", horizon=horizon)
        ids = torch.tensor(list(word), dtype=torch.long, device=hidden.device)
        tag, wl = choices(
            ids[:-1],
            hidden[1:],
            self.filing if self.learned else None,
            seed=seed,
            document=document,
            visit=visit,
            role="file",
        )
        query, ql = choices(
            ids,
            hidden,
            self.ask if self.learned else None,
            seed=seed,
            document=document,
            visit=visit,
            role="query",
        )
        rows = lookup(tag, query, horizon)
        bank = self.value(torch.cat((hidden[1:], self.literal.byte(ids[1:])), 1)).tanh()
        at = [t for t, j in enumerate(rows) if j is not None]
        changed = 0
        if at:
            q = torch.tensor(at, dtype=torch.long)
            src = [rows[t] for t in at]
            donors = (
                [
                    rotate_trace(word, t, j, horizon)
                    for t, j in zip(at, src, strict=True)
                ]
                if wrong_binding
                else src
            )
            changed = sum(j != k for j, k in zip(src, donors, strict=True))
            idx = torch.tensor(src, dtype=torch.long)
            val = bank[torch.tensor(donors, dtype=torch.long) - 1]
            code = self.literal.byte(ids[idx])
            old = lp[q]
            corr = self.delta(self.mix(torch.cat([hidden[q], code, val], 1)).tanh())
            dz = (old + corr).logsumexp(1, keepdim=True) - old.logsumexp(
                1, keepdim=True
            )
            lp = lp.index_copy(0, q, old + (corr - dz))
        return Rollout(lp, wl, ql, tag, query, rows, horizon, changed)


def objective(result: Rollout, targets, *, learned):
    """Unbiased sampled-loss gradients for the registered fixed-contact schedule.

    Here query/file distributions depend only on the causal host/observed bytes, not
    the private answer. Query can affect this byte; file j can affect [j+16,j+H).
    Truncation outside that interval is exact. Do NOT condition filing credit on hits.
    No later target enters the predictor or the sampler. Only the external byte is scored.
    """
    n = len(result.lp)
    if (
        targets.shape != (n,)
        or targets.dtype != torch.long
        or result.horizon not in HORIZONS
    ):
        raise ValueError("unaligned external targets")
    each = F.nll_loss(result.lp, targets, reduction="none")
    nll = each.mean()
    if not learned:
        return nll, nll
    advantage = each.detach().double() - 2.0
    cs = F.pad(advantage.cumsum(0), (1, 0))
    js = torch.arange(1, n)
    returns = (
        cs[(js + result.horizon).clamp_max(n)] - cs[(js + MIN_AGE).clamp_max(n)]
    ).float()
    score = (
        (result.query_lp * advantage.float()).sum() + (result.file_lp * returns).sum()
    ) / n
    return nll + (score - score.detach()), nll
