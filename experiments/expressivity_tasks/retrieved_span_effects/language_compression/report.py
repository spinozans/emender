"""Receipt-only scorer. Numerators are per-document sums in nats, not window means."""

import math
import numpy as np

from .settings import ARMS, SEEDS, N, need

CONTRASTS = (
    ("count", "local"),
    ("exact", "local"),
    ("indexed", "local"),
    ("indexed", "count"),
    ("indexed", "exact"),
)


def summarize(losses):
    docs = N["probe"]
    need(set(losses) == {f"{s}.{a}" for s in SEEDS for a in ARMS}, "incomplete fits")
    draws = np.random.default_rng(5750).multinomial(
        docs, np.full(docs, 1 / docs), size=10000
    )
    rates = {}
    means = {}
    prices = {}
    for a in ARMS:
        n = np.stack([losses[f"{s}.{a}"] for s in SEEDS])
        need(
            n.shape == (len(SEEDS), docs) and np.isfinite(n).all() and (n >= 0).all(),
            "bad document losses",
        )
        b = (n / math.log(2) + 8.0) / 4097.0
        rates[a] = b
        means[a] = b.mean().item()
        prices[a] = b.mean(1).tolist()
    comp = {}
    for a, b in CONTRASTS:
        d = rates[b] - rates[a]
        v = d.mean(0)
        lo, hi = np.quantile(draws @ v / docs, [0.025, 0.975])
        seeds = d.mean(1)
        comp[f"{a}:{b}"] = dict(
            gain=v.mean().item(),
            ci=[lo.item(), hi.item()],
            seeds=seeds.tolist(),
            positive=bool(lo > 0 and (seeds > 0).all()),
        )
    return dict(
        raw_bytes=docs * 4097,
        shifted_targets=docs * 4096,
        first_byte_bits=8 * docs,
        mean_bpb=means,
        per_seed_bpb=prices,
        contrasts=comp,
        primary_positive=comp["indexed:local"]["positive"],
    )
