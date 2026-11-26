"""Deterministic invented CPU comparison against exhaustive attention, not a corpus study."""

import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from .attention import (
    Local,
    exhaustive,
    gaussian_features,
    indexed,
    log_features,
    prepare,
)
from .index import EndpointIndex
from .plan import Layout, Plan


@torch.no_grad()
def run():
    if (
        os.getenv("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("GPU_LEASE_HELD")
        or not __debug__
        or any(
            os.getenv(n) != "1"
            for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        )
    ):
        raise RuntimeError("explicit one-thread CPU only")
    torch.set_num_threads(1)
    dtype = torch.float64
    rng = np.random.default_rng(5607)
    B = 40
    D = 12
    V = 53
    docs = []
    for i in range(64):
        a = rng.integers(2, 48, size=65)
        a[6::8] = 2 + i % 2
        a[7::8] = 1
        docs.append((str(i), a.tolist()))
    index = EndpointIndex(docs, tokenizer="invented-53")
    n = len(index)
    layout = Layout(index.version, n, 32)
    emb = torch.tensor(rng.normal(size=(V, D)), dtype=dtype)
    emb[50] = emb[1]  # semantic twin without lexical matches

    def encode(seq):
        z = emb[seq]
        p = F.pad(z[:-1], (0, 0, 1, 0))
        return F.normalize(z + 0.25 * p, dim=-1) * 2.8

    k = torch.cat([encode(d)[:-1] for _, d in docs])[torch.tensor(index.order)]
    v = F.one_hot(torch.tensor(index.successor), V).to(dtype)
    b = k.new_full((n,), -math.log(n))
    omega = gaussian_features(D, 32, 5608, dtype=dtype)
    bank = prepare(layout, k, v, b, omega)
    prefix = []
    q = []
    lk = []
    lv = []
    lb = []
    ln = []
    for i in range(B):
        p = rng.integers(2, 48, size=3 + i % 8).tolist()
        p[-1] = [1, 50, 4, 2, 1][i % 5]
        z = encode(p)
        prefix.append(p)
        q.append(z[-1])
        lk.append(z[:-1])
        lv.append(F.one_hot(torch.tensor(p[1:]), V).to(dtype))
        ln.append(len(p) - 1)
        lb.append(k.new_full((len(p) - 1,), -math.log(len(p) - 1)))
    q = torch.stack(q)
    L = Local(torch.cat(lk), torch.cat(lv), torch.cat(lb), tuple(ln))
    null = (k.new_full((B,), math.log(0.2)), k.new_full((B, V), 1 / V))
    ref = exhaustive(q, k, v, b, local=L, null=null)
    exact = q @ k.T / math.sqrt(D) + b
    feat = (log_features(q, omega)[:, None] + log_features(k, omega)[None]).logsumexp(
        -1
    ) + b
    c = {}
    rs = [[index.frontier(p).hit_range(1)] for p in prefix]
    off = int(sum(a == z for [(a, z)] in rs))
    for label, ranges in [
        ("background", [[] for _ in q]),
        ("frontier_mems", rs),
        ("full", [[(0, n)] for _ in q]),
    ]:
        plan = Plan.build(layout, ranges)
        y = indexed(q, bank, plan, local=L, null=null)
        row_tv = []
        direct_y = []
        direct_z = []
        start = 0
        for i, r in enumerate(ranges):
            h = feat[i].clone()
            for a, z in r:
                h[a:z] = exact[i, a:z]
            sl = slice(start, start + ln[i])
            start += ln[i]
            local_score = (L.keys[sl] @ q[i]) / math.sqrt(D) + L.bias[sl]
            ps = torch.cat((h, local_score, null[0][i : i + 1]))
            u = ps.softmax(0)
            gt = torch.cat((exact[i], local_score, null[0][i : i + 1])).softmax(0)
            row_tv.append(0.5 * (u - gt).abs().sum().item())
            direct_y.append(u @ torch.cat((v, L.values[sl], null[1][i : i + 1])))
            direct_z.append(h.logsumexp(0))
        err = (torch.stack(direct_y) - y.read).abs().max().item()
        torch.testing.assert_close(
            y.read, torch.stack(direct_y), rtol=2e-12, atol=2e-13
        )
        torch.testing.assert_close(
            y.source_log_z[:, 0], torch.stack(direct_z), rtol=2e-12, atol=2e-13
        )
        nt = 0.5 * (ref.read - y.read).abs().sum(-1)
        c[label] = dict(
            exact_occurrences=sum(b - a for _, a, b in plan.exact),
            explicit_tail_rows=sum(b - a for _, a, b in plan.edge),
            summary_nodes=len(plan.nodes),
            row_tv_mean=float(np.mean(row_tv)),
            row_tv_max=max(row_tv),
            token_tv_mean=nt.mean().item(),
            token_tv_max=nt.max().item(),
            abs_corpus_share_mean=(ref.shares[:, 0] - y.shares[:, 0])
            .abs()
            .mean()
            .item(),
            abs_corpus_log_z_mean=(ref.source_log_z[:, 0] - y.source_log_z[:, 0])
            .abs()
            .mean()
            .item(),
            implementation_max_token_abs=err,
        )
    if c["full"]["row_tv_max"] > 1e-12 or c["full"]["token_tv_max"] > 1e-12:
        raise RuntimeError("full coverage must recover exhaustive attention")
    return dict(
        scope="invented FP64 CPU; distribution fidelity, not language quality or timing",
        rows=n,
        queries=B,
        lexical_miss_queries=off,
        rank=len(omega),
        key_dim=D,
        value_dim=V,
        leaf=layout.leaf,
        source_priors="unit mass per corpus and per local source before content score; no-read mass .2",
        cells=c,
    )


if __name__ == "__main__":
    report = run()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(report, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
