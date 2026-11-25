"""PT invented-input price and short-device oracles. None of these functions is a calibration reader."""

from bisect import bisect_right
from collections import defaultdict
from math import fsum
from time import perf_counter
import numpy as np
import torch
from ..x48_muon_continuation.data import check
from . import (
    paired_correction as pc,
    paired_correction_cost as old,
    joint_reader as joint,
    literal_mark as ls,
    correction_transfer_score as score,
    correction_transfer_reader as reader,
)
from .endpoint import frozen_prefill
from .literal import DEPTHS, MIN_AGE
from .panels import SEEDS


def two_keys(n):
    pattern = [
        (0, 20),
        (0, 30),
        (0, 20),
        (1, 30),
        (1, 20),
        (0, 30),
        (0, 30),
        (1, 20),
        (0, 20),
        (1, 30),
    ]
    motif = b"".join(
        (b"ABC\0D\xffEk", b"pqr\x1esuVk")[k] + bytes([c]) + b"!@#$%&;"
        for k, c in pattern
    )
    return (motif * (n // len(motif) + 1))[:n], bytes(
        80 + (t // 16) % 2 if t % 16 == 7 else t % 7 for t in range(n)
    )


PRICE_LENGTHS = dict(
    uniform=4095,
    recurring_pairs=4095,
    one_key_wrong_fanout=4095,
    many_alternatives=4095,
    two_keys_short=1023,
    two_keys_long=4095,
)


@torch.inference_mode()
def audit_head(seed, e, head):
    ck, b = reader.contract(seed)
    check(ck == e["sha256"], "wrong PT price head")
    writer = joint._writer()
    gen = torch.Generator().manual_seed(4918)
    h = torch.randn(4095, head.direct.host_dim, generator=gen)
    noise = torch.rand(4095, 256, generator=gen)
    cases = old.cases()
    out = {}
    for i, (k, n) in enumerate(PRICE_LENGTHS.items()):
        w, q = cases[k] if k in cases else two_keys(n)
        w = w[:n]
        z = noise[:n].clone()
        z[torch.arange(n), list(q[:n])] = 4.0
        tic = perf_counter()
        lp, _, row = score._answers(
            head, w, h[:n], z, b, seed, i, writer, tic=tic, pre_s=0
        )
        wall = perf_counter() - tic
        s = row["seconds"]
        v = row["work"]
        check(
            set(lp) == set(score.ARMS) and row["targets"] == n,
            "incomplete PT invented comparison",
        )
        out[k] = dict(
            positions=n,
            posthost_total_s=wall,
            posthost_ERROR_s=s["shared_ERROR_s"],
            **{a + "_extra_s": s[a + "_branch_s"] for a in score.BRANCHES},
            comparison_only_s=s["comparison_only_s"],
            matched=v["matched_query_rows"],
            different_pair_prefix_rows=v["prefix_different_law_rows"],
            different_pair_swapped_rows=v["changed_conditional_rows"],
            prefix_pair_cells=v["prefix_peak_pair_cells"],
            true_pair_cells=v["pair_peak_pair_cells"],
        )
        del lp
    check(
        out["two_keys_long"]["different_pair_prefix_rows"] > 0
        and out["one_key_wrong_fanout"]["matched"] > 0
        and out["one_key_wrong_fanout"]["different_pair_prefix_rows"] == 0
        and out["recurring_pairs"]["different_pair_swapped_rows"] > 0,
        "uninformative PT invented pricing",
    )
    return dict(
        checkpoint_sha=ck, invented_packets=out, natural_documents=0, host_executions=0
    )


def prefix_reference(w, z, p, beta):
    """On a *synthetic* packet, evaluate the prefix contrast by address-list bisects and rfind.

    PC supplies only ERROR and its already independently checked unpaired column. For b != q
    that column is A*v_b. Renormalizing it times the prefix likelihood gives A*r_prefix.
    This oracle may inspect the invented window to construct sorted lists, but its counts are
    bounded by the current frontier. Neither the lists nor this code are used for real reads.
    """
    q = z.argmax(dim=-1).tolist()
    cols = defaultdict(list)
    edges = defaultdict(list)
    for j, (a, c) in enumerate(zip(q[:-1], w[1:], strict=True), 1):
        if a != c:
            cols[c].append(j)
            edges[a, c].append(j)
    rows = p.row.numpy()
    symbols = p.symbol.numpy()
    f0 = p.x[:, 0].numpy()
    f1 = np.zeros_like(f0)
    lo = 0
    for t in p.active.tolist():
        a = q[t]
        close = t - MIN_AGE
        j = -1
        for d in DEPTHS:
            if d > t + 1:
                continue
            k = w.rfind(w[t + 1 - d : t + 1], 0, close)
            if k >= 0:
                j = k + d
                break
        check(1 <= j <= close, "missing oracle latest")
        hi = np.searchsorted(rows, t, side="right")
        sym = symbols[lo:hi]
        col = f0[lo:hi]
        alpha = -float(col[sym == a].item())
        weights = []
        for b, v in zip(sym, col, strict=True):
            if b == a:
                weights.append(0.0)
                continue
            sub = int(q[j - 1] != w[j] and w[j] == b)
            den = bisect_right(cols[b], close) - sub
            num = bisect_right(edges[a, b], close) - sub * int(q[j - 1] == a)
            check(v > 0 and den > 0 and 0 <= num <= den, "invalid oracle alternatives")
            weights.append(v * num / den)
        total = fsum(weights)
        check(alpha > 0 and total > 0, "empty oracle law")
        f1[lo:hi] = alpha * np.asarray(weights) / total - np.maximum(col, 0)
        lo = hi
    check(lo == len(rows), "oracle lost sparse coordinates")
    adjusted = p.base.clone()
    adjusted[p.row, p.symbol] += torch.from_numpy(beta[0] * f0 + beta[1] * f1)
    return adjusted - (adjusted.logsumexp(-1) - p.base.logsumexp(-1))[:, None]


@torch.inference_mode()
def short_shapes(host, head, beta, lengths, width, writer):
    for i, n in enumerate(lengths):
        word = two_keys(n)[0]
        tic = perf_counter()
        h, z, pre = frozen_prefill(host, word, width)
        out, _, row = score._answers(
            head, word, h, z, beta, SEEDS[0], i, writer, tic=tic, pre_s=pre
        )
        p = pc.fields(
            head,
            word,
            h,
            z,
            beta["J"],
            beta["ERROR"],
            ls.text_tape,
            mode="panel",
            anchors=(beta["U"], beta["HISTORY"]),
        )
        ref = dict(
            p.anchors,
            **{
                a: pc.response(p, a, beta["PAIR" if a == "SWAP" else a])
                for a in (*pc.DIMS, "SWAP")
            },
            PREFIX=prefix_reference(word, z, p, beta["PAIR"]),
        )
        check(
            row["targets"] == n
            and row["work"]["changed_conditional_rows"]
            == p.work["changed_conditional_rows"]
            and set(out) == set(ref) == set(score.ARMS)
            and all(torch.allclose(out[a], ref[a], atol=3e-14, rtol=0) for a in ref),
            "PT short-device score parity failed",
        )
