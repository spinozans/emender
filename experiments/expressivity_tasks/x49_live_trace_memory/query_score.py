"""Score a single frozen endpoint and one full shifted TRAIN diagnostic window. No file/lease loader."""

from __future__ import annotations
from collections import Counter
from time import perf_counter
import torch
from torch.nn.functional import nll_loss
from ..x48_muon_continuation.data import check
from .endpoint import frozen_prefill
from .panels import SEEDS, T
from .query_depth import MAX_DEPTH, score

POLICIES = "JEM"
STRATA = ("literal_miss", "hit_stay", "move_unmatched", "move_matched")


def cycle(seed, rank):
    check(seed in SEEDS and type(rank) is int and rank >= 0, "foreign Q seed/order")
    i = (rank + SEEDS.index(seed)) % 3
    return POLICIES[i:] + POLICIES[:i]


def classify(word, extended, matched):
    tags = torch.zeros(len(word), dtype=torch.uint8)
    totals = {x: Counter() for x in STRATA}
    last = -1
    for e, m in zip(extended, matched, strict=True):
        t, j, k, s = m.t, m.latest, m.extended, m.selected
        move = k != j
        ctl = m.matched
        check(
            last < t < len(word)
            and (e.t, e.depth, e.latest, e.extended, e.latest_length, e.longest)
            == (t, m.depth, j, k, m.latest_length, m.longest)
            and e.selected == k
            and not e.matched,
            "different E/M extension map",
        )
        last = t
        check(
            0 <= k <= j <= t - 16
            and 0 <= s <= j
            and word[j] == word[k] == word[s]
            and m.depth in (1, 2, 4, 8)
            and m.depth <= m.latest_length <= m.longest <= MAX_DEPTH
            and move == (m.longest > m.latest_length)
            and (s != k) == ctl,
            "foreign value address",
        )
        check(
            not ctl
            or (move and s != j and (t - k).bit_length() == (t - s).bit_length()),
            "control is not an equal-bin substitution",
        )
        tag = 3 if ctl else 2 if move else 1
        tags[t] = tag
        g = totals[STRATA[tag]]
        g.update(
            literal_depth_sum=m.depth,
            latest_length_sum=m.latest_length,
            extended_length_sum=m.longest,
            latest_age_sum=t - j,
            extended_age_sum=t - k,
            control_age_sum=t - s,
            abs_control_age_gap_sum=abs(k - s),
            signed_control_minus_extended_age_sum=k - s,
        )
        g["max_abs_control_age_gap"] = max(g["max_abs_control_age_gap"], abs(k - s))
    return tags, totals


@torch.inference_mode()
def window_score(host, head, word, *, seed, rank, width=256):
    check(
        type(word) is bytes
        and len(word) == T + 1
        and not host.training
        and not head.training,
        "one full document-bound window and frozen endpoint required",
    )
    inputs = word[:-1]
    h, z, prefill = frozen_prefill(host, inputs, width)
    order = cycle(seed, rank)
    lp = {}
    cells = {}
    routes = {}
    for a in order:
        work = {}
        time = {}
        di = {}
        start = perf_counter()
        lp[a] = score(
            head,
            inputs,
            h,
            z,
            policy=a,
            horizon=T,
            work=work,
            diagnostics=di,
            timing=time,
        )
        elapsed = perf_counter() - start
        check(
            lp[a].shape == (T, 256)
            and lp[a].isfinite().all()
            and 0 <= time["extra_address_s"] <= elapsed,
            "invalid Q probabilities/time",
        )
        cells[a] = dict(policy_s=elapsed, **time, work=work)
        if a != "J":
            routes[a] = di["hits"]
    tags, geom = classify(inputs, routes["E"], routes["M"])
    counts = torch.bincount(tags.long(), minlength=4).tolist()
    check(
        sum(counts) == T
        and all(torch.equal(lp["J"][tags < 2], lp[a][tags < 2]) for a in "EM")
        and torch.equal(lp["E"][tags < 3], lp["M"][tags < 3]),
        "changed probability at identical binding",
    )
    # Scoring seam only, after J, E and M have all returned every next-byte score.
    ys = torch.tensor(list(word[1:]), dtype=torch.long)
    strata = {
        k: dict(targets=n, geometry=dict(geom[k]))
        for k, n in zip(STRATA, counts, strict=True)
    }
    for a in POLICIES:
        w = cells[a]["work"]
        check(
            w["literal_queries"] == w["literal_value_writes"] == T
            and counts[0] == w["literal_misses"]
            and sum(counts[1:]) == w["literal_value_reads"] == w["literal_hits"]
            and all(w[k] == v for k, v in cells["J"]["work"].items()),
            "unmatched literal byte/read work",
        )
        check(
            a == "J"
            or (
                w["extended_moves"] == counts[2] + counts[3]
                and w["matched_controls"] == (counts[3] if a == "M" else 0)
            ),
            "wrong actual switches",
        )
        loss = nll_loss(lp[a], ys, reduction="none").double()
        total = loss.sum().item()
        parts = {k: loss[tags == i].sum().item() for i, k in enumerate(STRATA)}
        check(
            0 <= total < float("inf") and abs(total - sum(parts.values())) < 1e-8,
            "strata not all target bytes",
        )
        cells[a].update(nll_sum=total, stratum_nll_sums=parts)
    return dict(
        order=order,
        host_prefills=1,
        targets=T,
        host_prefill_s=prefill,
        cells=cells,
        strata=strata,
    )
