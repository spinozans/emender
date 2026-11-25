"""Closed-receipt LS work/mark correlations and all-target document-paired reductions."""

from __future__ import annotations
from math import sqrt
from ..x48_muon_continuation.data import check
from . import surprise_analysis as old
from .literal_mark import DEPTHS
from .panels import SEEDS, T

ARMS = "NTHJSU"
BOOT = 4910
PAIRS = (
    ("J", "T"),
    ("J", "S"),
    ("H", "T"),
    ("T", "N"),
    ("H", "N"),
    ("S", "T"),
    ("J", "H"),
    ("T", "U"),
    ("J", "U"),
)


def correlation(w):
    # Compute the centered products in integer arithmetic before the single final division.
    n = w["literal_hits"]
    t = w["text_code_sum"]
    h = w["neural_code_sum"]
    tt = w["text_code_square_sum"]
    hh = w["neural_code_square_sum"]
    th = w["text_neural_code_product_sum"]
    vt = n * tt - t * t
    vh = n * hh - h * h
    cov = n * th - t * h
    check(vt >= 0 and vh >= 0 and cov * cov <= vt * vh, "invalid paired mark moments")
    return cov / sqrt(vt * vh) if vt and vh else None


def checked_text(x):
    w = x["work"]
    hit = w["literal_hits"]
    n = x["targets"]
    visits = sum(max(0, n - d) for d in DEPTHS)
    pairs = w["text_distinct_pairs"]
    contexts = w["text_distinct_contexts"]
    changed = w["text_vs_neural_changed"]
    check(
        n == T
        and x["host_prefills"] == 1
        and all(type(v) is int and v >= 0 for v in w.values())
        and w["text_order_zero_updates"]
        == w["text_mark_storage_bytes"]
        == w["mark_storage_bytes"]
        == n
        and w["text_arrivals_encoded"] == n - 1
        and w["text_context_lookups"] == w["text_context_updates"] == visits
        and w["text_seen_context_lookups"] + contexts == visits
        and contexts <= pairs <= visits
        and w["text_marks_saturated"] <= n - 1
        and w["text_mark_reads"] == w["U_mark_reads"] == w["S_mark_reads"] == hit
        and changed <= hit
        and w["text_read_code_saturated"] <= hit,
        "wrong LS text-table/tape work",
    )
    tt, hh, th = (
        w[k]
        for k in (
            "text_code_square_sum",
            "neural_code_square_sum",
            "text_neural_code_product_sum",
        )
    )
    t, h = (w[k] for k in ("text_code_sum", "neural_code_sum"))
    check(
        max(t, h) <= 255 * hit
        and t <= tt <= 255 * t
        and h <= hh <= 255 * h
        and th <= 255 * min(t, h)
        and changed <= tt + hh - 2 * th <= 255**2 * changed,
        "wrong LS paired code sums",
    )
    correlation(w)


def summarize(role, logs, hashes):
    check(role in ("fit", "probe"), "LS has no final")
    z = old.copy_summary(
        role,
        logs,
        hashes,
        [T] * len(hashes),
        arms=ARMS,
        pairs=PAIRS,
        seed=BOOT,
        responses=ARMS,
        extra_clocks=("text_build_s", "text_read_joint_fields_s"),
    )
    peaks = {}
    corr = {}
    for s in SEEDS:
        k = str(s)
        rows = logs[s]
        support = z["documents_with_support_by_seed"][k]
        w = z["work_by_seed"][k]
        for r, first in zip(rows, logs[SEEDS[0]], strict=True):
            checked_text(r)
            check(
                all(
                    first["work"][name] == v
                    for name, v in r["work"].items()
                    if name.startswith("text_")
                    and name
                    not in ("text_vs_neural_changed", "text_neural_code_product_sum")
                ),
                "same-document literal counts or codes changed across weights",
            )
        support.update(
            text_vs_neural_changed=sum(
                r["work"]["text_vs_neural_changed"] > 0 for r in rows
            ),
            saturated_text_write=sum(
                r["work"]["text_marks_saturated"] > 0 for r in rows
            ),
        )
        peaks[k] = {
            name: max(r["work"][name] for r in rows)
            for name in ("text_distinct_contexts", "text_distinct_pairs")
        }
        corr[k] = correlation(w)
    return dict(
        **z,
        peak_table_entries_per_window_by_seed=peaks,
        hit_weighted_pearson_by_seed=corr,
    )


def decision(cs):
    pos = {
        k: v["ci95"][0] > 0 and v["gain"] > 0 and all(x > 0 for x in v["by_seed"])
        for k, v in cs.items()
    }
    r = cs["H:T"]
    text_beats_neural = (
        r["ci95"][1] < 0 and r["gain"] < 0 and all(x < 0 for x in r["by_seed"])
    )
    residual = pos["J:T"] and pos["J:S"]
    return dict(
        residual_neural_information=residual,
        positive_checks=pos,
        text_beats_neural_matched=text_beats_neural,
        consider_joint_confirmation=residual and pos["J:H"] and pos["J:U"],
        consider_literal_rival=text_beats_neural and pos["T:U"],
        local_incumbent="unchanged confirmed U with its matched checkpoint; native TRAIN diagnostic",
    )


def analyze(root):
    r = old.analyze(root, study="LS")
    r["decision"] = decision(r["stages"]["probe"]["contrasts"])
    return r
