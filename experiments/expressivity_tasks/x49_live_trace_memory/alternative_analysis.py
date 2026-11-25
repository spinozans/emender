"""Receipt-only X49-V reductions. All shifted targets, one jointly resampled set of documents."""

from __future__ import annotations
from collections import Counter
from math import isfinite
import numpy as np
from ..x48_muon_continuation.data import check
from .panels import T, SEEDS
from . import surprise_analysis as old

ARMS = "UDVK"
PAIRS = (("V", "D"), ("V", "K"), ("V", "U"), ("D", "U"), ("K", "D"))
BOOT = 4909


def checked_work(role, row):
    w = row["work"]
    o = row["outcomes"]
    hit = w["literal_hits"]
    non = o["hit_noncopy"]
    scored = role == "probe"
    check(
        role in ("fit", "probe")
        and row["targets"] == T
        and row["host_prefills"] == 1
        and ("nll_sums" in row) == scored,
        "wrong V exposure/phase",
    )
    check(
        set(o)
        == {"copy_correct", "hit_noncopy", *(a + "_alternative_correct" for a in "DVK")}
        and all(type(v) is int and v >= 0 for v in (*w.values(), *o.values())),
        "foreign V counters",
    )
    check(
        hit + w["literal_misses"]
        == T
        == w["literal_queries"]
        == w["literal_value_writes"]
        == w["mark_storage_bytes"]
        == w["alternative_storage_bytes"]
        and hit
        == w["literal_value_reads"]
        == w["U_mark_reads"]
        == w["V_alternative_reads"]
        == w["K_alternative_reads"]
        and w["literal_payloads_reassigned"] == 0
        and "S_mark_reads" not in w
        and w["alternatives_encoded"]
        == w["arrivals_encoded"]
        == T - 1
        == w["mark_norms_from_current"] + w["mark_norms_extra"]
        and o["copy_correct"] + non == hit,
        "wrong V tape, copy or read work",
    )
    check(
        max(
            w[k]
            for k in (
                "B_candidate_code_saturated",
                "now_candidate_code_saturated",
                "U_read_code_saturated",
            )
        )
        <= hit
        and w["marks_saturated"] <= T - 1
        and w["K_codes_changed"] <= w["K_slots_displaced"] <= hit
        and max(w["V_equals_current"], w["K_equals_current"]) <= hit
        and abs(w["V_equals_current"] - w["K_equals_current"]) <= w["K_codes_changed"]
        and all(o[a + "_alternative_correct"] <= non for a in "DVK")
        and abs(o["V_alternative_correct"] - o["K_alternative_correct"])
        <= w["K_codes_changed"]
        and all(
            abs(o[a + "_alternative_correct"] - o["D_alternative_correct"])
            <= hit - w[a + "_equals_current"]
            for a in "VK"
        ),
        "wrong V proposal geometry",
    )
    seconds = {
        "prefill_s",
        "base_B_s",
        "shared_current_s",
        "shared_US_mark_build_s",
        "U_mark_read_s",
        "incumbent_U_response_s",
        "current_alternative_s",
        "alternative_build_s",
        "V_read_and_feature_s",
        "K_rotation_read_and_feature_s",
    } | ({"response_" + a + "_s" for a in "DVK"} if scored else {"fit_statistics_s"})
    check(
        set(row["seconds"]) == seconds
        and all(isfinite(v) and v >= 0 for v in row["seconds"].values()),
        "wrong V timings",
    )
    if scored:
        loss = row["nll_sums"]
        check(
            set(loss) == set(ARMS)
            and all(isfinite(v) and v >= 0 for v in loss.values())
            and (non or len(set(loss.values())) == 1),
            "wrong V all-byte sums",
        )


def summarize(role, logs, hashes):
    n = len(hashes)
    check(
        n and len(set(hashes)) == n and set(logs) == set(SEEDS), "unpaired V population"
    )
    weights = {}
    docs = {}
    clock = Counter()
    scored = role == "probe"
    losses = np.empty((len(SEEDS), n, len(ARMS))) if scored else None
    for si, s in enumerate(SEEDS):
        work = Counter()
        support = Counter()
        for i, (h, x) in enumerate(zip(hashes, logs[s], strict=True)):
            checked_work(role, x)
            w = x["work"]
            o = x["outcomes"]
            ref = logs[SEEDS[0]][i]
            check(
                x["document_sha"] == h
                and all(
                    ref["work"][k] == v
                    for k, v in w.items()
                    if k.startswith("literal_")
                    or k in ("K_slots_displaced", "mark_norms_from_current")
                )
                and all(
                    o[k] == ref["outcomes"][k] for k in ("copy_correct", "hit_noncopy")
                ),
                "V literal geometry or identity differs across seeds",
            )
            work.update(w)
            work.update(o)
            clock.update(x["seconds"])
            support.update(
                {
                    k: int(v > 0)
                    for k, v in dict(
                        hits=w["literal_hits"],
                        misses=w["literal_misses"],
                        displaced_slot=w["K_slots_displaced"],
                        changed_code=w["K_codes_changed"],
                        different_from_current=w["literal_hits"]
                        - w["V_equals_current"],
                        **o,
                    ).items()
                }
            )
            if scored:
                losses[si, i, :] = [x["nll_sums"][a] for a in ARMS]
        weights[str(s)] = dict(work)
        docs[str(s)] = dict(support)
    out = dict(
        paired_documents=n,
        real_targets_per_seed=n * T,
        work_by_seed=weights,
        documents_with_support_by_seed=docs,
        mean_seconds_per_seed_document={
            k: v / (len(SEEDS) * n) for k, v in clock.items()
        },
    )
    if scored:
        out.update(old.paired_table(losses, [T] * n, ARMS, PAIRS, BOOT))
    return out


def decision(cs):
    passes = {
        k: cs[k]["gain"] > 0
        and cs[k]["ci95"][0] > 0
        and all(v > 0 for v in cs[k]["by_seed"])
        for k in ("V:D", "V:K", "V:U")
    }
    return dict(
        contrasts_pass=passes,
        warrants_separate_confirmation=all(passes.values()),
        local_incumbent="U with its matched W-B checkpoint; this TRAIN diagnostic does not adopt V",
    )


def analyze(root):
    r = old.analyze(root, study="V")
    r["decision"] = decision(r["stages"]["probe"]["contrasts"])
    return r
