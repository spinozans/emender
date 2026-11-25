"""EH: the primary denominator is all shifted bytes, not mistakes or changed histograms."""

from collections import Counter
from math import isclose, isfinite
import numpy as np
from ..x48_muon_continuation.data import check
from .error_history import ARMS, DIMS
from .literal import DEPTHS, MIN_AGE
from .panels import SEEDS, T
from . import surprise_analysis as common

PAIRS = (
    ("ERROR", "HISTORY"),
    ("ERROR", "SHIFT"),
    ("ERROR", "SWAP"),
    ("ERROR", "J"),
    ("HISTORY", "COUNT"),
    ("HISTORY", "J"),
    ("COUNT", "J"),
    ("SHIFT", "HISTORY"),
    ("ERROR", "U"),
    ("J", "U"),
    ("U", "B"),
)
BOOT = 4915
CLOCKS = {
    "prefill_s",
    "base_B_s",
    "shared_current_s",
    "shared_US_mark_build_s",
    "U_mark_read_s",
    "text_build_s",
    "J_response_s",
    "anchor_BU_s",
    "guess_and_tape_prep_s",
    "true_pool_s",
    "shift_pool_s",
    "feature_s",
}
PEAKS = (
    "pool_peak_keys",
    "pool_peak_cells",
    "pool_packed_bytes",
    "shift_peak_keys",
    "shift_peak_cells",
    "shift_peak_donor_keys",
    "shift_packed_bytes",
    "text_distinct_contexts",
    "text_distinct_pairs",
    "fp64_feature_bytes",
)


def checked_history_work(w, length, right, *, tape_bytes, written, mistakes):
    hit = w["literal_hits"]
    a = w["pool_active_rows"]
    visits = sum(max(0, length - d) for d in DEPTHS)
    admit = sum(max(0, length - MIN_AGE - d) for d in DEPTHS)
    n = w["packed_feature_entries"]
    check(
        type(length) is int
        and 1 <= length <= T
        and type(right) is int
        and 0 <= right <= hit
        and all(type(v) is int and v >= 0 for v in w.values()),
        "wrong EH exposure, counters or phase",
    )
    check(
        length
        == hit + w["literal_misses"]
        == w["literal_queries"]
        == w["literal_value_writes"]
        == tape_bytes
        == w["text_mark_storage_bytes"]
        == w["mark_storage_bytes"]
        == w["text_order_zero_updates"]
        and hit
        == w["pool_hits"]
        == w["shift_hits"]
        == w["literal_value_reads"]
        == w["U_mark_reads"]
        and w["literal_payloads_reassigned"] == 0
        and "S_mark_reads" not in w
        and written
        == w["text_arrivals_encoded"]
        == w["arrivals_encoded"]
        == w["mark_norms_extra"] + w["mark_norms_from_current"]
        == length - 1
        and w["text_context_updates"]
        == w["text_context_lookups"]
        == visits
        == w["text_distinct_contexts"] + w["text_seen_context_lookups"]
        and w["text_distinct_contexts"] <= w["text_distinct_pairs"] <= visits,
        "wrong anchor, original notes, or guess-tape accounting",
    )
    check(
        all(
            w["pool_" + k] == w["shift_" + k]
            for k in (
                "key_admissions",
                "committed_records",
                "peak_keys",
                "hits",
                "active_rows",
                "older_occurrence_uses",
                "older_mistake_uses",
            )
        )
        and w["pool_key_admissions"]
        == admit
        == w["pool_peak_keys"] + w["pool_committed_records"]
        and w["pool_peak_cells"] <= 2 * w["pool_committed_records"]
        and w["shift_peak_cells"] <= w["shift_committed_records"]
        and a <= hit
        and a <= w["pool_older_occurrence_uses"] <= a * max(0, length - MIN_AGE - 1)
        and w["shift_older_mistake_uses"] <= w["pool_older_occurrence_uses"],
        "sham changed older occurrence geometry",
    )
    m = mistakes
    check(
        w["shift_mistakes_written"] == m <= length - 1
        and m == w["shift_peak_donor_keys"] + w["shift_donor_slots_written"]
        and w["shift_different_guesses_written"] <= w["shift_donor_slots_written"]
        and w["changed_histogram_rows"] <= min(a, w["shift_older_changed_uses"])
        and w["shift_older_changed_uses"]
        <= w["shift_older_displaced_uses"]
        <= w["shift_older_mistake_uses"]
        and (w["shift_different_guesses_written"] or not w["shift_older_changed_uses"]),
        "incorrect EH mistake/donor accounting",
    )
    pc, sc = w["pool_packed_cells"], w["shift_packed_cells"]
    check(
        a <= pc <= n <= pc + sc
        and sc <= n <= 256 * a
        and w["fp64_feature_bytes"] == n * 7 * 8
        and w["pool_packed_bytes"] == 5 * length + 24 * pc
        and w["shift_packed_bytes"] == 5 * length + 8 * sc,
        "incorrect EH packed or feature size",
    )


def checked_row(role, row):
    scored = role == "probe"
    w = row["work"]
    a = w["pool_active_rows"]
    check(
        role in ("fit", "probe")
        and row["targets"] == T
        and row["host_prefills"] == 1
        and ("nll_sums" in row) == scored,
        "wrong EH exposure, counters or phase",
    )
    checked_history_work(
        w,
        T,
        row["candidate_correct"],
        tape_bytes=w["guess_tape_bytes"],
        written=w["guesses_written"],
        mistakes=w["wrong_guesses_written"],
    )
    times = row["seconds"]
    check(
        set(times)
        == CLOCKS
        | (
            {k + "_response_s" for k in (*DIMS, "SWAP")}
            if scored
            else {"fit_statistics_s"}
        )
        and all(isfinite(v) and v >= 0 for v in times.values()),
        "wrong EH phase timings",
    )
    if scored:
        z = row["nll_sums"]
        check(
            set(z) == set(ARMS)
            and all(isfinite(v) and v >= 0 for v in z.values())
            and (a or all(z["J"] == z[k] for k in (*DIMS, "SWAP")))
            and (
                w["changed_histogram_rows"]
                or isclose(z["SWAP"], z["ERROR"], rel_tol=5e-14, abs_tol=5e-14)
            ),
            "incorrect all-byte EH answers or null intervention",
        )


def support(w, right):
    return dict(
        hits=w["literal_hits"],
        misses=w["literal_misses"],
        copy_correct=right,
        singleton=w["literal_hits"] - w["pool_active_rows"],
        older_history=w["pool_active_rows"],
        older_mistake=w["pool_older_mistake_uses"],
        donor_displaced=w["shift_older_displaced_uses"],
        changed_guess=w["shift_older_changed_uses"],
        changed_histogram=w["changed_histogram_rows"],
    )


SAME = {
    "pool_hits",
    "pool_active_rows",
    "pool_key_admissions",
    "pool_committed_records",
    "pool_peak_keys",
    "pool_older_occurrence_uses",
    "mark_norms_from_current",
}


def summarize(role, logs, hashes):
    return full_summary(
        role,
        logs,
        hashes,
        validator=checked_row,
        support_counts=support,
        peaks_to_keep=PEAKS,
        same=SAME,
        arms=ARMS,
        pairs=PAIRS,
        boot=BOOT,
    )


def full_summary(
    role,
    logs,
    hashes,
    *,
    validator,
    support_counts,
    peaks_to_keep,
    same,
    arms,
    pairs,
    boot,
    sizes=None,
):
    """Reduce complete document windows; sizes (when supplied) are actual shifted target counts."""
    n = len(hashes)
    sizes = [T] * n if sizes is None else sizes
    check(
        n
        and n == len(sizes) == len(set(hashes))
        and set(logs) == set(SEEDS)
        and all(type(v) is int and 1 <= v <= T for v in sizes),
        "unpaired history documents or exposure",
    )
    scored = role == "probe"
    loss = np.empty((len(SEEDS), n, len(arms))) if scored else None
    secs = Counter()
    work = {}
    docs = {}
    peaks = {}
    for si, s in enumerate(SEEDS):
        total = Counter()
        sup = Counter()
        for i, (h, r) in enumerate(zip(hashes, logs[s], strict=True)):
            validator(role, r)
            w = r["work"]
            ref = logs[SEEDS[0]][i]
            right = r["candidate_correct"]
            check(
                r["document_sha"] == h
                and r["targets"] == sizes[i]
                and right == ref["candidate_correct"]
                and all(
                    v == ref["work"][k]
                    for k, v in w.items()
                    if k.startswith(("text_", "literal_")) or k in same
                ),
                "foreign EH document or changed literal geometry",
            )
            total.update(w)
            total.update(candidate_correct=right)
            sup.update({k: int(bool(v)) for k, v in support_counts(w, right).items()})
            secs.update(r["seconds"])
            if scored:
                loss[si, i] = [r["nll_sums"][k] for k in arms]
        k = str(s)
        work[k] = dict(total)
        docs[k] = dict(sup)
        peaks[k] = {v: max(r["work"][v] for r in logs[s]) for v in peaks_to_keep}
    z = dict(
        paired_documents=n,
        real_targets_per_seed=sum(sizes),
        work_by_seed=work,
        documents_with_support_by_seed=docs,
        peak_per_window_by_seed=peaks,
        mean_seconds_per_seed_document={
            k: v / (len(SEEDS) * n) for k, v in secs.items()
        },
    )
    if scored:
        z.update(common.paired_table(loss, sizes, arms, pairs, boot))
    return z


def decision(cs):
    pos = {
        k: v["gain"] > 0 and v["ci95"][0] > 0 and all(x > 0 for x in v["by_seed"])
        for k, v in cs.items()
    }
    return dict(
        positive_checks=pos,
        added_bound_wrong_guess_identities=all(
            pos["ERROR:" + k] for k in ("HISTORY", "SHIFT", "SWAP", "J")
        ),
        pooled_scalar_history=pos["HISTORY:COUNT"] and pos["HISTORY:J"],
        scope="native TRAIN only; no default or scale decision",
    )


def analyze(root):
    out = common.analyze(root, study="EH")
    out["decision"] = decision(out["stages"]["probe"]["contrasts"])
    return out
