"""ET paired real-byte scores and complete chosen-path clocks. Never infer the cost of an index from its response clock."""

from collections import Counter
from math import isclose, isfinite
import numpy as np
from ..x48_muon_continuation.data import check
from . import error_history_stats as old
from .error_history_stats import ARMS as ARMS, PAIRS as PAIRS
from .error_transfer_score import BRANCHES, first
from .panels import SEEDS, T
from .surprise_analysis import paired_table

BOOT = 4916
SUB = ("prep", "pool", "feature", "response")
SHARED = (
    "prefill_s",
    "base_B_s",
    "shared_current_s",
    "shared_US_mark_build_s",
    "U_mark_read_s",
    "text_build_s",
    "J_response_s",
)
CLOCKS = {"shared_J_s", "comparison_only_s", *SHARED} | {
    a + "_" + k + "_s" for a in BRANCHES for k in (*SUB, "branch")
}
SAME = (
    "hits",
    "active_rows",
    "key_admissions",
    "committed_records",
    "peak_keys",
    "older_occurrence_uses",
    "older_mistake_uses",
)


def checked_row(row, s, i, length):
    w = row["work"]
    cw = row["chosen_work"]
    loss = row["nll_sums"]
    times = row["seconds"]
    hit = w["literal_hits"]
    a = w["pool_active_rows"]
    check(
        row["targets"] == length
        and row["host_prefills"] == 1
        and row["first_branch"] == first(s, i)
        and set(loss) == set(ARMS)
        and all(isfinite(v) and v >= 0 for v in loss.values()),
        "wrong ET order, scores or exposure",
    )
    old.checked_history_work(
        w,
        length,
        row["candidate_correct"],
        tape_bytes=w["top1_bytes"],
        written=w["top1_written"],
        mistakes=w["wrong_top1"],
    )
    check(
        (hit or len(set(loss.values())) == 1)
        and (a or all(loss[k] == loss["J"] for k in (*old.DIMS, "SWAP")))
        and (
            w["changed_histogram_rows"]
            or isclose(loss["ERROR"], loss["SWAP"], abs_tol=5e-14, rel_tol=5e-14)
        ),
        "changed ET miss, singleton or null intervention",
    )
    check(
        set(times) == CLOCKS
        and all(isfinite(v) and v >= 0 for v in times.values())
        and times["shared_J_s"] >= sum(times[k] for k in SHARED) > 0
        and times["comparison_only_s"] > 0,
        "wrong shared ET cost",
    )
    check(
        set(cw) == set(BRANCHES) and w["miss_flag_bytes"] == 0,
        "incorrect ET pool family",
    )
    for arm in BRANCHES:
        c = cw[arm]
        h = arm == "HISTORY"
        nc = c["packed_cells"]
        d = old.DIMS[arm]
        check(
            all(type(v) is int and v >= 0 for v in c.values())
            and all(c[k] == w["pool_" + k] for k in SAME)
            and not any(k.startswith("shift_") for k in c)
            and c["top1_bytes"] == length
            and c["top1_written"] == length - 1
            and c["wrong_top1"] == w["wrong_top1"]
            and c["miss_flag_bytes"] == (length if h else 0)
            and c["packed_feature_entries"] == nc
            and c["fp64_feature_bytes"] == 8 * d * nc
            and c["packed_bytes"] == 5 * length + 4 * d * nc
            and a <= nc <= w["pool_packed_cells"]
            and c["peak_cells"] <= w["pool_peak_cells"]
            and (not h or c["peak_cells"] <= c["committed_records"])
            and (
                h
                or (
                    nc == w["pool_packed_cells"]
                    and c["peak_cells"] == w["pool_peak_cells"]
                )
            ),
            "incorrect chosen-only ET index accounting",
        )
        check(
            times[arm + "_branch_s"]
            >= sum(times[arm + "_" + k + "_s"] for k in SUB)
            > 0,
            "incomplete chosen ET clock",
        )


def cost(sec, orders, nt):
    j = sec["shared_J_s"]
    h = sec["HISTORY_branch_s"]
    e = sec["ERROR_branch_s"]
    return dict(
        first_counts=dict(orders),
        shared_J_s=j,
        shared_plus_HISTORY_s=j + h,
        shared_plus_ERROR_s=j + e,
        HISTORY_extra_s_per_million_targets=h * 1e6 / nt,
        ERROR_extra_s_per_million_targets=e * 1e6 / nt,
        ERROR_minus_HISTORY_s_per_million_targets=(e - h) * 1e6 / nt,
        shared_path_HISTORY_over_J=(j + h) / j,
        shared_path_ERROR_over_J=(j + e) / j,
        shared_path_ERROR_over_HISTORY=(j + e) / (j + h),
    )


def summarize(logs, hashes, sizes):
    n = len(sizes)
    nt = sum(sizes)
    check(
        n
        and n == len(hashes) == len(set(hashes))
        and all(type(v) is int and 1 <= v < T for v in sizes)
        and set(logs) == set(SEEDS),
        "invalid ET paired short population",
    )
    loss = np.empty((len(SEEDS), n, len(ARMS)))
    work = {}
    chosen = {}
    peaks = {}
    sup = {}
    clocks = {}
    prices = {}
    for si, s in enumerate(SEEDS):
        total = Counter()
        ct = {k: Counter() for k in BRANCHES}
        docs = Counter()
        seconds = Counter()
        order = Counter()
        peak = {}
        for i, (r, h, length) in enumerate(zip(logs[s], hashes, sizes, strict=True)):
            checked_row(r, s, i, length)
            w = r["work"]
            cw = r["chosen_work"]
            ref = logs[SEEDS[0]][i]
            correct = r["candidate_correct"]
            check(
                r["document_sha"] == h
                and correct == ref["candidate_correct"]
                and all(
                    v == ref["work"][k]
                    for k, v in w.items()
                    if k.startswith(("literal_", "text_"))
                    or k
                    in (
                        "pool_hits",
                        "pool_active_rows",
                        "pool_older_occurrence_uses",
                        "pool_peak_keys",
                        "pool_key_admissions",
                        "pool_committed_records",
                        "mark_norms_from_current",
                    )
                ),
                "changed ET document or geometry",
            )
            loss[si, i] = [r["nll_sums"][k] for k in ARMS]
            total.update(w)
            total.update(candidate_correct=correct)
            docs.update({k: int(bool(v)) for k, v in old.support(w, correct).items()})
            seconds.update(r["seconds"])
            order[r["first_branch"]] += 1
            for name in old.PEAKS:
                peak[name] = max(peak.get(name, 0), w[name])
            for k, c in cw.items():
                ct[k].update(c)
                for name in (
                    "peak_keys",
                    "peak_cells",
                    "packed_bytes",
                    "fp64_feature_bytes",
                    "top1_bytes",
                    "miss_flag_bytes",
                ):
                    peak[k + "_" + name] = max(peak.get(k + "_" + name, 0), c[name])
        k = str(s)
        work[k] = dict(total)
        chosen[k] = {a: dict(v) for a, v in ct.items()}
        sup[k] = dict(docs)
        clocks[k] = dict(seconds)
        prices[k] = cost(seconds, order, nt)
        peaks[k] = peak
    return dict(
        paired_documents=n,
        real_targets_per_seed=nt,
        work_by_seed=work,
        chosen_work_by_seed=chosen,
        documents_with_support_by_seed=sup,
        peak_per_window_by_seed=peaks,
        seconds_by_seed=clocks,
        branch_cost_by_seed=prices,
        **paired_table(loss, sizes, ARMS, PAIRS, BOOT),
    )


def decision(stats):
    d = old.decision(stats["contrasts"])
    return dict(
        positive_checks=d["positive_checks"],
        replicated_bound_wrong_guesses=d["added_bound_wrong_guess_identities"],
        replicated_pooled_scalar_history=d["pooled_scalar_history"],
        scope="locked transfer on short native calibration only, no default or scale decision",
    )
