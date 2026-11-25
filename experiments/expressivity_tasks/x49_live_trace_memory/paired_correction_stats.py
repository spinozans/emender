"""PC role checks and all-target paired inference. An unqueried exchange contributes no conditional evidence."""

from math import fsum, isclose, isfinite
from ..x48_muon_continuation.data import check
from .paired_correction import ARMS, DIMS
from .literal import DEPTHS, MIN_AGE
from .panels import SEEDS, T
from . import error_history_stats as hist, surprise_analysis as common

BOOT = 4917
PAIRS = (
    ("PAIR", "MARG"),
    ("PAIR", "SHUF"),
    ("PAIR", "SWAP"),
    ("PAIR", "ERROR"),
    ("MARG", "ERROR"),
    ("SHUF", "MARG"),
    ("PAIR", "J"),
    ("ERROR", "HISTORY"),
    ("HISTORY", "J"),
    ("J", "U"),
    ("U", "B"),
)
SUB = {
    "base_B_s",
    "shared_current_s",
    "shared_US_mark_build_s",
    "U_mark_read_s",
    "text_build_s",
    "J_response_s",
    "guess_tape_s",
    "error_fields_response_s",
}
CLOCKS = SUB | {
    "prefill_s",
    "shared_posthost_ERROR_s",
    "marginal_s",
    "true_pair_index_s",
    "true_pair_feature_s",
    "shuf_index_s",
    "shuf_feature_support_s",
}
PEAKS = (
    "error_peak_keys",
    "error_peak_cells",
    "error_packed_bytes",
    "error_fp64_feature_bytes",
    "marginal_feature_bytes",
    "pair_feature_bytes",
    "panel_feature_bytes",
    "guess_tape_bytes",
    "query_tape_bytes",
    "shuf_peak_loose_keys",
    "text_distinct_contexts",
    "text_distinct_pairs",
    *(
        a + "_" + k
        for a in ("pair", "shuf")
        for k in ("peak_keys", "peak_guess_rows", "peak_pair_cells", "packed_bytes")
    ),
)
SAME = {
    "error_hits",
    "error_active_rows",
    "error_key_admissions",
    "error_committed_records",
    "error_peak_keys",
    "error_older_occurrence_uses",
    "mark_norms_from_current",
}


def checked_row(role, r):
    w = r["work"]
    hit = w["literal_hits"]
    active = w["matched_query_rows"]
    old = w["error_active_rows"]
    h = w["matched_mistake_uses"]
    entries = w["marginal_entries"]
    right = r["candidate_correct"]
    probe = role == "probe"
    check(
        role in ("fit", "probe")
        and r["targets"] == T
        and r["host_prefills"] == 1
        and ("nll_sums" in r) == probe
        and all(type(v) is int and v >= 0 for v in w.values())
        and type(right) is int
        and 0 <= right <= hit,
        "wrong PC exposure, counters or phase",
    )
    visits = sum(T - d for d in DEPTHS)
    admit = visits - len(DEPTHS) * MIN_AGE
    check(
        hit + w["literal_misses"]
        == w["literal_queries"]
        == w["literal_value_writes"]
        == w["guess_tape_bytes"]
        == w["query_tape_bytes"]
        == w["text_order_zero_updates"]
        == w["mark_storage_bytes"]
        == w["text_mark_storage_bytes"]
        == T
        and hit == w["U_mark_reads"] == w["literal_value_reads"] == w["error_hits"]
        and w["literal_payloads_reassigned"] == 0
        and "S_mark_reads" not in w
        and w["guesses_written"]
        == w["arrivals_encoded"]
        == w["text_arrivals_encoded"]
        == w["mark_norms_from_current"] + w["mark_norms_extra"]
        == T - 1
        and w["wrongs_written"] <= T - 1
        and visits
        == w["text_context_lookups"]
        == w["text_context_updates"]
        == w["text_distinct_contexts"] + w["text_seen_context_lookups"]
        and w["text_distinct_contexts"] <= w["text_distinct_pairs"] <= visits,
        "wrong PC anchor, tape or text work",
    )
    check(
        admit
        == w["error_key_admissions"]
        == w["error_committed_records"] + w["error_peak_keys"]
        and w["error_peak_cells"] <= 2 * w["error_committed_records"]
        and active <= old <= hit
        and old <= w["error_older_occurrence_uses"] <= T * old
        and active
        <= h
        <= w["error_older_mistake_uses"]
        <= w["error_older_occurrence_uses"]
        and h
        <= w["nonquery_alternative_uses_on_active"]
        <= w["error_older_mistake_uses"]
        and h + w["matched_correct_uses_on_active"] <= w["error_older_occurrence_uses"]
        and 2 * active <= entries <= min(256 * active, w["error_packed_cells"])
        and old
        <= w["error_packed_cells"]
        == w["error_packed_feature_entries"]
        <= 256 * old
        and w["error_fp64_feature_bytes"] == 48 * w["error_packed_cells"]
        and w["error_packed_bytes"] == 5 * T + 24 * w["error_packed_cells"]
        and w["marginal_feature_bytes"] == 8 * entries
        and w["pair_feature_bytes"] == 16 * entries
        and w["panel_feature_bytes"] == 24 * entries,
        "wrong PC selected-key geometry, marginal support or feature sizes",
    )
    for k in ("pair", "shuf"):
        e = {name[len(k) + 1 :]: v for name, v in w.items() if name.startswith(k + "_")}
        check(
            all(
                e[name] == w["error_" + name]
                for name in ("key_admissions", "committed_records", "peak_keys")
            )
            and active == e["active_rows"]
            and h == e["matched_mistake_uses"]
            and e["mistake_commits"]
            <= min(e["committed_records"], 4 * w["wrongs_written"])
            and e["peak_guess_rows"]
            <= e["final_pair_cells"]
            <= e["peak_pair_cells"]
            <= e["mistake_commits"]
            and active <= e["packed_cells"] <= min(h, entries - active)
            and e["packed_bytes"] == 5 * T + 6 * e["packed_cells"],
            "wrong PC sparse conditional table or sample size",
        )
    attempts = w["shuf_attempted_exchanges"]
    legal = w["shuf_legal_exchanges"]
    mc = w["pair_mistake_commits"]
    changed = w["changed_conditional_rows"]
    uses = w["changed_conditional_uses"]
    check(
        w["pair_final_pair_cells"] == w["pair_peak_pair_cells"]
        and w["shuf_mistake_commits"] == mc
        and w["pair_peak_guess_rows"] == w["shuf_peak_guess_rows"]
        and legal <= attempts
        and 0 <= mc - 2 * attempts <= w["shuf_peak_loose_keys"] <= w["shuf_peak_keys"]
        and (legal > 0 or uses == 0)
        and changed <= uses <= h
        and bool(changed) == bool(uses)
        and max(w["true_nonmarginal_rows"], w["shuf_nonmarginal_rows"], changed)
        <= active
        and changed <= w["true_nonmarginal_rows"] + w["shuf_nonmarginal_rows"],
        "incorrect PC exchange or changed conditional support",
    )
    sec = r["seconds"]
    needed = CLOCKS | (
        {"other_anchors_s", *(a + "_response_s" for a in (*DIMS, "SWAP"))}
        if probe
        else {"fit_statistics_s"}
    )
    check(
        set(sec) == needed
        and all(isfinite(v) and v >= 0 for v in sec.values())
        and sec["shared_posthost_ERROR_s"] >= fsum(sec[k] for k in SUB),
        "wrong PC clocks or phase",
    )
    if probe:
        z = r["nll_sums"]
        check(
            set(z) == set(ARMS)
            and all(isfinite(v) and v >= 0 for v in z.values())
            and (active or all(z[a] == z["ERROR"] for a in (*DIMS, "SWAP")))
            and (
                changed or isclose(z["PAIR"], z["SWAP"], rel_tol=5e-14, abs_tol=5e-14)
            ),
            "incorrect PC all-target answers or null control",
        )


def support(w, right):
    return dict(
        hits=w["literal_hits"],
        misses=w["literal_misses"],
        copy_correct=right,
        singleton=w["literal_hits"] - w["error_active_rows"],
        older_history=w["error_active_rows"],
        older_mistake=w["error_older_mistake_uses"],
        matched_query=w["matched_query_rows"],
        nonmarginal=w["true_nonmarginal_rows"],
        legal_exchange=w["shuf_legal_exchanges"],
        changed_conditional=w["changed_conditional_rows"],
    )


def summarize(role, logs, hashes):
    z = hist.full_summary(
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
    if role == "probe":
        price = {}
        nt = z["real_targets_per_seed"]
        for s in SEEDS:
            clocks = [r["seconds"] for r in logs[s]]
            sums = {k: fsum(c[k] for c in clocks) for k in clocks[0]}
            shared = sums["prefill_s"] + sums["shared_posthost_ERROR_s"]
            m = sums["marginal_s"] + sums["MARG_response_s"]
            p = (
                sums["marginal_s"]
                + sums["true_pair_index_s"]
                + sums["true_pair_feature_s"]
                + sums["PAIR_response_s"]
            )
            check(shared > 0, "missing shared ERROR time")
            price[str(s)] = dict(
                shared_ERROR_s=shared,
                MARG_extra_s=m,
                PAIR_extra_s=p,
                MARG_extra_s_per_million=m * 1e6 / nt,
                PAIR_extra_s_per_million=p * 1e6 / nt,
                shared_plus_PAIR_over_ERROR=(shared + p) / shared,
                shared_plus_PAIR_over_MARG=(shared + p) / (shared + m),
            )
        z["probe_path_estimates_by_seed"] = price
    return z


def decision(cs):
    pos = {
        k: v["gain"] > 0 and v["ci95"][0] > 0 and all(b > 0 for b in v["by_seed"])
        for k, v in cs.items()
    }
    success = all(pos["PAIR:" + a] for a in ("MARG", "SHUF", "SWAP", "ERROR"))
    return dict(
        positive_checks=pos,
        useful_queried_pair_table=success,
        useful_marginal_trigger=pos["MARG:ERROR"],
        consider_locked_pair_transfer=success,
        scope="new native TRAIN only; no default, final or scale decision",
    )


def analyze(root):
    z = common.analyze(root, study="PC")
    z["decision"] = decision(z["stages"]["probe"]["contrasts"])
    return z
