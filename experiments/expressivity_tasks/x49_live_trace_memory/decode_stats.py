"""DR: audit observed FIT writes separately from scored bytes. No payload/teacher access."""

from collections import Counter
from math import isfinite
import numpy as np
from ..x48_muon_continuation.data import check
from . import surprise_analysis as common
from .decode_record import ARMS
from .decode_fit import VALUE
from .panels import SEEDS, T

BOOT = 4913
PAIRS = tuple(tuple(x) for x in ("DP", "DS", "JP", "JD", "DU", "DB", "JU", "UB", "SP"))
COMMON_CLOCKS = {"prefill_s", "text_build_s", "neural_build_s", "full_vector_write_s"}
PROBE_CLOCKS = {
    "address_and_code_s",
    "direct_s",
    "current_host_field_s",
    "true_note_reads_s",
    "full_vector_read_s",
    "prototype_read_s",
    "decode_true_s",
    "decoder_sham_search_read_s",
    "decode_sham_s",
    *(a + s for a in "BPDS" for s in ("_head_s", "_trust_s")),
}


def checked_row(role, x):
    w = x["work"]
    times = x["seconds"]
    probe = role == "probe"
    check(
        role in ("fit", "probe")
        and x["targets"] == T
        and x["host_prefills"] == 1
        and all(
            (k in x) == probe
            for k in ("nll_sums", "squared_error_sums", "candidate_correct")
        )
        and ("fit_writes" in w) != probe,
        "wrong DR phase or exposure",
    )
    check(
        set(times) == COMMON_CLOCKS | (PROBE_CLOCKS if probe else {"fit_statistics_s"})
        and all(isfinite(v) and v >= 0 for v in times.values()),
        "incomplete DR timings",
    )
    visits = sum(T - i for i in (1, 2, 4, 8))
    check(
        all(type(v) is int and v >= 0 for v in w.values())
        and w["text_order_zero_updates"]
        == w["text_mark_storage_bytes"]
        == w["neural_storage_bytes"]
        == w["full_value_writes"]
        == T
        and w["neural_arrivals_encoded"]
        == w["text_arrivals_encoded"]
        == w["neural_norms_reused"] + w["neural_norms_extra"]
        == T - 1
        and max(w["neural_marks_saturated"], w["text_marks_saturated"]) <= T - 1
        and w["text_context_lookups"]
        == w["text_context_updates"]
        == visits
        == w["text_seen_context_lookups"] + w["text_distinct_contexts"]
        and w["text_distinct_contexts"] <= w["text_distinct_pairs"] <= visits,
        "invalid teacher/tape work",
    )
    if not probe:
        check(
            w["fit_writes"] == T - 1
            and w["neural_norms_reused"] == 0
            and not any(
                k in w
                for k in ("literal_queries", "full_value_reads", "mse_components")
            ),
            "invalid DR FIT write denominator or future selection",
        )
        return
    hit = w["literal_hits"]
    nc = hit * VALUE
    ss = x["squared_error_sums"]
    loss = x["nll_sums"]
    right = x["candidate_correct"]
    check(
        hit + w["literal_misses"] == T == w["literal_queries"]
        and hit == w["compact_note_pairs_read"] == w["full_value_reads"]
        and w["mse_components"] == nc
        and w["literal_payloads_reassigned"] == 0
        and w["sham_pairs_changed"] <= w["sham_positions_displaced"] <= hit
        and type(right) is int
        and 0 <= right <= hit,
        "invalid DR hit/slot/target conservation",
    )
    check(
        set(ss) == set("PDS")
        and all(isfinite(v) and 0 <= v <= 4 * nc for v in ss.values())
        and set(loss) == set(ARMS)
        and all(isfinite(v) and v >= 0 for v in loss.values())
        and (hit or len(set(loss.values())) == 1)
        and (
            w["sham_pairs_changed"] or (loss["S"] == loss["D"] and ss["S"] == ss["D"])
        ),
        "invalid DR reconstruction diagnostics or all-byte sums",
    )


def summarize(role, logs, hashes):
    return table_summary(
        role,
        logs,
        hashes,
        check_row=checked_row,
        arms=ARMS,
        pairs=PAIRS,
        seed=BOOT,
        mse="PDS",
        geometry=(
            "displaced_slot",
            "sham_positions_displaced",
            "changed_pair",
            "sham_pairs_changed",
        ),
    )


def table_summary(role, logs, hashes, *, check_row, arms, pairs, seed, mse, geometry):
    """Shared target and teacher-component denominators for fixed-table comparisons."""
    displaced_label, displaced, changed_label, changed = geometry
    n = len(hashes)
    probe = role == "probe"
    check(
        n == len(set(hashes)) and n > 0 and set(logs) == set(SEEDS),
        "wrong paired DR population",
    )
    stage = dict(
        paired_documents=n,
        real_input_positions_per_seed=n * T,
        work_by_seed={},
        documents_with_support_by_seed={},
        peak_table_entries_by_seed={},
    )
    if probe:
        stage.update(real_targets_per_seed=n * T, hit_component_MSE_by_seed={})
    else:
        stage["teacher_fit_writes_per_seed"] = n * (T - 1)
    clocks = Counter()
    loss = np.zeros((len(SEEDS), n, len(arms))) if probe else None
    for si, s in enumerate(SEEDS):
        k = str(s)
        work = Counter()
        ds = Counter()
        ss = Counter()
        for i, (h, x) in enumerate(zip(hashes, logs[s], strict=True)):
            check_row(role, x)
            w = x["work"]
            ref = logs[SEEDS[0]][i]
            check(
                x["document_sha"] == h
                and all(
                    ref["work"][a] == v
                    for a, v in w.items()
                    if a.startswith(("text_", "literal_"))
                    or a
                    in (
                        "sham_positions_displaced",
                        "fit_writes",
                        "full_value_reads",
                        "neural_norms_reused",
                    )
                ),
                "DR address, literal work or document changed with weight",
            )
            work.update(w)
            clocks.update(x["seconds"])
            support = dict(
                saturated_text=w["text_marks_saturated"],
                saturated_neural=w["neural_marks_saturated"],
            )
            if probe:
                check(
                    x["candidate_correct"] == ref["candidate_correct"],
                    "wrong same-byte copy target",
                )
                work.update(candidate_correct=x["candidate_correct"])
                support.update(
                    hits=w["literal_hits"],
                    misses=w["literal_misses"],
                    copy_correct=x["candidate_correct"],
                    **{displaced_label: w[displaced], changed_label: w[changed]},
                )
                ss.update(x["squared_error_sums"])
                loss[si, i, :] = [x["nll_sums"][a] for a in arms]
            ds.update({a: int(v > 0) for a, v in support.items()})
        stage["work_by_seed"][k] = dict(work)
        stage["documents_with_support_by_seed"][k] = dict(ds)
        stage["peak_table_entries_by_seed"][k] = {
            a: max(x["work"][a] for x in logs[s])
            for a in ("text_distinct_contexts", "text_distinct_pairs")
        }
        if probe:
            stage["hit_component_MSE_by_seed"][k] = {
                a: ss[a] / work["mse_components"] if work["mse_components"] else None
                for a in mse
            }
    stage["mean_seconds_per_seed_document"] = {
        a: v / (n * len(SEEDS)) for a, v in clocks.items()
    }
    if probe:
        stage.update(common.paired_table(loss, [T] * n, arms, pairs, seed))
    return stage


def decision(cs):
    ok = {
        k: v["ci95"][0] > 0 and v["gain"] > 0 and all(x > 0 for x in v["by_seed"])
        for k, v in cs.items()
    }
    reconstructed = ok["D:P"] and ok["D:S"]
    return dict(
        positive_checks=ok,
        prediction_relevant_reconstruction=reconstructed,
        full_vector_advantage=ok["J:D"],
        consider_lower_payload_evaluation=reconstructed and ok["D:U"],
        scope="TRAIN diagnostic only; B baseline, U confirmed, J replicated on calibration",
    )


def analyze(root):
    r = common.analyze(root, study="DR")
    r["decision"] = decision(r["stages"]["probe"]["contrasts"])
    r["explicit_historical_bytes_per_input"] = dict(B=128, U=129, J=130, P=2, D=2, S=2)
    r["installed_decoder_bytes_per_checkpoint"] = 100352
    r["prototype_table_bytes_per_checkpoint"] = 32768
    return r
