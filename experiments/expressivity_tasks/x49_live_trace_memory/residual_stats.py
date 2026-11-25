"""All real next bytes and conditional wrong-code controls; only closed document aggregates."""

from math import isfinite
import numpy as np
from ..x48_muon_continuation.data import check
from . import decode_stats as dr, surprise_analysis as common
from .residual_byte import ARMS
from .panels import T, SEEDS

PAIRS = tuple(
    tuple(v) for v in ("RD", "RG", "RS", "JR", "RU", "RB", "JD", "DU", "SG", "GD", "JU")
)
BOOT = 4914
COMMON = {
    "prefill_s",
    "full_vector_write_s",
    "text_build_s",
    "neural_build_s",
    "coarse_write_and_subtract_s",
    "encode_byte_s",
}
PROBE = {
    "address_and_code_s",
    "direct_s",
    "current_host_field_s",
    "full_vector_read_s",
    "bound_code_read_s",
    "coarse_decode_s",
    "sham_index_and_read_s",
    "read_diagnostics_s",
    *(a + "_residual_add_s" for a in "RGS"),
    *(a + k for a in "BDGRS" for k in ("_head_s", "_trust_s")),
}


def _hist(a, total):
    check(
        type(a) is list
        and len(a) == 256
        and all(type(v) is int and v >= 0 for v in a)
        and sum(a) == total,
        "wrong RB code histogram or exposure",
    )


def checked_row(role, x):
    probe = role == "probe"
    w = x["work"]
    times = x["seconds"]
    visits = sum(T - i for i in (1, 2, 4, 8))
    check(
        role in ("fit", "probe")
        and x["targets"] == T
        and x["host_prefills"] == 1
        and all(
            (k in x) == probe
            for k in (
                "nll_sums",
                "squared_error_sums",
                "candidate_correct",
                "read_code_counts",
            )
        )
        and ("fit_code_counts" in x) != probe
        and ("fit_writes" in w) != probe,
        "wrong RB phase or exposure",
    )
    check(
        set(times) == COMMON | (PROBE if probe else {"fit_statistics_s"})
        and all(isfinite(v) and v >= 0 for v in times.values()),
        "wrong RB timing",
    )
    check(
        all(type(v) is int and v >= 0 for v in w.values())
        and w["text_order_zero_updates"]
        == w["text_mark_storage_bytes"]
        == w["neural_storage_bytes"]
        == w["residual_storage_bytes"]
        == w["full_value_computations"]
        == T
        and w["encoded_residuals"]
        == w["neural_arrivals_encoded"]
        == w["text_arrivals_encoded"]
        == w["neural_norms_extra"] + w["neural_norms_reused"]
        == T - 1
        and max(w["neural_marks_saturated"], w["text_marks_saturated"]) <= T - 1
        and w["D_write_clipped_components"] <= 32 * (T - 1)
        and w["text_context_lookups"]
        == w["text_context_updates"]
        == visits
        == w["text_seen_context_lookups"] + w["text_distinct_contexts"]
        and w["text_distinct_contexts"] <= w["text_distinct_pairs"] <= visits,
        "invalid RB tape or write work",
    )
    if not probe:
        check(
            w["fit_writes"] == T - 1
            and w["neural_norms_reused"] == 0
            and not any(
                k in w
                for k in ("literal_queries", "mse_components", "true_residual_reads")
            ),
            "future hit selection in RB FIT",
        )
        _hist(x["fit_code_counts"], T - 1)
        return
    hits = w["literal_hits"]
    components = 32 * hits
    ss = x["squared_error_sums"]
    loss = x["nll_sums"]
    hist = x["read_code_counts"]
    right = x["candidate_correct"]
    check(
        hits + w["literal_misses"] == T == w["literal_queries"]
        and hits
        == w["full_value_reads"]
        == w["true_residual_reads"]
        == w["sham_residual_reads"]
        and w["mse_components"] == components
        and w["literal_payloads_reassigned"] == 0
        and w["sham_code_changed"] <= w["sham_displaced"] <= hits
        and w["sham_index_triples"] <= w["sham_index_inserts"] <= T - 17
        and all(w[k + "_unseen_code_reads"] <= hits for k in ("true", "sham"))
        and type(right) is int
        and 0 <= right <= hits
        and max(w[k + "_clipped_components"] for k in ("D_read", "G", "R", "S"))
        <= components,
        "invalid RB read or clipping work",
    )
    check(set(hist) == {"true", "sham"}, "foreign RB read histogram")
    _hist(hist["true"], hits)
    _hist(hist["sham"], hits)
    check(
        sum(abs(a - b) for a, b in zip(hist["true"], hist["sham"], strict=True))
        <= 2 * w["sham_code_changed"],
        "RB changed-code count incompatible with histograms",
    )
    check(
        set(ss) == set("DGRS")
        and all(isfinite(v) and 0 <= v <= 4 * components for v in ss.values())
        and set(loss) == set(ARMS)
        and all(isfinite(v) and v >= 0 for v in loss.values())
        and (hits or len(set(loss.values())) == 1)
        and (w["sham_code_changed"] or (loss["S"] == loss["R"] and ss["S"] == ss["R"]))
        and (w["true_unseen_code_reads"] < hits or loss["R"] == loss["G"]),
        "wrong RB MSE, all-target sums or fallback",
    )


def summarize(role, logs, hashes):
    z = dr.table_summary(
        role,
        logs,
        hashes,
        check_row=checked_row,
        arms=ARMS,
        pairs=PAIRS,
        seed=BOOT,
        mse="DGRS",
        geometry=(
            "displaced_slot",
            "sham_displaced",
            "changed_code",
            "sham_code_changed",
        ),
    )
    z["occupancy_by_seed"] = {}
    for s in SEEDS:
        k = str(s)
        occ = {}
        for label in ("fit",) if role == "fit" else ("true", "sham"):
            n = np.sum(
                [
                    x["fit_code_counts"]
                    if label == "fit"
                    else x["read_code_counts"][label]
                    for x in logs[s]
                ],
                axis=0,
                dtype=np.int64,
            )
            occ[label] = dict(
                count=int(n.sum()),
                used_codes=int((n > 0).sum()),
                singleton_codes=int((n == 1).sum()),
                max_count=int(n.max()),
            )
        z["occupancy_by_seed"][k] = occ
        if role == "probe":
            z["peak_table_entries_by_seed"][k].update(
                {
                    key: max(x["work"][key] for x in logs[s])
                    for key in ("sham_index_inserts", "sham_index_triples")
                }
            )
    return z


def decision(cs):
    pos = {
        k: v["gain"] > 0 and v["ci95"][0] > 0 and all(x > 0 for x in v["by_seed"])
        for k, v in cs.items()
    }
    useful = all(pos[k] for k in ("R:D", "R:G", "R:S"))
    return dict(
        positive_checks=pos,
        useful_bound_residual_byte=useful,
        consider_locked_calibration_transfer=useful and pos["R:U"],
        full_vector_advantage=pos["J:R"],
        scope="native TRAIN only; no adoption, equivalence, final or scale decision",
    )


def analyze(root):
    z = common.analyze(root, study="RB")
    z["decision"] = decision(z["stages"]["probe"]["contrasts"])
    z["explicit_historical_bytes_per_input"] = dict(
        B=128, U=129, J=130, D=2, G=2, R=3, S=3
    )
    z["installed_R_bytes_per_checkpoint"] = dict(
        coarse_decoder=100352, axes=1024, residual_table=32768, total=134144
    )
    return z
