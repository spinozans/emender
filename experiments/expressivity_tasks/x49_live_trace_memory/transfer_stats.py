"""JT: paired whole-document losses and measured shared/branch work, not a standalone serving clock."""

from __future__ import annotations
from collections import Counter
from math import isfinite
import numpy as np
from ..x48_muon_continuation.data import check
from .literal_mark import DEPTHS
from .panels import SEEDS, T
from .surprise_analysis import paired_table
from .transfer_score import first

ARMS = "BUHJ"
PAIRS = (("J", "U"), ("J", "H"), ("J", "B"), ("U", "B"), ("H", "U"))
BOOT = 4912
CLOCKS = {
    "prefill_s",
    "base_B_s",
    "shared_current_s",
    "shared_US_mark_build_s",
    "U_mark_read_s",
    "shared_s",
    "U_branch_s",
    "J_branch_s",
    "text_build_s",
    "H_control_s",
}


def decision(z):
    cs = z["contrasts"]
    ok = {
        k: cs[k]["gain"] > 0
        and cs[k]["ci95"][0] > 0
        and all(v > 0 for v in cs[k]["by_seed"])
        for k in ("J:U", "J:H")
    }
    return dict(
        checks=ok,
        replicated_fixed_joint_candidate=all(ok.values()),
        scope="locked short native calibration transfer only; U remains the native-final-confirmed reader",
    )


def summarize(logs, hashes, sizes):
    n = len(sizes)
    nt = sum(sizes)
    check(
        n == len(hashes) == len(set(hashes))
        and n > 0
        and all(type(v) is int and 1 <= v < T for v in sizes)
        and set(logs) == set(SEEDS),
        "invalid JT population",
    )
    losses = np.zeros((len(SEEDS), n, len(ARMS)))
    work = {}
    support = {}
    clocks = {}
    peaks = {}
    cost = {}
    for si, s in enumerate(SEEDS):
        k = str(s)
        total = Counter()
        docs = Counter()
        sec = Counter()
        peak = Counter()
        orders = Counter()
        for i, (h, x, length) in enumerate(zip(hashes, logs[s], sizes, strict=True)):
            w = x["work"]
            hit = w["literal_hits"]
            miss = w["literal_misses"]
            right = x["candidate_correct"]
            visits = sum(max(0, length - d) for d in DEPTHS)
            z = x["nll_sums"]
            times = x["seconds"]
            o = x["first_branch"]
            ref = logs[SEEDS[0]][i]
            check(
                x["document_sha"] == h
                and x["targets"] == length
                and x["host_prefills"] == 1
                and o == first(s, i)
                and set(z) == set(ARMS)
                and all(isfinite(v) and v >= 0 for v in z.values())
                and (hit or len(set(z.values())) == 1),
                "wrong JT score pairing, order or miss",
            )
            check(
                all(type(v) is int and v >= 0 for v in w.values())
                and type(right) is int
                and 0 <= right <= hit
                and hit + miss == length
                and w["literal_queries"]
                == w["literal_value_writes"]
                == w["mark_storage_bytes"]
                == w["text_mark_storage_bytes"]
                == w["text_order_zero_updates"]
                == length
                and w["literal_value_reads"] == w["U_mark_reads"] == hit
                and w["literal_payloads_reassigned"] == 0
                and not any(key.startswith("S_") for key in w)
                and w["text_arrivals_encoded"]
                == w["arrivals_encoded"]
                == w["mark_norms_extra"] + w["mark_norms_from_current"]
                == length - 1
                and w["text_marks_saturated"] <= length - 1
                and w["marks_saturated"] <= length - 1
                and w["text_context_lookups"] == w["text_context_updates"] == visits
                and w["text_distinct_contexts"] + w["text_seen_context_lookups"]
                == visits
                and w["text_distinct_contexts"] <= w["text_distinct_pairs"] <= visits,
                "unbound JT marks, tables or targets",
            )
            check(
                right == ref["candidate_correct"]
                and all(
                    ref["work"][a] == v
                    for a, v in w.items()
                    if a.startswith(("literal_", "text_"))
                ),
                "same bytes but different count/address evidence",
            )
            check(
                set(times) == CLOCKS
                and all(isfinite(v) and v >= 0 for v in times.values())
                and times["J_branch_s"] >= times["text_build_s"]
                and times["shared_s"] >= times["prefill_s"],
                "wrong JT cost decomposition",
            )
            losses[si, i] = [z[a] for a in ARMS]
            total.update(w)
            total.update(candidate_correct=right)
            docs.update(
                hits=int(hit > 0),
                misses=int(miss > 0),
                copy_correct=int(right > 0),
                saturated_text_write=int(w["text_marks_saturated"] > 0),
                saturated_neural_write=int(w["marks_saturated"] > 0),
            )
            sec.update(times)
            orders[o] += 1
            for a in ("text_distinct_contexts", "text_distinct_pairs"):
                peak[a] = max(peak[a], w[a])
        share = sec["shared_s"]
        j = sec["J_branch_s"]
        u = sec["U_branch_s"]
        check(share + u > 0, "unmeasured shared path")
        work[k] = dict(total)
        support[k] = dict(docs)
        clocks[k] = dict(sec)
        peaks[k] = dict(peak)
        cost[k] = dict(
            first_counts=dict(orders),
            shared_plus_U_s=share + u,
            shared_plus_J_s=share + j,
            J_minus_U_s_per_million_targets=(j - u) * 1e6 / nt,
            shared_path_J_over_U=(share + j) / (share + u),
        )
    return dict(
        paired_documents=n,
        real_targets_per_seed=nt,
        work_by_seed=work,
        documents_with_support_by_seed=support,
        peak_table_entries_per_window_by_seed=peaks,
        seconds_by_seed=clocks,
        branch_cost_by_seed=cost,
        **paired_table(losses, sizes, ARMS, PAIRS, BOOT),
    )
