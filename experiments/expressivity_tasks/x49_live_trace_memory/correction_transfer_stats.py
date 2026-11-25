"""PT short-document inference. Count every byte, not only queries with a visible pair-table difference."""

from collections import Counter
from math import fsum, isclose, isfinite
from ..x48_muon_continuation.data import check
from . import paired_correction_stats as pc, error_history_stats as hist
from .correction_transfer_score import ARMS as ARMS, BRANCHES, TV_TOL, order
from .literal import MIN_AGE
from .panels import SEEDS, T

BOOT = 4918
PAIRS = (*pc.PAIRS, ("PAIR", "PREFIX"), ("PREFIX", "MARG"), ("PREFIX", "ERROR"))
SHARED = pc.SUB | {
    "prefill_s",
    "shared_posthost_ERROR_s",
    "shared_ERROR_s",
    "comparison_only_s",
}
COMPONENTS = {
    "MARG": ("marginal", "response"),
    "PAIR": ("marginal", "index", "conditional", "response"),
    "PREFIX": ("marginal", "prefix_scan", "conditional", "response"),
}
CLOCKS = SHARED | {
    a + "_" + t + "_s" for a, ks in COMPONENTS.items() for t in (*ks, "branch")
}
PEAKS = tuple(
    k
    for k in pc.PEAKS
    if k not in ("marginal_feature_bytes", "pair_feature_bytes", "panel_feature_bytes")
) + (
    "shuf_feature_bytes",
    "prefix_peak_pair_cells",
    "prefix_peak_guess_rows",
    "prefix_peak_nonempty_actual_columns",
    "prefix_rate_bytes",
)
MARG = (
    "marginal_entries",
    "matched_query_rows",
    "matched_mistake_uses",
    "matched_correct_uses_on_active",
    "nonquery_alternative_uses_on_active",
)


def checked_row(role, r):
    n = r["targets"]
    w = r["work"]
    cw = r["chosen_work"]
    sec = r["seconds"]
    loss = r["nll_sums"]
    active = w["matched_query_rows"]
    e = w["marginal_entries"]
    tv = r["conditional_tv_sum"]
    changed = w["prefix_different_law_rows"]
    check(
        role == "probe"
        and 1 <= n < T
        and r["host_prefills"] == 1
        and set(loss) == set(ARMS)
        and all(isfinite(v) and v >= 0 for v in loss.values())
        and set(cw) == set(BRANCHES)
        and sorted(r["branch_order"]) == sorted(BRANCHES),
        "wrong PT short population, order or scores",
    )
    pc.checked_pair_work(w, n, r["candidate_correct"])
    check(
        (active or all(loss[a] == loss["ERROR"] for a in (*BRANCHES, "SHUF", "SWAP")))
        and (
            w["changed_conditional_rows"]
            or isclose(loss["PAIR"], loss["SWAP"], abs_tol=5e-14, rel_tol=5e-14)
        )
        and (w["literal_hits"] or len(set(loss.values())) == 1),
        "changed PT inactive rows or fixed null exchange",
    )
    writes = w["prefix_mistake_writes"]
    admit = w["prefix_admitted_records"]
    cells = w["prefix_peak_pair_cells"]
    look = w["prefix_alternative_bin_lookups"]
    check(
        admit == max(0, n - 1 - MIN_AGE)
        and 0 <= w["wrongs_written"] - writes <= n - 1 - admit
        and writes <= min(w["wrongs_written"], admit)
        and max(w["prefix_peak_guess_rows"], w["prefix_peak_nonempty_actual_columns"])
        <= min(cells, 256)
        and cells
        <= min(
            writes,
            255 * 256,
            w["prefix_peak_guess_rows"] * w["prefix_peak_nonempty_actual_columns"],
        )
        and w["prefix_active_rows"] == active
        and w["prefix_discounted_edge_reads"]
        <= w["prefix_discounted_column_reads"]
        <= w["prefix_latest_mistake_discounts"]
        <= active
        and active
        <= w["pair_packed_cells"]
        <= w["prefix_nonzero_queried_bins"]
        <= look
        == e - active
        and w["prefix_rate_bytes"] == 8 * e
        and w["shuf_feature_bytes"] == 16 * e,
        "incorrect PREFIX writes, discount, state or returned entries",
    )
    # A row counted as unchanged can have FP roundoff, and the tolerance is descriptive only.
    check(
        isfinite(tv)
        and 0 <= changed <= active
        and changed * TV_TOL
        <= tv
        <= changed + (active - changed) * TV_TOL + 2e-12 * active,
        "incorrect conditional distance or affected-query count",
    )
    check(
        set(sec) == CLOCKS
        and all(isfinite(v) and v >= 0 for v in sec.values())
        and sec["shared_ERROR_s"] >= sec["prefill_s"] + sec["shared_posthost_ERROR_s"]
        and sec["shared_posthost_ERROR_s"] >= fsum(sec[k] for k in pc.SUB) > 0
        and sec["comparison_only_s"] > 0,
        "incorrect PT shared or control clocks",
    )
    for a in BRANCHES:
        prefixes = set(MARG)
        if a == "PAIR":
            prefixes.update(k for k in w if k.startswith("pair_"))
            prefixes.add("true_nonmarginal_rows")
        elif a == "PREFIX":
            prefixes.update(
                k
                for k in w
                if k.startswith("prefix_") and k != "prefix_different_law_rows"
            )
        c = cw[a]
        check(
            set(c) == prefixes | {"marginal_feature_bytes", a + "_feature_bytes"}
            and all(type(v) is int and v >= 0 for v in c.values())
            and all(c[k] == w[k] for k in prefixes)
            and c["marginal_feature_bytes"] == 8 * e
            and c[a + "_feature_bytes"] == (8 if a == "MARG" else 16) * e,
            "PT chosen branch used foreign work or feature storage",
        )
        check(
            sec[a + "_branch_s"]
            >= fsum(sec[a + "_" + k + "_s"] for k in COMPONENTS[a])
            > 0,
            "incomplete PT chosen branch clock",
        )


def support(w, right):
    return dict(
        pc.support(w, right),
        prefix_different_law=w["prefix_different_law_rows"],
        prefix_discount=w["prefix_latest_mistake_discounts"],
    )


def summarize(logs, hashes, sizes):
    z = hist.full_summary(
        "probe",
        logs,
        hashes,
        sizes=sizes,
        validator=checked_row,
        support_counts=support,
        peaks_to_keep=PEAKS,
        same=pc.SAME,
        arms=ARMS,
        pairs=PAIRS,
        boot=BOOT,
    )
    nt = z["real_targets_per_seed"]
    cost = {}
    clocks = {}
    feature = {}
    tv = {}
    for s in SEEDS:
        rs = logs[s]
        k = str(s)
        check(
            all(tuple(r["branch_order"]) == order(s, i) for i, r in enumerate(rs)),
            "incorrect PT branch order",
        )
        sec = {t: fsum(r["seconds"][t] for r in rs) for t in CLOCKS}
        shared = sec["shared_ERROR_s"]
        names = Counter("_".join(r["branch_order"]) for r in rs)
        c = dict(
            shared_ERROR_s=shared,
            shared_ERROR_s_per_million=shared * 1e6 / nt,
            comparison_s_per_million=sec["comparison_only_s"] * 1e6 / nt,
            order_counts=dict(names),
        )
        peaks = {}
        for a in BRANCHES:
            b = sec[a + "_branch_s"]
            c[a + "_extra_s_per_million"] = b * 1e6 / nt
            c["shared_plus_" + a + "_s"] = shared + b
            c[a + "_over_ERROR"] = 1 + b / shared
            peaks[a] = max(r["chosen_work"][a][a + "_feature_bytes"] for r in rs)
        for a, b in [("PAIR", "MARG"), ("PREFIX", "MARG"), ("PAIR", "PREFIX")]:
            c[a + "_over_" + b] = (
                c["shared_plus_" + a + "_s"] / c["shared_plus_" + b + "_s"]
            )
        clocks[k] = sec
        cost[k] = c
        feature[k] = peaks
        tv[k] = fsum(r["conditional_tv_sum"] for r in rs)
    z.update(
        seconds_by_seed=clocks,
        branch_cost_by_seed=cost,
        peak_chosen_feature_bytes_by_seed=feature,
        conditional_tv_sum_by_seed=tv,
    )
    return z


def decision(stats):
    cs = stats["contrasts"]
    d = pc.decision(cs)
    pos = d["positive_checks"]
    x = cs["PAIR:PREFIX"]
    neg = x["gain"] < 0 and x["ci95"][1] < 0 and all(v < 0 for v in x["by_seed"])
    winner = "unresolved"
    if pos["PAIR:PREFIX"]:
        winner = "PAIR"
    elif neg:
        winner = "PREFIX"
    return dict(
        positive_checks=pos,
        replicated_queried_pair_table=d["useful_queried_pair_table"],
        replicated_marginal_trigger=d["useful_marginal_trigger"],
        prefix_better_than_marg_and_error=pos["PREFIX:MARG"] and pos["PREFIX:ERROR"],
        pair_vs_prefix=winner,
        scope="locked short native calibration, not a final, default change or scale authority",
    )
