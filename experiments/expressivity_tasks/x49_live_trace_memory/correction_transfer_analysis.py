"""PT receipt-only close: replay variable target support, chosen paths, all comparisons and actual weight bindings."""

import json
import os
from math import fsum, isclose
from ..x48_muon_continuation.data import check, sha
from . import (
    confirm_analysis as prior,
    correction_transfer_panel as p,
    correction_transfer_data as data,
    correction_transfer_stats as stats,
)
from .correction_transfer_stats import (
    BOOT as BOOT,
    decision as decision,
    summarize as summarize,
)
from .panels import SEEDS


def analyze(root):
    return prior.analyze(root, study="PT")


def audit(rows, stage, s):
    n = sum(r["targets"] for r in rows)
    check(
        n == stage["real_targets_per_seed"] and len(rows) == stage["paired_documents"],
        "wrong PT denominator",
    )
    q = stage["branch_cost_by_seed"][s]
    clocks = stage["seconds_by_seed"][s]
    w = stage["work_by_seed"][s]
    docs = stage["documents_with_support_by_seed"][s]
    for key, label in [
        ("matched_query_rows", "matched_query"),
        ("true_nonmarginal_rows", "nonmarginal"),
        ("shuf_legal_exchanges", "legal_exchange"),
        ("changed_conditional_rows", "changed_conditional"),
        ("prefix_different_law_rows", "prefix_different_law"),
        ("prefix_latest_mistake_discounts", "prefix_discount"),
    ]:
        counts = [r["work"][key] for r in rows]
        check(
            sum(counts) == w[key] and sum(v > 0 for v in counts) == docs[label],
            "incorrect PT affected-query or document count",
        )
    check(
        all(
            w[k] == sum(r["work"][k] for r in rows)
            for k in (
                "changed_conditional_uses",
                "prefix_mistake_writes",
                "prefix_nonzero_queried_bins",
                "prefix_discounted_edge_reads",
            )
        )
        and w["candidate_correct"] == sum(r["candidate_correct"] for r in rows)
        and isclose(
            sum(r["conditional_tv_sum"] for r in rows),
            stage["conditional_tv_sum_by_seed"][s],
            abs_tol=2e-12,
            rel_tol=5e-13,
        ),
        "incorrect PT conditional mass",
    )
    check(
        {k: max(r["work"][k] for r in rows) for k in stats.PEAKS}
        == stage["peak_per_window_by_seed"][s]
        and {
            a: max(r["chosen_work"][a][a + "_feature_bytes"] for r in rows)
            for a in stats.BRANCHES
        }
        == stage["peak_chosen_feature_bytes_by_seed"][s],
        "incorrect PT peak state",
    )

    def eq(x, y):
        return isclose(x, y, abs_tol=2e-12, rel_tol=5e-13)

    check(
        set(clocks) == stats.CLOCKS
        and all(eq(sum(r["seconds"][k] for r in rows), v) for k, v in clocks.items()),
        "incorrect PT clock aggregation",
    )
    e = fsum(r["seconds"]["shared_ERROR_s"] for r in rows)
    v = dict(
        shared_ERROR_s=e,
        shared_ERROR_s_per_million=1e6 * e / n,
        comparison_s_per_million=1e6
        * fsum(r["seconds"]["comparison_only_s"] for r in rows)
        / n,
    )
    for a in stats.BRANCHES:
        total = fsum(
            r["seconds"]["shared_ERROR_s"] + r["seconds"][a + "_branch_s"] for r in rows
        )
        v["shared_plus_" + a + "_s"] = total
        v[a + "_extra_s_per_million"] = 1e6 * (total - e) / n
        v[a + "_over_ERROR"] = total / e
    for a, b in [("PAIR", "MARG"), ("PREFIX", "MARG"), ("PAIR", "PREFIX")]:
        v[a + "_over_" + b] = (
            v["shared_plus_" + a + "_s"] / v["shared_plus_" + b + "_s"]
        )
    o = {}
    for i, r in enumerate(rows):
        seq = stats.order(int(s), i)
        check(seq == tuple(r["branch_order"]), "foreign PT branch order")
        name = "_".join(seq)
        o[name] = o.get(name, 0) + 1
    check(
        set(q) == set(v) | {"order_counts"}
        and o == q["order_counts"]
        and all(eq(y, q[k]) for k, y in v.items()),
        "incorrect PT shared-path price",
    )


def report():
    check(os.getenv("OPENBLAS_NUM_THREADS") == "1", "one BLAS thread for PT close")
    r = prior.report(study="PT")
    d = p.ROOT / p.JOB.name
    rows, nt = data.checked_rows(
        data.u.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    )
    hs = [x["sha256"] for x in rows]
    sizes = [x["length"] - 1 for x in rows]
    for s in SEEDS:
        audit(
            p.prior.journal(d / f"{s}.jsonl", hs, sizes=sizes)[1], r["analysis"], str(s)
        )
    check(
        nt == r["analysis"]["real_targets_per_seed"]
        and r["pc_fit_freeze_sha"] == data.FIT_SHA
        and r["pc_result_sha"] == data.reader.PUBLIC
        and r["pc_close_sha"] == data.reader.CLOSED
        and r["swap_fit"] == r["prefix_fit"] == "PAIR"
        and r["metadata_reservation"]["unreserved"] == 0
        and not (d / "windows.u8").exists(),
        "foreign or incomplete PT close",
    )
    r["qualification_sha"] = p.qualification_sha(r["ref"])
    r["verified"]["independent_prefix_support_and_branch_cost"] = True
    return r


if __name__ == "__main__":
    os.umask(0o077)
    r = report()
    f = p.ROOT.parent / "correction-transfer-result-1.json"
    with f.open("x") as out:
        json.dump(r, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print("X49-PT verified aggregate:", sha(f), f.stat().st_size, "bytes", flush=True)
