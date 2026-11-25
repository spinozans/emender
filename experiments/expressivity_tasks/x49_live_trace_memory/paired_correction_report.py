"""Independent receipt-only PC close: compare whole-document counts, pair support, peaks and clocks."""

import json
import os
from math import isclose, fsum
from ..x48_muon_continuation.data import check, sha
from . import (
    alternative_report as common,
    paired_correction_panel as p,
    paired_correction_data as data,
)
from .panels import SEEDS
from .surprise_panel import journal


def audit(role, rows, stage, s):
    w = [r["work"] for r in rows]
    z = stage["work_by_seed"][s]
    d = stage["documents_with_support_by_seed"][s]
    for field, label in [
        ("matched_query_rows", "matched_query"),
        ("true_nonmarginal_rows", "nonmarginal"),
        ("shuf_legal_exchanges", "legal_exchange"),
        ("changed_conditional_rows", "changed_conditional"),
    ]:
        check(
            z[field] == sum(r[field] for r in w)
            and d[label] == sum(r[field] > 0 for r in w),
            "incorrect PC affected-document reduction",
        )
    check(
        z["candidate_correct"] == sum(r["candidate_correct"] for r in rows)
        and z["changed_conditional_uses"]
        == sum(r["changed_conditional_uses"] for r in w)
        and all(
            v == max(r[k] for r in w)
            for k, v in stage["peak_per_window_by_seed"][s].items()
        ),
        "incorrect PC peaks or queried mass",
    )
    if role == "probe":
        n = sum(r["targets"] for r in rows)
        cs = [r["seconds"] for r in rows]
        err = fsum(c["shared_posthost_ERROR_s"] + c["prefill_s"] for c in cs)
        marg = fsum(c["marginal_s"] + c["MARG_response_s"] for c in cs)
        pair = fsum(
            c["true_pair_index_s"]
            + c["true_pair_feature_s"]
            + c["marginal_s"]
            + c["PAIR_response_s"]
            for c in cs
        )
        cost = stage["probe_path_estimates_by_seed"][s]
        v = dict(
            shared_ERROR_s=err,
            MARG_extra_s=marg,
            PAIR_extra_s=pair,
            PAIR_extra_s_per_million=pair * 1e6 / n,
            MARG_extra_s_per_million=marg * 1e6 / n,
            shared_plus_PAIR_over_ERROR=1 + pair / err,
            shared_plus_PAIR_over_MARG=1 + (pair - marg) / (err + marg),
        )
        check(
            set(v) == set(cost)
            and all(
                isclose(t, cost[k], abs_tol=2e-12, rel_tol=5e-13) for k, t in v.items()
            ),
            "incorrect PC branch-clock reduction",
        )


def report():
    r = common.report(study="PC")
    d = p.ROOT / p.JOB.name
    check(
        r["et_result_sha"] == data.ET_RESULT
        and r["eh_fit_freeze_sha"] == data.earlier.FIT
        and r["swap_fit"] == "PAIR"
        and r["fit_dimensions"] == p.DIMS
        and r["j_refits"] == r["history_refits"] == r["error_refits"] == 0,
        "foreign PC readout family",
    )
    roles = data.checked_roles(
        data.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    )
    for role, rs in roles.items():
        hs = [x["sha256"] for x in rs]
        stage = r["stages"][role]
        for s in SEEDS:
            audit(role, journal(d / f"{role}-{s}.jsonl", hs)[1], stage, str(s))
    r["verified"]["independent_conditional_support_peaks_and_cost"] = True
    return r


if __name__ == "__main__":
    os.umask(0o077)
    r = report()
    f = p.ROOT.parent / "paired-correction-result-1.json"
    with f.open("x") as out:
        json.dump(r, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print("X49-PC verified aggregate:", sha(f), f.stat().st_size, "bytes", flush=True)
