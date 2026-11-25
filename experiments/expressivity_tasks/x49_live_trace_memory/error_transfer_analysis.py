"""ET closed-receipt replay: verify full branch costs and intervention support, never reread validation text."""

import json
from math import fsum, isclose
import os
from ..x48_muon_continuation.data import check, sha
from . import (
    confirm_analysis as prior,
    error_transfer_panel as p,
    error_transfer_data as data,
)
from .error_transfer_stats import (
    BOOT as BOOT,
    decision as decision,
    summarize as summarize,
)
from .panels import SEEDS


def analyze(root):
    return prior.analyze(root, study="ET")


def audit(rows, stats, s):
    c = stats["branch_cost_by_seed"][s]
    n = sum(r["targets"] for r in rows)
    check(n == stats["real_targets_per_seed"], "incorrect ET denominator")
    j = fsum(r["seconds"]["shared_J_s"] for r in rows)
    hc = fsum(r["seconds"]["HISTORY_branch_s"] for r in rows)
    ec = fsum(r["seconds"]["ERROR_branch_s"] for r in rows)
    th = fsum(
        r["seconds"]["shared_J_s"] + r["seconds"]["HISTORY_branch_s"] for r in rows
    )
    te = fsum(r["seconds"]["shared_J_s"] + r["seconds"]["ERROR_branch_s"] for r in rows)
    delta = fsum(
        r["seconds"]["ERROR_branch_s"] - r["seconds"]["HISTORY_branch_s"] for r in rows
    )
    expect = dict(
        shared_J_s=j,
        shared_plus_HISTORY_s=th,
        shared_plus_ERROR_s=te,
        HISTORY_extra_s_per_million_targets=hc * 1e6 / n,
        ERROR_extra_s_per_million_targets=ec * 1e6 / n,
        ERROR_minus_HISTORY_s_per_million_targets=delta * 1e6 / n,
        shared_path_HISTORY_over_J=th / j,
        shared_path_ERROR_over_J=te / j,
        shared_path_ERROR_over_HISTORY=te / th,
    )
    check(
        all(isclose(v, c[k], rel_tol=5e-13, abs_tol=2e-12) for k, v in expect.items())
        and all(
            c["first_counts"].get(k, 0) == sum(r["first_branch"] == k for r in rows)
            for k in ("ERROR", "HISTORY")
        ),
        "incorrect ET shared-path cost reduction",
    )
    for label, name in [
        ("donor_displaced", "shift_older_displaced_uses"),
        ("changed_guess", "shift_older_changed_uses"),
        ("changed_histogram", "changed_histogram_rows"),
        ("older_history", "pool_active_rows"),
    ]:
        vals = [r["work"][name] for r in rows]
        check(
            sum(vals) == stats["work_by_seed"][s][name]
            and sum(bool(v) for v in vals)
            == stats["documents_with_support_by_seed"][s][label],
            "incorrect ET affected-document counts",
        )
    for k in ("ERROR", "HISTORY"):
        for name in ("peak_keys", "peak_cells", "packed_bytes", "fp64_feature_bytes"):
            check(
                max(r["chosen_work"][k][name] for r in rows)
                == stats["peak_per_window_by_seed"][s][k + "_" + name],
                "incorrect chosen-branch peak",
            )


def report():
    check(os.getenv("OPENBLAS_NUM_THREADS") == "1", "one BLAS thread for ET")
    r = prior.report(study="ET")
    rows, nt = data.checked_rows(
        data.u.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    )
    hs = [v["sha256"] for v in rows]
    sizes = [v["length"] - 1 for v in rows]
    d = p.ROOT / p.JOB.name
    for s in SEEDS:
        audit(
            p.prior.journal(d / f"{s}.jsonl", hs, sizes=sizes)[1], r["analysis"], str(s)
        )
    check(
        r["eh_result_sha"] == data.reader.PUBLIC
        and r["eh_fit_freeze_sha"] == data.FIT_SHA
        and r["swap_fit"] == "ERROR"
        and nt == r["analysis"]["real_targets_per_seed"]
        and not (d / "windows.u8").exists(),
        "foreign ET close",
    )
    r["verified"]["independent_branch_cost_and_support"] = True
    return r


if __name__ == "__main__":
    os.umask(0o077)
    r = report()
    f = p.ROOT.parent / "error-transfer-result-1.json"
    with f.open("x") as out:
        json.dump(r, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print("X49-ET verified aggregate:", sha(f), f.stat().st_size, "bytes", flush=True)
