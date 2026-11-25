"""Closed JT evidence only. Separately reconstruct both shared-path costs and variable-denominator intervals."""

from __future__ import annotations
import json
from math import fsum, isclose
import os
from ..x48_muon_continuation.data import check, sha
from . import confirm_analysis as prior, transfer_data as data, transfer_panel as p
from .transfer_stats import BOOT as BOOT, decision as decision, summarize as summarize
from .panels import SEEDS


def analyze(root):
    return prior.analyze(root, study="JT")


def audit_cost(rows, calc):
    n = sum(x["targets"] for x in rows)
    j = fsum(x["seconds"]["shared_s"] + x["seconds"]["J_branch_s"] for x in rows)
    u = fsum(x["seconds"]["shared_s"] + x["seconds"]["U_branch_s"] for x in rows)
    delta = fsum(x["seconds"]["J_branch_s"] - x["seconds"]["U_branch_s"] for x in rows)
    expected = dict(
        shared_plus_J_s=j,
        shared_plus_U_s=u,
        J_minus_U_s_per_million_targets=delta * 1e6 / n,
        shared_path_J_over_U=j / u,
    )
    check(
        all(
            isclose(v, calc[k], rel_tol=5e-13, abs_tol=1e-12)
            for k, v in expected.items()
        )
        and all(
            calc["first_counts"].get(a, 0)
            == sum(row["first_branch"] == a for row in rows)
            for a in "UJ"
        ),
        "incorrect JT shared-path time or denominator",
    )


def report():
    check(os.getenv("OPENBLAS_NUM_THREADS") == "1", "JT audit uses one BLAS thread")
    r = prior.report(study="JT")
    rows, nt = data.checked_rows(
        data.u.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    )
    hs = [x["sha256"] for x in rows]
    sizes = [x["length"] - 1 for x in rows]
    d = p.ROOT / p.JOB.name
    for s in SEEDS:
        _, journal = p.prior.journal(d / f"{s}.jsonl", hs, sizes=sizes)
        audit_cost(journal, r["analysis"]["branch_cost_by_seed"][str(s)])
    check(
        nt == r["analysis"]["real_targets_per_seed"]
        and not (d / "windows.u8").exists(),
        "incorrect JT exposure",
    )
    r["verified"]["independent_branch_cost_reduction"] = True
    return r


if __name__ == "__main__":
    os.umask(0o077)
    r = report()
    f = p.ROOT.parent / "joint-transfer-result-1.json"
    with f.open("x") as out:
        json.dump(r, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    print("X49-JT verified aggregate:", sha(f), f.stat().st_size, "bytes", flush=True)
