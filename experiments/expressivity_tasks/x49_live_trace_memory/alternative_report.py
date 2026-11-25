"""X49-V closed-receipt publication audit. Never load a model, fit on PROBE or reopen text."""

from __future__ import annotations
import json
import os
from ..x48_muon_continuation.data import check, load, sha
from . import alternative_panel
from .fit_phase import incumbent as phase
from .panel import cpu
from .surprise_report import check_fits, check_bootstrap
from .panels import SEEDS


def report(*, study="V"):
    cpu()
    data, p, run = phase(study)
    if study == "V":
        from . import alternative_analysis as analysis
    else:
        from . import literal_mark_analysis as analysis
    check(
        os.getenv("OPENBLAS_NUM_THREADS") == "1",
        f"one BLAS thread for the {study} audit",
    )
    root = p.ROOT
    d = root / p.JOB.name
    done = load(root / "done.json")
    check(
        done["complete"] is True and sha(root / "charged.json") == done["charges_sha"],
        f"unbound {study} ledger",
    )
    r = data.bound(root / "summary.json", done["summary_sha"])
    check(r == analysis.analyze(root), f"{study} closure not reproducible")
    p.validate(load(run.AUTH), r["ref"])
    initial = data.bound(d / "frozen.json", r["freeze_sha"])
    check(
        initial == p.plan(r["ref"], hash_weights=True),
        f"different {study} model bytes, incumbent or data",
    )
    sealed = data.bound(d / "fit-frozen.json", r["fit_freeze_sha"])
    residual = check_fits(
        d, sealed, dims=p.DIMS, fit_documents=data.N["fit"], conditioned=study == "LS"
    )
    book = data.bound(data.BASE / "reservation.json", r["data"]["rows_sha"])
    roles = data.checked_roles(book)
    check_bootstrap(
        d, r["stages"]["probe"], [x["sha256"] for x in roles["probe"]], study=study
    )
    prep = data.bound(data.BASE / "prepared.json", r["data"]["prepared_sha"])
    check(
        prep["fit_source_bytes"] == sum(x["length"] for x in roles["fit"])
        and prep["probe_documents_opened"] == 0
        and prep["qualified_ref"] == r["ref"]
        and not (d / "probe/windows.u8").exists(),
        f"changed {study} role boundary",
    )
    return dict(
        **r,
        seed_order=list(SEEDS),
        authority_sha=sha(run.AUTH),
        ridge=p.RIDGE,
        bootstrap=dict(
            seed=analysis.BOOT, joint_document_draws=10000, interval="linear percentile"
        ),
        preparation={
            k: prep[k]
            for k in (
                "eligible_train",
                "excluded_unique",
                "documents",
                "fit_source_bytes",
                "fit_window_bytes",
                "fit_targets",
                "probe_documents_opened",
                "wall_s",
                "cpu_s",
            )
        },
        verified=dict(
            done_sha=sha(root / "done.json"),
            charges_sha=done["charges_sha"],
            summary_sha=done["summary_sha"],
            independent_cholesky_residuals=residual,
            independent_paired_bootstrap=True,
            actual_weight_bytes_rehashed=3,
            driver_and_short_parents_cpu_s=done["driver_and_short_parents_cpu_s"],
        ),
    )


if __name__ == "__main__":
    os.umask(0o077)
    z = report()
    out = alternative_panel.ROOT.parent / "alternative-result-1.json"
    with out.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        "X49-V verified aggregate:", sha(out), out.stat().st_size, "bytes", flush=True
    )
