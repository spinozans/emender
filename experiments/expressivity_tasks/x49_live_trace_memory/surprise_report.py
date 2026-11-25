"""X49-U publication audit: terminal hashes, separately solved FIT systems and an independent document bootstrap. No corpus, model execution or reopening the probe."""

import json
from math import isclose
from pathlib import Path
import numpy as np
from experiments.expressivity_tasks.x48_muon_continuation.data import check, load, sha
from experiments.expressivity_tasks.x49_live_trace_memory import (
    surprise_analysis as ana,
    surprise_data as data,
    surprise_panel as p,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel import cpu
from experiments.expressivity_tasks.x49_live_trace_memory.surprise_run import AUTH

REF = "bcff5643047683b575f984504620a59dae6b17a2"
PUBLIC = "31f8e868142e53c8c2d819d6c91e8b8eb7dbdb1f"


def check_fits(folder, readouts):
    """Independent positive-definite solve by Cholesky, from pre-probe moments only."""
    result = {}
    for s in p.SEEDS:
        k = str(s)
        sealed = readouts["seeds"][k]
        fit = data.bound(folder / f"fit-{s}.json", sealed["receipt_sha"])["state"]
        check(fit["beta"] == sealed["beta"], "wrong U coefficients")
        n = fit["targets"]
        check(
            n == data.N["fit"] * p.T and fit["documents"] == data.N["fit"],
            "wrong U fit denominator",
        )
        result[k] = {}
        for arm, d in p.DIMS.items():
            gram = np.asarray(fit["gram"][arm])
            rhs = np.asarray(fit["rhs"][arm])
            m = (gram + gram.T) / (2 * n) + p.RIDGE * np.eye(d)
            chol = np.linalg.cholesky(m)
            beta = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs / n))
            orig = np.asarray(fit["beta"][arm])
            np.testing.assert_allclose(orig, beta, rtol=1e-12, atol=2e-14)
            result[k][arm] = float(np.linalg.norm(m @ orig - rhs / n, ord=np.inf))
    return result


def check_bootstrap(folder, stats, hashes):
    # Convert sampled document indices to multiplicities. Resample each arm's *absolute*
    # loss sum per seed, divide by that draw's target count, then take contrasts and average seeds.
    n = len(hashes)
    inds = np.random.default_rng(4907).integers(0, n, size=(10000, n))
    mul = np.asarray([np.bincount(row, minlength=n) for row in inds], dtype=np.float64)
    check(np.all(mul.sum(1) == n), "lost bootstrap mass")
    sums = {k: [] for k in ana.ARMS}
    draw = {k: [] for k in ana.ARMS}
    support = stats["documents_with_support_by_seed"]
    for s in p.SEEDS:
        _, rows = p.journal(folder / f"probe-{s}.jsonl", hashes)
        nt = np.asarray([x["targets"] for x in rows])
        den = mul @ nt
        check(
            nt.sum() == stats["real_targets_per_seed"] == n * p.T, "incorrect exposure"
        )
        seed = str(s)
        work = stats["work_by_seed"][seed]
        check(
            sum(x["candidate_correct"] for x in rows) == work["candidate_correct"]
            and sum(x["work"]["S_codes_changed"] > 0 for x in rows)
            == support[seed]["changed_code"],
            "incorrect mark support",
        )
        for a in ana.ARMS:
            v = np.asarray([x["nll_sums"][a] for x in rows])
            sums[a].append(float(v.sum() / nt.sum()))
            draw[a].append((mul @ v) / den)
    for a in ana.ARMS:
        r = stats["policies"][a]
        np.testing.assert_allclose(r["by_seed"], sums[a], atol=5e-14, rtol=0)
        check(
            isclose(r["nats_per_byte"], np.mean(sums[a]), abs_tol=5e-14, rel_tol=0),
            "absolute NLL mismatch",
        )
    for a, b in ana.PAIRS:
        r = stats["contrasts"][a + ":" + b]
        ds = np.subtract(sums[b], sums[a])
        boots = np.subtract(draw[b], draw[a]).mean(0)
        np.testing.assert_allclose(r["by_seed"], ds, rtol=0, atol=5e-14)
        np.testing.assert_allclose(
            r["ci95"],
            np.percentile(boots, [2.5, 97.5], method="linear"),
            rtol=0,
            atol=5e-14,
        )
        check(
            isclose(r["gain"], ds.mean(), abs_tol=5e-14, rel_tol=0),
            "contrast sign/denominator mismatch",
        )


def report():
    cpu()
    root = p.ROOT
    folder = root / p.JOB.name
    check(not (root / "stop.json").exists(), "do not close partial U")
    done = load(root / "done.json")
    check(
        done["complete"] is True
        and done["summary_sha"] == sha(root / "summary.json")
        and done["charges_sha"] == sha(root / "charged.json"),
        "U not closed",
    )
    summary = data.bound(root / "summary.json", done["summary_sha"])
    check(
        summary["ref"] == REF and summary == ana.analyze(root),
        "U closed aggregate not reproducible",
    )
    auth = load(AUTH)
    p.validate(auth, REF)
    initial = data.bound(folder / "frozen.json", summary["freeze_sha"])
    check(
        initial == p.plan(REF, hash_weights=True)
        and initial["data"] == summary["data"] == auth["data"],
        "unbound U base",
    )
    fit = data.bound(folder / "fit-frozen.json", summary["fit_freeze_sha"])
    residual = check_fits(folder, fit)
    book = data.bound(data.BASE / "reservation.json", summary["data"]["rows_sha"])
    roles = data.checked_roles(book)
    check_bootstrap(
        folder, summary["stages"]["probe"], [x["sha256"] for x in roles["probe"]]
    )
    prep = data.bound(data.BASE / "prepared.json", summary["data"]["prepared_sha"])
    check(
        prep["fit_source_bytes"] == sum(r["length"] for r in roles["fit"])
        and prep["probe_documents_opened"] == 0
        and prep["qualified_ref"] == REF,
        "invalid pre-probe preparation",
    )
    check(
        not (folder / "probe/windows.u8").exists(),
        "unexpected persistent probe window file",
    )
    return dict(
        **summary,
        public_source=PUBLIC,
        registration="514f475e7675cac8c67cd5d94e7c43c9121fec6d",
        authority_sha=sha(AUTH),
        seed_order=list(p.SEEDS),
        history=p.T,
        regularization=p.RIDGE,
        bootstrap=dict(
            joint_document_draws=10000,
            seed=4907,
            interval="2.5 and 97.5 percentiles, linear",
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
            summary_sha=done["summary_sha"],
            charges_sha=done["charges_sha"],
            w_b_checkpoints_rehashed=3,
            independent_cholesky_residuals=residual,
            independent_bootstrap=True,
            driver_and_short_parents_cpu_s=done["driver_and_short_parents_cpu_s"],
        ),
    )


if __name__ == "__main__":
    import os

    os.umask(0o077)
    z = report()
    out = Path(p.OUT) / "surprise-result-1.json"
    with out.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print("X49-U verified aggregate:", sha(out), out.stat().st_size, "bytes")
