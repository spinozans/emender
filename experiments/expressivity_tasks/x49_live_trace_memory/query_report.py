"""Receipt-only X49-Q closure and independent paired analysis; no text or model execution."""

import json
from math import isclose
from pathlib import Path
import numpy as np
from experiments.expressivity_tasks.x47_xsa_attention.w3_launch import fields
from experiments.expressivity_tasks.x48_muon_continuation.data import check, load, sha
from experiments.expressivity_tasks.x49_live_trace_memory import (
    query_analysis,
    query_data as data,
    query_panel as p,
)
from experiments.expressivity_tasks.x49_live_trace_memory.panel import cpu

REF = "76f9dbde8565edb04292537d8fc96f8c64c7c70b"
PUBLIC = "4d7320aa0d8307a72d945192e208b3c9a361789e"
PREP = "8d54639d1e4b6a1df5f97eb953c8ea76fa010fd240c5c9bb6fe859f18a981637"


def check_cost(d, summary):
    ident = load(d / "identity.json")
    started = load(d / "start.json")
    complete = load(d / "complete.json")
    progress = load(d / "progress.json")
    la = load(d / "launch.json")
    live = load(d / "live.json")
    obs = [json.loads(s) for s in (d / "survival.jsonl").read_bytes().splitlines()]
    ini, after, last, end = (
        fields(d / k) for k in ("initial", "after", "last", "released")
    )
    charge = summary["charge"]
    n = 3 * data.N
    check(
        started["name"] == p.JOB.name
        and started["ref"] == ident["ref"] == REF
        and started["cap_s"] == p.TOTAL
        and started["kind"] == complete["kind"] == p.JOB.kind
        and ident["job"] == p.JOB.name,
        "foreign stage",
    )
    check(
        ini["adopted"] == "1"
        and all(
            ini[k] == after[k] == last[k] == ident["owner"][k]
            for k in ("pid", "starttime", "gpu", "created", "adopted")
        ),
        "foreign lease",
    )
    check(
        la["pid"] == int(ini["pid"])
        and la["start"] == ini["starttime"]
        and live in obs
        and len(obs) >= 2,
        "missing live owner",
    )
    check(
        all(
            o["pid"] == la["pid"]
            and o["ppid"] == 1
            and o["start"] == la["start"]
            and o["trainer"] == ident["trainer"]
            and o["trainer_start"] == ident["trainer_start"]
            for o in obs
        ),
        "lost witness identity",
    )
    check(
        all(
            a["units"] < b["units"] and int(a["heartbeat"]) <= int(b["heartbeat"])
            for a, b in zip(obs, obs[1:])
        )
        and 0 < obs[0]["units"] <= live["units"]
        and int(ini["heartbeat"]) < int(live["heartbeat"]) <= int(after["heartbeat"]),
        "no actual witnessed renewal",
    )
    check(
        progress == dict(pid=ident["trainer"], units=n, scored_targets=n * p.T)
        and complete["targets"] == n * p.T
        and obs[-1]["units"] == complete["units"] == charge["witnessed_units"] == n
        and complete["updates"] == 0,
        "incomplete Q",
    )
    check(
        load(d / "charge.json") == charge
        and sha(d / "complete.json") == charge["complete_sha"]
        and (d / "watch-ack").is_file()
        and int(end["rc"]) == charge["rc"] == 0
        and 0
        < charge["upper_card_s"]
        == int(end["released_epoch"]) - int(ini["created"]) + 1
        <= p.TOTAL,
        "incorrect terminal charge",
    )
    check(
        isclose(
            charge["inclusive_wall_s"],
            (int(end["end_monotonic_ns"]) - started["t0"]) * 1e-9,
            abs_tol=1e-12,
            rel_tol=0,
        ),
        "incorrect elapsed time",
    )
    return dict(
        upper_card_s=charge["upper_card_s"],
        inclusive_wall_s=charge["inclusive_wall_s"],
        owner_and_children_cpu_s=charge["owner_and_waited_children_cpu_s"],
        worker_cpu_core_s=summary["cpu_core_s"],
        peak_device_allocated=summary["max_gpu_alloc"],
        peak_device_reserved=summary["max_gpu_reserved"],
        peak_RSS_KiB=summary["process_RSS_peak_KiB"],
        real_progress_observations=len(obs),
        terminal_units=n,
        device=ident["device"],
        torch=ident["torch"],
        python=ident["python"],
        cpu_threads=ident["cpu_threads"],
    )


def independently_reduce(d, analysis):
    """Different reduction: resample whole docs per seed/policy, normalize then take contrasts."""
    seeds = p.SEEDS
    n = data.N
    policies = "JEM"
    rs = [load(d / "complete.json")["endpoints"][str(s)] for s in seeds]
    logs = []
    for s, r in zip(seeds, rs, strict=True):
        path = d / f"{s}.jsonl"
        check(
            sha(path) == r["scores_sha"]
            and len(r["scores_sha"]) == 64
            and r["windows"] == n,
            "changed doc sums",
        )
        logs.append(list(map(json.loads, path.read_bytes().splitlines())))
    ordered = [[x["document_sha"] for x in rows] for rows in logs]
    check(
        ordered[0] == ordered[1] == ordered[2] and len(set(ordered[0])) == n,
        "unpaired Q seeds",
    )
    nt = np.array([x["targets"] for x in logs[0]])
    check(np.all(nt == p.T) and nt.sum() == 524288, "incomplete denominator")
    ix = np.random.default_rng(4906).integers(0, n, size=(10000, n))
    den = nt[ix].sum(1)
    av = {}
    boots = {}
    for a in policies:
        mat = np.array([[x["cells"][a]["nll_sum"] for x in rows] for rows in logs])
        av[a] = mat.sum(1) / nt.sum()
        boots[a] = np.mean([v[ix].sum(axis=1) / den for v in mat], axis=0)
        np.testing.assert_allclose(
            av[a], analysis["policies"][a]["by_seed"], rtol=0, atol=5e-14
        )
    for a, b in query_analysis.PAIRS:
        z = analysis["contrasts"][a + ":" + b]
        ds = av[b] - av[a]
        ci = np.percentile(boots[b] - boots[a], [2.5, 97.5], method="linear")
        np.testing.assert_allclose(ds, z["by_seed"], rtol=0, atol=5e-14)
        np.testing.assert_allclose(ci, z["ci95"], rtol=0, atol=5e-14)
        check(
            isclose(z["gain"], float(ds.mean()), abs_tol=5e-14, rel_tol=0),
            "wrong all-byte gain",
        )
    # Never export individual geometry where a stratum is isolated to a single document.
    for k, v in analysis["strata"].items():
        if v["documents_with_support"] < 2:
            v["geometry_per_seed"] = {}
    check(
        sum(v["real_targets_per_seed"] for v in analysis["strata"].values())
        == nt.sum(),
        "bad stratum accounting",
    )


def report():
    cpu()
    d = p.ROOT / p.JOB.name
    check(not (p.ROOT / "stop.json").exists(), "no Q after partial")
    terminal = load(p.ROOT / "done.json")
    check(
        terminal["complete"] is True
        and terminal["summary_sha"] == sha(p.ROOT / "summary.json")
        and terminal["charges_sha"] == sha(p.ROOT / "charged.json"),
        "Q campaign not closed",
    )
    summary = load(p.ROOT / "summary.json")
    check(summary == query_analysis.analyze(p.ROOT), "Q aggregate not reproduced")
    check(
        summary["ref"] == REF
        and data.metadata() == summary["data"]
        and sha(data.BASE / "prepared.json") == PREP,
        "foreign source or reservation",
    )
    prep = load(data.BASE / "prepared.json")
    a = p.OUT / "query-authorization-1.json"
    p.validate(load(a), REF)
    plan = load(d / "frozen.json")
    check(
        sha(d / "frozen.json") == summary["freeze_sha"]
        and plan == p.plan(REF, hash_weights=True),
        "frozen checkpoint bytes changed",
    )
    lcost = check_cost(d, summary)
    analysis = summary["analysis"]
    independently_reduce(d, analysis)
    return dict(
        scope="X49-Q, inference-only native TRAIN diagnostic, not a held-out final",
        qualified_source=REF,
        public_source=PUBLIC,
        spec_sha=data.SPEC,
        w_result_sha=data.W_RESULT,
        authority_sha=sha(a),
        data=summary["data"],
        preparation={
            k: prep[k]
            for k in (
                "excluded_unique",
                "eligible_train",
                "documents",
                "source_train_bytes",
                "window_bytes",
                "targets",
                "wall_s",
                "cpu_s",
                "calibration_or_test_documents_opened",
            )
        },
        checkpoints=summary["checkpoints"],
        frozen_sha=summary["freeze_sha"],
        cost=lcost,
        seed_order=list(p.SEEDS),
        history_limit=p.T,
        updates=0,
        host_prefills=3 * data.N,
        policy_evaluations=9 * data.N,
        bootstrap=dict(
            joint_document_draws=10000,
            seed=4906,
            interval="2.5 and 97.5 percentiles, linear",
        ),
        analysis=analysis,
        verified=dict(
            summary_sha=terminal["summary_sha"],
            charges_sha=terminal["charges_sha"],
            terminal_sha=sha(p.ROOT / "done.json"),
            strictly_positive_release_checked=True,
            weight_files_hashed=3,
            independent_paired_reduction=True,
        ),
    )


if __name__ == "__main__":
    import os

    os.umask(0o077)
    out = Path(p.OUT) / "query-result-1.json"
    z = report()
    with out.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print("X49-Q closed aggregate:", sha(out), out.stat().st_size, "bytes")
