"""X49-U qualified one-shot local runner. All three fits must be sealed before the probe reader."""

from __future__ import annotations
import json
import os
import re
import resource
import shutil
import subprocess
from time import monotonic
import torch
from ..x48_muon_continuation.data import ROOT as REPO, check, load, sha, write_json
from ..x48_muon_continuation.train import cpu_seconds
from . import (
    followup_loop as loop,
    panel,
    query_run,
    surprise_panel as p,
    surprise_data as data,
)
from .meter import MODULE, PY, HERE
from .fit_phase import study as phase
from .panels import OUT, T, SEEDS
from .panel_train import append
from .surprise_fit import Moments as Moments, document as document

MOD = MODULE + ".surprise_run"
AUTH = OUT / "surprise-authorization-1.json"
GATES = 96
Q_GATE_SHA = "5f07a5aa90e7040d7f05f5efc5d57691f7af13b6f528c7ac00db52795fe6229d"


def source():
    check(os.getenv("OPENBLAS_NUM_THREADS") == "1", "U requires one BLAS thread")
    ref = query_run.source()
    data.checked_spec()
    return ref


def qualification():
    return qualify_frozen(
        source,
        "surprise",
        data.SPEC,
        GATES,
        "U synthetic FIT/PROBE lifecycle and frozen real CPU strict loads; no U data",
    )


def qualify_frozen(
    check_source, name, spec, gates, scope, *, writer_build=None, head_audit=None
):
    panel.cpu()
    ref = check_source()
    d = OUT / ("qualify-" + name + "-" + ref)
    d.mkdir()
    f = d / "gate.log"
    tic = monotonic()
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTEST_ADDOPTS="")
    if writer_build is not None:
        env["X49_JOINT_TEST_BUILD"] = str(writer_build.resolve())
    torch.set_num_threads(4)
    with f.open("xb") as out:
        for cmd in (
            [
                PY,
                "-m",
                "pytest",
                "-q",
                str(HERE),
                "tests/test_memory_program_direct.py",
                "tests/test_x49_relative.py",
            ],
            [PY, "-m", MODULE + ".native_check"],
        ):
            subprocess.run(
                cmd, cwd=REPO, env=env, check=True, stdout=out, stderr=subprocess.STDOUT
            )
    text = f.read_text()
    native = [json.loads(s) for s in text.splitlines() if s.startswith("{")]
    check(
        re.findall(r"=+ (\d+) passed in [^\n]+ =+", text) == [str(gates)]
        and len(native) == 1
        and native[0]["work"]["matched_frontiers"] == 654,
        "incomplete U gate",
    )
    extra = {}

    def visit(s, e, c):
        extra[str(s)] = head_audit(s, e, c)

    states = query_run.strict_cpu_states(
        visit_head=visit if head_audit is not None else None
    )
    parent = data.checked_spec()
    old = data.bound(
        OUT / ("qualify-query-" + parent["qualified_source"]) / "passed.json",
        Q_GATE_SHA,
    )
    check(
        states == old["states"]
        and {s: x["checkpoint"] for s, x in states.items()} == parent["checkpoints"]
        and not torch.cuda.is_initialized()
        and check_source() == ref,
        "unbound W-B CPU states",
    )
    z = dict(
        scope=scope,
        ref=ref,
        spec=spec,
        passed=gates,
        log_sha=sha(f),
        states=states,
        **(dict(head_audit=extra) if head_audit is not None else {}),
        natural_windows_opened=0,
        wall_s=monotonic() - tic,
    )
    write_json(d / "passed.json", z)
    return z


def checked_fixture_ref():
    return checked_gate(source, "surprise", data.SPEC, GATES)


def checked_gate(check_source, name, spec, gates):
    ref = check_source()
    d = OUT / ("qualify-" + name + "-" + ref)
    z = load(d / "passed.json")
    check(
        z["ref"] == ref
        and z["spec"] == spec
        and z["passed"] == gates
        and z["log_sha"] == sha(d / "gate.log")
        and z["natural_windows_opened"] == 0
        and all(
            z["states"][str(s)]["checkpoint"]
            == p.checkpoint(s, hash_weights=False)[1]["sha256"]
            for s in SEEDS
        ),
        "U qualification changed",
    )
    return ref


def authority():
    check(AUTH.is_file(), "no separate U fit/probe budget")
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient U disk")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong U stage")
    return loop.admit(job, p, authority)


def model(seed, endpoint):
    from .query_panel import module

    m, c = module(seed, endpoint)
    return m.to("cuda"), c


def document_options(plan, seed):
    return {}  # U learns its own readouts, starting at ordinary B.


def evaluate(job, ctx, *, study="U"):
    data, p, runner = phase(study)
    d = ctx["folder"]
    plan_sha = load(d / "start.json")["freeze_sha"]
    plan = data.bound(d / "frozen.json", plan_sha)
    check(
        job == p.JOB
        and plan["ref"] == ctx["ref"]
        and plan["units"] == p.units()
        and plan["neural_updates"] == 0
        and plan["horizon"] == T,
        f"invalid {study} setup",
    )
    cost = {}
    c0 = cpu_seconds()
    progress = 0
    probes = {}
    fit_sha = None
    for role in ("fit", "probe"):
        tic = monotonic()
        if role == "fit":
            wins = data.fit_windows(job)
            frozen = None
        else:
            _, wins, frozen = data.extract("probe", job=job, fit_sha=fit_sha)
        cost[role + "_stream_s"] = monotonic() - tic
        check(len(wins) == data.N[role], f"incomplete {study} stream")
        for seed in SEEDS:
            key = str(seed)
            e = plan["endpoints"][key]
            tic = monotonic()
            m, c = runner.model(seed, e)
            options = (
                runner.document_options(plan, seed, c)
                if study == "RB"
                else runner.document_options(plan, seed)
            )
            cost[f"{role}_{seed}_model_s"] = monotonic() - tic
            log = d / f"{role}-{seed}.jsonl"
            log.touch(mode=0o600, exist_ok=False)
            mom = runner.Moments() if role == "fit" else None
            beta = None
            if role == "probe":
                entry = frozen["seeds"][key]
                beta = (
                    runner.load_decoder(d, seed, entry)
                    if study in ("DR", "RB")
                    else entry["beta"]
                )
            for h, word in wins:
                x = runner.document(m, c, word, fit=mom, beta=beta, **options)
                check(
                    x["targets"] == T and x["host_prefills"] == 1,
                    f"wrong {study} exposure",
                )
                append(log, dict(document_sha=h, **x))
                progress += 1
                panel.progress(d, progress, progress * T, root=p.ROOT)
            note = dict(
                seed=seed,
                checkpoint_sha=e["sha256"],
                journal_sha=sha(log),
                plan_sha=plan_sha,
            )
            if role == "fit":
                tic = monotonic()
                if study in ("DR", "RB"):
                    state, times = runner.finish_fit(d, seed, mom, m, c, options)
                    cost.update({f"{k}_{seed}_s": v for k, v in times.items()})
                else:
                    state = mom.freeze()
                    cost[f"solve_{seed}_s"] = monotonic() - tic
                write_json(
                    d / f"fit-{seed}.json",
                    dict(
                        **note,
                        state=state,
                        fit_identity_sha=plan["data"]["fit_identity_sha"],
                    ),
                )
            else:
                probes[key] = dict(**note, fit_freeze_sha=fit_sha, windows=len(wins))
            del m, c, mom
            torch.cuda.empty_cache()
        del wins, word
        if role == "fit":
            tic = monotonic()
            fit_sha = p.freeze_readouts(d, plan)
            cost["fit_freeze_s"] = monotonic() - tic
    check(progress == plan["units"], f"missing {study} work")
    return dict(
        kind=job.kind,
        units=progress,
        targets=progress * T,
        neural_updates=0,
        fitted_readouts=len(p.DIMS) * len(SEEDS),
        freeze_sha=plan_sha,
        fit_freeze_sha=fit_sha,
        probes=probes,
        probe_identity_sha=sha(d / "probe/identity.json"),
        stage_seconds=cost,
        cpu_core_s=[b - a for a, b in zip(c0, cpu_seconds(), strict=True)],
        gpu_peak_alloc=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved=torch.cuda.max_memory_reserved(),
        process_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32)
