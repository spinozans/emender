"""X49-Q source qualification, licensed detached inference only, no natural-data training."""

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
from . import followup_loop as loop, panel as p, query_panel as q, query_data as data
from .meter import PY, HERE, MODULE
from .panels import OUT, SEEDS, T
from .panel_train import append
from .query_score import window_score

MOD = MODULE + ".query_run"
AUTH = OUT / "query-authorization-1.json"
GATES = 85


def source():
    ref = p.source()
    data.checked_spec()
    tests = ["tests/test_memory_program_direct.py", "tests/test_x49_relative.py"]
    subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *tests], cwd=REPO, check=True
    )
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", *tests],
        cwd=REPO,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return ref


def qualification():
    from .endpoint import state_digest

    p.cpu()
    ref = source()
    tic = monotonic()
    d = OUT / ("qualify-query-" + ref)
    d.mkdir()
    f = d / "gate.log"
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTEST_ADDOPTS="")
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
            [PY, "-m", MODULE + ".query_native"],
            [PY, "-m", MODULE + ".native_check"],
        ):
            subprocess.run(
                cmd, cwd=REPO, env=env, check=True, stdout=out, stderr=subprocess.STDOUT
            )
    text = f.read_text()
    obj = [json.loads(v) for v in text.splitlines() if v.startswith("{")]
    check(
        re.findall(r"=+ (\d+) passed in [^\n]+ =+", text) == [str(GATES)]
        and len(obj) == 2
        and obj[0]["work"]["policy_frontiers"] == 108
        and obj[0]["work"]["matched_substitutions"] == 4
        and obj[1]["work"]["matched_frontiers"] == 654,
        "incomplete Q gate",
    )
    states = {}
    for seed in SEEDS:
        _, e = q.checkpoint(seed, hash_weights=False)
        m, c = q.module(seed, e)
        check(
            not (m.training or c.training)
            and all(
                not t.requires_grad and t.device.type == "cpu"
                for t in (*m.parameters(), *c.parameters())
            ),
            "unfrozen qualification model",
        )
        states[str(seed)] = dict(
            checkpoint=e["sha256"],
            host=state_digest(m.state_dict()),
            head=state_digest(c.state_dict()),
        )
        del m, c
    check(
        not torch.cuda.is_initialized() and source() == ref,
        "device initialized or source changed during Q qualification",
    )
    z = dict(
        scope="Q synthetic + strict frozen W-B CPU loads; no diagnostic windows read",
        ref=ref,
        spec=data.SPEC,
        passed=GATES,
        log_sha=sha(f),
        states=states,
        natural_Q_windows_read=0,
        wall_s=monotonic() - tic,
    )
    write_json(d / "passed.json", z)
    return z


def checked_fixture_ref():
    ref = source()
    d = OUT / ("qualify-query-" + ref)
    z = load(d / "passed.json")
    check(
        z["ref"] == ref
        and z["spec"] == data.SPEC
        and z["passed"] == GATES
        and z["log_sha"] == sha(d / "gate.log")
        and z["natural_Q_windows_read"] == 0
        and all(
            z["states"][str(s)]["checkpoint"]
            == q.checkpoint(s, hash_weights=False)[1]["sha256"]
            for s in SEEDS
        ),
        "Q qualification not bound to this source",
    )
    return ref


def authority():
    check(AUTH.is_file(), "no separate Q inference budget")
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient Q disk")
    ref = checked_fixture_ref()
    q.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(
        job == q.JOB and shutil.disk_usage(OUT).free >= 32 << 30,
        "wrong Q stage or inadequate disk",
    )
    return loop.admit(job, q, authority)


def evaluate(job, ctx):
    d = ctx["folder"]
    f = d / "frozen.json"
    check(
        job == q.JOB and sha(f) == load(d / "start.json")["freeze_sha"],
        "Q plan changed before worker",
    )
    plan = load(f)
    check(
        plan["ref"] == ctx["ref"]
        and plan["units"] == 3 * data.N
        and plan["updates"] == 0
        and plan["horizon"] == T,
        "wrong Q setup",
    )
    c0 = cpu_seconds()
    tic = monotonic()
    windows = data.diagnostic_windows(job)
    source_s = monotonic() - tic
    check(len(windows) == data.N, "incorrect number of Q windows")
    write_json(
        d / "stream.json",
        dict(
            scope="bound prepared TRAIN only",
            windows=len(windows),
            targets_per_endpoint=data.N * T,
            wall_s=source_s,
            **plan["data"],
        ),
    )
    n = 0
    endpoints = {}
    for seed in SEEDS:
        e = plan["endpoints"][str(seed)]
        m, c = q.module(seed, e)
        m = m.to("cuda")
        torch.set_num_threads(4)
        log = d / f"{seed}.jsonl"
        log.touch(mode=0o600, exist_ok=False)
        for i, (h, word) in enumerate(windows):
            x = window_score(m, c, word, seed=seed, rank=i)
            check(x["targets"] == T and x["host_prefills"] == 1, "wrong Q work")
            append(log, dict(document_sha=h, **x))
            n += 1
            p.progress(d, n, n * T, root=q.ROOT)
        endpoints[str(seed)] = dict(
            checkpoint_sha=e["sha256"], scores_sha=sha(log), windows=len(windows)
        )
        del m, c
        torch.cuda.empty_cache()
    check(n == plan["units"], "incomplete Q exposure")
    return dict(
        kind=job.kind,
        units=n,
        targets=n * T,
        updates=0,
        endpoints=endpoints,
        stream_sha=sha(d / "stream.json"),
        freeze_sha=sha(f),
        cpu_core_s=[b - a for a, b in zip(c0, cpu_seconds(), strict=True)],
        gpu_peak_alloc=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved=torch.cuda.max_memory_reserved(),
        process_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def inside(job):
    ctx = p.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(q, MOD, authority, qualification, inside, 32)
