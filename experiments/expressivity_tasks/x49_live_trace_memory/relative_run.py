"""CPU qualification/admission and detached single-use driver for X49-R, not the v2 experiment."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
import re
import shutil
import subprocess
import time

from ..x48_muon_continuation.data import ROOT as REPO, check, load, sha, write_json
from ..x48_muon_continuation.lease import launch
from . import panel as p, relative_panel as r
from .meter import HERE, MODULE, PY
from .panels import OUT
from .relative_data import BASE, SPEC, RESERVATION, metadata
from .train_stream import TrainStream

MOD = MODULE + ".relative_run"
TEST = "tests/test_x49_relative.py"
GATES = 54
AUTH = OUT / "relative-authorization-1.json"


def source():
    ref = p.source()
    check(
        sha(HERE / "RELATIVE.md") == SPEC
        and sum(j.cap for j in r.JOBS.values()) == r.TOTAL,
        "changed R scope",
    )
    subprocess.run(["git", "diff", "--quiet", "HEAD", "--", TEST], cwd=REPO, check=True)
    subprocess.run(
        ["git", "ls-files", "--error-unmatch", TEST],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        check=True,
    )
    return ref


def qualification():
    p.cpu()
    ref = source()
    d = OUT / ("qualify-relative-" + ref)
    d.mkdir()
    log = d / "gate.log"
    begin = time.monotonic()
    q = load(BASE / "prepared.json")
    check(
        q["raw_bytes_opened"] == 0
        and q["fresh"] == 137
        and q["spec"] == SPEC
        and q["reservation_sha"] == sha(RESERVATION)
        and load(RESERVATION) == metadata(),
        "incorrect fresh reservation",
    )
    r.old_summary()
    stream = TrainStream()
    check(len(stream.docs) == 1024, "wrong TRAIN stream")
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTEST_ADDOPTS="")
    with log.open("xb") as f:
        for cmd in (
            [
                PY,
                "-m",
                "pytest",
                "-q",
                str(HERE),
                "tests/test_memory_program_direct.py",
                TEST,
            ],
            *(
                [PY, "-m", MODULE + "." + m]
                for m in (
                    "native_check",
                    "trace_native",
                    "coupled",
                    "posthoc_credit_control",
                )
            ),
        ):
            subprocess.run(
                cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env, check=True
            )
    text = log.read_text()
    passes = re.findall(r"=+ (\d+) passed in [^\n]+ =+", text)
    objs = [json.loads(s) for s in text.splitlines() if s.startswith("{")]
    check(
        passes == [str(GATES)]
        and len(objs) == 4
        and objs[0]["work"]["matched_frontiers"] == 654
        and objs[1]["counts"]["frontiers"] == 542
        and objs[2]["cuda_initialized"] is False
        and objs[3]["hard_read"]["birth"]["relative_policy_grad"] == 0
        and objs[3]["hard_read"]["nonzero"]["relative_policy_grad"] > 0
        and source() == ref,
        "incomplete R gate",
    )
    out = dict(
        ref=ref,
        spec=SPEC,
        reservation_sha=q["reservation_sha"],
        parent_summary=r.SUMMARY,
        passed=GATES,
        log_sha=sha(log),
        wall_s=time.monotonic() - begin,
        evaluation_windows_opened=0,
    )
    write_json(d / "passed.json", out)
    return out


def authority():
    check(AUTH.is_file(), "no separate R campaign budget")
    ref = source()
    h = sha(RESERVATION)
    r.validate(load(AUTH), ref, h)
    qdir = OUT / ("qualify-relative-" + ref)
    q = load(qdir / "passed.json")
    check(
        q["ref"] == ref
        and q["spec"] == SPEC
        and q["reservation_sha"] == h
        and q["parent_summary"] == r.SUMMARY
        and q["passed"] == GATES
        and q["evaluation_windows_opened"] == 0
        and q["log_sha"] == sha(qdir / "gate.log"),
        "R source not qualified",
    )
    r.old_summary()
    return ref


def pre(job):
    ref = authority()
    p.prior(job, root=r.ROOT, jobs=r.JOBS, total=r.TOTAL)
    return ref


def admit(job):
    ref = pre(job)
    d = r.ROOT / job.name
    start = load(d / "start.json")
    ticket = load(r.ROOT / (job.name + ".ticket.json"))
    check(
        ticket["job"] == job.name
        and ticket["ref"] == ref
        and ticket["t0"] == start["t0"]
        and start["kind"] == job.kind
        and start["cap_s"] == job.cap
        and start["ref"] == ref
        and start["name"] == job.name
        and not (d / "released").exists()
        and not (r.ROOT / "stop.json").exists(),
        "unlicensed or closed R job",
    )
    return d, ref, start


def inside(job):
    from .panel_train import train_admitted
    from .panel_eval import _documents, evaluate_admitted

    ctx = p.enter_admitted(job, *admit(job))
    d = ctx["folder"]
    progress = partial(p.progress, root=r.ROOT)
    if job.kind == "train":
        out = train_admitted(
            job,
            ctx,
            progress,
            r.binding,
            lambda birth: r.check_first(job.seed, birth=birth),
            lambda result: r.check_first(job.seed, result=result),
        )
    else:
        out = evaluate_admitted(job, ctx, _documents, progress, r.resolve)
    write_json(d / "complete.json", out)
    for _ in range(40):
        if (d / "watch-ack").is_file():
            return
        time.sleep(0.2)
    raise RuntimeError("no independent R completion acknowledgment")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "phase",
        choices=[
            "qualify",
            "check",
            "launch",
            "observe",
            "drive",
            "one",
            "prelease",
            "inside",
        ],
    )
    ap.add_argument("--job", choices=list(r.JOBS))
    a = ap.parse_args()
    os.umask(0o077)
    os.chdir(REPO)
    job = r.JOBS.get(a.job)
    check(
        (job is not None) == (a.phase in ("one", "prelease", "inside")),
        "invalid R job selector",
    )
    if a.phase == "inside":
        return inside(job)
    p.cpu()
    if a.phase == "qualify":
        print(json.dumps(qualification()))
        return
    if a.phase == "observe":
        p.observe(r.ROOT, r.JOBS)
        return
    if a.phase == "prelease":
        admit(job)
        print(job.cap)
        return
    if a.phase == "one":
        p.dispatch_one(job, r.ROOT, pre(job), r.freeze)
        return
    ref = authority()
    if a.phase == "drive":
        for _ in range(50):
            if os.getppid() == 1 and (r.ROOT / "driver.json").is_file():
                break
            time.sleep(0.1)
        p.run_schedule(r.ROOT, ref, r.JOBS, MOD, r.TOTAL, r.finish)
        return
    check(
        not r.ROOT.exists() and shutil.disk_usage(OUT).free >= 80 << 30,
        "R already admitted or insufficient disk",
    )
    if a.phase == "check":
        print(ref)
        return
    r.ROOT.mkdir(mode=0o700)
    print(launch([PY, "-m", MOD, "drive"], r.ROOT / "console", r.ROOT / "driver.json"))


if __name__ == "__main__":
    main()
