"""CPU qualification/admission and detached single-use driver for X49-R, not the v2 experiment."""

from __future__ import annotations

from functools import partial
import json
import os
import re
import subprocess
import time

from ..x48_muon_continuation.data import ROOT as REPO, check, load, sha, write_json
from . import followup_loop as loop, panel as p, relative_panel as r
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


def admit(job):
    return loop.admit(job, r, authority)


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
    loop.complete(d, out)


def main():
    loop.cli(r, MOD, authority, qualification, inside, 80)


if __name__ == "__main__":
    main()
