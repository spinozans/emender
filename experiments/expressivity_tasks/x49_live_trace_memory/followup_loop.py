"""Shared process/lease lifecycle for single-use follow-ups, independent of learner or analysis."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from ..x48_muon_continuation.data import ROOT as REPO, check, load, write_json
from ..x48_muon_continuation.lease import launch
from . import panel as p
from .meter import PY
from .panels import OUT


def pre(job, spec, authority):
    ref = authority()
    p.prior(job, root=spec.ROOT, jobs=spec.JOBS, total=spec.TOTAL)
    return ref


def admit(job, spec, authority):
    ref = pre(job, spec, authority)
    d = spec.ROOT / job.name
    start = load(d / "start.json")
    ticket = load(spec.ROOT / (job.name + ".ticket.json"))
    check(
        ticket["job"] == job.name
        and ticket["ref"] == ref
        and ticket["t0"] == start["t0"]
        and start["kind"] == job.kind
        and start["cap_s"] == job.cap
        and start["ref"] == ref
        and start["name"] == job.name
        and not (d / "released").exists()
        and not (spec.ROOT / "stop.json").exists(),
        "unlicensed or closed job: " + job.name,
    )
    return d, ref, start


def complete(d, out):
    write_json(d / "complete.json", out)
    for _ in range(40):
        if (d / "watch-ack").is_file():
            return
        time.sleep(0.2)
    raise RuntimeError("no independent completion acknowledgment")


def cli(spec, mod, authority, qualify, inside, min_disk_gib, *, targets_for=None):
    ap = argparse.ArgumentParser(
        description="Single-use registered local campaign: " + mod
    )
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
    ap.add_argument("--job", choices=list(spec.JOBS))
    a = ap.parse_args()
    os.umask(0o077)
    os.chdir(REPO)
    job = spec.JOBS.get(a.job)
    check(
        (job is not None) == (a.phase in ("one", "prelease", "inside")),
        "invalid phase/job",
    )
    if a.phase == "inside":
        return inside(job)
    p.cpu()
    if a.phase == "qualify":
        print(json.dumps(qualify()))
        return
    if a.phase == "observe":
        p.observe(spec.ROOT, spec.JOBS)
        return
    if a.phase == "prelease":
        admit(job, spec, authority)
        print(job.cap)
        return
    if a.phase == "one":
        p.dispatch_one(job, spec.ROOT, pre(job, spec, authority), spec.freeze)
        return
    ref = authority()
    if a.phase == "drive":
        for _ in range(50):
            if os.getppid() == 1 and (spec.ROOT / "driver.json").is_file():
                break
            time.sleep(0.1)
        p.run_schedule(
            spec.ROOT,
            ref,
            spec.JOBS,
            mod,
            spec.TOTAL,
            spec.finish,
            targets_for=targets_for,
        )
        return
    check(
        not spec.ROOT.exists() and shutil.disk_usage(OUT).free >= min_disk_gib << 30,
        "already admitted or insufficient disk",
    )
    if a.phase == "check":
        print(ref)
        return
    spec.ROOT.mkdir(mode=0o700)
    print(
        launch(
            [PY, "-m", mod, "drive"], spec.ROOT / "console", spec.ROOT / "driver.json"
        )
    )
