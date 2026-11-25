"""X49-R exact schedule, frozen mixed-source endpoints and fresh-test analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from ..x48_muon_continuation.data import check, load, sha, write_json
from .panel_spec import (
    Job,
    ROOT as OLD,
    JOBS as OLD_JOBS,
    SEEDS,
    T,
    STEPS,
    binding as old_binding,
    common_wall,
)
from .panel_freeze import read_bound_training
from .panel_report import TRAIN_REF, SUMMARY
from .panels import OUT, fingerprint
from .relative_data import SPEC, N, RESERVATION
from .train_stream import V2_IDENTITY

ROOT = OUT / "relative-panel-1"
JOBS = {f"R-{s}": Job(f"R-{s}", "train", 1380, "R", s) for s in SEEDS}
FINAL = "r-final"
JOBS[FINAL] = Job(FINAL, "final_test", 3600, "RBFC")
TOTAL = 7740
PAIRS = (("R", "C"), ("R", "F"), ("R", "B"))


def name(arm, seed):
    check(arm in ("R", "B", "F", "C") and seed in SEEDS, "foreign R/control identity")
    return f"{arm}-{seed}"


def binding(ref, arm, seed):
    name(arm, seed)
    if arm != "R":
        return old_binding(TRAIN_REF, arm, seed)
    return dict(
        old_binding(ref, arm, seed),
        schema="x49-relative-v1",
        credit="same-weight-literal-nll-v1",
        protocol_sha256=SPEC,
    )


def resolve(arm, seed, i, ref):
    k = name(arm, seed)
    check(
        type(i) is int and 32 <= i <= STEPS and not i % 32,
        "not an R/retained analysis milestone",
    )
    return (ROOT if arm == "R" else OLD) / k / f"weights-{i:04d}.pt", binding(
        ref, arm, seed
    )


def expected(ref, reservation_sha):
    return dict(
        scope="x49-relative-credit-local-v1",
        qualified_ref=ref,
        spec=SPEC,
        parent_summary=SUMMARY,
        train_identity=V2_IDENTITY,
        reservation_sha=reservation_sha,
        total_card_seconds=TOTAL,
        job_caps={k: j.cap for k, j in JOBS.items()},
        max_acquisitions=4,
        retries=0,
        remote=False,
        gpus=1,
        seeds=list(SEEDS),
        final_documents=N,
    )


def validate(a, ref, reservation_sha):
    e = expected(ref, reservation_sha)
    check(
        type(a) is dict
        and set(a) == set(e) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in e}) == fingerprint(e),
        "foreign/incomplete R campaign authority",
    )


def old_summary():
    done = load(OLD / "done.json")
    check(
        done["complete"]
        and not (OLD / "stop.json").exists()
        and done["summary_sha"] == sha(OLD / "summary.json") == SUMMARY
        and done["charges_sha"] == sha(OLD / "charged.json"),
        "changed v2 evidence",
    )
    return load(OLD / "summary.json")


def runs_for(ref):
    parent = old_summary()
    runs = {}
    for a in "RBFC":
        for seed in SEEDS:
            k = name(a, seed)
            d = (ROOT if a == "R" else OLD) / k
            if a != "R":
                check(
                    load(d / "complete.json") == parent["train"][k]
                    and load(d / "charge.json") == parent["charges"][k],
                    "foreign historical control",
                )
            cap = (JOBS if a == "R" else OLD_JOBS)[k].cap
            runs[k] = read_bound_training(d, binding(ref, a, seed), cap)
    for s in SEEDS:
        birth = [runs[name(a, s)]["birth"] for a in "RFCB"]
        check(
            birth[0] == birth[1] == birth[2]
            and birth[0]["host"] == birth[3]["host"]
            and birth[0]["literal"] == birth[3]["literal"],
            "mismatched paired origins",
        )
    return runs


def freeze(job, ref):
    check(job == JOBS[FINAL], "only R fresh final may be frozen")
    f = ROOT / FINAL / "frozen.json"
    check(not f.exists(), "already frozen")
    runs = runs_for(ref)
    walls = common_wall(runs, pairs=PAIRS, name=name)
    endpoints = {}

    def pin(a, s, i):
        k = name(a, s)
        key = f"{k}@{i}"
        if key in endpoints:
            return
        p, b = resolve(a, s, i, ref)
        m = runs[k]["milestones"][i // 32 - 1]
        h = m["sha256"]
        size = m["bytes"]
        check(
            m["step"] == i
            and load(p.with_suffix(".json"))
            == dict(binding=b, bytes=size, file=p.name, sha256=h, step=i)
            and p.stat().st_size == size
            and sha(p) == h,
            "unbound fresh-test model",
        )
        endpoints[key] = dict(arm=a, seed=s, step=i, bytes=size, sha256=h, binding=b)

    for s in SEEDS:
        for a in "RBFC":
            pin(a, s, STEPS)
        for pair in walls.values():
            for a, v in pair[str(s)]["selected"].items():
                if v:
                    pin(a, s, v["step"])
    check(12 <= len(endpoints) <= 21, "incorrect R endpoint count")
    write_json(
        f,
        dict(
            ref=ref,
            job=FINAL,
            split="final_test",
            visit=0,
            horizons=[128, 1024, T],
            documents=N,
            units=N * len(endpoints),
            endpoints=endpoints,
            wall_pairs=walls,
            credit_spec=SPEC,
            train_identity=V2_IDENTITY,
            reservation_sha=sha(RESERVATION),
            parent_summary_sha=SUMMARY,
            training_charges={k: v["charge_sha"] for k, v in runs.items()},
        ),
    )
    return sha(f)


def check_first(seed, birth=None, result=None):
    # Locked TRAIN-only reference (the summary hash covers the original trajectory log hash).
    k = name("F", seed)
    s = old_summary()
    d = OLD / k
    term = s["train"][k]
    if birth is not None:
        check(
            birth
            == load(d / "start-state.json")["birth"]
            == load(OLD / name("C", seed) / "start-state.json")["birth"],
            "foreign R birth",
        )
    if result is not None:
        check(
            sha(d / "steps.jsonl") == term["steps_sha256"],
            "foreign fixed-key TRAIN receipt",
        )
        with (d / "steps.jsonl").open() as f:
            old = json.loads(f.readline())
        check(
            all(
                result[x] == old[x]
                for x in (
                    "step",
                    "targets",
                    "preupdate_nll",
                    "work",
                    "host_lr",
                    "head_lr",
                )
            )
            and abs(result["preclip"] - old["preclip"]) <= 1e-6 * (1 + old["preclip"]),
            "R birth update differs from fixed-key control: scores/work/preclip",
        )


def fresh_descriptor(job, frozen):
    from .relative_run import admit
    from .panel import licensed_admitted_split
    from .relative_data import metadata

    check(
        job == JOBS[FINAL]
        and licensed_admitted_split(job, *admit(job)) == frozen
        and frozen["reservation_sha"] == sha(RESERVATION)
        and frozen["credit_spec"] == SPEC
        and frozen["parent_summary_sha"] == SUMMARY
        and frozen["train_identity"] == V2_IDENTITY,
        "unlicensed or foreign R fresh split",
    )
    r = metadata()
    check(
        r == load(RESERVATION) and len(r["rows"]) == frozen["documents"] == N,
        "R split changed before source open",
    )
    return ROOT / FINAL, r["source"], r["rows"]


def compare(t, wall):
    from .panel_analysis import summarize

    draw = np.random.default_rng(4904).integers(0, N, size=(10000, N))
    out = {}
    shams = {}

    def vals(a, h, sh="none"):
        return np.stack([t[f"{name(a, s)}@1024"][h, sh] for s in SEEDS])

    def summ(x):
        return summarize(x, draw, n=N)

    for r, b in PAIRS:
        pair = f"{r}:{b}"
        v = {h: vals(b, h) - vals(r, h) for h in (128, 1024, T)}
        m = []
        w = wall[pair]
        for s in SEEDS:
            sel = w[str(s)]["selected"]
            if all(sel[a] for a in (r, b)):
                m.append(
                    t[f"{name(b, s)}@{sel[b]['step']}"][T, "none"]
                    - t[f"{name(r, s)}@{sel[r]['step']}"][T, "none"]
                )
        out[pair] = dict(
            full={str(h): summ(x) for h, x in v.items()},
            common_wall=summ(m) if len(m) == 3 else None,
            wall_conclusive=len(m) == 3 and all(x["conclusive"] for x in w.values()),
            horizon_interaction=summ(v[T] - v[128]),
        )
    for a in "RBFC":
        shams[a] = {
            sh: summ(vals(a, T, sh) - vals(a, T))
            for sh in (["literal"] if a == "B" else ["literal", "trace"])
        }
    return dict(contrasts=out, binding=shams)


def finish(root):
    from .panel_analysis import read_stage

    check(Path(root) == ROOT, "foreign R root")
    stage = JOBS[FINAL]
    f = load(root / FINAL / "frozen.json")
    check(f["reservation_sha"] == sha(RESERVATION), "changed fresh membership")
    r = load(RESERVATION)
    t, st, p = read_stage(root, FINAL, r["rows"], job=stage)
    runs = runs_for(p["ref"])
    wall = common_wall(runs, pairs=PAIRS, name=name)
    check(p["wall_pairs"] == wall, "changed R clocks")
    charges = load(root / "charged.json")
    check(
        set(charges) == set(JOBS)
        and all(
            load(root / k / "charge.json") == c
            and 0 < c["upper_card_s"] <= JOBS[k].cap
            and c["rc"] == 0
            for k, c in charges.items()
        ),
        "invalid R charges",
    )
    write_json(
        root / "summary.json",
        dict(
            scope="x49-R v1; no remote work; no original test reuse",
            analysis=compare(t, wall),
            final_stage=st,
            training={
                k: load(root / k / "complete.json")
                for k, j in JOBS.items()
                if j.kind == "train"
            },
            wall=wall,
            new_charges=charges,
            new_card_s=sum(c["upper_card_s"] for c in charges.values()),
            parent_summary_sha=SUMMARY,
            freeze_sha=sha(root / FINAL / "frozen.json"),
        ),
    )
