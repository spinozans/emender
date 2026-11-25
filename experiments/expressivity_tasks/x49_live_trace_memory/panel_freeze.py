"""Bind endpoints, exposure and elapsed time from training receipts alone. Never read evaluation text or scores."""

from __future__ import annotations
import json
from ..x48_muon_continuation.data import load, sha, write_json
from .panel_spec import (
    JOBS,
    ROOT,
    SEEDS,
    STEPS,
    T,
    binding,
    check,
    common_wall,
    training,
)
from .train_stream import V2_IDENTITY


def read_training(root, arm, seed, ref):
    d = root / training(arm, seed)
    complete = load(d / "complete.json")
    paid = load(d / "charge.json")
    b = binding(ref, arm, seed)
    check(
        complete["binding"] == b
        and complete["kind"] == "train"
        and complete["units"] == STEPS
        and complete["targets"] == STEPS * T
        and paid["rc"] == 0
        and paid["witnessed_units"] == STEPS
        and paid["renewed"]
        and 0 < paid["upper_card_s"] <= JOBS[d.name].cap
        and paid["complete_sha"] == sha(d / "complete.json"),
        "incomplete or foreign training trajectory",
    )
    check(
        complete["steps_sha256"] == sha(d / "steps.jsonl")
        and complete["clock_sha256"] == sha(d / "clock.jsonl"),
        "changed training receipts",
    )
    steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
    check(
        [s["step"] for s in steps] == list(range(1, STEPS + 1))
        and all(
            s["targets"] == T and 0 <= s["preupdate_nll"] < float("inf") for s in steps
        ),
        "incorrect training dose",
    )
    clock = [json.loads(s) for s in (d / "clock.jsonl").read_text().splitlines()]
    rec = load(d / "latest.json")
    check(
        rec["step"] == STEPS
        and rec["binding"] == b
        and rec["file"] == "step-1024.pt"
        and rec["bytes"] == (d / "step-1024.pt").stat().st_size,
        "invalid full recovery cursor",
    )
    start = load(d / "start-state.json")
    check(
        start["binding"] == b
        and [m["step"] for m in clock] == list(range(32, STEPS + 1, 32))
        and complete["final_weights"] == clock[-1]["sha256"]
        and all(0 < m["wall_s"] <= paid["inclusive_wall_s"] for m in clock)
        and all(x["wall_s"] < y["wall_s"] for x, y in zip(clock, clock[1:]))
        and complete["final_recovery"] == rec["sha256"],
        "clock or recovery mismatch",
    )
    return dict(
        wall_s=paid["inclusive_wall_s"],
        milestones=clock,
        birth=start["birth"],
        charge_sha=sha(d / "charge.json"),
    )


def freeze(job, ref, root=ROOT):
    check(job.kind in ("calibration", "final_test"), "not an evaluation stage")
    p = root / job.name / "frozen.json"
    check(not p.exists(), "evaluation already frozen")
    runs = {
        training(k, s): read_training(root, k, s, ref) for k in job.arms for s in SEEDS
    }
    endpoints = {}
    walls = common_wall(runs) if job.kind == "final_test" else {}

    def pin(k, s, step):
        name = training(k, s)
        key = f"{name}@{step}"
        if key in endpoints:
            return
        m = runs[name]["milestones"][step // 32 - 1]
        d = root / name
        filename = f"weights-{step:04d}.pt"
        path = d / filename
        note = load(path.with_suffix(".json"))
        b = binding(ref, k, s)
        check(
            m["step"] == step
            and note
            == dict(
                binding=b,
                bytes=m["bytes"],
                file=filename,
                sha256=m["sha256"],
                step=step,
            )
            and path.stat().st_size == m["bytes"]
            and sha(path) == m["sha256"],
            "selected weights not bound to actual bytes",
        )
        endpoints[key] = dict(
            arm=k, seed=s, step=step, bytes=m["bytes"], sha256=m["sha256"], binding=b
        )

    for s in SEEDS:
        births = {k: runs[training(k, s)]["birth"] for k in job.arms}
        check(
            len({v["host"] for v in births.values()})
            == len({v["literal"] for v in births.values()})
            == 1
            and all(
                len({births[k]["head"] for k in group if k in births}) <= 1
                for group in ("AB", "FC")
            ),
            "unequal paired origin",
        )
        for k in job.arms:
            pin(k, s, STEPS)
        for pairs in walls.values():
            for k, v in pairs[str(s)]["selected"].items():
                if v:
                    pin(k, s, v["step"])
    n = 110 if job.kind == "calibration" else 128
    check(
        len(endpoints) <= len(job.arms) * len(SEEDS) + (12 if walls else 0),
        "too many common-wall endpoints",
    )
    stages = {}
    if job.kind == "final_test":
        for name in ("cal-AB", "cal-FC"):
            d = root / name
            r = load(d / "complete.json")
            c = load(d / "charge.json")
            check(
                r["kind"] == "calibration"
                and c["rc"] == 0
                and c["complete_sha"] == sha(d / "complete.json")
                and c["witnessed_units"] == r["units"] == 6 * 110,
                "uncertified calibration stage",
            )
            stages[name] = sha(d / "charge.json")
    frozen = dict(
        schema="x49-frozen-v2",
        ref=ref,
        job=job.name,
        split=job.kind,
        train_identity=V2_IDENTITY,
        visit=0,
        horizons=[128, 1024, T],
        documents=n,
        units=n * len(endpoints),
        endpoints=endpoints,
        wall_pairs=walls,
        training_charges={k: v["charge_sha"] for k, v in runs.items()},
        calibration_charges=stages,
    )
    write_json(p, frozen)
    return sha(p)
