"""Pure panel schedule and wall-matching rules; never open score/endpoint payloads here."""

from __future__ import annotations
from dataclasses import dataclass
import math
from .panels import SEEDS, T, OUT, check, fingerprint
from .meter_verify import PAIRS
from .train_stream import V2_IDENTITY

ROOT = OUT / "natural-panel-v2"
ARMS = ("A", "B", "F", "C")
STEPS = 1024
SHARED_S = 4.45 / 12
TOTAL = 22860
MIN_DISK_GiB = 320


@dataclass(frozen=True)
class Job:
    name: str
    kind: str
    cap: int
    arms: str
    seed: int | None = None


def training(arm, seed):
    check(arm in ARMS and seed in SEEDS, "unknown training identity")
    return f"{arm}-{seed}"


def schedule():
    jobs = []
    for pair, cap in [("AB", 900), ("FC", 1200)]:
        for s in SEEDS:
            jobs.extend(Job(training(k, s), "train", 1380, k, s) for k in pair)
        jobs.append(Job("cal-" + pair, "calibration", cap, pair))
    jobs.append(Job("final", "final_test", 4200, "ABFC"))
    return tuple(jobs)


JOBS = {j.name: j for j in schedule()}


def binding(ref, arm, seed):
    from .coupled import ORIGIN

    return dict(
        schema="x49-natural-v2",
        ref=ref,
        origin=ORIGIN,
        train_identity=V2_IDENTITY,
        arm=arm,
        seed=seed,
        horizon=T,
    )


def choose(history, budget):
    check(
        [m["step"] for m in history] == list(range(32, 1025, 32))
        and all(math.isfinite(m["wall_s"]) and m["wall_s"] > 0 for m in history)
        and all(a["wall_s"] < b["wall_s"] for a, b in zip(history, history[1:])),
        "invalid milestone clock",
    )
    admitted = [x for x in history if x["wall_s"] + SHARED_S <= budget]
    if not admitted:
        return None  # inconclusive, never invent an earlier endpoint
    m = admitted[-1]
    wall = m["wall_s"] + SHARED_S
    return dict(
        step=m["step"],
        targets=m["step"] * T,
        checkpoint_sha=m["sha256"],
        wall_s=wall,
        slack=(budget - wall) / budget,
    )


def common_wall(runs, *, pairs=PAIRS, name=training):
    out = {}
    for cand, control in pairs:
        pair = cand + ":" + control
        out[pair] = {}
        for seed in SEEDS:
            selected = {k: runs[name(k, seed)] for k in (cand, control)}
            check(
                all(
                    math.isfinite(t["wall_s"])
                    and t["wall_s"] >= t["milestones"][-1]["wall_s"]
                    for t in selected.values()
                ),
                "terminal wall predates checkpoint",
            )
            budget = min(t["wall_s"] for t in selected.values()) + SHARED_S
            picks = {k: choose(o["milestones"], budget) for k, o in selected.items()}
            out[pair][str(seed)] = dict(
                budget_s=budget,
                selected=picks,
                conclusive=all(
                    v is not None and v["slack"] <= 0.05 for v in picks.values()
                ),
            )
    return out


def validate_authorization(a, ref, protocol_sha):
    expected = dict(
        scope="x49-registered-natural-panel-v2",
        qualified_ref=ref,
        protocol_sha256=protocol_sha,
        total_card_seconds=TOTAL,
        job_caps={j.name: j.cap for j in JOBS.values()},
        gpus=1,
        max_acquisitions=15,
        retries=0,
        remote=False,
        seeds=list(SEEDS),
        train_identity=V2_IDENTITY,
        single_frozen_final=True,
    )
    check(
        isinstance(a, dict)
        and set(a) == set(expected) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in expected}) == fingerprint(expected),
        "different or incomplete natural-panel authorization",
    )
