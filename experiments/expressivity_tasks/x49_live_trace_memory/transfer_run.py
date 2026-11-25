"""JT: one separately admitted inference-only stage for short native calibration. Never fit on this split."""

from __future__ import annotations
from dataclasses import replace
import os
import shutil
import torch
from ..x48_muon_continuation.data import check, load, sha, write_json
from . import (
    confirm_run as prior,
    followup_loop as loop,
    joint_reader as joint,
    literal_mark as ls,
    panel,
    surprise_run as frozen,
    transfer_data as data,
    transfer_panel as p,
    transfer_score as score,
)
from .meter import MODULE
from .panels import OUT, SEEDS

MOD = MODULE + ".transfer_run"
AUTH = OUT / "joint-transfer-authorization-1.json"
GATES = 132
CORE_GATE = "3f38b734ac348081e05ed830ae9af87059dc790b912b4cb70bb47dcaa22793be"
WIDTH = 256


def source():
    ref = frozen.source()
    p.endpoints(hash_weights=False)
    check(
        sha(OUT / "joint-transfer-score-gate-1.log") == CORE_GATE,
        "unbound JT score/clock fixture",
    )
    return ref


def qualification():
    # Check that the receipt-bound build is actually callable before any reservation.
    panel.cpu()
    t, w = joint._writer()(b"\x00\xff\xff")
    py, work = ls.text_tape(b"\x00\xff\xff")
    check(torch.equal(t, py) and w == work, "invalid qualified native writer")
    return frozen.qualify_frozen(
        source,
        "joint-transfer",
        data.SPEC,
        GATES,
        "JT exact frozen short scores, cost clocks, no-fit calibration-only lifecycle, variable target receipts and strict CPU host/head loads. No JT natural access.",
        writer_build=joint.COST / "build",
    )


def checked_fixture_ref():
    return frozen.checked_gate(source, "joint-transfer", data.SPEC, GATES)


def authority():
    check(
        AUTH.is_file() and shutil.disk_usage(OUT).free >= 32 << 30,
        "no separate JT budget or insufficient disk",
    )
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong JT job")
    return loop.admit(job, p, authority)


def scorer(seed, e):
    writer = joint._writer()
    beta = e["beta"]
    return lambda host, head, word, rank: score.document(
        host, head, word, beta, seed, rank, writer, width=WIDTH
    )


@torch.inference_mode()
def check_shapes(d, plan_sha, host, head, beta):
    f = d / "shape-check.json"
    check(
        not f.exists() and not (d / "source-open-attempted").exists(),
        "late or repeated JT shape check",
    )
    fixture = b"then: \0\xffabc\x1e abc " * 300
    writer = joint._writer()
    for i, n in enumerate(p.LENGTHS):
        obs = fixture[:n]
        lp, packet, row = score.answers(
            host, head, obs, beta, SEEDS[0], i, writer, width=WIDTH
        )
        # Check the actual short GPU-host-derived packets, at one synthetic prefill per length.
        # Reference uses the Python writer and LS response, NOT the native scored J output.
        tape, work = ls.text_tape(obs)
        native, nw = writer(obs)
        h = packet.phi["U"]
        t = tape[packet.latest].double()[:, None] / 64
        ref = replace(
            packet, phi=dict(H=h, J=torch.cat((h[:, :3], t, h[:, 3:]), dim=1))
        )
        check(
            row["targets"] == n
            and row["host_prefills"] == 1
            and torch.equal(tape, native)
            and nw == work
            and all(row["work"][k] == v for k, v in work.items())
            and torch.equal(lp["B"], packet.base.double())
            and all(
                torch.equal(lp[a], ls.response(ref, a, beta[a]))
                for a in ("H", "U", "J")
            ),
            "JT device shape/score mismatch",
        )
    write_json(
        f,
        dict(
            plan_sha=plan_sha,
            trainer=os.getpid(),
            trainer_start=load(d / "identity.json")["trainer_start"],
            seed=SEEDS[0],
            lengths=p.LENGTHS,
            natural_targets=0,
            synthetic_prefills=len(p.LENGTHS),
        ),
    )


def evaluate(job, ctx):
    return prior.evaluate(job, ctx, study="JT")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32, targets_for=p.targets_for)
