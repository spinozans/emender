"""Only a no-fit JT job and its exact short calibration population. U, LS and the native writer stay frozen."""

from __future__ import annotations
from ..x48_muon_continuation.data import check, write_json
from . import (
    confirm_panel as old,
    literal_mark_panel as ls,
    joint_reader as joint,
    local_u,
    transfer_data as data,
    surprise_panel as prior,
)
from .literal_mark import DIMS
from .surprise import checked_vector
from .panel_spec import Job
from .panels import OUT, SEEDS, fingerprint

ROOT = OUT / "joint-transfer-panel-1"
TOTAL = 1200
JOB = Job("jt-short-cal", "calibration", TOTAL, "BUHJ")
JOBS = {JOB.name: JOB}
LENGTHS = [1, 16, 64, 65, 4095]


def lineage():
    note = data.u.bound(joint.RECEIPTS / "joint_reader_cost.json", data.COST_SHA)
    return dict(
        ls_result_sha=data.RESULT,
        u_confirmation_sha=local_u.CONFIRMATION,
        writer_cost_sha=data.COST_SHA,
        writer_binary_sha=note["build"]["library_sha256"],
    )


def endpoints(*, hash_weights):
    r = data.parent()
    seal = data.u.bound(ls.ROOT / ls.JOB.name / "fit-frozen.json", data.FIT_SHA)
    check(
        seal["spec"] == data.ls.SPEC
        and seal["plan_sha"] == r["freeze_sha"]
        and set(seal["seeds"]) == set(r["beta"]) == set(map(str, SEEDS)),
        "foreign fixed LS coefficients",
    )
    out = {}
    for s in SEEDS:
        k = str(s)
        _, w = prior.checkpoint(s, hash_weights=hash_weights)
        e = seal["seeds"][k]
        jb, j = joint.contract(s)
        ub, u = local_u.contract(s)
        check(
            e["checkpoint_sha"] == r["checkpoints"][k] == w["sha256"] == jb == ub
            and set(e["beta"]) == set(DIMS)
            and fingerprint(e["beta"]) == fingerprint(r["beta"][k])
            and list(j) == e["beta"]["J"]
            and list(u) == r["u_beta"][k],
            "JT checkpoint/seed/coefficient mismatch",
        )
        for a in ("H", "J"):
            checked_vector(DIMS, a, e["beta"][a])
        out[k] = dict(weights=w, beta=dict(U=list(u), H=e["beta"]["H"], J=list(j)))
    return out


def plan(ref, *, hash_weights):
    return old.plan(ref, hash_weights=hash_weights, study="JT")


def expected(ref):
    return old.expected(ref, study="JT")


def validate(a, ref):
    return old.validate(a, ref, study="JT")


def freeze(job, ref):
    return old.freeze(job, ref, study="JT")


def targets_for(job, d, start):
    return old.targets_for(job, d, start, study="JT")


def fresh_descriptor(job, initial):
    return old.fresh_descriptor(job, initial, study="JT")


def finish(root):
    from .transfer_analysis import analyze

    check(root == ROOT, "wrong JT closure root")
    write_json(root / "summary.json", analyze(root))
