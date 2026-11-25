"""One no-fit ET calibration transfer, with the weight-specific predictors and donor rule sealed before text."""

from ..x48_muon_continuation.data import check, write_json
from . import (
    error_transfer_data as data,
    error_transfer_reader as reader,
    confirm_panel as old,
    surprise_panel as prior,
    transfer_panel as jt,
)
from .error_history import DIMS as DIMS, ARMS
from .panel_spec import Job
from .panels import OUT, SEEDS

ROOT = OUT / "error-transfer-panel-1"
TOTAL = 1200
JOB = Job("et-short-cal", "calibration", TOTAL, ",".join(ARMS))
JOBS = {JOB.name: JOB}
LENGTHS = [1, 16, 64, 65, 4095]


def lineage():
    return dict(
        **jt.lineage(),
        eh_result_sha=reader.PUBLIC,
        eh_close_sha=reader.CLOSED,
        swap_fit="ERROR",
    )


def endpoints(*, hash_weights):
    r = data.parent()
    out = {}
    for s in SEEDS:
        ck, b = reader.contract(s)
        k = str(s)
        _, w = prior.checkpoint(s, hash_weights=hash_weights)
        check(
            ck == w["sha256"] == r["checkpoints"][k]
            and b == dict(U=r["u_beta"][k], J=r["j_beta"][k], **r["beta"][k])
            and set(b) == {"U", "J", *DIMS},
            "ET checkpoint/coefficient mismatch",
        )
        out[k] = dict(weights=w, beta=b)
    return out


def plan(ref, *, hash_weights):
    return old.plan(ref, hash_weights=hash_weights, study="ET")


def expected(ref):
    return old.expected(ref, study="ET")


def validate(a, ref):
    return old.validate(a, ref, study="ET")


def freeze(job, ref):
    return old.freeze(job, ref, study="ET")


def targets_for(job, d, start):
    return old.targets_for(job, d, start, study="ET")


def fresh_descriptor(job, initial):
    return old.fresh_descriptor(job, initial, study="ET")


def finish(root):
    from .error_transfer_analysis import analyze

    check(root == ROOT, "wrong ET root")
    write_json(root / "summary.json", analyze(root))
