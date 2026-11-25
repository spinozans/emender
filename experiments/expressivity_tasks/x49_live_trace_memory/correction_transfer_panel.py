"""One no-refit PT transfer. Tie all three deployed alternatives to the original seven vectors and weight."""

from ..x48_muon_continuation.data import check, sha, write_json
from . import (
    confirm_panel as old,
    correction_transfer_data as data,
    correction_transfer_reader as reader,
    transfer_panel as jt,
    surprise_panel as prior,
)
from .correction_transfer_score import ARMS
from .paired_correction import DIMS as DIMS
from .panels import OUT, SEEDS
from .panel_spec import Job

ROOT = OUT / "correction-transfer-panel-1"
TOTAL = 1200
JOB = Job("pt-short-cal", "calibration", TOTAL, ",".join(ARMS))
JOBS = {JOB.name: JOB}
LENGTHS = [1, 16, 64, 65, 4095]


def lineage():
    return dict(
        **jt.lineage(),
        pc_result_sha=reader.PUBLIC,
        pc_close_sha=reader.CLOSED,
        eh_fit_freeze_sha=data.pc.earlier.FIT,
        et_result_sha=data.pc.ET_RESULT,
        swap_fit="PAIR",
        prefix_fit="PAIR",
    )


def endpoints(*, hash_weights):
    parent = data.parent()
    out = {}
    for seed in SEEDS:
        s = str(seed)
        ck, b = reader.contract(seed)
        _, e = prior.checkpoint(seed, hash_weights=hash_weights)
        check(
            ck == e["sha256"] == parent["checkpoints"][s]
            and all(b[a] == parent["beta"][s][a] for a in DIMS)
            and set(b) == {"U", "J", "HISTORY", "ERROR", *DIMS},
            "PT weight or prior coefficient mismatch",
        )
        out[s] = dict(weights=e, beta=b)
    return out


def qualification_sha(ref):
    return sha(OUT / ("qualify-correction-transfer-" + ref) / "passed.json")


def plan(ref, *, hash_weights):
    return dict(
        old.plan(ref, hash_weights=hash_weights, study="PT"),
        qualification_sha=qualification_sha(ref),
    )


def expected(ref):
    return dict(old.expected(ref, study="PT"), qualification_sha=qualification_sha(ref))


def validate(a, ref):
    return old.validate(a, ref, study="PT")


def freeze(job, ref):
    return old.freeze(job, ref, study="PT")


def targets_for(job, d, start):
    return old.targets_for(job, d, start, study="PT")


def fresh_descriptor(job, initial):
    return old.fresh_descriptor(job, initial, study="PT")


def finish(root):
    from .correction_transfer_analysis import analyze

    check(root == ROOT, "wrong PT root")
    write_json(root / "summary.json", analyze(root))
