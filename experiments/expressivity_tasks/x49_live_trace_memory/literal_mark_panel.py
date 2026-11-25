"""LS receipts: separate native TRAIN cohort/job; seal all 15 fits before access to any PROBE text."""

from __future__ import annotations
from ..x48_muon_continuation.data import check, write_json
from . import (
    literal_mark_data as data,
    alternative_panel as previous,
    surprise_panel as old,
)
from .literal_mark import DIMS as DIMS
from .literal_mark_fit import Moments
from .surprise_fit import RIDGE as RIDGE
from .panels import SEEDS, OUT
from .panel_spec import Job

ROOT = OUT / "literal-surprise-probe-1"
SCOPE = "x49-ls-bound-text-and-neural-surprise-v1"
TOTAL = 1200
JOB = Job("ls-fit-probe", "fit_then_probe", TOTAL, "NTHJSU")
JOBS = {JOB.name: JOB}


def units():
    return len(SEEDS) * sum(data.N.values())


def bindings(*, hash_weights):
    return previous.bindings(hash_weights=hash_weights, study="LS")


def plan(ref, *, hash_weights):
    return previous.plan(ref, hash_weights=hash_weights, study="LS")


def expected(ref):
    return previous.expected(ref, study="LS")


def validate(a, ref):
    return previous.validate(a, ref, study="LS")


def freeze(job, ref):
    return previous.freeze(job, ref, study="LS")


def checked_fit(state):
    old.stored_fit(state, Moments, data.N["fit"])


def fit_row_ok(row):
    from .literal_mark_analysis import checked_text

    checked_text(row)
    return old.fit_row_ok(row)


def readout_map(d, initial):
    return old.readout_map(d, initial, study="LS")


def freeze_readouts(d, initial):
    return old.freeze_readouts(d, initial, study="LS")


def checked_readouts(d, initial, fit_sha):
    return old.checked_readouts(d, initial, fit_sha, study="LS")


def finish(root):
    from .literal_mark_analysis import analyze

    check(root == ROOT, "wrong LS root")
    write_json(root / "summary.json", analyze(root))
