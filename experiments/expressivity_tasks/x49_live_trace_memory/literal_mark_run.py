"""Single local LS FIT/freeze/PROBE job. No neural updates, incumbent refits or final reader."""

from __future__ import annotations
import shutil
from ..x48_muon_continuation.data import check, load, sha
from . import (
    followup_loop as loop,
    panel,
    literal_mark_data as data,
    literal_mark_panel as p,
    surprise_run as prior,
)
from .literal_mark_fit import Moments as Moments, document as document
from .surprise_run import model as model
from .alternative_run import document_options as document_options
from .meter import MODULE
from .panels import OUT

MOD = MODULE + ".literal_mark_run"
AUTH = OUT / "literal-surprise-authorization-1.json"
GATES = 120
CORE_GATE = "ceef17c820f2d200ae7091649dc4a46d0142753387bcf4423cb5f2ac768f7219"


def source():
    ref = prior.source()
    p.bindings(hash_weights=False)
    check(
        sha(OUT / "literal-surprise-mechanism-gate-1.log") == CORE_GATE,
        "unbound LS mechanism gate",
    )
    return ref


def qualification():
    return prior.qualify_frozen(
        source,
        "literal-surprise",
        data.SPEC,
        GATES,
        "LS dense literal/mark/fit oracles, cohort/phase isolation, paired reductions and strict original U host/head loads; no natural LS access",
    )


def checked_fixture_ref():
    return prior.checked_gate(source, "literal-surprise", data.SPEC, GATES)


def authority():
    check(AUTH.is_file(), "no separate LS FIT/PROBE budget")
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient LS disk")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong LS stage")
    return loop.admit(job, p, authority)


def evaluate(job, ctx):
    return prior.evaluate(job, ctx, study="LS")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32)
