"""X49-V runner. No remote, no new final, one registered two-phase inference lease."""

from __future__ import annotations
import shutil
from ..x48_muon_continuation.data import check, load, sha
from . import (
    followup_loop as loop,
    panel,
    alternative_data as data,
    alternative_panel as p,
    surprise_run as prior,
    local_u,
)
from .alternative_fit import Moments as Moments, document as document
from .surprise_run import model as model
from .meter import MODULE
from .panels import OUT

MOD = MODULE + ".alternative_run"
AUTH = OUT / "alternative-authorization-1.json"
GATES = 109
CORE_GATE = "e980fcbe0bc63da2a48701244d16e9ac23291dd6b703b213b7160cfcf21f601f"


def source():
    ref = prior.source()
    p.bindings(hash_weights=False)
    check(
        sha(OUT / "alternative-mechanism-gate-1.log") == CORE_GATE,
        "unbound V mechanism gate",
    )
    return ref


def qualification():
    return prior.qualify_frozen(
        source,
        "alternative",
        data.SPEC,
        GATES,
        "V synthetic tape, non-copy scoring/Fisher, paired analysis, two-phase lifecycle and strict U model loads; no V text",
    )


def checked_fixture_ref():
    return prior.checked_gate(source, "alternative", data.SPEC, GATES)


def authority():
    check(AUTH.is_file(), "no separate V FIT/PROBE budget")
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient V disk")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong V stage")
    return loop.admit(job, p, authority)


def document_options(plan, seed):
    ck, b = local_u.contract(seed)
    key = str(seed)
    check(
        plan["u_confirmation_sha"] == local_u.CONFIRMATION
        and plan["u_refits"] == 0
        and plan["endpoints"][key]["sha256"] == ck
        and plan["u_beta"][key] == list(b),
        "foreign seed, U base or U coefficients",
    )
    return dict(u_beta=b)


def evaluate(job, ctx):
    return prior.evaluate(job, ctx, study="V")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32)
