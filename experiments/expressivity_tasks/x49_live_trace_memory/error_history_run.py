"""One registered EH job; paid work requires separate local authority and the published CPU pass."""

from math import isfinite
import shutil
from ..x48_muon_continuation.data import check, load, sha
from . import (
    error_history_data as data,
    error_history_panel as p,
    followup_loop as loop,
    panel,
    surprise_run as prior,
    decode_run,
    joint_reader as joint,
)
from .error_history_fit import Moments as Moments, document as document
from .error_history_cost import audit_head
from .surprise_run import model as model
from .meter import MODULE
from .panels import OUT, SEEDS, T

MOD = MODULE + ".error_history_run"
AUTH = OUT / "error-history-authorization-1.json"
GATES = 164
CORE = "9492084fa9bcac9137ddfcda197deba85374639727e911c46bf831dd01291496"


def source():
    ref = prior.source()
    p.bindings(hash_weights=False)
    joint._writer()
    check(
        sha(OUT / "error-history-mechanism-gate-1.log") == CORE,
        "changed EH synthetic gate",
    )
    return ref


def qualification():
    return prior.qualify_frozen(
        source,
        "error-history",
        data.SPEC,
        GATES,
        "EH exhaustive live-count, all-score/Fisher and lifecycle checks; three strict CPU loads and invented full-size packet pricing, no natural EH text",
        writer_build=joint.COST / "build",
        head_audit=audit_head,
    )


def checked_fixture_ref():
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient EH disk")
    ref = prior.checked_gate(source, "error-history", data.SPEC, GATES)
    z = load(OUT / ("qualify-error-history-" + ref) / "passed.json")
    aud = z["head_audit"]
    check(set(aud) == set(map(str, SEEDS)), "missing EH price head")
    for k, r in aud.items():
        cases = r["invented_cpu_packets"]
        check(
            r["checkpoint_sha"] == z["states"][k]["checkpoint"]
            and r["natural_windows"] == r["host_executions"] == 0
            and set(cases)
            == {"recurring_mixed", "uniform_bytes", "one_key_many_wrong_bytes"}
            and all(
                v["positions"] == T
                and v["active_rows"] > 0
                and all(
                    isfinite(v[m]) and v[m] > 0
                    for m in ("extra_fit_s", "extra_probe_s")
                )
                for v in cases.values()
            ),
            "missing EH invented price check",
        )
    prior.check_public_qualification("error-history", ref, data.SPEC, aud)
    return ref


def authority():
    check(AUTH.is_file(), "no separate EH FIT/PROBE budget")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong EH stage")
    return loop.admit(job, p, authority)


def document_options(plan, seed):
    check(
        plan["fit_dimensions"] == p.DIMS
        and plan["swap_fit"] == "ERROR"
        and plan["spec"] == data.SPEC,
        "foreign EH features or swap rule",
    )
    return decode_run.document_options(plan, seed)


def evaluate(job, ctx):
    return prior.evaluate(job, ctx, study="EH")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32)
