"""One PC two-phase job. Public CPU qualification precedes natural reservation; a distinct numeric authority precedes a lease."""

from math import isfinite
import shutil
from ..x48_muon_continuation.data import check, load, sha
from . import (
    paired_correction_panel as p,
    paired_correction_data as data,
    surprise_run as prior,
    panel,
    followup_loop as loop,
    joint_reader as joint,
)
from .paired_correction_cost import audit_head
from .paired_correction_fit import Moments as Moments, document as document
from .surprise_run import model as model
from .meter import MODULE
from .panels import OUT, SEEDS, T

MOD = MODULE + ".paired_correction_run"
AUTH = OUT / "paired-correction-authorization-1.json"
GATES = 192
CORE = "30a13b0884b9493958054086897569ac6671191e51889a67b72f338fae44f3b4"


def source():
    ref = prior.source()
    p.bindings(hash_weights=False)
    joint._writer()
    check(
        sha(OUT / "paired-correction-mechanism-gate-1.log") == CORE,
        "changed PC prefix/Fisher gate",
    )
    return ref


def qualification():
    return prior.qualify_frozen(
        source,
        "paired-correction",
        data.SPEC,
        GATES,
        "PC prefix, exact margins, dense-score/Fisher and one-shot lifecycle; three strict CPU loads and full-size invented conditional pricing, no PC text",
        writer_build=joint.COST / "build",
        head_audit=audit_head,
    )


def checked_fixture_ref():
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient PC disk")
    ref = prior.checked_gate(source, "paired-correction", data.SPEC, GATES)
    z = load(OUT / ("qualify-paired-correction-" + ref) / "passed.json")
    a = z["head_audit"]
    check(set(a) == set(map(str, SEEDS)), "missing PC price heads")
    for k, v in a.items():
        cases = v["invented_cpu_packets"]
        check(
            v["checkpoint_sha"] == z["states"][k]["checkpoint"]
            and v["natural_windows"] == v["host_executions"] == 0
            and set(cases)
            == {
                "uniform",
                "recurring_pairs",
                "one_key_wrong_fanout",
                "many_alternatives",
            }
            and cases["recurring_pairs"]["changed_conditional_rows"] > 0
            and all(
                r["positions"] == T
                and 0 <= r["changed_conditional_rows"] <= r["matched_query_rows"] <= T
                and all(
                    isfinite(r[b]) and r[b] > 0
                    for b in (
                        "extra_fit_s",
                        "extra_probe_s",
                        "MARG_chosen_extra_s",
                        "PAIR_chosen_extra_s",
                        "posthost_fit_s",
                        "posthost_probe_s",
                    )
                )
                for r in cases.values()
            ),
            "incorrect PC posthost pricing",
        )
    prior.check_public_qualification("paired-correction", ref, data.SPEC, a)
    return ref


def authority():
    check(AUTH.is_file(), "no separate PC FIT/PROBE budget")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong PC stage")
    return loop.admit(job, p, authority)


def document_options(plan, seed):
    ck, b = p.fixed(seed)
    s = str(seed)
    check(
        plan["fit_dimensions"] == p.DIMS
        and plan["swap_fit"] == "PAIR"
        and plan["spec"] == data.SPEC
        and plan["eh_fit_freeze_sha"] == data.earlier.FIT
        and plan["et_result_sha"] == data.ET_RESULT,
        "foreign PC rules, seal or parent",
    )
    check(
        plan["endpoints"][s]["sha256"] == ck
        and plan["writer_binary_sha"] == p.jt.lineage()["writer_binary_sha"]
        and all(plan[name][s] == b[a] for a, name in p.FIXED.items()),
        "foreign seed, weight, writer or PC anchor",
    )
    return dict(
        ub=b["U"], jb=b["J"], hb=b["HISTORY"], eb=b["ERROR"], writer=joint._writer()
    )


def evaluate(job, ctx):
    return prior.evaluate(job, ctx, study="PC")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32)
