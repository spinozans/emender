"""PT one-shot locked calibration runner. CPU qualification and publication are distinct from natural admission."""

from math import isfinite, fsum
import os
import shutil
from ..x48_muon_continuation.data import check, load, sha, write_json
from . import (
    confirm_run as prior,
    correction_transfer_panel as p,
    correction_transfer_reader as reader,
    correction_transfer_data as data,
    correction_transfer_check as fixtures,
    correction_transfer_score as score,
    joint_reader as joint,
    followup_loop as loop,
    panel,
    surprise_run as checked,
)
from .meter import MODULE
from .panels import OUT, SEEDS

MOD = MODULE + ".correction_transfer_run"
AUTH = OUT / "correction-transfer-authorization-1.json"
GATES = 211
WIDTH = 256
FIRST = "3bda0492933ca1851b07dd04d26d46f384f6a9144f2c0775dbfc934215c078ce"
CORE = "7f9d793025a9149b84c55579ccadb2e95d254cb29278c778d83054da32abc900"


def source():
    ref = checked.source()
    p.endpoints(hash_weights=False)
    joint._writer()
    check(
        sha(OUT / "correction-transfer-score-gate-1.log") == FIRST
        and sha(OUT / "correction-transfer-score-gate-2.log") == CORE,
        "changed PT fixture history",
    )
    return ref


def qualification():
    check(
        not (data.BASE.exists() or p.ROOT.exists() or AUTH.exists()),
        "late PT qualification",
    )
    return checked.qualify_frozen(
        source,
        "correction-transfer",
        data.SPEC,
        GATES,
        "PT frozen short-score, prefix law and lifecycle checks, three strict CPU loads and invented full-size prices; no PT reservation or text",
        writer_build=joint.COST / "build",
        head_audit=fixtures.audit_head,
    )


def checked_fixture_ref():
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient PT disk")
    ref = checked.checked_gate(source, "correction-transfer", data.SPEC, GATES)
    z = load(OUT / ("qualify-correction-transfer-" + ref) / "passed.json")
    aud = z["head_audit"]
    check(set(aud) == set(map(str, SEEDS)), "missing PT price head")
    for k, a in aud.items():
        v = a["invented_packets"]
        check(
            a["checkpoint_sha"] == z["states"][k]["checkpoint"]
            and a["host_executions"] == a["natural_documents"] == 0
            and {s: r["positions"] for s, r in v.items()} == fixtures.PRICE_LENGTHS
            and v["two_keys_long"]["different_pair_prefix_rows"] > 0,
            "wrong PT inventoried packets",
        )
        for r in v.values():
            costs = [
                r[k]
                for k in (
                    "posthost_ERROR_s",
                    "MARG_extra_s",
                    "PAIR_extra_s",
                    "PREFIX_extra_s",
                    "comparison_only_s",
                )
            ]
            check(
                all(isfinite(t) and t > 0 for t in costs)
                and isfinite(r["posthost_total_s"])
                and r["posthost_total_s"] >= fsum(costs),
                "wrong PT synthetic price",
            )
    checked.check_public_qualification("correction-transfer", ref, data.SPEC, aud)
    return ref


def authority():
    check(AUTH.is_file(), "no separate PT short-calibration budget")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong PT scope")
    return loop.admit(job, p, authority)


def scorer(seed, e):
    ck, b = reader.contract(seed)
    check(ck == e["weights"]["sha256"] and b == e["beta"], "foreign PT reader or seed")
    w = joint._writer()
    return lambda host, head, word, rank: score.document(
        host, head, word, b, seed, rank, w, width=WIDTH
    )


def check_shapes(d, plan_sha, host, head, beta):
    f = d / "shape-check.json"
    check(
        not f.exists() and not (d / "source-open-attempted").exists(),
        "late or repeated PT device check",
    )
    fixtures.short_shapes(host, head, beta, p.LENGTHS, WIDTH, joint._writer())
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
    return prior.evaluate(job, ctx, study="PT")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32, targets_for=p.targets_for)
