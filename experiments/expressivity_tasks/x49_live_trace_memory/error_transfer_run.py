"""ET: one short-calibration source open, after published CPU checks and bound no-fit device parity."""

from math import isfinite
import os
import shutil
from time import perf_counter
import torch
from ..x48_muon_continuation.data import check, load, sha, write_json
from . import (
    confirm_run as prior,
    error_transfer_data as data,
    error_transfer_panel as p,
    error_transfer_score as score,
    error_transfer_reader as reader,
    error_history as eh,
    error_history_cost as synthetic,
    followup_loop as loop,
    joint_reader as joint,
    literal_mark as ls,
    panel,
    surprise_run as checked,
)
from .endpoint import frozen_prefill
from .meter import MODULE
from .panels import OUT, SEEDS

MOD = MODULE + ".error_transfer_run"
AUTH = OUT / "error-transfer-authorization-1.json"
GATES = 179
WIDTH = 256
CORE = "f2885fff4495d824ab657d1b0293eb58e423193e4b8254b409b2cbb42b91dac6"


def source():
    ref = checked.source()
    p.endpoints(hash_weights=False)
    joint._writer()
    check(
        sha(OUT / "error-transfer-score-gate-1.log") == CORE,
        "unbound ET short score gate",
    )
    return ref


@torch.inference_mode()
def audit_head(seed, endpoint, head):
    ck, b = reader.contract(seed)
    check(ck == endpoint["sha256"], "foreign ET price head")
    cases = synthetic.invented()
    writer = joint._writer()
    g = torch.Generator().manual_seed(4916)
    h = torch.randn(4095, head.direct.host_dim, generator=g)
    z = torch.randn(4095, 256, generator=g)
    clocks = {}
    for i, (name, n, key) in enumerate(
        [
            ("mixed_short", 1023, "recurring_mixed"),
            ("uniform_short", 1023, "uniform_bytes"),
            ("mixed_long", 4095, "recurring_mixed"),
            ("one_key_stress", 4095, "one_key_many_wrong_bytes"),
        ]
    ):
        w = cases[key][:n]
        if name == "one_key_stress":
            z = torch.zeros(n, 256)
            z[torch.arange(n), torch.arange(n) % 256] = 3.0
        tic = perf_counter()
        lp, _, row = score._answers(
            head, w, h[:n], z[:n], b, seed, i, writer, pre_s=0.0, tic=tic
        )
        wall = perf_counter() - tic
        clock = row["seconds"]
        check(
            set(lp) == set(eh.ARMS)
            and row["targets"] == n
            and row["work"]["pool_active_rows"] > 0,
            "incomplete ET price packet",
        )
        clocks[name] = dict(
            positions=n,
            posthost_total_s=wall,
            posthost_J_s=clock["shared_J_s"],
            HISTORY_extra_s=clock["HISTORY_branch_s"],
            ERROR_extra_s=clock["ERROR_branch_s"],
            comparison_only_s=clock["comparison_only_s"],
            active_rows=row["work"]["pool_active_rows"],
        )
        del lp
    return dict(
        checkpoint_sha=ck,
        invented_packets=clocks,
        natural_documents=0,
        host_executions=0,
    )


def qualification():
    return checked.qualify_frozen(
        source,
        "error-transfer",
        data.SPEC,
        GATES,
        "ET locked short predictions, cost and lifecycle checks; three strict CPU loads and post-host pricing on invented packets, no ET source read",
        writer_build=joint.COST / "build",
        head_audit=audit_head,
    )


def checked_fixture_ref():
    check(shutil.disk_usage(OUT).free >= 32 << 30, "insufficient ET disk")
    ref = checked.checked_gate(source, "error-transfer", data.SPEC, GATES)
    z = load(OUT / ("qualify-error-transfer-" + ref) / "passed.json")
    aud = z["head_audit"]
    check(set(aud) == set(map(str, SEEDS)), "missing ET price head")
    for k, a in aud.items():
        v = a["invented_packets"]
        check(
            a["checkpoint_sha"] == z["states"][k]["checkpoint"]
            and a["host_executions"] == a["natural_documents"] == 0
            and {name: r["positions"] for name, r in v.items()}
            == {
                "mixed_short": 1023,
                "uniform_short": 1023,
                "mixed_long": 4095,
                "one_key_stress": 4095,
            }
            and all(
                r["active_rows"] > 0
                and all(
                    isfinite(r[t]) and r[t] > 0
                    for t in (
                        "posthost_total_s",
                        "posthost_J_s",
                        "HISTORY_extra_s",
                        "ERROR_extra_s",
                        "comparison_only_s",
                    )
                )
                and r["posthost_total_s"]
                >= sum(
                    r[t]
                    for t in (
                        "posthost_J_s",
                        "HISTORY_extra_s",
                        "ERROR_extra_s",
                        "comparison_only_s",
                    )
                )
                for r in v.values()
            ),
            "invalid ET CPU price packet",
        )
    checked.check_public_qualification("error-transfer", ref, data.SPEC, aud)
    return ref


def authority():
    check(AUTH.is_file(), "no separate ET short-calibration budget")
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong ET stage")
    return loop.admit(job, p, authority)


def scorer(seed, e):
    ck, b = reader.contract(seed)
    check(
        ck == e["weights"]["sha256"] and b == e["beta"],
        "foreign ET deployment coefficients",
    )
    w = joint._writer()
    return lambda host, head, word, rank: score.document(
        host, head, word, b, seed, rank, w, width=WIDTH
    )


@torch.inference_mode()
def check_shapes(d, plan_sha, host, head, beta):
    f = d / "shape-check.json"
    check(
        not f.exists() and not (d / "source-open-attempted").exists(),
        "late/repeated ET device check",
    )
    writer = joint._writer()
    fixture = b" key=\0\xff\x1e; color: b! key=\0\xff\x1e; color: r! " * 130
    for i, n in enumerate(p.LENGTHS):
        w = fixture[:n]
        tic = perf_counter()
        h, z, pre = frozen_prefill(host, w, WIDTH)
        out, _, row = score._answers(
            head, w, h, z, beta, SEEDS[0], i, writer, pre_s=pre, tic=tic
        )
        ref = eh.fields(head, w, h, z, beta["J"], ls.text_tape, ub=beta["U"])
        expected = dict(
            ref.anchors,
            **{
                a: eh.response(ref, a, beta["ERROR" if a == "SWAP" else a])
                for a in (*eh.DIMS, "SWAP")
            },
        )
        check(
            row["targets"] == n
            and row["work"]["changed_histogram_rows"]
            == ref.work["changed_histogram_rows"]
            and all(torch.equal(out[k], ref.anchors[k]) for k in ref.anchors)
            and all(
                torch.allclose(out[k], expected[k], rtol=0, atol=3e-14) for k in eh.ARMS
            ),
            "ET short native/device score mismatch",
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
    return prior.evaluate(job, ctx, study="ET")


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32, targets_for=p.targets_for)
