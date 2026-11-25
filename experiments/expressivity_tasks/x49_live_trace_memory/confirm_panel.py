"""Exactly one no-refit final using U's coefficient bytes, W-B checkpoints and fresh short membership."""

from __future__ import annotations
from itertools import accumulate
import os
from ..x48_muon_continuation.data import check, load, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from . import confirm_data as data, surprise_panel as prior
from .panels import OUT, T, SEEDS, fingerprint
from .panel_spec import Job
from .surprise import checked_beta, DIMS
from .short_phase import study as phase

ROOT = OUT / "confirm-panel-1"
TOTAL = 1200
JOB = Job("uc-final", "final_test", TOTAL, "BNUS")
JOBS = {JOB.name: JOB}
LENGTHS = [1, 16, 64, 65, 4095]


def endpoints(*, hash_weights):
    r = data.parent()
    sealed = data.u.bound(prior.ROOT / prior.JOB.name / "fit-frozen.json", data.FIT_SHA)
    check(
        sealed["spec"] == data.u.SPEC
        and sealed["plan_sha"] == r["freeze_sha"]
        and set(sealed["seeds"]) == set(r["beta"]) == set(map(str, SEEDS)),
        "foreign fixed coefficients",
    )
    out = {}
    for s in SEEDS:
        k = str(s)
        _, w = prior.checkpoint(s, hash_weights=hash_weights)
        e = sealed["seeds"][k]
        beta = e["beta"]
        check(
            e["checkpoint_sha"] == r["checkpoints"][k] == w["sha256"]
            and fingerprint(beta) == fingerprint(r["beta"][k])
            and set(beta) == set(DIMS),
            "U checkpoint/seed/coefficient mismatch",
        )
        for a, v in beta.items():
            checked_beta(a, v)
        out[k] = dict(weights=w, beta=beta)
    return out


def plan(ref, *, hash_weights, study="UC"):
    dat, p, _, _ = phase(study)
    m = dat.metadata(ref)
    parent = {"u_result_sha": dat.RESULT} if study == "UC" else p.lineage()
    return dict(
        ref=ref,
        job=p.JOB.name,
        kind=p.JOB.kind,
        spec=dat.SPEC,
        **parent,
        fit_freeze_sha=dat.FIT_SHA,
        history=T,
        new_fits=0,
        neural_updates=0,
        data=m,
        units=m["documents"] * len(SEEDS),
        targets_per_endpoint=m["targets"],
        endpoints=p.endpoints(hash_weights=hash_weights),
    )


def expected(ref, *, study="UC"):
    dat, p, _, _ = phase(study)
    z = p.plan(ref, hash_weights=False)
    parent = {"u_result_sha": dat.RESULT} if study == "UC" else p.lineage()
    return dict(
        scope={
            "UC": "x49-u-frozen-native-short-confirmation-v1",
            "JT": "x49-jt-fixed-short-calibration-v1",
            "ET": "x49-et-locked-short-error-histories-v1",
            "PT": "x49-pt-locked-short-correction-pairs-v1",
        }[study],
        ref=ref,
        spec=dat.SPEC,
        **parent,
        fit_freeze_sha=dat.FIT_SHA,
        data=z["data"],
        units=z["units"],
        checkpoint_and_beta_id=fingerprint(z["endpoints"]),
        new_fits=0,
        neural_updates=0,
        split="new_native_short_final"
        if study == "UC"
        else "new_native_short_calibration_no_fitting",
        local_gpus=1,
        max_acquisitions=1,
        total_card_seconds=p.TOTAL,
        job_caps={p.JOB.name: p.TOTAL},
        retries=0,
        remote=False,
    )


def validate(a, ref, *, study="UC"):
    _, p, _, _ = phase(study)
    e = p.expected(ref)
    check(
        type(a) is dict
        and set(a) == set(e) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in e}) == fingerprint(e),
        f"wrong {study} confirmation authority",
    )


def freeze(job, ref, *, study="UC"):
    _, p, _, _ = phase(study)
    f = p.ROOT / p.JOB.name / "frozen.json"
    check(job == p.JOB and not f.exists(), "foreign/repeated confirmation freeze")
    write_json(f, p.plan(ref, hash_weights=True))
    flush_dir(f.parent)
    return sha(f)


def targets_for(job, d, start, *, study="UC"):
    dat, p, _, _ = phase(study)
    z = dat.u.bound(d / "frozen.json", start["freeze_sha"])
    rows, nt = dat.checked_rows(
        dat.u.bound(dat.BASE / "reservation.json", z["data"]["rows_sha"])
    )
    counts = list(accumulate([r["length"] - 1 for r in rows] * len(SEEDS), initial=0))
    check(
        job == p.JOB
        and start["units"] == z["units"] == len(counts) - 1
        and counts[-1] == len(SEEDS) * nt
        and z["targets_per_endpoint"] == nt,
        "wrong target witness schedule",
    )
    return counts.__getitem__


def fresh_descriptor(job, initial, *, study="UC"):
    from .panel import licensed_admitted_split

    dat, p, run, _ = phase(study)
    check(job == p.JOB, "wrong frozen short reader")
    d, ref, start = run.admit(job)
    check(
        d == p.ROOT / p.JOB.name
        and licensed_admitted_split(job, d, ref, start)
        == initial
        == p.plan(ref, hash_weights=True),
        "unlicensed or changed frozen confirmation",
    )
    check(
        (d / "shape-check.json").is_file(),
        "missing short device check before raw access",
    )
    z = load(d / "shape-check.json")
    ident = load(d / "identity.json")
    check(
        z
        == dict(
            plan_sha=start["freeze_sha"],
            trainer=os.getpid(),
            trainer_start=ident["trainer_start"],
            seed=SEEDS[0],
            lengths=p.LENGTHS,
            natural_targets=0,
            synthetic_prefills=len(p.LENGTHS),
        ),
        "missing bound short device check",
    )
    book = dat.u.bound(dat.BASE / "reservation.json", initial["data"]["rows_sha"])
    rows, nt = dat.checked_rows(book)
    check(nt == initial["targets_per_endpoint"], "wrong raw short-final exposure")
    return d, book["source"], rows


def finish(root):
    from .confirm_analysis import analyze

    check(root == ROOT, "wrong confirmation root")
    write_json(root / "summary.json", analyze(root))
