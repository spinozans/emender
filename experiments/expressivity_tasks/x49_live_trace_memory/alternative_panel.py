"""One registered X49-V job. Freeze both the original U predictor and the new D/V/K fits."""

from __future__ import annotations
from ..x48_muon_continuation.data import check, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from . import alternative_data as data, surprise_panel as old, local_u
from .alternative import DIMS as DIMS
from .alternative_fit import Moments
from .surprise_fit import RIDGE
from .panels import T, SEEDS, OUT, fingerprint
from .panel_spec import Job
from .fit_phase import incumbent as phase

ROOT = OUT / "alternative-probe-1"
SCOPE = "x49-v-old-noncopy-alternative-v1"
TOTAL = 1200
JOB = Job("v-fit-probe", "fit_then_probe", TOTAL, "UDVK")
JOBS = {JOB.name: JOB}


def units():
    return len(SEEDS) * sum(data.N.values())


def bindings(*, hash_weights, study="V"):
    dat, _, _ = phase(study)
    dat.checked_spec()
    weights = {}
    coeff = {}
    for s in SEEDS:
        ck, b = local_u.contract(s)
        _, w = old.checkpoint(s, hash_weights=hash_weights)
        check(w["sha256"] == ck, "wrong frozen U base file")
        weights[str(s)] = w
        coeff[str(s)] = list(b)
    return weights, coeff


def plan(ref, *, hash_weights, study="V"):
    dat, p, _ = phase(study)
    m = dat.metadata()
    check(m["prepared_source"] == ref, f"{study} data source changed")
    w, b = p.bindings(hash_weights=hash_weights)
    return dict(
        job=p.JOB.name,
        kind=p.JOB.kind,
        ref=ref,
        spec=dat.SPEC,
        **({"v_result_sha": dat.V_RESULT} if study == "LS" else {}),
        u_confirmation_sha=local_u.CONFIRMATION,
        u_beta=b,
        endpoints=w,
        horizon=T,
        data=m,
        documents=dict(dat.N),
        units=p.units(),
        ridge=RIDGE,
        neural_updates=0,
        u_refits=0,
    )


def expected(ref, *, study="V"):
    dat, p, _ = phase(study)
    z = p.plan(ref, hash_weights=False)
    return dict(
        scope=p.SCOPE,
        spec=dat.SPEC,
        **({"v_result_sha": z["v_result_sha"]} if study == "LS" else {}),
        ref=ref,
        u_confirmation_sha=local_u.CONFIRMATION,
        endpoints_id=fingerprint([z["endpoints"], z["u_beta"]]),
        data=z["data"],
        role_documents=z["documents"],
        units=z["units"],
        ridge=RIDGE,
        fitted_scalars_per_seed=dict(p.DIMS),
        neural_updates=0,
        u_refits=0,
        split="native_train_fit_and_disjoint_probe",
        local_gpus=1,
        max_acquisitions=1,
        total_card_seconds=p.TOTAL,
        job_caps={p.JOB.name: p.TOTAL},
        remote=False,
        retries=0,
    )


def validate(a, ref, *, study="V"):
    _, p, _ = phase(study)
    e = p.expected(ref)
    check(
        type(a) is dict
        and set(a) == set(e) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in e}) == fingerprint(e),
        f"wrong {study} authority",
    )


def freeze(job, ref, *, study="V"):
    _, p, _ = phase(study)
    f = p.ROOT / p.JOB.name / "frozen.json"
    check(job == p.JOB and not f.exists(), f"wrong/repeated {study} plan")
    write_json(f, p.plan(ref, hash_weights=True))
    flush_dir(f.parent)
    return sha(f)


def checked_fit(state):
    old.stored_fit(state, Moments, data.N["fit"])


def fit_row_ok(row):
    from .alternative_analysis import checked_work

    checked_work("fit", row)
    return True


def readout_map(d, initial):
    return old.readout_map(d, initial, study="V")


def freeze_readouts(d, initial):
    return old.freeze_readouts(d, initial, study="V")


def checked_readouts(d, initial, fit_sha):
    return old.checked_readouts(d, initial, fit_sha, study="V")


def finish(root):
    from .alternative_analysis import analyze

    check(root == ROOT, "wrong V root")
    write_json(root / "summary.json", analyze(root))
