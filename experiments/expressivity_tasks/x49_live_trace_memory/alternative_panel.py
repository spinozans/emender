"""One registered X49-V job. Freeze both the original U predictor and the new D/V/K fits."""

from __future__ import annotations
from ..x48_muon_continuation.data import check, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from . import alternative_data as data, surprise_panel as old, local_u
from .alternative import DIMS
from .alternative_fit import Moments
from .surprise_fit import RIDGE
from .panels import T, SEEDS, OUT, fingerprint
from .panel_spec import Job

ROOT = OUT / "alternative-probe-1"
TOTAL = 1200
JOB = Job("v-fit-probe", "fit_then_probe", TOTAL, "UDVK")
JOBS = {JOB.name: JOB}


def units():
    return len(SEEDS) * sum(data.N.values())


def bindings(*, hash_weights):
    data.checked_spec()
    weights = {}
    coeff = {}
    for s in SEEDS:
        ck, b = local_u.contract(s)
        _, w = old.checkpoint(s, hash_weights=hash_weights)
        check(w["sha256"] == ck, "wrong frozen U base file")
        weights[str(s)] = w
        coeff[str(s)] = list(b)
    return weights, coeff


def plan(ref, *, hash_weights):
    m = data.metadata()
    check(m["prepared_source"] == ref, "V data source changed")
    w, b = bindings(hash_weights=hash_weights)
    return dict(
        job=JOB.name,
        kind=JOB.kind,
        ref=ref,
        spec=data.SPEC,
        u_confirmation_sha=local_u.CONFIRMATION,
        u_beta=b,
        endpoints=w,
        horizon=T,
        data=m,
        documents=dict(data.N),
        units=units(),
        ridge=RIDGE,
        neural_updates=0,
        u_refits=0,
    )


def expected(ref):
    z = plan(ref, hash_weights=False)
    return dict(
        scope="x49-v-old-noncopy-alternative-v1",
        spec=data.SPEC,
        ref=ref,
        u_confirmation_sha=local_u.CONFIRMATION,
        endpoints_id=fingerprint([z["endpoints"], z["u_beta"]]),
        data=z["data"],
        role_documents=z["documents"],
        units=z["units"],
        ridge=RIDGE,
        fitted_scalars_per_seed=dict(DIMS),
        neural_updates=0,
        u_refits=0,
        split="native_train_fit_and_disjoint_probe",
        local_gpus=1,
        max_acquisitions=1,
        total_card_seconds=TOTAL,
        job_caps={JOB.name: TOTAL},
        remote=False,
        retries=0,
    )


def validate(a, ref):
    e = expected(ref)
    check(
        type(a) is dict
        and set(a) == set(e) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in e}) == fingerprint(e),
        "wrong V authority",
    )


def freeze(job, ref):
    f = ROOT / JOB.name / "frozen.json"
    check(job == JOB and not f.exists(), "wrong/repeated V plan")
    write_json(f, plan(ref, hash_weights=True))
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
