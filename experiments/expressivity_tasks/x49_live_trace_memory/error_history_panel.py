"""EH alone: same frozen U/J, four new coefficient vectors per weight and no SWAP fit."""

from ..x48_muon_continuation.data import check, sha, write_json
from . import (
    error_history_data as data,
    decode_panel as dr,
    alternative_panel as prior,
    surprise_panel as old,
)
from .error_history import DIMS as DIMS, ARMS
from .error_history_fit import Moments
from .surprise_fit import RIDGE as RIDGE
from .panels import OUT, SEEDS, T, fingerprint
from .panel_spec import Job

ROOT = OUT / "error-history-probe-1"
TOTAL = 1200
SCOPE = "x49-eh-live-older-mistake-histograms-v1"
JOB = Job("eh-fit-probe", "fit_then_probe", TOTAL, ",".join(ARMS))
JOBS = {JOB.name: JOB}


def units():
    return len(SEEDS) * sum(data.N.values())


def bindings(*, hash_weights):
    r = data.checked_spec()
    w, b = dr.bindings(hash_weights=hash_weights)
    check(
        r["checkpoints"] == {k: e["sha256"] for k, e in w.items()}
        and all(
            r[a.lower() + "_beta"] == {k: v[a] for k, v in b.items()}
            for a in ("U", "J")
        )
        and r["writer_binary_sha"] == dr.jt.lineage()["writer_binary_sha"],
        "foreign RB checkpoint/readout/writer anchor",
    )
    return w, b


def qualification_sha(ref):
    return sha(OUT / ("qualify-error-history-" + ref) / "passed.json")


def plan(ref, *, hash_weights):
    m = data.metadata()
    check(m["prepared_source"] == ref, "EH data source changed")
    w, b = bindings(hash_weights=hash_weights)
    return dict(
        ref=ref,
        job=JOB.name,
        kind=JOB.kind,
        spec=data.SPEC,
        qualification_sha=qualification_sha(ref),
        rb_result_sha=data.RB_RESULT,
        jt_result_sha=data.rb.dr.JT_RESULT,
        **dr.jt.lineage(),
        endpoints=w,
        u_beta={k: v["U"] for k, v in b.items()},
        j_beta={k: v["J"] for k, v in b.items()},
        ridge=RIDGE,
        fit_dimensions=dict(DIMS),
        swap_fit="ERROR",
        data=m,
        documents=dict(data.N),
        units=units(),
        horizon=T,
        neural_updates=0,
        u_refits=0,
        j_refits=0,
    )


def expected(ref):
    z = plan(ref, hash_weights=False)
    return dict(
        scope=SCOPE,
        ref=ref,
        spec=data.SPEC,
        rb_result_sha=data.RB_RESULT,
        qualification_sha=z["qualification_sha"],
        writer_binary_sha=z["writer_binary_sha"],
        weight_and_readouts_id=fingerprint(
            [z[k] for k in ("endpoints", "u_beta", "j_beta")]
        ),
        data=z["data"],
        role_documents=z["documents"],
        units=z["units"],
        fit_scalars_per_seed=dict(DIMS),
        swap_fit="ERROR",
        ridge=RIDGE,
        neural_updates=0,
        u_refits=0,
        j_refits=0,
        split="native_train_older_error_fit_and_disjoint_probe",
        local_gpus=1,
        max_acquisitions=1,
        total_card_seconds=TOTAL,
        job_caps={JOB.name: TOTAL},
        remote=False,
        retries=0,
    )


def validate(a, ref):
    return prior.validate(a, ref, study="EH")


def freeze(job, ref):
    return prior.freeze(job, ref, study="EH")


def checked_fit(state):
    return old.stored_fit(state, Moments, data.N["fit"])


def fit_row_ok(row):
    from .error_history_stats import checked_row

    checked_row("fit", row)
    return True


def readout_map(d, initial):
    return old.readout_map(d, initial, study="EH")


def freeze_readouts(d, initial):
    return old.freeze_readouts(d, initial, study="EH")


def checked_readouts(d, initial, fit_sha):
    return old.checked_readouts(d, initial, fit_sha, study="EH")


def finish(root):
    from .error_history_stats import analyze

    check(root == ROOT, "wrong EH root")
    write_json(root / "summary.json", analyze(root))
