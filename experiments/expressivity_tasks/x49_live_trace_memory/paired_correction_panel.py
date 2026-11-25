"""One local PC FIT/PROBE, with frozen anchors verified against ET and their original pre-PROBE seal."""

from ..x48_muon_continuation.data import check, sha, write_json
from . import (
    paired_correction_data as data,
    alternative_panel as prior,
    surprise_panel as old,
    transfer_panel as jt,
)
from .paired_correction import DIMS as DIMS, ARMS
from .paired_correction_fit import Moments
from .surprise_fit import RIDGE as RIDGE
from .panels import OUT, SEEDS, T, fingerprint
from .panel_spec import Job

ROOT = OUT / "paired-correction-probe-1"
TOTAL = 1200
SCOPE = "x49-pc-forecast-keyed-corrections-v1"
JOB = Job("pc-fit-probe", "fit_then_probe", TOTAL, ",".join(ARMS))
JOBS = {JOB.name: JOB}
FIXED = {"U": "u_beta", "J": "j_beta", "HISTORY": "history_beta", "ERROR": "error_beta"}


def units():
    return len(SEEDS) * sum(data.N.values())


def fixed(seed):
    r = data.checked_spec()
    ck, b = data.earlier.contract(seed)
    check(
        r["predictors"][str(seed)] == dict(checkpoint=ck, beta=b)
        and r["writer_binary_sha"] == jt.lineage()["writer_binary_sha"],
        "wrong seed, byte writer or pre-ET readouts",
    )
    return ck, {k: b[k] for k in FIXED}


def bindings(*, hash_weights):
    weights = {}
    coeff = {}
    for s in SEEDS:
        ck, b = fixed(s)
        _, e = old.checkpoint(s, hash_weights=hash_weights)
        check(e["sha256"] == ck, "wrong fixed PC checkpoint")
        weights[str(s)] = e
        coeff[str(s)] = b
    return weights, coeff


def qualification_sha(ref):
    return sha(OUT / ("qualify-paired-correction-" + ref) / "passed.json")


def plan(ref, *, hash_weights):
    m = data.metadata()
    check(m["prepared_source"] == ref, "PC data source changed")
    w, b = bindings(hash_weights=hash_weights)
    return dict(
        ref=ref,
        job=JOB.name,
        kind=JOB.kind,
        spec=data.SPEC,
        qualification_sha=qualification_sha(ref),
        et_result_sha=data.ET_RESULT,
        et_close_sha=data.ET_CLOSE,
        eh_result_sha=data.earlier.PUBLIC,
        eh_fit_freeze_sha=data.earlier.FIT,
        jt_result_sha=data.et.dr.JT_RESULT,
        **jt.lineage(),
        endpoints=w,
        **{name: {s: v[a] for s, v in b.items()} for a, name in FIXED.items()},
        ridge=RIDGE,
        fit_dimensions=dict(DIMS),
        swap_fit="PAIR",
        data=m,
        documents=dict(data.N),
        units=units(),
        horizon=T,
        neural_updates=0,
        u_refits=0,
        j_refits=0,
        history_refits=0,
        error_refits=0,
    )


def expected(ref):
    z = plan(ref, hash_weights=False)
    return dict(
        scope=SCOPE,
        ref=ref,
        **{
            k: z[k]
            for k in (
                "spec",
                "qualification_sha",
                "et_result_sha",
                "et_close_sha",
                "eh_fit_freeze_sha",
                "writer_binary_sha",
                "data",
                "units",
                "swap_fit",
                "ridge",
                "neural_updates",
                "u_refits",
                "j_refits",
                "history_refits",
                "error_refits",
            )
        },
        weight_and_readouts_id=fingerprint(
            [z[k] for k in ("endpoints", *FIXED.values())]
        ),
        role_documents=z["documents"],
        fit_scalars_per_seed=dict(DIMS),
        split="native_train_paired_correction_fit_and_disjoint_probe",
        local_gpus=1,
        max_acquisitions=1,
        total_card_seconds=TOTAL,
        job_caps={JOB.name: TOTAL},
        remote=False,
        retries=0,
    )


def validate(a, ref):
    return prior.validate(a, ref, study="PC")


def freeze(job, ref):
    return prior.freeze(job, ref, study="PC")


def checked_fit(state):
    return old.stored_fit(state, Moments, data.N["fit"])


def fit_row_ok(row):
    from .paired_correction_stats import checked_row

    checked_row("fit", row)
    return True


def readout_map(d, initial):
    return old.readout_map(d, initial, study="PC")


def freeze_readouts(d, initial):
    return old.freeze_readouts(d, initial, study="PC")


def checked_readouts(d, initial, fit_sha):
    return old.checked_readouts(d, initial, fit_sha, study="PC")


def finish(root):
    from .paired_correction_stats import analyze

    check(root == ROOT, "wrong PC root")
    write_json(root / "summary.json", analyze(root))
