"""One DR FIT/seal/PROBE job: identity-bind three private decoder tables separately from probes."""

from ..x48_muon_continuation.data import check, write_json
from . import (
    decode_data as data,
    decode_store as store,
    transfer_panel as jt,
    surprise_panel as old,
    alternative_panel as prior,
)
from .decode_fit import RIDGE as RIDGE
from .panel_spec import Job
from .panels import OUT, SEEDS, T, fingerprint

ROOT = OUT / "decode-record-probe-1"
SCOPE = "x49-dr-two-note-decoder-v1"
TOTAL = 1200
JOB = Job("dr-fit-probe", "fit_then_probe", TOTAL, "BUJPDS")
JOBS = {JOB.name: JOB}
DIMS = {
    "D": 256 * (32 + 2 + 2 * 32)
}  # installed FP32 scalars; one decoder, not 256 readout fits


def units():
    return len(SEEDS) * sum(data.N.values())


def bindings(*, hash_weights):
    r = data.checked_spec()
    w = {}
    b = {}
    for k, e in jt.endpoints(hash_weights=hash_weights).items():
        check(
            r["predictors"][k]
            == dict(checkpoint=e["weights"]["sha256"], beta=e["beta"]),
            "wrong JT seed or coefficients",
        )
        w[k] = e["weights"]
        b[k] = {a: e["beta"][a] for a in ("U", "J")}
    return w, b


def plan(ref, *, hash_weights):
    m = data.metadata()
    check(m["prepared_source"] == ref, "DR data source changed")
    w, b = bindings(hash_weights=hash_weights)
    return dict(
        ref=ref,
        job=JOB.name,
        kind=JOB.kind,
        spec=data.SPEC,
        jt_result_sha=data.JT_RESULT,
        **jt.lineage(),
        endpoints=w,
        u_beta={k: x["U"] for k, x in b.items()},
        j_beta={k: x["J"] for k, x in b.items()},
        horizon=T,
        documents=dict(data.N),
        data=m,
        units=units(),
        ridge=RIDGE,
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
        jt_result_sha=data.JT_RESULT,
        writer_binary_sha=z["writer_binary_sha"],
        weight_and_readouts_id=fingerprint(
            [z[k] for k in ("endpoints", "u_beta", "j_beta")]
        ),
        data=z["data"],
        role_documents=z["documents"],
        units=z["units"],
        fit_writes_per_seed=data.N["fit"] * (T - 1),
        installed_decoder_scalars=DIMS["D"],
        ridge=RIDGE,
        neural_updates=0,
        u_refits=0,
        j_refits=0,
        split="native_train_fit_writes_and_disjoint_scored_probe",
        local_gpus=1,
        max_acquisitions=1,
        total_card_seconds=TOTAL,
        job_caps={JOB.name: TOTAL},
        remote=False,
        retries=0,
    )


def validate(a, ref):
    return prior.validate(a, ref, study="DR")


def freeze(job, ref):
    return prior.freeze(job, ref, study="DR")


def fit_row_ok(row):
    from .decode_stats import checked_row

    checked_row("fit", row)
    return True


def fitted_entry(d, s, state, logs):
    proof = state["synthetic"]
    check(
        proof
        == dict(
            observed_sha=fixture_sha(),
            hit_rows=fixture_hits(),
            all_256_B_U_J_equal=True,
            all_256_teacher_disabled_D_equal=True,
            synthetic_host_prefills=3,
        ),
        "missing DR teacher-free parity",
    )
    return dict(
        **store.check_state(d, s, state, sum(x["work"]["fit_writes"] for x in logs)),
        synthetic=proof,
    )


def fixture_word():
    return (b"\xff\0 color: blue \x1e color: black aaababa " + bytes(range(16))) * 3


def fixture_sha():
    from .panels import digest

    return digest(fixture_word())


def fixture_hits():
    from .literal import literal_rows

    return sum(x.latest is not None for x in literal_rows(fixture_word(), T))


def readout_map(d, initial):
    return old.readout_map(d, initial, study="DR")


def freeze_readouts(d, initial):
    return old.freeze_readouts(d, initial, study="DR")


def checked_readouts(d, initial, fit_sha):
    return old.checked_readouts(d, initial, fit_sha, study="DR")


def finish(root):
    from .decode_stats import analyze

    check(root == ROOT, "wrong DR closure root")
    write_json(root / "summary.json", analyze(root))
