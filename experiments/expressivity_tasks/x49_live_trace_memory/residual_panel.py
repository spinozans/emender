"""RB authority only. The old D files are immutable inputs; the new tables are sealed outputs."""

from math import isfinite, ulp
from ..x48_muon_continuation.data import check, load, write_json
from . import (
    residual_data as data,
    residual_store as store,
    decode_store,
    decode_panel as dr,
    surprise_panel as old,
    alternative_panel as prior,
)
from .panel_spec import Job
from .panels import OUT, SEEDS, T, fingerprint

ROOT = OUT / "residual-byte-probe-1"
TOTAL = 1200
JOB = Job("rb-fit-probe", "fit_then_probe", TOTAL, "BUJDGRS")
JOBS = {JOB.name: JOB}
DIMS = {"R": 256 * 32}
SCOPE = "x49-rb-actual-residual-byte-v1"


def units():
    return len(SEEDS) * sum(data.N.values())


def coarse_folder():
    return dr.ROOT / dr.JOB.name


def bound_coarse(seed, h):
    return decode_store.tables(coarse_folder(), seed, {"tables_sha": h})


def bindings(*, hash_weights):
    r = data.checked_spec()
    f = data.bound(coarse_folder() / "fit-frozen.json", data.DR_SEAL)
    check(
        f["seeds"] == r["decoders"] and f["spec"] == data.dr.SPEC, "foreign DR fit map"
    )
    w, b = dr.bindings(hash_weights=hash_weights)
    coarse = {}
    for s in SEEDS:
        k = str(s)
        e = f["seeds"][k]
        check(
            w[k]["sha256"] == e["checkpoint_sha"] == r["checkpoints"][k]
            and b[k] == dict(U=r["u_beta"][k], J=r["j_beta"][k]),
            "wrong DR seed/reader",
        )
        coarse[k] = e["tables_sha"]
        bound_coarse(s, coarse[k])
    return w, b, coarse


def checked_axes(ref, w, coarse):
    z = load(OUT / ("qualify-residual-byte-" + ref) / "passed.json")
    a = z["head_audit"]
    check(
        z["ref"] == ref
        and z["spec"] == data.SPEC
        and z["natural_windows_opened"] == 0
        and set(a) == set(map(str, SEEDS)),
        "no RB axis qualification",
    )
    for k, e in a.items():
        h = e["axes_sha256"]
        check(
            e["checkpoint_sha"] == z["states"][k]["checkpoint"] == w[k]["sha256"]
            and e["coarse_sha"] == coarse[k]
            and len(h) == 64
            and f"{int(h, 16):064x}" == h
            and all(
                isfinite(e[v]) and e[v] >= 0
                for v in (
                    "minimum_relative_leading_gap",
                    "gram_relative_residual_inf",
                    "orthogonality_error_inf",
                )
            )
            and e["minimum_relative_leading_gap"] > 64 * ulp(1.0)
            and max(e["gram_relative_residual_inf"], e["orthogonality_error_inf"])
            <= 64 * 32 * ulp(1.0),
            "foreign qualified RB axes",
        )
    return a


def plan(ref, *, hash_weights):
    m = data.metadata()
    check(m["prepared_source"] == ref, "RB data source changed")
    w, b, c = bindings(hash_weights=hash_weights)
    return dict(
        ref=ref,
        job=JOB.name,
        kind=JOB.kind,
        spec=data.SPEC,
        dr_result_sha=data.DR_RESULT,
        dr_fit_seal=data.DR_SEAL,
        jt_result_sha=data.dr.JT_RESULT,
        **dr.jt.lineage(),
        endpoints=w,
        u_beta={k: e["U"] for k, e in b.items()},
        j_beta={k: e["J"] for k, e in b.items()},
        coarse_sha=c,
        qualified_axes=checked_axes(ref, w, c),
        data=m,
        documents=dict(data.N),
        units=units(),
        horizon=T,
        bits=8,
        neural_updates=0,
        u_refits=0,
        j_refits=0,
        d_refits=0,
    )


def expected(ref):
    z = plan(ref, hash_weights=False)
    return dict(
        scope=SCOPE,
        ref=ref,
        spec=data.SPEC,
        dr_result_sha=data.DR_RESULT,
        dr_fit_seal=data.DR_SEAL,
        writer_binary_sha=z["writer_binary_sha"],
        weight_and_readouts_id=fingerprint(
            [
                z[k]
                for k in (
                    "endpoints",
                    "u_beta",
                    "j_beta",
                    "coarse_sha",
                    "qualified_axes",
                )
            ]
        ),
        data=z["data"],
        role_documents=z["documents"],
        units=z["units"],
        fit_writes_per_seed=data.N["fit"] * (T - 1),
        bits=8,
        neural_updates=0,
        u_refits=0,
        j_refits=0,
        d_refits=0,
        split="native_train_residual_writes_and_disjoint_scored_probe",
        local_gpus=1,
        max_acquisitions=1,
        total_card_seconds=TOTAL,
        job_caps={JOB.name: TOTAL},
        remote=False,
        retries=0,
    )


def validate(a, ref):
    return prior.validate(a, ref, study="RB")


def freeze(job, ref):
    return prior.freeze(job, ref, study="RB")


def fit_row_ok(x):
    from .residual_stats import checked_row

    checked_row("fit", x)
    return True


def fitted_entry(d, s, state, logs, initial):
    k = str(s)
    ident = dict(
        axes_sha256=initial["qualified_axes"][k]["axes_sha256"],
        coarse_sha=initial["coarse_sha"][k],
    )
    proof = dict(
        observed_sha=dr.fixture_sha(),
        hit_rows=dr.fixture_hits(),
        synthetic_host_prefills=4,
        full_anchors_equal=True,
        one_write_standalone_R_equal=True,
    )
    check(
        state["input_tables"] == ident and state["synthetic"] == proof,
        "foreign axis/decoder or incomplete RB parity",
    )
    return dict(
        **store.check_state(d, s, state, logs), input_tables=ident, synthetic=proof
    )


def readout_map(d, initial):
    return old.readout_map(d, initial, study="RB")


def freeze_readouts(d, initial):
    return old.freeze_readouts(d, initial, study="RB")


def checked_readouts(d, initial, fit_sha):
    return old.checked_readouts(d, initial, fit_sha, study="RB")


def finish(root):
    from .residual_stats import analyze

    check(root == ROOT, "wrong RB closure root")
    write_json(root / "summary.json", analyze(root))
