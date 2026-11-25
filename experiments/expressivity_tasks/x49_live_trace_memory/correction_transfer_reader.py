"""PT frozen reader identity. Inspect only closed receipts and seals, never old windows or learned table contents."""

from ..x48_muon_continuation.data import check, sha
from . import (
    paired_correction_data as source,
    paired_correction_panel as panel,
    surprise as u,
)
from .panels import OUT, SEEDS

PROTOCOL = "78b7e7cdd5b5b05fc93438f57b97b23f5ad1e5a0d06348d8d3092ef596126a0a"
PUBLIC = "80609171412a6c5682b9b57469a7750d5cbd432918994eac26a51fcfca60631d"
CLOSED = "91ef90da942bad7f68e59fa26895a43e553e15dac35e23e3c7872588950ba71e"
FIT = "787a79320e6ad0ad0dd7d6c38100fea870d82e1c386e9e91a872b811b873f356"


def parent():
    check(
        sha(source.old.HERE / "CORRECTION_TRANSFER.md") == PROTOCOL,
        "changed PT registration",
    )
    pub = source.bound(
        source.old.HERE / "receipts/paired_correction_result.json", PUBLIC
    )
    r = source.bound(OUT / "paired-correction-result-1.json", CLOSED)
    check(
        pub["private_close_sha256"] == CLOSED
        and all(
            pub[k] == v for k, v in r.items() if k not in ("beta", "u_beta", "j_beta")
        )
        and r["spec"] == source.SPEC
        and r["fit_freeze_sha"] == FIT
        and r["et_result_sha"] == source.ET_RESULT
        and r["eh_fit_freeze_sha"] == source.earlier.FIT
        and r["verified"]["independent_paired_bootstrap"]
        and r["verified"]["independent_conditional_support_peaks_and_cost"]
        and r["decision"]["consider_locked_pair_transfer"]
        and r["decision"]["useful_marginal_trigger"]
        and r["swap_fit"] == "PAIR"
        and r["cost"]["rc"]
        == r["neural_updates"]
        == r["history_refits"]
        == r["error_refits"]
        == 0,
        "foreign or incomplete PC parent",
    )
    return r


def contract(seed):
    check(type(seed) is int and seed in SEEDS, "unknown PT checkpoint")
    s = str(seed)
    r = parent()
    d = panel.ROOT / panel.JOB.name
    plan = source.bound(d / "frozen.json", r["freeze_sha"])
    seal = source.bound(d / "fit-frozen.json", FIT)
    entry = seal["seeds"][s]
    ck, base = panel.fixed(seed)
    check(
        ck
        == entry["checkpoint_sha"]
        == plan["endpoints"][s]["sha256"]
        == r["checkpoints"][s]
        and seal["spec"] == plan["spec"] == source.SPEC
        and plan["ref"] == r["ref"]
        and plan["data"] == r["data"]
        and plan["fit_dimensions"] == r["fit_dimensions"] == panel.DIMS
        and plan["swap_fit"] == r["swap_fit"] == "PAIR"
        and seal["fit_units"] == 768
        and seal["plan_sha"] == r["freeze_sha"]
        and seal["fit_identity_sha"] == r["data"]["fit_identity_sha"]
        and all(plan[k][s] == base[a] for a, k in panel.FIXED.items())
        and plan["u_beta"][s] == r["u_beta"][s]
        and plan["j_beta"][s] == r["j_beta"][s]
        and set(entry["beta"]) == set(panel.DIMS)
        and entry["beta"] == r["beta"][s],
        "foreign PT checkpoint, anchor or sealed PC coefficients",
    )
    for a in panel.DIMS:
        u.checked_vector(panel.DIMS, a, entry["beta"][a])
    return ck, dict(base, **entry["beta"])
