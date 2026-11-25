"""ET's immutable readout contract. Reads small closed receipts, not text, memberships or FIT moments."""

from ..x48_muon_continuation.data import check, sha
from . import (
    error_history as eh,
    error_history_data as source,
    error_history_panel as panel,
    joint_reader as joint,
    local_u,
    surprise as u,
)
from .panels import OUT, SEEDS

PROTOCOL = "a0d9d7ad73b848f1bf506d70be205b34b5bf4c65fd5573d51ce95bad8d5b826c"
PUBLIC = "60c270920302709dc57664c018db963db4946bc834f59f48f237d5d114ce00ec"
CLOSED = "be743653fa8bf84b82184d3cd5617b217f05b7a505313188f65b1ec017ac20b3"
FIT = "a8e297fd58ece00be45fbc3b3b80bf2f72c993226f0ac1092c52617843b96567"


def parent():
    check(
        sha(joint.RECEIPTS.parent / "ERROR_TRANSFER.md") == PROTOCOL,
        "changed ET question",
    )
    pub = source.bound(joint.RECEIPTS / "error_history_result.json", PUBLIC)
    r = source.bound(OUT / "error-history-result-1.json", CLOSED)
    check(
        pub["closed_private_aggregate_sha256"] == CLOSED
        and all(
            r[k] == v
            for k, v in pub.items()
            if k not in ("publication_scope", "closed_private_aggregate_sha256")
        )
        and r["spec"] == source.SPEC
        and r["fit_freeze_sha"] == FIT
        and r["verified"]["independent_paired_bootstrap"]
        and r["decision"]["added_bound_wrong_guess_identities"]
        and r["decision"]["pooled_scalar_history"]
        and r["neural_updates"]
        == r["u_refits"]
        == r["j_refits"]
        == r["cost"]["rc"]
        == 0,
        "foreign or partial EH parent",
    )
    return r


def contract(seed):
    check(type(seed) is int and seed in SEEDS, "unknown ET checkpoint")
    s = str(seed)
    r = parent()
    d = panel.ROOT / panel.JOB.name
    plan = source.bound(d / "frozen.json", r["freeze_sha"])
    seal = source.bound(d / "fit-frozen.json", FIT)
    entry = seal["seeds"][s]
    e = plan["endpoints"][s]
    uck, ub = local_u.contract(seed)
    jck, jb = joint.contract(seed)
    b = entry["beta"]
    check(
        seal["spec"] == plan["spec"] == source.SPEC
        and seal["plan_sha"] == r["freeze_sha"]
        and seal["fit_identity_sha"]
        == plan["data"]["fit_identity_sha"]
        == r["data"]["fit_identity_sha"]
        and seal["fit_units"] == 768
        and plan["data"] == r["data"]
        and uck == jck == e["sha256"] == entry["checkpoint_sha"] == r["checkpoints"][s]
        and list(ub) == r["u_beta"][s] == plan["u_beta"][s]
        and list(jb) == r["j_beta"][s] == plan["j_beta"][s]
        and plan["fit_dimensions"] == r["fit_dimensions"] == eh.DIMS
        and plan["swap_fit"] == r["swap_fit"] == "ERROR"
        and b == r["beta"][s]
        and set(b) == set(eh.DIMS),
        "foreign ET weight, fit identity or coefficient set",
    )
    for a in eh.DIMS:
        u.checked_vector(eh.DIMS, a, b[a])
    return uck, dict(U=list(ub), J=list(jb), **b)
