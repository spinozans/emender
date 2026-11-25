"""LS: verified next full native TRAIN rank after V. No final, calibration or prior text access."""

from __future__ import annotations
import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha
from . import alternative_data as v, surprise_data as old, local_u, query_data
from .surprise_data import bound
from .panels import OUT, fingerprint

BASE = OUT / "literal-surprise-data-1"
SPEC = "4a859bd3f484968e2c2d8a8118ad1ad10729f00079d83558cf5632c867bc7f70"
V_RESULT = "80770a965e37f9f70c8af1be107319c13108c345c9f2d1fbd1293a040e2da3c6"
N = {"fit": 256, "probe": 128}


def checked_spec():
    check(sha(old.HERE / "LITERAL_SURPRISE.md") == SPEC, "changed LS registration")
    v.checked_spec()
    result = bound(old.HERE / "receipts/alternative_result.json", V_RESULT)
    check(
        result["spec"] == v.SPEC
        and result["u_confirmation_sha"] == local_u.CONFIRMATION
        and result["u_refits"] == result["neural_updates"] == 0
        and result["verified"]["independent_paired_bootstrap"] is True,
        "wrong V closure",
    )
    return result


def next_block(book, records, pre, quota):
    roles = v.checked_roles(book)
    prior = roles["fit"] + roles["probe"]
    return query_data.select_new(
        records,
        base_ex=pre,
        prior={"train": prior},
        quota=quota,
        size=book["source"]["size"],
    )


def selection():
    r = checked_spec()
    book, records, pre = v.selection_context()
    check(
        v.metadata() == r["data"]
        and bound(v.BASE / "reservation.json", r["data"]["rows_sha"]) == book,
        "foreign spent V reservation",
    )
    rows, ex, supply = next_block(book, records, pre, sum(N.values()))
    check(len(ex) == 42555 and supply == 23571, "changed LS TRAIN supply")
    return dict(
        schema="x49-ls-native-train",
        spec=SPEC,
        v_result_sha=V_RESULT,
        v_rows_sha=r["data"]["rows_sha"],
        u_confirmation_sha=local_u.CONFIRMATION,
        manifest_sha=book["manifest_sha"],
        source=book["source"],
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        eligible_train=supply,
        roles={"fit": rows[: N["fit"]], "probe": rows[N["fit"] :]},
    )


def checked_roles(book):
    return old.role_rows(book, "x49-ls-native-train", SPEC, N, MIN_OFFSET)


def prepare():
    return old.prepare(study="LS")


def metadata():
    return old.metadata(study="LS")


def licensed(job):
    return old.licensed(job, study="LS")


def fit_windows(job):
    return old.fit_windows(job, study="LS")


def extract(role, *, job=None, fit_sha=None):
    return old.extract(role, job=job, fit_sha=fit_sha, study="LS")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
