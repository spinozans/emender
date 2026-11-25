"""RB: fresh native TRAIN after all DR and earlier roles. No previous payload is reopened."""

import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha
from . import decode_data as dr, query_data, surprise_data as old
from .surprise_data import bound as bound
from .panels import OUT, fingerprint

BASE = OUT / "residual-byte-data-1"
SPEC = "ec7021d2128a12969638e9cdfad07bbde5f11ac5dd5c0905842c97ae1071ec36"
DR_RESULT = "d7d826da0d41f90fececc1262a62ad9b278617e506c9c8b447ca86f106c6f948"
DR_SEAL = "90917b65673b8d4d8fbe401a719665c57dd57cbc817fb39b770b13527b94fa47"
N = {"fit": 256, "probe": 128}


def checked_spec():
    check(sha(old.HERE / "RESIDUAL_BYTE.md") == SPEC, "changed RB specification")
    dr.checked_spec()
    r = bound(old.HERE / "receipts/decode_record_result.json", DR_RESULT)
    check(
        r["spec"] == dr.SPEC
        and r["jt_result_sha"] == dr.JT_RESULT
        and r["fit_freeze_sha"] == DR_SEAL
        and r["verified"]["independent_MSE_denominator"] is True
        and r["neural_updates"] == r["u_refits"] == r["j_refits"] == 0,
        "foreign DR result",
    )
    return r


def next_block(book, records, pre, quota):
    roles = dr.checked_roles(book)
    return query_data.select_new(
        records,
        base_ex=pre,
        prior={"train": roles["fit"] + roles["probe"]},
        quota=quota,
        size=book["source"]["size"],
    )


def selection_context():
    """Reproduce the spent RB descriptors, manifest and pre-RB exclusions; no text."""
    r = checked_spec()
    b, records, pre = dr.selection_context()
    check(
        dr.metadata() == r["data"]
        and b == bound(dr.BASE / "reservation.json", r["data"]["rows_sha"]),
        "changed spent DR reservation",
    )
    rows, ex, supply = next_block(b, records, pre, sum(N.values()))
    check(len(ex) == 43704 and supply == 22803, "incorrect clean RB supply")
    descriptor = dict(
        schema="x49-rb-native-train",
        spec=SPEC,
        dr_result_sha=DR_RESULT,
        dr_rows_sha=r["data"]["rows_sha"],
        manifest_sha=b["manifest_sha"],
        source=b["source"],
        eligible_train=supply,
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        roles=dict(fit=rows[: N["fit"]], probe=rows[N["fit"] :]),
    )
    return descriptor, records, ex


def selection():
    return selection_context()[0]


def checked_roles(book):
    return old.role_rows(book, "x49-rb-native-train", SPEC, N, MIN_OFFSET)


def prepare():
    return old.prepare(study="RB")


def metadata():
    return old.metadata(study="RB")


def licensed(job):
    return old.licensed(job, study="RB")


def fit_windows(job):
    return old.fit_windows(job, study="RB")


def extract(role, *, job=None, fit_sha=None):
    return old.extract(role, job=job, fit_sha=fit_sha, study="RB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
