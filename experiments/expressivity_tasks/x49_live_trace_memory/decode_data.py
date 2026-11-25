"""DR: next complete native TRAIN FIT/PROBE blocks, excluding JT too. No final or prior payload reader."""

import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha
from . import (
    transfer_data as jt,
    surprise_data as old,
    literal_mark_data as ls,
    alternative_data as v,
)
from .surprise_data import bound as bound
from .panels import OUT, fingerprint

BASE = OUT / "decode-record-data-1"
SPEC = "fc8567f0a92d341b82bbcfd664d1fe95c5e741d9d292f0b70eadc098a8a7c371"
JT_RESULT = "62db2e0beb85bee456b1fea2005d98b4f5684f6d41f493841ac3fc499b9a24d6"
N = {"fit": 256, "probe": 128}


def checked_spec():
    check(sha(old.HERE / "DECODE_RECORD.md") == SPEC, "changed DR registration")
    r = bound(old.HERE / "receipts/joint_transfer_result.json", JT_RESULT)
    jt.parent()
    check(
        r["spec"] == jt.SPEC
        and r["decision"]["replicated_fixed_joint_candidate"] is True
        and r["ls_fit_freeze_sha"] == jt.FIT_SHA
        and r["ls_result_sha"] == jt.RESULT
        and r["writer_cost_sha"] == jt.COST_SHA
        and r["verified"]["independent_paired_bootstrap"] is True
        and r["neural_updates"] == r["new_fits"] == 0,
        "foreign JT closure",
    )
    return r


def next_block(records, pre, roles, cal, quota, size):
    check(
        all(
            x["split"] == "calibration" and int(x["sha256"], 16) % 100 == 0 for x in cal
        ),
        "wrong spent JT split",
    )
    return v.next_block(records, pre, roles, cal, quota, size)


def selection():
    r = checked_spec()
    book, records, pre, lsbook = jt.selection_context()
    check(
        jt.metadata(r["ref"]) == r["data"]
        and book == bound(jt.BASE / "reservation.json", r["data"]["rows_sha"]),
        "foreign spent JT reservation",
    )
    cal, nt = jt.checked_rows(book)
    check(len(cal) == 381 and nt == 525674, "wrong spent JT dose")
    rows, ex, supply = next_block(
        records,
        pre,
        ls.checked_roles(lsbook),
        cal,
        sum(N.values()),
        book["source"]["size"],
    )
    check(len(ex) == 43320 and supply == 23187, "incorrect DR TRAIN supply")
    return dict(
        schema="x49-dr-native-train",
        spec=SPEC,
        jt_result_sha=JT_RESULT,
        jt_rows_sha=r["data"]["rows_sha"],
        ls_rows_sha=book["ls_rows_sha"],
        source=book["source"],
        manifest_sha=book["manifest_sha"],
        eligible_train=supply,
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        roles=dict(fit=rows[: N["fit"]], probe=rows[N["fit"] :]),
    )


def checked_roles(book):
    return old.role_rows(book, "x49-dr-native-train", SPEC, N, MIN_OFFSET)


def prepare():
    return old.prepare(study="DR")


def metadata():
    return old.metadata(study="DR")


def licensed(job):
    return old.licensed(job, study="DR")


def fit_windows(job):
    return old.fit_windows(job, study="DR")


def extract(role, *, job=None, fit_sha=None):
    return old.extract(role, job=job, fit_sha=fit_sha, study="DR")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
