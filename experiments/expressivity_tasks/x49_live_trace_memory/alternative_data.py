"""X49-V native TRAIN roles after every old reservation incl. the frozen U native final."""

from __future__ import annotations
import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha
from . import surprise_data as old, confirm_data as final, local_u, query_data
from .panels import OUT, fingerprint
from .surprise_data import bound

BASE = OUT / "alternative-data-1"
SPEC = "08b1691a1adcba52de0c839791bc0fe7f9377a703220a025a8bc1ea4e8f0b6a2"
N = {"fit": 256, "probe": 128}


def checked_spec():
    check(sha(old.HERE / "FORECAST_ALTERNATIVE.md") == SPEC, "changed V protocol")
    r = bound(local_u.RECEIPT, local_u.CONFIRMATION)
    check(
        r["spec"] == final.SPEC
        and r["u_result_sha"] == final.RESULT
        and r["u_fit_freeze_sha"] == final.FIT_SHA
        and r["decision"]["recommend_local_frozen_U"] is True
        and r["new_fits"] == r["neural_updates"] == 0,
        "wrong confirmed incumbent",
    )
    return r


def next_block(records, pre, roles, short, quota, size):
    final_shas = {row["sha256"] for row in short}
    us = roles["fit"] + roles["probe"]
    used = {row["sha256"] for row in us}
    check(
        len(final_shas) == len(short)
        and len(used) == len(us)
        and not final_shas.intersection(pre | used)
        and not (used & pre),
        "overlapping earlier V exclusions",
    )
    return query_data.select_new(
        records, base_ex=pre | final_shas, prior={"train": us}, quota=quota, size=size
    )


def selection():
    r = checked_spec()
    b, records, pre, roles = final.selection_context()
    check(
        final.metadata(r["ref"]) == r["data"]
        and b == bound(final.BASE / "reservation.json", r["data"]["rows_sha"]),
        "changed prior native final reservation",
    )
    short, _ = final.checked_rows(b)
    check(
        len(short) == 342 and len(roles["fit"]) == 256 and len(roles["probe"]) == 128,
        "prior V exclusions changed",
    )
    rows, ex, supply = next_block(
        records, pre, roles, short, sum(N.values()), b["source"]["size"]
    )
    check(len(ex) == 42171 and supply == 23955, "wrong unspent TRAIN supply for V")
    return dict(
        schema="x49-v-native-train",
        spec=SPEC,
        confirmation_sha=local_u.CONFIRMATION,
        confirmation_rows_sha=r["data"]["rows_sha"],
        u_rows_sha=b["u_rows_sha"],
        manifest_sha=b["manifest_sha"],
        source=b["source"],
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        eligible_train=supply,
        roles={"fit": rows[: N["fit"]], "probe": rows[N["fit"] :]},
    )


def checked_roles(book):
    return old.role_rows(book, "x49-v-native-train", SPEC, N, MIN_OFFSET)


def prepare():
    return old.prepare(study="V")


def metadata():
    return old.metadata(study="V")


def licensed(job):
    return old.licensed(job, study="V")


def fit_windows(job):
    return old.fit_windows(job, study="V")


def extract(role, *, job=None, fit_sha=None):
    return old.extract(role, job=job, fit_sha=fit_sha, study="V")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
