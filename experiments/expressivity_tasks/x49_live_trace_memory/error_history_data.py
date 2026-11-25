"""EH metadata chain and single-use native TRAIN reader. RB windows, tables and axes are not EH inputs."""

import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha
from . import residual_data as rb, query_data, surprise_data as old
from .surprise_data import bound as bound
from .panels import OUT, fingerprint

BASE = OUT / "error-history-data-1"
SPEC = "fc1980513d22bfb2d93353253c5efd6e902c283eedd71daec534953d4626ef31"
RB_RESULT = "5a8f76f6e2f9cb20e8f426b2a86957327d33b9a81d2d9c86229f1ef31af7849d"
N = {"fit": 256, "probe": 128}


def checked_spec():
    check(sha(old.HERE / "ERROR_HISTORY.md") == SPEC, "changed EH registration")
    rb.checked_spec()
    r = bound(old.HERE / "receipts/residual_byte_result.json", RB_RESULT)
    check(
        r["spec"] == rb.SPEC
        and r["dr_result_sha"] == rb.DR_RESULT
        and r["verified"]["independent_paired_bootstrap"] is True
        and r["cost"]["rc"] == 0
        and r["neural_updates"] == r["u_refits"] == r["j_refits"] == 0,
        "foreign RB receipt",
    )
    return r


def next_block(book, records, pre, quota):
    roles = rb.checked_roles(book)
    return query_data.select_new(
        records,
        base_ex=pre,
        prior={"train": roles["fit"] + roles["probe"]},
        quota=quota,
        size=book["source"]["size"],
    )


def selection_context():
    """Rebuild EH descriptors and pre-EH exclusions for a later study; no corpus read."""
    r = checked_spec()
    book, records, pre = rb.selection_context()
    m = r["data"]
    check(
        rb.metadata() == m
        and book == bound(rb.BASE / "reservation.json", m["rows_sha"]),
        "spent RB membership changed",
    )
    rows, ex, supply = next_block(book, records, pre, sum(N.values()))
    check((len(ex), supply) == (44088, 22419), "wrong EH exclusions or TRAIN supply")
    descriptor = dict(
        schema="x49-eh-native-train",
        spec=SPEC,
        rb_result_sha=RB_RESULT,
        rb_rows_sha=m["rows_sha"],
        manifest_sha=book["manifest_sha"],
        source=book["source"],
        eligible_train=supply,
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        roles=dict(fit=rows[: N["fit"]], probe=rows[N["fit"] :]),
    )
    return descriptor, records, ex


def selection():
    return selection_context()[0]


def checked_roles(book):
    return old.role_rows(book, "x49-eh-native-train", SPEC, N, MIN_OFFSET)


def prepare():
    return old.prepare(study="EH")


def metadata():
    return old.metadata(study="EH")


def licensed(job):
    return old.licensed(job, study="EH")


def fit_windows(job):
    return old.fit_windows(job, study="EH")


def extract(role, *, job=None, fit_sha=None):
    return old.extract(role, job=job, fit_sha=fit_sha, study="EH")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
