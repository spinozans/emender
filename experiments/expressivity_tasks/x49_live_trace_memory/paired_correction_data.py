"""PC: the next contiguous TRAIN block after EH, also excluding the entire closed ET calibration cohort."""

import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha
from . import (
    error_transfer_data as et,
    error_transfer_reader as earlier,
    query_data,
    surprise_data as old,
)
from .surprise_data import bound as bound
from .panels import OUT, SEEDS, fingerprint

BASE = OUT / "paired-correction-data-1"
SPEC = "107ffdd012a60416159995634c1c9b2786610fbf758a6ae1bb94b0354aa64c49"
ET_RESULT = "0d58f3aa50fb6830e998747530043796f48a2c0d636b53452c48a4c3f480ef1f"
ET_CLOSE = "24da7fa08c8b2c740a073c676acd6a76df5ef7fb48894d3791e6df8991fc66b0"
N = {"fit": 256, "probe": 128}


def checked_spec():
    check(sha(old.HERE / "PAIRED_CORRECTION.md") == SPEC, "changed PC registration")
    pub = bound(old.HERE / "receipts/error_transfer_result.json", ET_RESULT)
    r = bound(OUT / "error-transfer-result-1.json", ET_CLOSE)
    check(
        pub["private_close_sha256"] == ET_CLOSE
        and all(pub[k] == v for k, v in r.items() if k != "predictors")
        and pub["checkpoints"]
        == {k: e["checkpoint"] for k, e in r["predictors"].items()}
        and set(pub["checkpoints"]) == set(map(str, SEEDS))
        and r["spec"] == et.SPEC
        and r["eh_result_sha"] == earlier.PUBLIC
        and r["eh_fit_freeze_sha"] == earlier.FIT
        and r["eh_close_sha"] == earlier.CLOSED
        and r["verified"]["independent_branch_cost_and_support"]
        and r["verified"]["independent_paired_bootstrap"]
        and r["swap_fit"] == "ERROR"
        and r["decision"]["replicated_bound_wrong_guesses"]
        and r["decision"]["replicated_pooled_scalar_history"]
        and r["cost"]["rc"] == r["neural_updates"] == r["new_fits"] == 0,
        "foreign ET result or fixed readout lineage",
    )
    return r


def next_block(earlier_train, spent_short, records, after_eh, quota):
    roles = et.eh.checked_roles(earlier_train)
    short, _ = et.checked_rows(spent_short)
    train = roles["fit"] + roles["probe"]
    train_hs = {r["sha256"] for r in train}
    short_hs = {r["sha256"] for r in short}
    check(
        len(train) == len(train_hs)
        and len(short) == len(short_hs)
        and train_hs <= after_eh
        and not (short_hs & after_eh)
        and earlier_train["source"] == spent_short["source"],
        "missing EH or overlapping ET reservation",
    )
    # Also prove exact TRAIN rank continuity, not just disjoint windows at different offsets.
    return query_data.select_new(
        records,
        base_ex=(after_eh - train_hs) | short_hs,
        prior={"train": train},
        quota=quota,
        size=spent_short["source"]["size"],
    )


def selection_context():
    r = checked_spec()
    b, records, ex, train = et.selection_context()
    m = r["data"]
    check(
        m == et.metadata(r["ref"])
        and b == bound(et.BASE / "reservation.json", m["rows_sha"]),
        "spent ET descriptors changed",
    )
    rows, exclude, supply = next_block(train, b, records, ex, sum(N.values()))
    check(
        (len(exclude), supply) == (44833, 22035),
        "wrong post-ET exclusions or full TRAIN supply",
    )
    descriptor = dict(
        schema="x49-pc-native-train",
        spec=SPEC,
        et_result_sha=ET_RESULT,
        et_rows_sha=m["rows_sha"],
        eh_rows_sha=b["eh_rows_sha"],
        manifest_sha=b["manifest_sha"],
        source=b["source"],
        eligible_train=supply,
        excluded_unique=len(exclude),
        excluded_sha=fingerprint(sorted(exclude)),
        roles=dict(fit=rows[: N["fit"]], probe=rows[N["fit"] :]),
    )
    return descriptor, records, exclude


def selection():
    return selection_context()[0]


def checked_roles(book):
    return old.role_rows(book, "x49-pc-native-train", SPEC, N, MIN_OFFSET)


def prepare():
    return old.prepare(study="PC")


def metadata():
    return old.metadata(study="PC")


def licensed(job):
    return old.licensed(job, study="PC")


def fit_windows(job):
    return old.fit_windows(job, study="PC")


def extract(role, *, job=None, fit_sha=None):
    return old.extract(role, job=job, fit_sha=fit_sha, study="PC")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
