"""ET's next contiguous short-calibration block. Metadata only, after the published ET qualification."""

import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check
from . import (
    error_transfer_reader as reader,
    error_history_data as eh,
    transfer_data as jt,
    decode_data as dr,
    confirm_data as short,
    surprise_data as u,
)
from .panels import OUT, fingerprint, membership
from .transfer_data import rank as rank
from .error_transfer_reader import parent as parent

BASE = OUT / "error-transfer-data-1"
SPEC = reader.PROTOCOL
FIT_SHA = reader.FIT
GOAL = 524288


def next_short(records, ex, spent, size, goal, lo):
    prior = {r["sha256"] for r in spent}
    check(
        prior and len(prior) == len(spent) and prior <= ex,
        "missing spent JT short-calibration cohort",
    )
    rows, n, nt = jt.choose(records, ex, size, goal, lo)
    old_nt = sum(r["length"] - 1 for r in spent)
    both, full_n, full_nt = jt.choose(records, ex - prior, size, old_nt + goal, lo)
    check(
        both == spent + rows and full_n == n + len(spent) and full_nt == nt + old_nt,
        "not the contiguous block after JT",
    )
    return rows, n, nt


def spent_context(meta):
    book, records, pre = eh.selection_context()
    check(
        eh.metadata() == meta
        and book == u.bound(eh.BASE / "reservation.json", meta["rows_sha"]),
        "changed spent EH reservation",
    )
    old = set(membership(eh.checked_roles(book)))
    check(len(old) == sum(eh.N.values()) and not old & pre, "wrong spent EH roles")
    return book, records, pre | old


def selection():
    meta = parent()["data"]
    book, records, ex = spent_context(meta)
    j = dr.checked_spec()
    jrows, _ = jt.checked_rows(
        u.bound(jt.BASE / "reservation.json", j["data"]["rows_sha"])
    )
    rows, n, nt = next_short(
        records, ex, jrows, book["source"]["size"], GOAL, MIN_OFFSET
    )
    check(
        (len(ex), n, nt) == (44472, 558, 810074),
        "wrong ET short supply or historical exclusions",
    )
    return dict(
        spec=SPEC,
        eh_result_sha=reader.PUBLIC,
        eh_rows_sha=meta["rows_sha"],
        jt_rows_sha=j["data"]["rows_sha"],
        source=book["source"],
        manifest_sha=book["manifest_sha"],
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        eligible_documents=n,
        eligible_targets=nt,
        rows=rows,
    )


def checked_rows(book):
    return jt.checked_short_calibration(book, spec=SPEC, lo=MIN_OFFSET, goal=GOAL)


def reserve():
    return short.reserve(study="ET")


def metadata(ref):
    return short.metadata(ref, study="ET")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["reserve"])
    ap.parse_args()
    print(json.dumps(reserve(), sort_keys=True))
