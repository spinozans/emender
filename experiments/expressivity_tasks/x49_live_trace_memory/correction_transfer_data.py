"""PT: the complete short calibration remainder, after both consecutive short blocks and all of PC."""

import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check
from . import (
    paired_correction_data as pc,
    correction_transfer_reader as reader,
    confirm_data as short,
    transfer_data as jt,
)
from . import surprise_data as u
from .transfer_data import rank as rank
from .correction_transfer_reader import parent as parent
from .panels import OUT, fingerprint, membership

BASE = OUT / "correction-transfer-data-1"
SPEC = reader.PROTOCOL
FIT_SHA = reader.FIT
DOCS = 197
GOAL = 285520


def remainder(records, ex, first, second, size, docs, targets, lo):
    check(first and second, "both earlier short prefixes required")
    # next_short independently reconstructs the concatenation of the old prefixes and the new rows.
    rows, n, nt = pc.et.next_short(records, ex, first + second, size, targets, lo)
    check(
        len(rows) == n == docs and sum(r["length"] - 1 for r in rows) == nt == targets,
        "PT must exhaust the exact remaining inventory",
    )
    return rows


def spent_context(meta):
    book, records, ex = pc.selection_context()
    check(
        pc.metadata() == meta
        and book == u.bound(pc.BASE / "reservation.json", meta["rows_sha"]),
        "changed spent PC reservation",
    )
    old = set(membership(pc.checked_roles(book)))
    check(
        len(old) == sum(pc.N.values()) and not old & ex,
        "incomplete or overlapping spent PC roles",
    )
    return book, records, ex | old


def selection():
    from .correction_transfer_run import checked_fixture_ref

    checked_fixture_ref()  # not even a natural metadata census until the public gate
    m = parent()["data"]
    book, records, ex = spent_context(m)
    et = u.bound(pc.et.BASE / "reservation.json", book["et_rows_sha"])
    second, _ = pc.et.checked_rows(et)
    jb = u.bound(jt.BASE / "reservation.json", et["jt_rows_sha"])
    first, _ = jt.checked_rows(jb)
    check(
        len(ex) == 45217
        and book["source"] == et["source"] == jb["source"]
        and book["manifest_sha"] == et["manifest_sha"] == jb["manifest_sha"],
        "wrong post-PC exclusions or manifest",
    )
    rows = remainder(
        records, ex, first, second, book["source"]["size"], DOCS, GOAL, MIN_OFFSET
    )
    return dict(
        spec=SPEC,
        pc_result_sha=reader.PUBLIC,
        pc_rows_sha=m["rows_sha"],
        et_rows_sha=book["et_rows_sha"],
        jt_rows_sha=et["jt_rows_sha"],
        source=book["source"],
        manifest_sha=book["manifest_sha"],
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        eligible_documents=DOCS,
        eligible_targets=GOAL,
        rows=rows,
    )


def checked_rows(book):
    rows, nt = jt.checked_short_calibration(book, spec=SPEC, lo=MIN_OFFSET, goal=GOAL)
    check(
        len(rows) == book["eligible_documents"] == DOCS
        and nt == book["eligible_targets"] == GOAL,
        "PT is not the complete short calibration remainder",
    )
    return rows, nt


def reserve():
    return short.reserve(study="PT")


def metadata(ref):
    return short.metadata(ref, study="PT")


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("phase", choices=["reserve"])
    a.parse_args()
    print(json.dumps(reserve(), sort_keys=True))
