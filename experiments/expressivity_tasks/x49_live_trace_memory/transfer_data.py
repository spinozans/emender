"""JT: untouched, native short calibration, reserved on metadata only. No coefficient calibration or final reader."""

from __future__ import annotations
import argparse
import json
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha
from . import (
    confirm_data as prior,
    joint_reader as joint,
    literal_mark_data as ls,
    local_u,
    surprise_data as u,
)
from .panels import OUT, T, digest, fingerprint, membership

BASE = OUT / "joint-transfer-data-1"
SPEC = "047b7573bfd933cf66479a9d5492affa9e28268c8bc9e1f04734eb49f771c2d9"
COST_SHA = "25389c636ec9524888bf63f0a8301abaab449ba45aa6e4b427f0474bfc660c0c"
RESULT = joint.RESULT
FIT_SHA = joint.FIT
GOAL = 524288


def parent():
    check(sha(u.HERE / "JOINT_TRANSFER.md") == SPEC, "changed JT registration")
    ls.checked_spec()
    r = u.bound(joint.RECEIPTS / "literal_surprise_result.json", RESULT)
    cost = u.bound(joint.RECEIPTS / "joint_reader_cost.json", COST_SHA)
    check(
        r["fit_freeze_sha"] == FIT_SHA
        and r["spec"] == ls.SPEC
        and r["u_confirmation_sha"] == local_u.CONFIRMATION
        and r["decision"]["consider_joint_confirmation"] is True
        and cost["synthetic_parity"] is True
        and cost["ls_result_sha256"] == RESULT
        and sha(joint.COST / "aggregate.json") == COST_SHA
        and sha(joint.COST / "build/text_tape.so") == cost["build"]["library_sha256"]
        and r["neural_updates"] == r["u_refits"] == 0,
        "foreign JT predictors or native writer",
    )
    return r


def rank(h):
    return digest(b"x49-jt-short-cal-v1:" + h.encode("ascii"))


def choose(records, excluded, size, goal, lo):
    # Manifest/split validity checked by parent selection. For duplicates keep earliest eligible offset.
    first = {}
    check(type(goal) is int and goal > 0, "positive JT target dose required")
    for r in records:
        h = r["sha256"]
        n = r["byte_length"]
        off = r["byte_offset"]
        if r["split"] != "calibration" or not 2 <= n <= T or off < lo or h in excluded:
            continue
        check(
            type(n) is type(off) is int
            and off == r["document"]
            and off + n <= size
            and len(h) == 64
            and f"{int(h, 16):064x}" == h
            and int(h, 16) % 100 == 0,
            "malformed JT calibration descriptor",
        )
        if h in first:
            check(first[h]["length"] == n, "inconsistent JT duplicate length")
        if h not in first or off < first[h]["offset"]:
            first[h] = dict(
                doc=off, offset=off, length=n, split="calibration", sha256=h, start=0
            )
    ordered = sorted(
        first.values(), key=lambda r: (rank(r["sha256"]), r["sha256"], r["doc"])
    )
    total = sum(r["length"] - 1 for r in ordered)
    check(
        total >= goal,
        f"insufficient clean short calibration targets: {len(ordered)} documents, {total} targets, need {goal}; no replacement",
    )
    n = 0
    for i, r in enumerate(ordered):
        n += r["length"] - 1
        if n >= goal:
            return ordered[: i + 1], len(ordered), total
    raise AssertionError("crossing not found")


def selection():
    r = parent()
    b, records, pre = ls.selection_context()
    check(
        r["data"] == ls.metadata()
        and b == u.bound(ls.BASE / "reservation.json", r["data"]["rows_sha"]),
        "foreign spent LS cohorts",
    )
    old = set(membership(ls.checked_roles(b)))
    check(
        len(old) == 384 and not old.intersection(pre) and len(pre) == 42555,
        "wrong pre-JT exclusions",
    )
    ex = pre | old
    rows, n, nt = choose(records, ex, b["source"]["size"], GOAL, MIN_OFFSET)
    return dict(
        spec=SPEC,
        ls_result_sha=RESULT,
        ls_rows_sha=r["data"]["rows_sha"],
        source=b["source"],
        manifest_sha=b["manifest_sha"],
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        eligible_documents=n,
        eligible_targets=nt,
        rows=rows,
    )


def checked_rows(book):
    rows = book["rows"]
    check(book["spec"] == SPEC and rows, "incorrect JT reservation")
    seen = set()
    ids = set()
    order = []
    for x in rows:
        h = x["sha256"]
        off = x["offset"]
        n = x["length"]
        check(
            x["split"] == "calibration"
            and len(h) == 64
            and f"{int(h, 16):064x}" == h
            and int(h, 16) % 100 == 0
            and type(n) is type(off) is int
            and off == x["doc"] >= MIN_OFFSET
            and off + n <= book["source"]["size"]
            and 2 <= n <= T
            and x["start"] == 0
            and h not in seen
            and off not in ids,
            "incorrect whole short calibration membership",
        )
        order.append((rank(h), h, off))
        ids.add(off)
        seen.add(h)
    total = sum(x["length"] - 1 for x in rows)
    check(
        order == sorted(order)
        and total >= GOAL > total - rows[-1]["length"] + 1
        and book["eligible_documents"] >= len(rows)
        and book["eligible_targets"] >= total,
        "incorrect JT rank crossing",
    )
    return rows, total


def reserve():
    return prior.reserve(study="JT")


def metadata(ref):
    return prior.metadata(ref, study="JT")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["reserve"])
    ap.parse_args()
    print(json.dumps(reserve(), sort_keys=True))
