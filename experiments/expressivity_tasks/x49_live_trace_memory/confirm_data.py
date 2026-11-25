"""Native held-out U confirmation: single-use short-document reservation, metadata only."""

from __future__ import annotations
import argparse
import json
import os
from time import monotonic, process_time
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from . import surprise_data as u
from .panels import OUT, T, digest, fingerprint, membership
from .write_data import choose_short

BASE = OUT / "confirm-data-1"
SPEC = "b0b5c38827e167d8f80efe4aba05d0995a464841c2e2969a166c8b48340ace46"
RESULT = "2acc7cba0b6e63de8c6db705a74499da04b08138ec3e0777f5a02b99903615a6"
FIT_SHA = "67656fe3d1053a30519574c70e0c6258c4bced91089be171ffb1d832075156b9"
GOAL = 524288


def parent():
    u.checked_spec()
    check(
        sha(u.HERE / "SURPRISE_CONFIRMATION.md") == SPEC,
        "changed confirmation registration",
    )
    r = u.bound(u.HERE / "receipts/surprise_result.json", RESULT)
    check(
        r["fit_freeze_sha"] == FIT_SHA and r["neural_updates"] == 0, "foreign U parent"
    )
    return r


def next_short(records, excluded, spent, goal, *, lo=MIN_OFFSET):
    """Choose by the old short rank; independently reconstruct the combined old + new prefix."""
    prev = {x["sha256"] for x in spent}
    check(len(prev) == len(spent) and prev <= excluded, "missing W reservation")
    rows, n, nt = choose_short(records, excluded, goal, min_offset=lo)
    earlier = excluded - prev
    first = {}
    for r in records:
        if (
            r["split"] == "final_test"
            and r["byte_offset"] >= lo
            and 2 <= r["byte_length"] <= T
            and r["sha256"] not in earlier
        ):
            first.setdefault(r["sha256"], r)
    order = sorted(
        first,
        key=lambda h: (
            digest(b"x49-w-short-v1:" + h.encode("ascii")),
            h,
            first[h]["document"],
        ),
    )
    both = spent + rows
    check(
        [x["sha256"] for x in both] == order[: len(both)]
        and all(
            x
            == dict(
                doc=first[h]["document"],
                offset=first[h]["byte_offset"],
                start=0,
                split="final_test",
                length=first[h]["byte_length"],
                sha256=h,
            )
            for h, x in zip(order[: len(both)], both, strict=True)
        ),
        "not the next segment of W short rank",
    )
    shifts = sum(r["length"] - 1 for r in rows)
    check(
        shifts >= goal > shifts - (rows[-1]["length"] - 1)
        and n == len(order) - len(prev)
        and nt == sum(first[h]["byte_length"] - 1 for h in order[len(prev) :]),
        "incorrect short crossing",
    )
    return rows, n, nt


def selection():
    r = parent()
    book, records, pre, w = u.selection_context()
    check(
        book == u.bound(u.BASE / "reservation.json", r["data"]["rows_sha"]),
        "changed U reservation",
    )
    previous = set(membership(u.checked_roles(book)))
    check(len(previous) == 384 and not (previous & pre), "invalid prior U exclusion")
    ex = pre | previous
    rows, n, nt = next_short(records, ex, w["panels"]["final_test"], GOAL)
    check((n, nt) == (663, 1002290) and len(ex) == 41829, "changed confirmation supply")
    return dict(
        spec=SPEC,
        u_rows_sha=r["data"]["rows_sha"],
        source=book["source"],
        manifest_sha=book["manifest_sha"],
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        eligible_documents=n,
        eligible_targets=nt,
        rows=rows,
    )


def checked_rows(book):
    rows = book["rows"]
    hs = set()
    docs = set()
    rank = []
    check(book["spec"] == SPEC and rows, "foreign or empty final")
    for x in rows:
        h = x["sha256"]
        off = x["offset"]
        n = x["length"]
        check(
            x["split"] == "final_test"
            and int(h, 16) % 100 == 1
            and x["doc"] == off >= MIN_OFFSET
            and 2 <= n <= T
            and off + n <= book["source"]["size"]
            and x["start"] == 0
            and h not in hs
            and off not in docs,
            "incorrect whole short final membership",
        )
        hs.add(h)
        docs.add(off)
        rank.append((digest(b"x49-w-short-v1:" + h.encode()), h, off))
    total = sum(x["length"] - 1 for x in rows)
    check(
        rank == sorted(rank) and total >= GOAL > total - (rows[-1]["length"] - 1),
        "incorrect frozen short prefix",
    )
    return rows, total


def reserve():
    from .panel import cpu
    from .confirm_run import checked_fixture_ref

    cpu()
    ref = checked_fixture_ref()
    os.umask(0o077)
    tic = monotonic()
    c0 = process_time()
    BASE.mkdir(mode=0o700)
    book = selection()
    rows, total = checked_rows(book)
    f = BASE / "reservation.json"
    write_json(f, book)
    flush_dir(BASE)
    h = sha(f)
    check(u.bound(f, h) == selection(), "reservation did not rederive")
    z = dict(
        spec=SPEC,
        ref=ref,
        rows_sha=h,
        documents=len(rows),
        targets=total,
        unreserved=book["eligible_documents"] - len(rows),
        excluded_unique=book["excluded_unique"],
        payload_documents_opened=0,
        wall_s=monotonic() - tic,
        cpu_s=process_time() - c0,
    )
    write_json(BASE / "reserved.json", z)
    return z


def metadata(ref):
    parent()
    raw = (BASE / "reserved.json").read_bytes()
    z = json.loads(raw)
    book = u.bound(BASE / "reservation.json", z["rows_sha"])
    rows, total = checked_rows(book)
    check(
        z["spec"] == SPEC
        and z["ref"] == ref
        and z["payload_documents_opened"] == 0
        and (z["documents"], z["targets"]) == (len(rows), total),
        "unqualified or foreign final reservation",
    )
    return dict(
        reserved_sha=digest(raw),
        rows_sha=z["rows_sha"],
        documents=len(rows),
        targets=total,
    )


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("phase", choices=["reserve"])
    a.parse_args()
    print(json.dumps(reserve(), sort_keys=True))
