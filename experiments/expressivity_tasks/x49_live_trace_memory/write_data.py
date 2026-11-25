"""X49-W: authoritative metadata selection, TRAIN-only extraction and a bound stream. No short-test payload reader."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from time import monotonic, process_time

from ..x48_muon_continuation.data import (
    FLAG,
    MAN_SHA,
    MIN_OFFSET,
    load,
    sha,
    write_json,
)
from .panels import (
    OUT,
    SEEDS,
    T,
    check,
    digest,
    early_panels,
    fingerprint,
    membership,
    order,
    prior_sets,
    select,
    verify_selection,
    window,
)
from .relative_data import PARENT, RESERVATION as R_RES, metadata as r_metadata

SPEC = "2767ed0c1c42d8fbe6f04581b2df7aafbc1610f131c2b23291453d8db2b49394"
R_SHA = "abf0f3cc4b354787ef561f4ff09fbef51da43591732e548515a0134c2a09388d"
BASE = OUT / "write-data-1"
ROWS = BASE / "reservation.json"
TARGET_BUDGET = 561152


def choose_short(records, excluded, budget, *, min_offset=MIN_OFFSET):
    """After select() has validated the input manifest identities: full short documents, fixed rank."""
    check(type(budget) is int and budget > 0, "positive target budget required")
    by = {}
    for r in records:
        h = r["sha256"]
        i = r["byte_offset"]
        n = r["byte_length"]
        if (
            r["split"] == "final_test"
            and 2 <= n <= T
            and i >= min_offset
            and h not in excluded
        ):
            check(
                type(n) is int
                and type(i) is int
                and i == r["document"]
                and int(h, 16) % 100 == 1,
                "malformed short member",
            )
            by.setdefault(
                h,
                dict(doc=i, offset=i, length=n, split="final_test", start=0, sha256=h),
            )
    ranked = sorted(
        by.values(),
        key=lambda x: (
            digest(("x49-w-short-v1:" + x["sha256"]).encode("ascii")),
            x["sha256"],
            x["offset"],
        ),
    )
    chosen = []
    total = 0
    for r in ranked:
        chosen.append(r)
        total += r["length"] - 1
        if total >= budget:
            break
    check(
        total >= budget > total - (chosen[-1]["length"] - 1),
        "insufficient unspent short targets",
    )
    return chosen, len(ranked), sum(x["length"] - 1 for x in ranked)


def selection():
    check(
        sha(Path(__file__).with_name("WRITE_READ.md")) == SPEC
        and sha(R_RES) == R_SHA
        and sha(OUT / "v2/reservation.json") == PARENT,
        "foreign W registration or parent",
    )
    r = load(R_RES)
    check(r == r_metadata(), "foreign R membership")
    v2 = load(OUT / "v2/reservation.json")
    manifest = FLAG / "documents.jsonl.gz"
    check(sha(manifest) == MAN_SHA, "changed source manifest")
    with gzip.open(manifest, "rt") as f:
        records = list(map(json.loads, f))
    ex, _ = prior_sets()
    ex.update(early_panels(records))
    ex.update(membership(v2["panels"]))
    check(fingerprint(sorted(ex)) == r["excluded_sha"], "wrong exclusion history")
    ex.update(membership(r["rows"]))
    check(len(ex) == 39916, "wrong exclusion union")
    size = r["source"]["size"]
    quota = {"train": 1024}
    panels, counts = select(records, ex, quota, size)
    verify_selection(records, ex, panels, quota, counts)
    short, docs, nt = choose_short(records, ex, TARGET_BUDGET)
    panels["final_test"] = short
    check(
        counts == {"train": 25491}
        and (docs, nt) == (1040, 1565002)
        and len(set(membership(panels))) == 1024 + len(short)
        and not ex.intersection(membership(panels)),
        "changed W supply or overlapping reservation",
    )
    # Independent suffix witness: a different set-first construction in source-identity order.
    clean = [
        v
        for v in records
        if v["split"] == "final_test"
        and v["byte_offset"] >= MIN_OFFSET
        and 2 <= v["byte_length"] <= T
        and v["sha256"] not in ex
    ]
    uniq = {v["sha256"]: v for v in reversed(clean)}
    hs = sorted(uniq, key=lambda h: (digest(b"x49-w-short-v1:" + h.encode()), h))
    n = len(short)
    check(
        len(uniq) == docs
        and [x["sha256"] for x in short] == hs[:n]
        and all(
            x["doc"] == uniq[h]["document"]
            and x["length"] == uniq[h]["byte_length"]
            and x["start"] == 0
            for x, h in zip(short, hs[:n], strict=True)
        )
        and sum(uniq[h]["byte_length"] - 1 for h in hs[: n - 1])
        < TARGET_BUDGET
        <= sum(uniq[h]["byte_length"] - 1 for h in hs[:n]),
        "short test rank/hash/byte count mismatch",
    )
    return dict(
        schema="x49-write-v1",
        spec=SPEC,
        v2_sha=PARENT,
        r_sha=R_SHA,
        manifest_sha=MAN_SHA,
        excluded_sha=fingerprint(sorted(ex)),
        source=r["source"],
        eligible_train=counts["train"],
        eligible_short_final=docs,
        eligible_short_targets=nt,
        panels=panels,
    )


def prepare():
    check(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.environ.get("GPU_LEASE_HELD"),
        "W CPU prep only",
    )
    os.umask(0o077)
    BASE.mkdir(mode=0o700)
    start = monotonic()
    cpu = process_time()
    m = selection()
    write_json(ROWS, m)
    h = sha(ROWS)
    short = m["panels"]["final_test"]
    n = sum(x["length"] - 1 for x in short)
    reserve = dict(
        scope="metadata only",
        spec=SPEC,
        rows_sha=h,
        eligible_train=m["eligible_train"],
        eligible_short=m["eligible_short_final"],
        reserved_train=1024,
        reserved_short=len(short),
        reserved_short_targets=n,
        unused_short=m["eligible_short_final"] - len(short),
        payload_bytes_opened=0,
        wall_s=monotonic() - start,
        cpu_s=process_time() - cpu,
    )
    write_json(BASE / "reserved.json", reserve)
    # Raw open is for TRAIN descriptors only, after rebinding the immutable reservation, never short text.
    check(
        load(ROWS) == selection() and sha(ROWS) == h,
        "changed W reservation before TRAIN read",
    )
    rows = m["panels"]["train"]
    src = m["source"]
    tx = BASE / "train.preparing"
    tx.mkdir(mode=0o700)
    hs = []
    source_bytes = 0

    def verify(f):
        s = os.fstat(f.fileno())
        check(
            (s.st_ino, s.st_size, s.st_mtime_ns)
            == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "changed source descriptor",
        )

    with open(src["path"], "rb", buffering=0) as f, (tx / "windows.u8").open("xb") as z:
        verify(f)
        for row in rows:
            check(
                row["split"] == "train" and int(row["sha256"], 16) % 100 >= 2,
                "refuse non-TRAIN descriptor",
            )
            raw = os.pread(f.fileno(), row["length"], row["offset"])
            w = window(raw, row)
            source_bytes += len(raw)
            check(len(w) == T + 1, "incomplete W train shift")
            z.write(w)
            hs.append(
                dict(window=digest(w), inputs=digest(w[:-1]), targets=digest(w[1:]))
            )
        verify(f)
        z.flush()
        os.fsync(z.fileno())
    write_json(tx / "shifts.json", hs)
    ident = dict(
        schema="x49-write-train-v1",
        spec=SPEC,
        rows_sha=h,
        raw_sha=sha(tx / "windows.u8"),
        shifts_sha=sha(tx / "shifts.json"),
        orders={str(s): fingerprint(order(rows, s)) for s in SEEDS},
        windows=len(rows),
        targets=len(rows) * T,
        evaluation_windows_opened=0,
    )
    write_json(tx / "identity.json", ident)
    tx.rename(BASE / "train")
    out = dict(
        scope="TRAIN extracted only; test metadata reserved",
        reservation=reserve,
        spec=SPEC,
        rows_sha=h,
        short_documents=len(short),
        short_targets=n,
        train_identity=sha(BASE / "train/identity.json"),
        train_raw_source_bytes=source_bytes,
        train_window_bytes=len(rows) * (T + 1),
        train_targets=len(rows) * T,
        evaluation_windows_opened=0,
        total_prep_wall_s=monotonic() - start,
        total_prep_cpu_s=process_time() - cpu,
    )
    write_json(BASE / "prepared.json", out)
    return out


class WriteStream:
    def __init__(self, identity):
        f = BASE / "train"
        ip = f / "identity.json"
        check(sha(ip) == identity, "foreign W training identity")
        q = load(ip)
        check(
            q["schema"] == "x49-write-train-v1"
            and q["spec"] == SPEC
            and q["rows_sha"] == sha(ROWS)
            and q["targets"] == T * 1024
            and q["evaluation_windows_opened"] == 0,
            "foreign W registration",
        )
        m = load(ROWS)
        rr = m["panels"]["train"]
        self.data = (f / "windows.u8").read_bytes()
        check(
            digest(self.data) == q["raw_sha"]
            and sha(f / "shifts.json") == q["shifts_sha"]
            and len(rr) == q["windows"] == 1024
            and len(self.data) == len(rr) * (T + 1),
            "truncated W stream",
        )
        hs = load(f / "shifts.json")
        self.docs = tuple(v["doc"] for v in rr)
        self.orders = {s: tuple(order(rr, s)) for s in SEEDS}
        check(
            len(set(self.docs)) == 1024
            and {str(s): fingerprint(v) for s, v in self.orders.items()} == q["orders"],
            "wrong W seed order",
        )
        for i, (r, x) in enumerate(zip(rr, hs, strict=True)):
            w = self.row(i)
            check(
                r["split"] == "train"
                and x
                == dict(window=digest(w), inputs=digest(w[:-1]), targets=digest(w[1:])),
                "misaligned W shift",
            )

    def row(self, i):
        return self.data[i * (T + 1) : (i + 1) * (T + 1)]

    def sample(self, seed, index):
        check(
            type(seed) is int
            and seed in self.orders
            and type(index) is int
            and 1 <= index <= 1024,
            "bad W visit",
        )
        i = self.orders[seed][index - 1]
        return self.docs[i], self.row(i)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
