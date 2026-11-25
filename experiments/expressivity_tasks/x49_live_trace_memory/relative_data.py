"""X49-R fresh test reservation. Metadata only; no raw corpus/split loader."""

from __future__ import annotations

import gzip
import json
import os
from time import monotonic, process_time
from pathlib import Path
from ..x48_muon_continuation.data import FLAG, MAN_SHA, load, sha, write_json
from .panels import (
    OUT,
    check,
    early_panels,
    fingerprint,
    membership,
    prior_sets,
    reserve_data,
    select,
    verify_selection,
)
from .train_stream import V2_IDENTITY

SPEC = "59890e741228a29c46cd987f0e8d1f0c1c2381a0e43e9ce81af9b00c971a43fa"
PARENT = "cdec56106c43b9b5abcdfe0d42a24888ec2c62c925fc1df9cb40802a952c2f1b"
N = 137
BASE = OUT / "relative-data-1"
RESERVATION = BASE / "reservation.json"


def remaining(records, excluded, old_panels, source_size, n):
    """Exclude all v2 memberships before taking the entire remaining eligible native final set."""
    old = set(membership(old_panels))
    old_final = old_panels["final_test"]
    combined = excluded | old
    quota = {"final_test": n}
    fresh, counts = select(records, combined, quota, source_size)
    check(counts == quota, "unexpected number of fresh final documents")
    verify_selection(records, combined, fresh, quota, counts)
    original, counts0 = select(
        records, excluded, {"final_test": n + len(old_final)}, source_size
    )
    check(
        counts0 == {"final_test": n + len(old_final)}
        and original["final_test"] == old_final + fresh["final_test"]
        and not set(membership(fresh)) & old,
        "not the exact unspent suffix of the original ranked final",
    )
    return fresh["final_test"], fingerprint(sorted(combined))


def metadata():
    check(
        sha(Path(__file__).with_name("RELATIVE.md")) == SPEC, "changed R registration"
    )
    p = OUT / "v2/reservation.json"
    ip = OUT / "v2/train/identity.json"
    check(
        sha(ip) == V2_IDENTITY and sha(p) == load(ip)["reservation_sha"] == PARENT,
        "foreign v2 source",
    )
    v2 = load(p)
    check(v2 == reserve_data("v2"), "changed v2 parent or exclusions")
    f = FLAG / "documents.jsonl.gz"
    check(sha(f) == v2["manifest_sha"] == MAN_SHA, "foreign raw document manifest")
    with gzip.open(f, "rt") as handle:
        records = list(map(json.loads, handle))
    ex, _ = prior_sets()
    ex |= early_panels(records)
    check(fingerprint(sorted(ex)) == v2["excluded_sha"], "foreign exclusions")
    rows, excluded = remaining(records, ex, v2["panels"], v2["source"]["size"], N)
    return dict(
        schema="x49-relative-fresh-final-v1",
        spec=SPEC,
        manifest_sha=MAN_SHA,
        parent_sha=PARENT,
        source=v2["source"],
        excluded_sha=excluded,
        rows=rows,
    )


def prepare():
    check(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.environ.get("GPU_LEASE_HELD"),
        "CPU metadata only",
    )
    os.umask(0o077)
    start = monotonic()
    cpu = process_time()
    BASE.mkdir(mode=0o700)
    r = metadata()
    write_json(RESERVATION, r)
    receipt = dict(
        spec=SPEC,
        reservation_sha=sha(RESERVATION),
        v2_eligible=265,
        v2_spent=128,
        fresh=len(r["rows"]),
        native_split="final_test",
        raw_bytes_opened=0,
        wall_s=monotonic() - start,
        cpu_core_s=process_time() - cpu,
    )
    write_json(BASE / "prepared.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True))
