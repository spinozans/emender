"""X49 metadata reservation and TRAIN-only extraction. No evaluation byte reader here."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re

from ..x43_fork_resolved.c14_documents import DocumentWindows
from ..x48_muon_continuation.data import (
    FLAG,
    MAN_SHA,
    MIN_OFFSET,
    OLD_SHA,
    PANEL,
    REG_SHA,
    ROOT,
    check,
    sha,
    write_json,
)

T = 4096
N = {"train": 1024, "calibration": 128, "final_test": 128}
SEEDS = (4901, 4902, 4903)
OUT = ROOT / "output/x49-live"
X43R = FLAG.parent
ROWS48 = ROOT / "output/x48-muon/data/rows.jsonl"
ROWS48_SHA = "54c38b57716b8204c3aae8b4983b7a88a0fa76193752f9fc1af4dcb17d4b52bd"
# Historical membership only. Never consume an old target payload, route tape or model score.
EXTRA = {
    X43R
    / "p100_heldout/selection.json": "d28d42bf484699e0295201117b741e9b2401643313918fd3e63ae52622008a0d",
    X43R
    / "xsa_controller_v1/manifest.json": "f547152c8e1535fe0114f5cf540dcc65a64179d22bee554c60f2d2d8ca9ec421",
    X43R
    / "xsa_neural_evidence_v1/targets.json": "b194151f39c279e07d309befa00f4a8a182e0613b11375e7a8cc00e930807136",
    X43R
    / "natural_position_v1/documents.json": "f3fc30c5f612f7da047e9a3b45682a291a5d825b10de6eed144c2bc0d07debb6",
    X43R
    / "xsa_read_survey_v1/final_reservation.jsonl": "4443eb30e70c8468ac7a8f0f0d7e4eb7b1953017c90ce94fc20eb982f089514b",
    X43R
    / "xsa_budget_campaign_v1/2048/manifest.json": "62df78536087b348d5db995cd8e09cca848fe46f5a57948e8b46743730d38933",
    Path(
        "/mnt/nvme2n1/erikg/xsa_budget_campaign_v1/8192/manifest.json"
    ): "41d61b56874c3948c67469025a78b1bf8b78acf6edc585b08fe731345afabba5",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(obj):
    return digest(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def membership(obj):
    """Only document descriptors, not arbitrary fields named sha256 (e.g. a source-file hash)."""
    if isinstance(obj, list):
        for r in obj:
            yield from membership(r)
    elif isinstance(obj, dict):
        if (
            ("doc" in obj or "document" in obj)
            and ("offset" in obj or "byte_offset" in obj)
            and "sha256" in obj
        ):
            h = obj["sha256"]
            check(
                isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h),
                "invalid historical document SHA",
            )
            yield h
        else:
            for v in obj.values():
                yield from membership(v)


def prior_sets():
    hashes, receipts = set(), {}

    def include(p, expected, *, w3=False):
        raw = p.read_bytes()
        check(digest(raw) == expected, f"changed exclusion manifest: {p}")
        rows = (
            [json.loads(line) for line in raw.splitlines()]
            if p.suffix == ".jsonl"
            else json.loads(raw)
        )
        found = set(membership(rows["panels"] if w3 else rows))
        check(found, f"unrecognized historical membership: {p}")
        if p == ROWS48:
            check(len(rows) == len(found) == 32768, "incomplete X48 exclusion")
        if p.name == "final_reservation.jsonl":
            check(len(rows) == len(found) == 24, "incomplete survey reservation")
        hashes.update(found)
        receipts[str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p)] = dict(
            sha256=expected, documents=len(found)
        )
        return rows

    parent = include(PANEL, OLD_SHA, w3=True)
    check(
        parent["manifest_sha"] == MAN_SHA and parent["registration_sha"] == REG_SHA,
        "foreign W3 parent",
    )
    for name, expected in parent["exclusions"].items():
        include(ROOT / name, expected)
    for p, expected in [*EXTRA.items(), (ROWS48, ROWS48_SHA)]:
        include(p, expected)
    return hashes, receipts


def early_panels(records):
    """Reconstruct old source-order memberships without opening their raw test/score logs."""

    def take(split, length, start, stop):
        return {
            r["sha256"]
            for r in islice(
                (
                    r
                    for r in records
                    if r["split"] == split and r["byte_length"] >= length
                ),
                start,
                stop,
            )
        }

    seen = take("calibration", 2049, 0, 32)  # X43 session signal
    seen |= take(
        "final_test", 129, 0, 64
    )  # K15 D/E/F: first full 128-byte window per doc
    seen |= take("calibration", 1025, 128, 192) | take(
        "train", 1025, 0, 64
    )  # recovery/reach G/H
    seen |= take("train", 2049, 0, 256)  # K15 H
    # K15 D/G: 4096 full 128-byte windows at the far end of the train region, not 4096 docs.
    remaining = 4096
    for r in records:
        if r["split"] != "train" or r["byte_offset"] < 101450000000:
            continue
        windows = (r["byte_length"] - 1) // 128
        if windows > 0:
            seen.add(r["sha256"])
            remaining -= min(windows, remaining)
        if not remaining:
            break
    check(remaining == 0, "incomplete D/G historical train region")
    return seen


def select(records, excluded, quotas, source_size):
    by_hash, ids, eligible = {}, set(), {s: [] for s in quotas}
    end = 0
    for r in records:
        i, start, n, h, s = [
            r[k] for k in ("document", "byte_offset", "byte_length", "sha256", "split")
        ]
        check(
            type(i) is int
            and type(start) is int
            and type(n) is int
            and i == start
            and i not in ids
            and 0 <= end <= start <= start + n <= source_size,
            "bad or repeated source identity/boundary",
        )
        ids.add(i)
        end = start + n
        check(
            type(h) is str and re.fullmatch("[0-9a-f]{64}", h) and s in N,
            "invalid split or digest",
        )
        check(
            s == {0: "calibration", 1: "final_test"}.get(int(h, 16) % 100, "train"),
            "document split conflict",
        )
        old = by_hash.get(h)
        check(old in (None, (s, n)), "conflicting duplicate content")
        by_hash[h] = s, n
        if s in eligible and start >= MIN_OFFSET and n > T and h not in excluded:
            rank = digest(("x49-live-v1:" + h).encode("ascii"))
            eligible[s].append((rank, h, i, r))
    chosen = {}
    counts = {}
    for s, values in eligible.items():
        found = set()
        selected = []
        for _, h, i, r in sorted(values, key=lambda v: v[:3]):
            if h in found:
                continue
            found.add(h)
            if len(selected) < quotas[s]:
                j = int(h[16:32], 16) % ((r["byte_length"] - 1) // T)
                selected.append(
                    dict(
                        doc=i,
                        offset=i,
                        length=r["byte_length"],
                        split=s,
                        sha256=h,
                        start=j * T,
                    )
                )
        counts[s] = len(found)
        chosen[s] = selected
    check(
        all(len(chosen[s]) == n for s, n in quotas.items()),
        f"insufficient clean documents: {counts}, requested {quotas}",
    )
    return chosen, counts


def reserve_data():
    regpath, manpath = FLAG / "registration.json", FLAG / "documents.jsonl.gz"
    check(sha(regpath) == REG_SHA and sha(manpath) == MAN_SHA, "wrong flagship stream")
    reg = json.loads(regpath.read_bytes())
    check(reg["manifest_sha256"] == MAN_SHA, "unbound manifest")
    with gzip.open(manpath, "rt") as f:
        rows = list(map(json.loads, f))
    check(
        dict(Counter(r["split"] for r in rows))
        == {s: v["documents"] for s, v in reg["splits"].items()},
        "flagship split size mismatch",
    )
    excluded, registry = prior_sets()
    extra = early_panels(rows)
    excluded.update(extra)
    panels, counts = select(rows, excluded, N, reg["source"]["size"])
    return dict(
        schema="x49-v1",
        manifest_sha=MAN_SHA,
        registration_sha=REG_SHA,
        source=reg["source"],
        exclusions=registry,
        early_membership_sha=fingerprint(sorted(extra)),
        excluded_unique=len(excluded),
        eligible=counts,
        panels=panels,
    )


def freeze(root=OUT):
    root.mkdir(exist_ok=True, parents=True, mode=0o700)
    p = root / "reservation.json"
    reservation = reserve_data()
    with p.open("x") as f:
        os.fchmod(f.fileno(), 0o600)
        json.dump(reservation, f, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    return dict(
        reservation_sha=sha(p),
        eligible=reservation["eligible"],
        reserved=N,
        excluded_unique=reservation["excluded_unique"],
        payloads_opened=0,
    )


def window(raw, row):
    check(
        row["split"] == "train"
        and len(raw) == row["length"]
        and row["start"] % T == 0
        and 0 <= row["start"] <= len(raw) - 1 - T,
        "not a registered full TRAIN window",
    )
    st = row["start"]
    receipt = dict(split="train", source_index=row["doc"], sha256=row["sha256"])
    w = next(
        islice(DocumentWindows(T).windows(raw, receipt, split="train"), st // T, None)
    )
    check(
        w.start == st
        and w.document == row["doc"]
        and w.inputs == raw[st : st + T]
        and w.targets == raw[st + 1 : st + T + 1],
        "wrong C14 shift",
    )
    return w.inputs + w.targets[-1:]


def order(rows, seed):
    check(seed in SEEDS, "unregistered seed")
    return sorted(
        range(len(rows)),
        key=lambda i: (
            digest(f"x49-order:{seed}:{rows[i]['sha256']}".encode("ascii")),
            rows[i]["sha256"],
        ),
    )


def extract_train(root=OUT):
    p = root / "reservation.json"
    reservation = json.loads(p.read_bytes())
    check(reservation == reserve_data(), "reservation mismatch before source open")
    final = root / "train"
    tmp = root / "train.preparing"
    check(not final.exists(), "TRAIN stream already exists")
    tmp.mkdir(mode=0o700)
    src = reservation["source"]
    rows = reservation["panels"]["train"]
    identity = []

    def verify(f):
        s = os.fstat(f.fileno())
        check(
            (s.st_ino, s.st_size, s.st_mtime_ns)
            == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "changed open corpus identity",
        )

    with (
        Path(src["path"]).open("rb", buffering=0) as f,
        (tmp / "windows.u8").open("xb") as out,
    ):
        verify(f)
        for r in rows:
            w = window(os.pread(f.fileno(), r["length"], r["offset"]), r)
            out.write(w)
            identity.append(
                dict(window=digest(w), inputs=digest(w[:-1]), targets=digest(w[1:]))
            )
        verify(f)
        out.flush()
        os.fsync(out.fileno())
    write_json(tmp / "hashes.json", identity)
    result = dict(
        schema="x49-train-v1",
        reservation_sha=sha(p),
        raw_sha=sha(tmp / "windows.u8"),
        shifts_sha=sha(tmp / "hashes.json"),
        rows=len(rows),
        targets=T * len(rows),
        source_bytes_read=sum(r["length"] for r in rows),
        order_sha={str(s): fingerprint(order(rows, s)) for s in SEEDS},
        evaluation_windows_read=0,
    )
    write_json(tmp / "identity.json", result)
    tmp.rename(final)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["reserve", "train"])
    a = ap.parse_args()
    check(
        os.getenv("CUDA_VISIBLE_DEVICES") == "" and not os.getenv("GPU_LEASE_HELD"),
        "CPU only",
    )
    os.umask(0o077)
    print(
        json.dumps(
            (freeze if a.phase == "reserve" else extract_train)(), sort_keys=True
        )
    )
