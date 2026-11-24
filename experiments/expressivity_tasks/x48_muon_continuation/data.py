"""One private, hash-checked, document-bounded TRAIN stream. No evaluation split loader."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from itertools import islice
from pathlib import Path
from time import monotonic

from ..x43_fork_resolved.c14_documents import DocumentWindows

ROOT = Path(__file__).resolve().parents[3]
X47 = ROOT / "experiments/expressivity_tasks/x47_xsa_attention"
FLAG = X47.parent / "x43_fork_resolved/receipts/flagship_stream"
OUT = ROOT / "output/x48-muon"
PANEL = X47 / "receipts/w3/panels.json"
REG_SHA = "c36f84cfd1a59530a7991a4be6c1f68b424a79c5b7de169eabfc90d3cd31cc31"
MAN_SHA = "48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414"
OLD_SHA = "60144b7ccb2ec1cc2a08c3db9c75a54eaa0c5ab52429288089098e9f83760c38"
B, T, STEPS = 32, 128, 1024
MIN_OFFSET = 100_600_000_000


def check(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load(p):
    return json.loads(p.read_text())


def write_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, allow_nan=False, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def doc_hashes(obj):
    if isinstance(obj, list):
        for x in obj:
            yield from doc_hashes(x)
    elif isinstance(obj, dict):
        if "sha256" in obj:
            yield obj["sha256"]
        for v in obj.values():
            yield from doc_hashes(v)


def select(records, excluded, count):
    seen = set(excluded)
    blocked = set()
    rows = []
    for d in records:
        h, n, off = d["sha256"], d["byte_length"], d["byte_offset"]
        check(d["split"] in ("train", "calibration", "final_test"), "unknown split")
        if d["split"] != "train":
            blocked.add(h)  # metadata only; exclude even a malformed conflicting duplicate
            continue
        if off < MIN_OFFSET or n <= T or h in seen:
            continue
        seen.add(h)
        j = int(h[16:32], 16) % ((n - 1) // T)
        rank = hashlib.sha256(("x48-muon-v1:" + h).encode()).digest()
        rows.append(
            (rank, dict(doc=d["document"], offset=off, length=n, split="train", sha256=h, start=j * T))
        )
    chosen = list(
        islice((r for _, r in sorted(rows, key=lambda x: x[0]) if r["sha256"] not in blocked), count)
    )
    check(len(chosen) == count == len({r["doc"] for r in chosen}), "insufficient distinct TRAIN documents")
    return chosen


def cut(raw, row):
    start = row["start"]
    check(
        row["split"] == "train" and 0 <= start < start + T < len(raw) == row["length"] and start % T == 0,
        "invalid TRAIN window",
    )
    it = DocumentWindows(T).windows(
        raw, dict(sha256=row["sha256"], source_index=row["doc"], split="train"), split="train"
    )
    w = next(islice(it, start // T, None))
    check(
        w.document == row["doc"]
        and w.start == start
        and w.inputs == raw[start : start + T]
        and w.targets == raw[start + 1 : start + T + 1],
        "C14 shift or boundary mismatch",
    )
    return w.inputs + w.targets[-1:]


def prepare(root=OUT):
    os.umask(0o077)
    t0 = monotonic()
    root.mkdir(exist_ok=True, mode=0o700)
    work = root / "data.preparing"
    check(not (root / "data").exists(), "data already frozen")
    work.mkdir(mode=0o700)
    regp, mp = FLAG / "registration.json", FLAG / "documents.jsonl.gz"
    check(
        sha(regp) == REG_SHA and sha(mp) == MAN_SHA and sha(PANEL) == OLD_SHA,
        "unregistered corpus or panel metadata",
    )
    reg = load(regp)
    old = load(PANEL)
    src = reg["source"]
    check(src == old["source"] and reg["manifest_sha256"] == old["manifest_sha"] == MAN_SHA, "unbound source")
    excluded = set(doc_hashes(old["panels"]))
    for p, h in old["exclusions"].items():
        p = ROOT / Path(p)
        check(sha(p) == h, "changed historical exclusion")
        excluded.update(doc_hashes(load(p)))
    with gzip.open(mp, "rt") as f:
        rows = select(map(json.loads, f), excluded, B * STEPS)

    def source_ok(f):
        s = os.fstat(f.fileno())
        check(
            (s.st_ino, s.st_size, s.st_mtime_ns) == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "changed source fd",
        )

    with (
        Path(src["path"]).open("rb", buffering=0) as f,
        (work / "windows.u8").open("xb") as dst,
        (work / "rows.jsonl").open("x") as out,
    ):
        source_ok(f)
        for r in rows:
            window = cut(os.pread(f.fileno(), r["length"], r["offset"]), r)
            dst.write(window)
            r["window_sha"] = hashlib.sha256(window).hexdigest()
            out.write(json.dumps(r, sort_keys=True) + "\n")
        source_ok(f)
    ident = dict(
        schema="x48-train-v1",
        registration=REG_SHA,
        manifest=MAN_SHA,
        exclusions_parent=OLD_SHA,
        rows_sha=sha(work / "rows.jsonl"),
        windows_sha=sha(work / "windows.u8"),
        documents=len(rows),
        context=T,
        targets=B * T * STEPS,
        raw_bytes=len(rows) * (T + 1),
        cross_document_targets=0,
        source=src,
        seconds=monotonic() - t0,
    )
    write_json(work / "identity.json", ident)
    work.rename(root / "data")
    print(json.dumps(ident, sort_keys=True), flush=True)


def stream_id(identity):
    # Elapsed extraction time is evidence, not part of the data. An independently
    # rebuilt byte-identical stream must be loadable without forging its clock.
    fields = {k: v for k, v in identity.items() if k != "seconds"}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Stream:
    def __init__(self, root, expected):
        d = root / "data"
        x = load(d / "identity.json")
        check(stream_id(x) == expected, "changed prepared stream")
        check(
            x["schema"] == "x48-train-v1"
            and x["documents"] == B * STEPS
            and x["manifest"] == MAN_SHA
            and x["registration"] == REG_SHA
            and x["exclusions_parent"] == OLD_SHA,
            "wrong training manifest",
        )
        self.rows = list(map(json.loads, (d / "rows.jsonl").read_text().splitlines()))
        self.raw = (d / "windows.u8").read_bytes()
        check(
            sha(d / "rows.jsonl") == x["rows_sha"]
            and hashlib.sha256(self.raw).hexdigest() == x["windows_sha"]
            and len(self.raw) == B * STEPS * (T + 1)
            and len(self.rows) == B * STEPS,
            "changed batch payload",
        )

    def batch(self, step):
        check(type(step) is int and 1 <= step <= STEPS, "no wraparound")
        i = (step - 1) * B
        rows = self.rows[i : i + B]
        words = [self.raw[j * (T + 1) : (j + 1) * (T + 1)] for j in range(i, i + B)]
        check(
            all(
                r["split"] == "train"
                and r["offset"] >= MIN_OFFSET
                and r["start"] + T < r["length"]
                and hashlib.sha256(w).hexdigest() == r["window_sha"]
                for r, w in zip(rows, words, strict=True)
            ),
            "unbound TRAIN batch",
        )
        return rows, words


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
