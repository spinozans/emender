"""One independent TRAIN-only reservation; seal all FIT states before PROBE payload."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np

from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MAN_SHA,
    MIN_OFFSET,
    check,
    sha,
    write_json,
)
from .program import ARMS, T

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "output/x50-read-tape"
RUN = OUT / "run-1"
SPEC = "506f13a56b308bfc00186e2a22e1c52d0bf19a056195fd4e43746f015a97e3d6"
REG = "815feef90f88e1536b3c3856fce520455318b48e"
SEEDS = (5001, 5002, 5003)
N = {"fit": 512, "probe": 256}
PT_PUBLIC = "25dbd6ee5dc29b71a770b040e6e661bdf28dcb5503cf7d734b67553ae5e52fc6"
PT_ROWS = "b6808bc4419a5653ed6be6996dab7ff820c5ee79fd8eb9272aba91faaed69fed"
ROW_PREFIX = "experiments/expressivity_tasks/x50_read_tape"


def cpu():
    check(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD")
        and os.getenv("OPENBLAS_NUM_THREADS") == "1"
        and not sys.flags.optimize,
        "unqualified CPU environment",
    )


def read(path):
    return json.loads(path.read_bytes())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def sources():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def registration():
    cpu()
    check(
        sha(HERE / "PROTOCOL.md") == SPEC
        and (OUT / "registered-ref").read_text().strip() == REG
        and git("rev-parse", REG + ":" + ROW_PREFIX + "/PROTOCOL.md")
        == git("hash-object", str(HERE / "PROTOCOL.md")),
        "wrong registration or protocol",
    )


def qualified():
    registration()
    q = read(OUT / "qualified.json")
    pub = (OUT / "qualified-ref").read_text().strip()
    receipt = ROW_PREFIX + "/receipts/qualification.json"
    check(
        q["sources"] == sources()
        and q["registration"] == REG
        and q["spec"] == SPEC
        and q["fit_documents"] == q["probe_documents"] == q["gpu_acquisitions"] == 0
        and git("rev-parse", "HEAD") == q["local_ref"]
        and git("rev-parse", pub + "^") == REG
        and git("rev-parse", pub + ":" + receipt)
        == git("hash-object", str(OUT / "qualified.json"))
        and not git("status", "--porcelain", "--", ROW_PREFIX),
        "missing or stale public qualification",
    )
    for name in q["sources"]:
        check(
            git("rev-parse", pub + ":" + ROW_PREFIX + "/" + name)
            == git("hash-object", str(HERE / name)),
            "public source mismatch",
        )
    return q


def select():
    # Historical selectors rederive receipts and reservations, not payloads.
    from experiments.expressivity_tasks.x49_live_trace_memory import (
        correction_transfer_data as pt,
        panels,
    )

    parent = pt.u.bound(
        pt.u.HERE / "receipts/correction_transfer_result.json", PT_PUBLIC
    )
    ptbook = pt.u.bound(pt.BASE / "reservation.json", PT_ROWS)
    prows, nt = pt.checked_rows(ptbook)
    prior, records, excluded = pt.spent_context(pt.parent()["data"])
    hs = {r["sha256"] for r in prows}
    check(
        parent["data"]["rows_sha"] == PT_ROWS
        and parent["metadata_reservation"]["unreserved"] == 0
        and prior["source"] == ptbook["source"]
        and prior["manifest_sha"] == ptbook["manifest_sha"] == MAN_SHA
        and len(excluded) == 45217
        and ptbook["excluded_sha"] == panels.fingerprint(sorted(excluded))
        and len(hs) == 197
        and nt == 285520
        and not hs & excluded,
        "wrong post-PT exclusion set",
    )
    ex = excluded | hs
    quota = {"train": sum(N.values())}
    chosen, supply = panels.select(records, ex, quota, prior["source"]["size"])
    panels.verify_selection(records, ex, chosen, quota, supply)
    rows = chosen["train"]
    return dict(
        spec=SPEC,
        manifest=MAN_SHA,
        excluded=len(ex),
        excluded_sha=panels.fingerprint(sorted(ex)),
        source=prior["source"],
        supply=supply["train"],
        fit=rows[: N["fit"]],
        probe=rows[N["fit"] :],
    )


def reserve():
    qualified()
    (RUN / "reservation-attempted").touch(exist_ok=False)
    book = select()
    write_json(RUN / "reservation.json", book)
    write_json(
        RUN / "reserved.json",
        dict(
            rows_sha=sha(RUN / "reservation.json"),
            spec=SPEC,
            role_sizes=N,
            excluded=book["excluded"],
            payload_documents=0,
        ),
    )


def state_keys():
    return {f"{seed}-{arm}" for seed in SEEDS for arm in ARMS}


def frozen():
    z = read(RUN / "frozen.json")
    check(
        z["spec"] == SPEC
        and z["rows_sha"] == sha(RUN / "reservation.json")
        and z["fit_identity_sha"] == sha(RUN / "fit.json")
        and z["qualification_sha"] == sha(OUT / "qualified.json")
        and set(z["states"]) == state_keys()
        and all(z["states"][key] == sha(RUN / (key + ".pt")) for key in state_keys()),
        "incomplete or changed FIT seal",
    )
    return z


def extract(role):
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import window

    qualified()
    if role not in N:
        raise ValueError("only reserved TRAIN fit/probe")
    if role == "probe":
        frozen()
    else:
        check(not (RUN / "frozen.json").exists(), "FIT already frozen")
    b = read(RUN / "reservation.json")
    m = read(RUN / "reserved.json")
    check(
        m["rows_sha"] == sha(RUN / "reservation.json")
        and m["spec"] == b["spec"] == SPEC
        and b["manifest"] == MAN_SHA
        and b["excluded"] == 45414
        and all(len(b[k]) == n for k, n in N.items())
        and len({r["sha256"] for k in N for r in b[k]}) == sum(N.values()),
        "changed metadata reservation",
    )
    (RUN / f"{role}-source-open-attempted").touch(exist_ok=False)
    src = b["source"]

    def verify(fd):
        st = os.fstat(fd)
        check(
            (st.st_ino, st.st_size, st.st_mtime_ns)
            == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "foreign source",
        )

    tic = monotonic()
    chunks = []
    nraw = 0
    with open(src["path"], "rb", buffering=0) as f:
        verify(f.fileno())
        for row in b[role]:
            check(
                row["split"] == "train"
                and int(row["sha256"], 16) % 100 >= 2
                and row["doc"] == row["offset"] >= MIN_OFFSET,
                "non-TRAIN or in-archive document",
            )
            raw = os.pread(f.fileno(), row["length"], row["offset"])
            nraw += len(raw)
            w = window(raw, row)[
                : T + 1
            ]  # whole-document SHA and C14 shift checked by window()
            check(
                len(w) == T + 1 and w == raw[row["start"] : row["start"] + T + 1],
                "crossed document or wrong native shift",
            )
            chunks.append(w)
        verify(f.fileno())
    payload = b"".join(chunks)
    ident = dict(
        role=role,
        documents=len(chunks),
        real_targets=T * len(chunks),
        source_bytes=nraw,
        window_stream_sha=hashlib.sha256(payload).hexdigest(),
        rows_sha=m["rows_sha"],
        wall_s=monotonic() - tic,
        fit_seal_sha=sha(RUN / "frozen.json") if role == "probe" else None,
    )
    write_json(RUN / (role + ".json"), ident)
    return np.frombuffer(payload, np.uint8).reshape(len(chunks), T + 1)
