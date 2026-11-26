"""Single-use, post-X51 TRAIN rank block. Only metadata from closed studies is used."""

import hashlib
import os
from pathlib import Path
from time import monotonic

import numpy as np

from experiments.expressivity_tasks.x51_read_retention import data as last
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    check,
    sha,
    write_json,
    MAN_SHA,
    MIN_OFFSET,
)
from .reader import ARMS, T

HERE = Path(__file__).resolve().parent
ROOT = last.ROOT
PREFIX = HERE.relative_to(ROOT).as_posix()
OUT = ROOT / "output/x52-sparse-read"
RUN = OUT / "run-1"
SPEC = "8fe5f12899b8260cae1fa154b9e43dfe6cf61dbcafc24ff8b3290f6fd50d3766"
REG = "f38e2500a18b7677e926a063098f50ecf419a991"
PARENT = "d76607369456e3f99dd956239411c0eb3b82b6fc"
PREV_PUBLIC = "bb8ea0b7fb85ecdccb0cdc58e04e3a04e164fc22c82af3c4abe1b92ae39bb4b0"
PREV_ROWS = "e8507b0ddf6bf5c4ffc7bdf96dc8b058e62758f6a38b7cde6f92cd19a737ace3"
PREV_CLOSE = "0ee2aefde3c383437d8970a3f29e2eb460ac39a21788931cbb2b380ce360263c"
N = {"fit": 512, "probe": 1024}
SEEDS = (5201, 5202, 5203)
EXCLUDED = 47718
git, read, cpu = last.git, last.read, last.parent.cpu


def sources():
    return {p.name: sha(p) for p in sorted(HERE.glob("*.py"))}


def registration():
    cpu()
    check(
        sha(HERE / "PROTOCOL.md") == SPEC
        and (OUT / "registered-ref").read_text().strip() == REG
        and git("rev-parse", REG + "^") == PARENT
        and git("rev-parse", REG + ":" + PREFIX + "/PROTOCOL.md")
        == git("hash-object", str(HERE / "PROTOCOL.md")),
        "wrong registration",
    )
    deps = [
        p.relative_to(ROOT).as_posix()
        for folder in (last.HERE, last.parent.HERE)
        for p in folder.glob("*.py")
    ]
    check(not git("diff", "--name-only", PARENT, "--", *deps), "changed parent code")


def qualified():
    registration()
    q = read(OUT / "qualified.json")
    pub = (OUT / "qualified-ref").read_text().strip()
    check(
        q["spec"] == SPEC
        and q["registration"] == REG
        and q["sources"] == sources()
        and q["natural_documents"] == q["gpu_acquisitions"] == 0
        and git("rev-parse", "HEAD") == q["local_ref"]
        and not git("status", "--porcelain", "--", PREFIX)
        and git("rev-parse", pub + "^") == REG
        and git("rev-parse", pub + ":" + PREFIX + "/receipts/qualification.json")
        == git("hash-object", str(OUT / "qualified.json")),
        "missing public qualification",
    )
    for name in q["sources"]:
        check(
            git("rev-parse", pub + ":" + PREFIX + "/" + name)
            == git("hash-object", str(HERE / name)),
            "qualified file mismatch",
        )
    return q


def reserve():
    from experiments.expressivity_tasks.x49_live_trace_memory import (
        correction_transfer_data as pt,
        panels,
    )

    qualified()
    (RUN / "reservation-attempted").touch(exist_ok=False)
    books = []
    for module, rsha, psha, csha in [
        (last.parent, last.PARENT_ROWS, last.PARENT_RESULT, last.PARENT_CLOSE),
        (last, PREV_ROWS, PREV_PUBLIC, PREV_CLOSE),
    ]:
        pub = pt.u.bound(module.HERE / "receipts/result.json", psha)
        book = pt.u.bound(module.RUN / "reservation.json", rsha)
        w = read(module.RUN / "witnessed.json")
        check(
            pub["private_close_sha256"] == w["result_sha"] == csha
            and w["receipt_only_recomputed"]
            and w["owner_exited"]
            and book["spec"] == module.SPEC
            and read(module.RUN / "reserved.json")["rows_sha"] == rsha
            and all(len(book[k]) == n for k, n in module.N.items()),
            "wrong closed parent",
        )
        books.append(book)
    b50, b51 = books
    prior, records, pre = pt.spent_context(pt.parent()["data"])
    ptbook = pt.u.bound(pt.BASE / "reservation.json", last.parent.PT_ROWS)
    rows, _ = pt.checked_rows(ptbook)
    pt_hs = {r["sha256"] for r in rows}
    check(
        len(pre) == 45217
        and len(pt_hs) == 197
        and not pre & pt_hs
        and panels.fingerprint(sorted(pre)) == ptbook["excluded_sha"],
        "wrong PT exclusions",
    )
    base = pre | pt_hs
    spent50 = b50["fit"] + b50["probe"]
    spent51 = b51["fit"] + b51["probe"]
    pre51 = base | {r["sha256"] for r in spent50}
    check(
        len(base) == b50["excluded"] == 45414
        and len(pre51) == b51["excluded"] == 46182
        and panels.fingerprint(sorted(base)) == b50["excluded_sha"]
        and panels.fingerprint(sorted(pre51)) == b51["excluded_sha"]
        and b51["parent_rows"] == last.PARENT_ROWS
        and prior["source"] == ptbook["source"] == b50["source"] == b51["source"]
        and prior["manifest_sha"]
        == ptbook["manifest_sha"]
        == b50["manifest"]
        == b51["manifest"]
        == MAN_SHA,
        "changed inherited source or exclusions",
    )
    fresh, ex, supply = last.next_block(
        records, base, spent50 + spent51, sum(N.values()), b51["source"]["size"]
    )
    check(len(ex) == EXCLUDED, "wrong complete exclusion count")
    write_json(
        RUN / "reservation.json",
        dict(
            spec=SPEC,
            parent=PREV_ROWS,
            manifest=MAN_SHA,
            source=b51["source"],
            supply=supply,
            excluded=len(ex),
            excluded_sha=panels.fingerprint(sorted(ex)),
            fit=fresh[: N["fit"]],
            probe=fresh[N["fit"] :],
        ),
    )
    write_json(
        RUN / "reserved.json",
        dict(
            spec=SPEC,
            rows_sha=sha(RUN / "reservation.json"),
            roles=N,
            payload_documents=0,
        ),
    )


def state_keys():
    return {f"{s}-{a}" for s in SEEDS for a in ARMS}


def frozen():
    z = read(RUN / "frozen.json")
    check(
        z["spec"] == SPEC
        and set(z["states"]) == state_keys()
        and all(sha(RUN / (k + ".pt")) == v for k, v in z["states"].items())
        and z["qualification_sha"] == sha(OUT / "qualified.json")
        and z["rows_sha"] == sha(RUN / "reservation.json")
        and z["fit_sha"] == sha(RUN / "fit.json")
        and z["prior_sha"] == sha(RUN / "unigram.json"),
        "changed or incomplete fit freeze",
    )
    return z


def extract(role):
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import window

    qualified()
    check(role in N, "unknown data role")
    if role == "probe":
        z = frozen()
        ack = read(RUN / "freeze-ack.json")
        check(
            ack["seal"] == sha(RUN / "frozen.json")
            and ack["targets"] == z["fit_targets"]
            and ack["owner"] == read(RUN / "owner.json"),
            "unwitnessed freeze",
        )
    else:
        check(not (RUN / "frozen.json").exists(), "fit closed")
    b = read(RUN / "reservation.json")
    r = read(RUN / "reserved.json")
    check(
        b["spec"] == r["spec"] == SPEC
        and b["manifest"] == MAN_SHA
        and b["parent"] == PREV_ROWS
        and b["excluded"] == EXCLUDED
        and r["roles"] == N
        and r["rows_sha"] == sha(RUN / "reservation.json")
        and all(len(b[k]) == n for k, n in N.items())
        and len({v["sha256"] for k in N for v in b[k]}) == sum(N.values()),
        "changed reservation",
    )
    (RUN / (role + "-source-open-attempted")).touch(exist_ok=False)
    tic = monotonic()
    src = b["source"]
    chunks = []
    size = 0

    def verify(fd):
        st = os.fstat(fd)
        check(
            (st.st_ino, st.st_size, st.st_mtime_ns)
            == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "wrong source identity",
        )

    with open(src["path"], "rb", buffering=0) as f:
        verify(f.fileno())
        for row in b[role]:
            check(
                row["split"] == "train"
                and int(row["sha256"], 16) % 100 >= 2
                and row["doc"] == row["offset"] >= MIN_OFFSET,
                "wrong split or offset",
            )
            raw = os.pread(f.fileno(), row["length"], row["offset"])
            w = window(raw, row)[: T + 1]
            size += len(raw)
            check(
                len(w) == T + 1 and w == raw[row["start"] : row["start"] + T + 1],
                "wrong C14 shift",
            )
            chunks.append(w)
        verify(f.fileno())
    payload = b"".join(chunks)
    write_json(
        RUN / (role + ".json"),
        dict(
            role=role,
            documents=len(chunks),
            targets=len(chunks) * T,
            source_bytes=size,
            rows_sha=r["rows_sha"],
            shifts_sha=hashlib.sha256(payload).hexdigest(),
            wall_s=monotonic() - tic,
            fit_seal=sha(RUN / "frozen.json") if role == "probe" else None,
        ),
    )
    return np.frombuffer(payload, np.uint8).reshape(-1, T + 1)
