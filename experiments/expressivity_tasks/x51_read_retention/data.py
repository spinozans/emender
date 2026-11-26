"""Replay the full spent rank chain from metadata only, then extract each new role once."""

from pathlib import Path
import hashlib
import os
from time import monotonic

import numpy as np

from experiments.expressivity_tasks.x50_read_tape import data as parent
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MAN_SHA,
    MIN_OFFSET,
    check,
    sha,
    write_json,
)
from .reader import ARMS, T

HERE = Path(__file__).resolve().parent
ROOT = parent.ROOT
OUT = ROOT / "output/x51-read-retention"
RUN = OUT / "run-1"
SPEC = "2eb5d4b06c0adfbb8effeba033e3784fdd879c3f6be51fed518af5c3a48e94dd"
REG = "e3e8ebb1c8322bcec63d424875c94ce394d0f09a"
PARENT = "71cf07e3f8ba6e8c1693cd219fdc0ee25f2d7d09"
PARENT_RESULT = "5440696c48fb6647e9f1efd1d1f1e046b69d82582906895d10600e8ecdfae260"
PARENT_ROWS = "fc18de10c3010ab14c1e49c40cff6fe03b88e919a4af4d6160bf54a08f55bab3"
PARENT_CLOSE = "177fd64924fe363278ea21fea72711fcd171ebb604ba66c4cf13ccd38eef529b"
N = {"fit": 512, "probe": 1024}
SEEDS = (5101, 5102, 5103)
git, read = parent.git, parent.read
PREFIX = HERE.relative_to(ROOT).as_posix()


def source_files():
    return {f.name: sha(f) for f in sorted(HERE.glob("*.py"))}


def registration():
    parent.cpu()
    check(
        sha(HERE / "PROTOCOL.md") == SPEC
        and (OUT / "registered-ref").read_text().strip() == REG
        and git("rev-parse", REG + "^") == PARENT
        and git("rev-parse", REG + ":" + PREFIX + "/PROTOCOL.md")
        == git("hash-object", str(HERE / "PROTOCOL.md")),
        "registration changed",
    )
    for f in ("kernel.py", "program.py", "mount.py", "data.py", "study.py", "run.py"):
        p = parent.HERE / f
        check(
            git("rev-parse", PARENT + ":" + p.relative_to(ROOT).as_posix())
            == git("hash-object", str(p)),
            "foreign X50 dependency",
        )


def qualified():
    registration()
    q = read(OUT / "qualified.json")
    pub = (OUT / "qualified-ref").read_text().strip()
    check(
        q["spec"] == SPEC
        and q["registration"] == REG
        and q["sources"] == source_files()
        and q["natural_documents"] == 0
        and git("rev-parse", "HEAD") == q["local_ref"]
        and not git("status", "--porcelain", "--", PREFIX)
        and git("rev-parse", pub + "^") == REG
        and git("rev-parse", pub + ":" + PREFIX + "/receipts/qualification.json")
        == git("hash-object", str(OUT / "qualified.json")),
        "unqualified source or access before publication",
    )
    for name in q["sources"]:
        check(
            git("rev-parse", pub + ":" + PREFIX + "/" + name)
            == git("hash-object", str(HERE / name)),
            "public source mismatch",
        )
    return q


def next_block(records, base, spent, n, size):
    from experiments.expressivity_tasks.x49_live_trace_memory import panels

    hs = {r["sha256"] for r in spent}
    check(
        len(hs) == len(spent) and not hs & base, "duplicated or already blocked parent"
    )
    ex = base | hs
    q = {"train": n}
    chosen, supply = panels.select(records, ex, q, size)
    panels.verify_selection(records, ex, chosen, q, supply)
    both, _ = panels.select(records, base, {"train": len(spent) + n}, size)
    check(
        both["train"] == spent + chosen["train"],
        "not the contiguous suffix after both X50 roles",
    )
    return chosen["train"], ex, supply["train"]


def reserve():
    from experiments.expressivity_tasks.x49_live_trace_memory import (
        correction_transfer_data as pt,
        panels,
    )

    qualified()
    (RUN / "reservation-attempted").touch(exist_ok=False)
    pub = pt.u.bound(parent.HERE / "receipts/result.json", PARENT_RESULT)
    book = pt.u.bound(parent.RUN / "reservation.json", PARENT_ROWS)
    witness = read(parent.RUN / "witnessed.json")
    check(
        pub["private_close_sha256"] == witness["result_sha"] == PARENT_CLOSE
        and witness["receipt_only_recomputed"]
        and witness["owner_exited"]
        and read(parent.RUN / "reserved.json")["rows_sha"] == PARENT_ROWS
        and book["spec"] == parent.SPEC,
        "wrong or incomplete X50 parent",
    )
    prior, records, ex = pt.spent_context(pt.parent()["data"])
    short = pt.u.bound(pt.BASE / "reservation.json", parent.PT_ROWS)
    short_rows, _ = pt.checked_rows(short)
    check(
        len(ex) == 45217
        and panels.fingerprint(sorted(ex)) == short["excluded_sha"]
        and not ex & {r["sha256"] for r in short_rows},
        "wrong PT context",
    )
    ex = ex | {r["sha256"] for r in short_rows}
    check(
        len(ex) == book["excluded"] == 45414
        and panels.fingerprint(sorted(ex)) == book["excluded_sha"]
        and prior["source"] == short["source"] == book["source"]
        and prior["manifest_sha"]
        == short["manifest_sha"]
        == book["manifest"]
        == MAN_SHA
        and all(len(book[k]) == n for k, n in parent.N.items()),
        "wrong X50 exclusion or source",
    )
    rows, ex, supply = next_block(
        records,
        ex,
        book["fit"] + book["probe"],
        sum(N.values()),
        book["source"]["size"],
    )
    check(len(ex) == 46182, "wrong total excluded")
    z = dict(
        spec=SPEC,
        manifest=MAN_SHA,
        parent_rows=PARENT_ROWS,
        excluded=len(ex),
        excluded_sha=panels.fingerprint(sorted(ex)),
        supply=supply,
        source=book["source"],
        fit=rows[: N["fit"]],
        probe=rows[N["fit"] :],
    )
    write_json(RUN / "reservation.json", z)
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
        and z["states"].keys() == state_keys()
        and all(sha(RUN / (k + ".pt")) == v for k, v in z["states"].items())
        and z["qualification_sha"] == sha(OUT / "qualified.json")
        and z["rows_sha"] == sha(RUN / "reservation.json")
        and z["fit_sha"] == sha(RUN / "fit.json")
        and z["prior_sha"] == sha(RUN / "unigram.json"),
        "incomplete or changed fit seal",
    )
    return z


def extract(role):
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import window

    qualified()
    if role not in N:
        raise ValueError("only the registered TRAIN roles are available")
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
        check(not (RUN / "frozen.json").exists(), "FIT closed")
    book = read(RUN / "reservation.json")
    m = read(RUN / "reserved.json")
    check(
        book["spec"] == m["spec"] == SPEC
        and m["rows_sha"] == sha(RUN / "reservation.json")
        and m["roles"] == N
        and book["manifest"] == MAN_SHA
        and book["excluded"] == 46182
        and all(len(book[k]) == n for k, n in N.items())
        and len({v["sha256"] for k in N for v in book[k]}) == sum(N.values()),
        "changed reservation",
    )
    tic = monotonic()
    (RUN / (role + "-source-open-attempted")).touch(exist_ok=False)
    src = book["source"]
    chunks = []
    total = 0

    def verify(fd):
        st = os.fstat(fd)
        check(
            (st.st_ino, st.st_size, st.st_mtime_ns)
            == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "wrong raw source",
        )

    with open(src["path"], "rb", buffering=0) as f:
        verify(f.fileno())
        for row in book[role]:
            check(
                row["split"] == "train"
                and int(row["sha256"], 16) % 100 >= 2
                and row["doc"] == row["offset"] >= MIN_OFFSET,
                "wrong split or in-archive target",
            )
            raw = os.pread(f.fileno(), row["length"], row["offset"])
            w = window(raw, row)[: T + 1]
            total += len(raw)
            check(
                len(w) == T + 1 and w == raw[row["start"] : row["start"] + T + 1],
                "C14 shift mismatch",
            )
            chunks.append(w)
        verify(f.fileno())
    payload = b"".join(chunks)
    write_json(
        RUN / (role + ".json"),
        dict(
            role=role,
            documents=len(chunks),
            targets=T * len(chunks),
            rows_sha=m["rows_sha"],
            source_bytes=total,
            shifts_sha=hashlib.sha256(payload).hexdigest(),
            wall_s=monotonic() - tic,
            fit_seal=sha(RUN / "frozen.json") if role == "probe" else None,
        ),
    )
    return np.frombuffer(payload, np.uint8).reshape(-1, T + 1)
