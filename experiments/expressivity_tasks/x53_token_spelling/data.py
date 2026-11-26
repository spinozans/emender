"""Post-X52 hash-disjoint selection and one opening of each new role."""

import hashlib
import os
from pathlib import Path
from time import monotonic

from experiments.expressivity_tasks.x52_sparse_read import data as p52
from experiments.expressivity_tasks.x48_muon_continuation.data import (
    MAN_SHA,
    MIN_OFFSET,
    check,
    sha,
    write_json,
)

HERE = Path(__file__).resolve().parent
ROOT = p52.ROOT
PREFIX = HERE.relative_to(ROOT).as_posix()
OUT = ROOT / "output/x53-token-spelling"
RUN = OUT / "run-1"
SPEC = "87ced744b7b586927b1bfd1061b0da8d73d6706d66bff781c407b8a93a6092c9"
REG = "3285f49c707359dabeb1cd80a3439515091eab5d"
PARENT = "891517bf953182d4385a4813059ff0002dbdae3e"
PREV_ROWS = "abba72b4d6b780c0c5c5a1a9db94a59aeb9c6911c1823a9689c8880378fad606"
PREV_CLOSE = "940d6c2f21ac838b9622b73b79a25f559ad07216cae2e3e8576983129478feee"
PREV_PUBLIC = "0fec043d5454240b4e9d2fc8d23da60628f38043a75d291353637826e9114bc4"
N = {"build": 512, "probe": 1024}
LENGTH = {"build": 4097, "probe": 257}
EXCLUDED = 49254
git, read = p52.git, p52.read


def cpu():
    p52.cpu()
    check(
        all(os.getenv(k) == "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")),
        "single-thread environment required",
    )


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
        (HERE.parent / "token_suffix_fanout" / "index.py"),
        p52.HERE / "data.py",
        p52.last.HERE / "data.py",
        p52.last.parent.HERE / "data.py",
    ]
    check(
        not git(
            "diff",
            "--name-only",
            PARENT,
            "--",
            *[p.relative_to(ROOT).as_posix() for p in deps],
        ),
        "changed dependency",
    )


def qualified():
    registration()
    q = read(OUT / "qualified.json")
    pub = (OUT / "qualified-ref").read_text().strip()
    check(
        q["spec"] == SPEC
        and q["registration"] == REG
        and q["natural_documents"] == q["gpu_acquisitions"] == 0
        and q["sources"] == sources()
        and q["local_ref"] == git("rev-parse", "HEAD")
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
            "different public source",
        )
    return q


def reserve():
    from experiments.expressivity_tasks.x49_live_trace_memory import (
        correction_transfer_data as pt,
        panels,
    )

    qualified()
    (RUN / "reservation-attempted").touch(exist_ok=False)
    prev, records, ex = pt.spent_context(pt.parent()["data"])
    p51 = p52.last
    p50 = p51.parent
    short = pt.u.bound(pt.BASE / "reservation.json", p50.PT_ROWS)
    shorts, _ = pt.checked_rows(short)
    extra = {z["sha256"] for z in shorts}
    check(
        len(extra) == 197
        and len(ex) == 45217
        and panels.fingerprint(sorted(ex)) == short["excluded_sha"]
        and prev["source"] == short["source"]
        and prev["manifest_sha"] == short["manifest_sha"] == MAN_SHA
        and not ex & extra,
        "wrong PT context",
    )
    base = ex | extra
    ex = base.copy()
    spent = []
    for mod, rs, ps, cs, before in [
        (p50, p51.PARENT_ROWS, p51.PARENT_RESULT, p51.PARENT_CLOSE, 45414),
        (p51, p52.PREV_ROWS, p52.PREV_PUBLIC, p52.PREV_CLOSE, 46182),
        (p52, PREV_ROWS, PREV_PUBLIC, PREV_CLOSE, 47718),
    ]:
        book = pt.u.bound(mod.RUN / "reservation.json", rs)
        pub = pt.u.bound(mod.HERE / "receipts/result.json", ps)
        w = read(mod.RUN / "witnessed.json")
        rows = book["fit"] + book["probe"]
        hs = {z["sha256"] for z in rows}
        check(
            w["owner_exited"]
            and w["receipt_only_recomputed"]
            and w["result_sha"] == pub["private_close_sha256"] == cs
            and read(mod.RUN / "reserved.json")["rows_sha"] == rs
            and book["spec"] == mod.SPEC
            and book["source"] == prev["source"]
            and book["manifest"] == MAN_SHA
            and len(ex) == book["excluded"] == before
            and book["excluded_sha"] == panels.fingerprint(sorted(ex))
            and len(hs) == len(rows)
            and not ex & hs
            and all(len(book[k]) == n for k, n in mod.N.items()),
            "wrong predecessor or spent cohort",
        )
        ex |= hs
        spent.extend(rows)
    rows, blocked, supply = p51.next_block(
        records, base, spent, sum(N.values()), prev["source"]["size"]
    )
    check(ex == blocked and len(ex) == EXCLUDED, "post-X52 exclusions mismatch")
    write_json(
        RUN / "reservation.json",
        dict(
            spec=SPEC,
            parent=PREV_ROWS,
            manifest=MAN_SHA,
            excluded=len(ex),
            excluded_sha=panels.fingerprint(sorted(ex)),
            source=prev["source"],
            supply=supply,
            build=rows[: N["build"]],
            probe=rows[N["build"] :],
        ),
    )
    write_json(
        RUN / "reserved.json",
        dict(
            rows_sha=sha(RUN / "reservation.json"),
            spec=SPEC,
            roles=N,
            payload_documents=0,
        ),
    )


def seal():
    f = read(RUN / "frozen.json")
    check(
        f["spec"] == SPEC
        and f["qualification_sha"] == sha(OUT / "qualified.json")
        and f["rows_sha"] == sha(RUN / "reservation.json")
        and f["image_sha"] == sha(RUN / "atlas.npz")
        and f["build_sha"] == sha(RUN / "build.json")
        and f["geometry"]["documents"] == N["build"]
        and f["geometry"]["byte_targets"] == N["build"] * (LENGTH["build"] - 1),
        "invalid frozen image",
    )
    return f


def windows(role):
    from experiments.expressivity_tasks.x49_live_trace_memory.panels import window

    qualified()
    check(role in N, "unexpected split")
    if role == "probe":
        seal()
        ack = read(RUN / "freeze-ack.json")
        check(
            ack["sha"] == sha(RUN / "frozen.json")
            and ack["owner"] == read(RUN / "owner.json"),
            "unwitnessed freeze",
        )
    else:
        check(not (RUN / "frozen.json").exists(), "BUILD closed")
    b = read(RUN / "reservation.json")
    r = read(RUN / "reserved.json")
    check(
        b["spec"] == r["spec"] == SPEC
        and b["manifest"] == MAN_SHA
        and b["excluded"] == EXCLUDED
        and b["parent"] == PREV_ROWS
        and r["rows_sha"] == sha(RUN / "reservation.json")
        and r["roles"] == N
        and all(len(b[k]) == n for k, n in N.items())
        and len({row["sha256"] for k in N for row in b[k]}) == sum(N.values()),
        "wrong allocation",
    )
    (RUN / (role + "-source-open-attempted")).touch(exist_ok=False)
    src = b["source"]
    digest = hashlib.sha256()
    size = elapsed = 0

    def identity(f):
        st = os.fstat(f.fileno())
        check(
            (st.st_ino, st.st_size, st.st_mtime_ns)
            == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "wrong raw source",
        )

    with open(src["path"], "rb", buffering=0) as f:
        identity(f)
        for row in b[role]:
            start = monotonic()
            check(
                row["split"] == "train"
                and int(row["sha256"], 16) % 100 >= 2
                and row["doc"] == row["offset"] >= MIN_OFFSET,
                "invalid TRAIN source",
            )
            raw = os.pread(f.fileno(), row["length"], row["offset"])
            v = window(raw, row)[: LENGTH[role]]
            check(
                len(v) == LENGTH[role]
                and raw[row["start"] : row["start"] + len(v)] == v,
                "C14 window mismatch",
            )
            size += len(raw)
            digest.update(v)
            elapsed += monotonic() - start
            yield v
        identity(f)
    write_json(
        RUN / (role + ".json"),
        dict(
            spec=SPEC,
            rows_sha=r["rows_sha"],
            role=role,
            windows=N[role],
            length=LENGTH[role],
            targets=N[role] * (LENGTH[role] - 1),
            source_bytes=size,
            window_bytes_sha=digest.hexdigest(),
            read_s=elapsed,
            fit_seal=sha(RUN / "frozen.json") if role == "probe" else None,
        ),
    )
