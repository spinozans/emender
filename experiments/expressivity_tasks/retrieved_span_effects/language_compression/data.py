"""Fresh checked raw-byte excerpts; membership and sample arrays remain private."""

import hashlib
import os
import sys
import time

import numpy as np

from .settings import DATA, JOB, N, OLD, MAN, sha, cpu, get, put, need, qualified


def reserve():
    from ...x49_live_trace_memory import correction_transfer_data as pt, panels
    from ...x53_token_spelling import data as s53
    from ..natural_data import DATA as PREVIOUS

    ref = qualified()
    cpu()
    DATA.mkdir(mode=0o700)
    s52 = s53.p52
    s51 = s52.last
    s50 = s51.parent
    book, records, ex = pt.spent_context(pt.parent()["data"])
    short = pt.u.bound(pt.BASE / "reservation.json", s50.PT_ROWS)
    short_rows, _ = pt.checked_rows(short)
    base = ex | {r["sha256"] for r in short_rows}
    need(len(base) == 45414, "incorrect spent base")
    spent = []
    for mod, h, key in [
        (s50, s51.PARENT_ROWS, "fit"),
        (s51, s52.PREV_ROWS, "fit"),
        (s52, s53.PREV_ROWS, "fit"),
        (
            s53,
            "3aa0469581837dd6e31172a59cd4130685af0b5138be72614e770f496226b130",
            "build",
        ),
    ]:
        r = pt.u.bound(mod.RUN / "reservation.json", h)
        spent += r[key] + r["probe"]
    prev = pt.u.bound(PREVIOUS / "reservation.json", OLD)
    ex = base | {r["sha256"] for r in spent}
    need(
        len(ex) == prev["excluded"] == 50790
        and panels.fingerprint(sorted(ex)) == prev["excluded_sha"]
        and prev["manifest"] == MAN
        and prev["source"] == book["source"],
        "previous cohort or manifest mismatch",
    )
    for role, n in [("build", 2048), ("fit", 8192), ("probe", 1024)]:
        rows = prev["roles"][role]
        need(len(rows) == n, "incomplete spent natural panel")
        spent += rows
    rows, blocked, supply = s51.next_block(
        records, base, spent, sum(N.values()), book["source"]["size"]
    )
    need(len(blocked) == 62054, "not the complete fresh suffix")
    offset = 0
    roles = {}
    for role, n in N.items():
        roles[role] = rows[offset : offset + n]
        offset += n
    put(
        DATA / "rows.json",
        dict(
            qualified=ref,
            manifest=MAN,
            previous=OLD,
            source=book["source"],
            excluded=len(blocked),
            excluded_sha=panels.fingerprint(sorted(blocked)),
            supply=supply,
            roles=roles,
        ),
    )
    put(
        DATA / "reservation.json",
        dict(rows_sha=sha(DATA / "rows.json"), qualified=ref, sizes=N, payloads=0),
    )


def extract(role):
    from ...x49_live_trace_memory.panels import window

    cpu()
    ref = qualified()
    r = get(DATA / "rows.json")
    h = sha(DATA / "rows.json")
    need(
        ref == r["qualified"] == get(DATA / "reservation.json")["qualified"]
        and h == get(DATA / "reservation.json")["rows_sha"]
        and len(r["roles"][role]) == N[role],
        "unbound new role",
    )
    if role == "probe":
        need(
            get(JOB / "freeze-ack.json")["frozen_sha"] == sha(JOB / "frozen.json")
            and get(JOB / "frozen.json")["rows_sha"] == h,
            "probe before witnessed all-fit freeze",
        )
    else:
        need(not (JOB / "frozen.json").exists(), "FIT/BUILD closed")
    (DATA / (role + "-opened")).touch(exist_ok=False)
    tic = time.monotonic()
    src = r["source"]
    width = 129 if role == "build" else 4097
    arr = np.empty((N[role], width), np.uint8)
    full = hashlib.sha256()
    with open(src["path"], "rb", buffering=0) as f:

        def check():
            st = os.fstat(f.fileno())
            need(
                (st.st_ino, st.st_size, st.st_mtime_ns)
                == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
                "source changed",
            )

        check()
        for j, row in enumerate(r["roles"][role]):
            raw = os.pread(f.fileno(), row["length"], row["offset"])
            need(
                hashlib.sha256(raw).hexdigest() == row["sha256"],
                "full document hash mismatch",
            )
            w = window(raw, row)
            need(len(w) == 4097, "wrong C14 window")
            full.update(w)
            arr[j] = np.frombuffer(w[:width], np.uint8)
        check()
    path = DATA / (role + ".npy")
    with path.open("xb") as f:
        np.save(f, arr, allow_pickle=False)
    put(
        DATA / (role + ".json"),
        dict(
            rows_sha=h,
            array_sha=sha(path),
            excerpt_sha=full.hexdigest(),
            shape=list(arr.shape),
            bytes=arr.nbytes,
            wall_s=time.monotonic() - tic,
        ),
    )


def load(role):
    meta = get(DATA / (role + ".json"))
    p = DATA / (role + ".npy")
    need(
        meta["rows_sha"] == sha(DATA / "rows.json") and meta["array_sha"] == sha(p),
        "changed role cache",
    )
    x = np.load(p, allow_pickle=False)
    need(x.dtype == np.uint8 and list(x.shape) == meta["shape"], "wrong raw bytes")
    return x


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        reserve()
        extract("build")
        extract("fit")
        put(
            DATA / "ready.json",
            dict(
                rows=sha(DATA / "rows.json"),
                build=sha(DATA / "build.json"),
                fit=sha(DATA / "fit.json"),
            ),
        )
        print("Fresh FIT and BUILD sealed; PROBE closed.")
    elif sys.argv[1] == "probe":
        extract("probe")
    else:
        raise ValueError("unknown action")
