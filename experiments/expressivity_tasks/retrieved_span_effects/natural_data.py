"""Fresh post-X53 TRAIN roles. Tokenize BUILD/FIT once; PROBE only after all freezes."""

import hashlib
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .natural_model import Shape
from .token_postings import index
from .window import BASE, HERE, ROOT, get, need, put

DATA = BASE / "natural-data"
SIZES = {"build": 2048, "fit": 8192, "probe": 1024}
CACHE = Path(
    "/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-long-document-anchor-v1/tiktoken-cache"
)
OBJ = "ec7223a39ce59f226a68acc30dc1af2788490e15"
TOKEN_SHA = "94b5ca7dff4d00767bc256fdd1b27e5b17361d7b8a5f968547f9f23eb70d2069"
FILES = (
    "NATURAL_PROTOCOL.md",
    "natural_model.py",
    "natural_oracle.py",
    "token_postings.py",
    "triton_loss.py",
    "natural_data.py",
    "natural_panel.py",
    "natural_meter.py",
    "cuda_step.py",
    "natural_score.py",
    "test_natural.py",
    "test_triton_read.py",
    "window.py",
    "window_leased.sh",
    "meter.py",
    "triton_read.py",
    "triton_controls.py",
    "triton_scan.py",
)


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def sources():
    return {name: sha(HERE / name) for name in FILES}


def public_gate():
    q = get(BASE / "natural-qualification.json")
    pub = (BASE / "natural-qualified-public-ref").read_text().strip()
    prefix = HERE.relative_to(ROOT)
    need(
        q["sources"] == sources()
        and q["natural_documents"] == q["natural_memberships"] == 0
        and q["cpu_passed"] > 0
        and sha(BASE / "natural-cpu.log") == q["tests_sha256"],
        "unqualified CPU sources",
    )
    need(
        not subprocess.run(
            ["git", "diff", "--quiet", pub, "--", *[str(prefix / n) for n in FILES]],
            cwd=ROOT,
        ).returncode
        and subprocess.check_output(
            ["git", "show", f"{pub}:{prefix}/receipts/natural_qualification.json"],
            cwd=ROOT,
        )
        == (BASE / "natural-qualification.json").read_bytes(),
        "no public qualification",
    )


def cpu():
    need(
        __debug__
        and os.uname().nodename == "lambda01"
        and os.getenv("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD")
        and all(
            os.getenv(x) == "1"
            for x in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
        ),
        "single-thread CPU admission required",
    )


def tokenizer():
    import tiktoken

    need(sha(CACHE / OBJ) == TOKEN_SHA, "p50k asset mismatch")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(CACHE)
    enc = tiktoken.get_encoding("p50k_base")
    need(enc.n_vocab == Shape().vocab, "vocab mismatch")
    return enc


def encode(enc, raw):
    # Arbitrary aligned raw byte windows can start/end inside a UTF-8 scalar.
    parts = re.split("([\udc80-\udcff]+)", raw.decode("utf8", "surrogateescape"))
    ids = []
    for i, p in enumerate(parts):
        if i % 2:
            ids.extend(enc.encode_single_token(bytes([ord(ch) - 0xDC00])) for ch in p)
        else:
            ids.extend(enc.encode_ordinary(p))
    need(enc.decode_bytes(ids) == raw, "lossy tokenization")
    return np.asarray(ids, np.int32)


def reserve():
    from ..x49_live_trace_memory import correction_transfer_data as pt, panels
    from ..x53_token_spelling import data as d53
    from ..x48_muon_continuation.data import MAN_SHA

    cpu()
    public_gate()
    need(
        get(BASE / "natural-meter/complete.json")["admitted"]
        and get(BASE / "natural-meter/charge.json")["release_verified"],
        "no closed meter",
    )
    (DATA / "reservation-attempted").touch(exist_ok=False)
    d52 = d53.p52
    d51 = d52.last
    d50 = d51.parent
    book, records, ex = pt.spent_context(pt.parent()["data"])
    short = pt.u.bound(pt.BASE / "reservation.json", d50.PT_ROWS)
    shorts, _ = pt.checked_rows(short)
    need(
        len(ex) == 45217 and panels.fingerprint(sorted(ex)) == short["excluded_sha"],
        "bad PC/PT chain",
    )
    base = ex | {r["sha256"] for r in shorts}
    spent = []
    ex = base.copy()
    for mod, rhash, phash, rkey, prev in [
        (d50, d51.PARENT_ROWS, d51.PARENT_RESULT, "fit", 45414),
        (d51, d52.PREV_ROWS, d52.PREV_PUBLIC, "fit", 46182),
        (d52, d53.PREV_ROWS, d53.PREV_PUBLIC, "fit", 47718),
        (
            d53,
            "3aa0469581837dd6e31172a59cd4130685af0b5138be72614e770f496226b130",
            "891b4407b9f793b2769ff73295d39ae1caefb4e9fde3b5c0284cf2b4357efbd8",
            "build",
            49254,
        ),
    ]:
        b = pt.u.bound(mod.RUN / "reservation.json", rhash)
        p = pt.u.bound(mod.HERE / "receipts/result.json", phash)
        w = get(mod.RUN / "witnessed.json")
        hs = {r["sha256"] for r in b[rkey] + b["probe"]}
        need(
            len(ex) == b["excluded"] == prev
            and b["excluded_sha"] == panels.fingerprint(sorted(ex))
            and not hs & ex
            and len(hs) == sum(mod.N.values())
            and b["manifest"] == MAN_SHA
            and b["source"] == book["source"]
            and w["result_sha"] == p["private_close_sha256"]
            and w["owner_exited"]
            and w["receipt_only_recomputed"],
            "bad spent role",
        )
        ex |= hs
        spent += b[rkey] + b["probe"]
    rows, blocked, supply = d51.next_block(
        records, base, spent, sum(SIZES.values()), book["source"]["size"]
    )
    need(ex == blocked and len(ex) == 50790, "exclusion chain mismatch")
    cursor = 0
    roles = {}
    for role, size in SIZES.items():
        roles[role] = rows[cursor : cursor + size]
        cursor += size
    put(
        DATA / "reservation.json",
        dict(
            excluded=len(ex),
            excluded_sha=panels.fingerprint(sorted(ex)),
            supply=supply,
            manifest=MAN_SHA,
            source=book["source"],
            roles=roles,
        ),
    )
    put(
        DATA / "reserved.json",
        dict(
            reservation_sha=sha(DATA / "reservation.json"), sizes=SIZES, payload_docs=0
        ),
    )


def windows(doc, time_len):
    for i in range(0, len(doc) - 1, time_len):
        n = min(time_len, len(doc) - i - 1)
        w = np.zeros(time_len + 1, np.int32)
        w[: n + 1] = doc[i : i + n + 1]
        yield w, n


def cached(role):
    meta = get(DATA / (role + ".json"))
    path = DATA / (role + ".npz")
    need(
        meta["image_sha"] == sha(path)
        and meta["rows_sha"]
        == sha(DATA / "reservation.json")
        == get(DATA / "reserved.json")["reservation_sha"]
        and meta["tokenizer_sha"] == TOKEN_SHA,
        "changed cached role",
    )
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def extract(role):
    from ..x49_live_trace_memory.panels import window

    cpu()
    public_gate()
    need(role in SIZES, "unknown role")
    frozen = BASE / "panel/frozen.json"
    if role == "probe":
        need(
            frozen.is_file()
            and get(BASE / "panel/freeze-ack.json")["freeze_sha"] == sha(frozen),
            "PROBE blocked until all checkpoints frozen and witnessed",
        )
    else:
        need(not frozen.exists(), "FIT/BUILD closed")
    book = get(DATA / "reservation.json")
    rh = sha(DATA / "reservation.json")
    need(
        get(DATA / "reserved.json")["reservation_sha"] == rh
        and len(book["roles"][role]) == SIZES[role],
        "unbound reservation",
    )
    (DATA / (role + "-open-attempted")).touch(exist_ok=False)
    enc = tokenizer()
    sh = Shape()
    docs = []
    digest = hashlib.sha256()
    byte_count = 0
    src = book["source"]
    start = time.monotonic()
    with open(src["path"], "rb", buffering=0) as f:

        def identity():
            st = os.fstat(f.fileno())
            need(
                (st.st_ino, st.st_size, st.st_mtime_ns)
                == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
                "source identity changed",
            )

        identity()
        for i, r in enumerate(book["roles"][role]):
            raw = os.pread(f.fileno(), r["length"], r["offset"])
            sample = window(raw, r)
            tok = encode(enc, sample)
            docs.append(tok)
            digest.update(sample)
            byte_count += len(raw)
            if (i + 1) % 128 == 0:
                put(
                    DATA / "extract-progress.json",
                    dict(role=role, docs=i + 1, pid=os.getpid()),
                )
        identity()
    p = DATA / (role + ".npz")
    info = {}
    if role == "build":
        image = index(docs, sh.page)
        np.savez(p, **vars(image))
        info.update(
            pages=len(image.pages), codes=len(image.codes), postings=len(image.postings)
        )
    else:
        ws = []
        ns = []
        di = []
        for i, doc in enumerate(docs):
            for w, n in windows(doc, sh.time):
                ws.append(w)
                ns.append(n)
                di.append(i)
        need(sum(ns) == sum(len(doc) - 1 for doc in docs), "lost shifted targets")
        np.savez(
            p,
            windows=np.stack(ws),
            count=np.array(ns, np.int32),
            doc=np.array(di, np.int32),
        )
        info.update(windows=len(ws), targets=sum(ns))
    put(
        DATA / (role + ".json"),
        dict(
            **info,
            docs=len(docs),
            tokens=sum(map(len, docs)),
            window_sha=digest.hexdigest(),
            source_bytes_checked=byte_count,
            tokenizer_sha=TOKEN_SHA,
            rows_sha=rh,
            image_sha=sha(p),
            wall_s=time.monotonic() - start,
        ),
    )


def prepare():
    cpu()
    public_gate()
    need(
        get(BASE / "natural-meter/complete.json")["admitted"]
        and get(BASE / "natural-meter/charge.json")["rc"] == 0,
        "natural meter not closed",
    )
    DATA.mkdir()
    reserve()
    extract("build")
    extract("fit")
    print(
        "Reserved disjoint roles; verified BUILD/FIT; PROBE remains sealed.", flush=True
    )


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "probe":
        extract("probe")
    else:
        raise ValueError("unknown phase")
