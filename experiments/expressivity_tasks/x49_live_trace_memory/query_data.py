"""X49-Q single-use native TRAIN diagnostic reservation and checked extraction. No test split loader."""

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
    check,
    load,
    sha,
    write_json,
)
from ..x48_muon_continuation.train import flush_dir
from .panels import (
    OUT,
    T,
    digest,
    early_panels,
    fingerprint,
    membership,
    prior_sets,
    select,
    verify_selection,
    window,
)
from . import write_data as wd
from .write_panel import ROW_SHA as W_ROWS_SHA

SPEC = "a7aee64f8ce4584e3740b8c5f1000159994400bc4763200375c5f71598d2e5f4"
W_RESULT = "538811441a9fb0dba5502a10d6b9ca550a179c463003b623a86433794f6b8b0d"
W_REF = "a8eda5bf15228c1f198176cb40bfd68cfb7900fc"
BASE = OUT / "query-data-1"
N = 128


def checked_spec():
    check(
        sha(Path(__file__).with_name("ADDRESS_DEPTH.md")) == SPEC
        and sha(Path(__file__).with_name("receipts") / "write_result.json") == W_RESULT,
        "changed Q registration or W result",
    )


def select_new(records, *, base_ex, prior, quota, size):
    spent = set(membership(prior))
    ex = set(base_ex) | spent
    check(not base_ex.intersection(spent), "invalid W exclusion union")
    q = {"train": quota}
    panels, counts = select(records, ex, q, size)
    verify_selection(records, ex, panels, q, counts)
    # Q is the exact next suffix of the same rank used to select W TRAIN; no offset reuse.
    n = len(prior["train"])
    combined, _ = select(records, base_ex, {"train": n + quota}, size)
    check(
        combined["train"] == prior["train"] + panels["train"],
        "not the unspent TRAIN rank suffix",
    )
    return panels["train"], ex, counts["train"]


def selection():
    checked_spec()
    check(sha(wd.ROWS) == W_ROWS_SHA, "foreign W membership")
    w = load(wd.ROWS)
    check(w == wd.selection(), "changed W parent")
    man = FLAG / "documents.jsonl.gz"
    check(sha(man) == MAN_SHA, "changed raw document manifest")
    with gzip.open(man, "rt") as f:
        records = list(map(json.loads, f))
    ex, _ = prior_sets()
    ex.update(early_panels(records))
    ex.update(membership(load(OUT / "v2/reservation.json")["panels"]))
    ex.update(membership(load(wd.R_RES)["rows"]))
    check(
        len(ex) == 39916 and fingerprint(sorted(ex)) == w["excluded_sha"],
        "historical membership mismatch",
    )
    rows, ex, left = select_new(
        records, base_ex=ex, prior=w["panels"], quota=N, size=w["source"]["size"]
    )
    check(left == 24467 and len(ex) == 41317, "changed Q supply")
    return dict(
        schema="x49-q-diagnostic-train",
        spec=SPEC,
        parent_rows_sha=W_ROWS_SHA,
        manifest_sha=MAN_SHA,
        source=w["source"],
        excluded_unique=len(ex),
        excluded_sha=fingerprint(sorted(ex)),
        eligible_train=left,
        rows=rows,
    )


def fixture_ref():
    from .query_run import checked_fixture_ref

    return checked_fixture_ref()


def prepare():
    check(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD"),
        "Q CPU preparation only",
    )
    ref = fixture_ref()
    os.umask(0o077)
    t0 = monotonic()
    c0 = process_time()
    BASE.mkdir(mode=0o700)
    f = BASE / "reservation.json"
    m = selection()
    write_json(f, m)
    flush_dir(BASE)
    rs = sha(f)
    rows = m["rows"]
    check(
        load(f) == selection() and sha(f) == rs and len(rows) == N,
        "Q reservation changed before raw TRAIN read",
    )
    src = m["source"]
    seen = set()
    for r in rows:
        h = r["sha256"]
        check(
            r["split"] == "train"
            and int(h, 16) % 100 >= 2
            and r["offset"] == r["doc"] >= MIN_OFFSET
            and r["length"] >= T + 1
            and r["start"] == T * (int(h[16:32], 16) % ((r["length"] - 1) // T))
            and h not in seen,
            "foreign native TRAIN descriptor",
        )
        seen.add(h)
    folder = BASE / "stream.preparing"
    folder.mkdir()
    hs = []
    source_bytes = 0

    def verify(fh):
        st = os.fstat(fh.fileno())
        check(
            (st.st_size, st.st_mtime_ns, st.st_ino)
            == tuple(src[k] for k in ("size", "mtime_ns", "inode")),
            "changed Q source descriptor",
        )

    (BASE / "raw-train-open-attempted").touch(mode=0o600, exist_ok=False)
    with (
        open(src["path"], "rb", buffering=0) as fh,
        (folder / "windows.u8").open("xb") as payload,
    ):
        verify(fh)
        for r in rows:
            raw = os.pread(fh.fileno(), r["length"], r["offset"])
            w = window(raw, r)
            check(len(w) == T + 1, "incomplete diagnostic shift")
            payload.write(w)
            source_bytes += len(raw)
            hs.append(
                dict(
                    document_sha=r["sha256"],
                    start=r["start"],
                    window=digest(w),
                    inputs=digest(w[:-1]),
                    targets=digest(w[1:]),
                )
            )
        verify(fh)
        payload.flush()
        os.fsync(payload.fileno())
    write_json(folder / "shifts.json", hs)
    idp = folder / "identity.json"
    write_json(
        idp,
        dict(
            schema="x49-q-checked-shifts",
            spec=SPEC,
            rows_sha=rs,
            count=N,
            targets=N * T,
            raw_sha=sha(folder / "windows.u8"),
            shifts_sha=sha(folder / "shifts.json"),
        ),
    )
    ident = sha(idp)
    flush_dir(folder)
    folder.rename(BASE / "stream")
    flush_dir(BASE)
    r = dict(
        scope="new native TRAIN diagnostic only; no scoring",
        spec=SPEC,
        fixture_ref=ref,
        rows_sha=rs,
        identity_sha=ident,
        excluded_unique=m["excluded_unique"],
        eligible_train=m["eligible_train"],
        documents=N,
        targets=N * T,
        source_train_bytes=source_bytes,
        window_bytes=N * (T + 1),
        calibration_or_test_documents_opened=0,
        wall_s=monotonic() - t0,
        cpu_s=process_time() - c0,
    )
    write_json(BASE / "prepared.json", r)
    return r


def metadata():
    """Private receipt identities, NO window/shift payload reads, no recomputation of selection."""
    checked_spec()
    pf = BASE / "prepared.json"
    q = load(pf)
    f = BASE / "reservation.json"
    idp = BASE / "stream/identity.json"
    ident = load(idp)
    check(
        q["spec"] == ident["spec"] == SPEC
        and ident["schema"] == "x49-q-checked-shifts"
        and q["rows_sha"] == sha(f) == ident["rows_sha"]
        and q["identity_sha"] == sha(idp)
        and q["documents"] == ident["count"] == N
        and q["targets"] == ident["targets"] == N * T
        and q["calibration_or_test_documents_opened"] == 0,
        "unbound Q preparation",
    )
    return dict(prepared_sha=sha(pf), rows_sha=sha(f), identity_sha=sha(idp))


def verify_payload(rows, ident, shifts, data):
    check(
        len(rows) == len(shifts) == ident["count"] == N
        and len(data) == N * (T + 1)
        and digest(data) == ident["raw_sha"]
        and len(set(r["sha256"] for r in rows)) == len({r["doc"] for r in rows}) == N,
        "truncated, duplicate or foreign Q stream",
    )
    out = []
    for i, (r, h) in enumerate(zip(rows, shifts, strict=True)):
        w = data[i * (T + 1) : (i + 1) * (T + 1)]
        check(
            r["split"] == "train"
            and int(r["sha256"], 16) % 100 >= 2
            and h
            == dict(
                document_sha=r["sha256"],
                start=r["start"],
                window=digest(w),
                inputs=digest(w[:-1]),
                targets=digest(w[1:]),
            ),
            "incorrect TRAIN identity/shift",
        )
        out.append((r["sha256"], w))
    return out


def diagnostic_windows(job):
    """Only the admitted trainer under the actual live lease owner may open the prepared stream."""
    from .query_run import admit
    from . import query_panel as q
    from .panel import licensed_admitted_reader

    check(job == q.JOB, "wrong Q reader")
    d, ref, start = admit(job)
    f = licensed_admitted_reader(job, d, ref, start)
    check(
        f == q.plan(ref, hash_weights=True) and d == q.ROOT / job.name,
        "unlicensed Q payload access",
    )
    (d / "source-open-attempted").touch(mode=0o600, exist_ok=False)
    folder = BASE / "stream"

    def bound_json(path, expected):
        raw = path.read_bytes()
        check(digest(raw) == expected, "Q stream metadata changed since freeze")
        return json.loads(raw)

    i = bound_json(folder / "identity.json", f["data"]["identity_sha"])
    return verify_payload(
        bound_json(BASE / "reservation.json", f["data"]["rows_sha"])["rows"],
        i,
        bound_json(folder / "shifts.json", i["shifts_sha"]),
        (folder / "windows.u8").read_bytes(),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
