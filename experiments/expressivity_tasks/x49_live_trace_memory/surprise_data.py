"""X49-U: one native TRAIN reservation. CPU extracts FIT; PROBE raw access requires all frozen fits."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from time import monotonic, process_time
from ..x48_muon_continuation.data import MIN_OFFSET, check, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from . import query_data as q
from .fit_phase import study as phase
from .panels import OUT, T, digest, fingerprint, window

BASE = OUT / "surprise-data-1"
N = {"fit": 256, "probe": 128}
SPEC = "12d8c45849549427a2de44b175b03b85ac523caf648e33512bd9ed88bda6e980"
Q_RESULT = "8d35f000f281659c84b3b7252b09b6f00cbc97d176a6a6cc021c30a1f68ff761"
HERE = Path(__file__).resolve().parent


def checked_spec():
    q.checked_spec()
    check(sha(HERE / "PAST_SURPRISE.md") == SPEC, "changed U registration")
    return bound(HERE / "receipts/query_result.json", Q_RESULT)


def bound(path, expected):
    raw = path.read_bytes()
    check(digest(raw) == expected, "changed bound receipt: " + path.name)
    return json.loads(raw)


def selection_context():
    """Reconstruct U and its earlier exclusions from metadata; never load a payload."""
    result = checked_spec()
    qr = bound(q.BASE / "reservation.json", result["data"]["rows_sha"])
    records, ex, w = q.parent_context()
    qrows, pre_q, left = q.select_new(
        records, base_ex=ex, prior=w["panels"], quota=q.N, size=w["source"]["size"]
    )
    check(
        qr["rows"] == qrows
        and qr["source"] == w["source"]
        and qr["spec"] == q.SPEC
        and qr["parent_rows_sha"] == q.W_ROWS_SHA
        and qr["eligible_train"] == left
        and qr["excluded_sha"] == fingerprint(sorted(pre_q)),
        "foreign Q rank/parent",
    )
    rows, excluded, supply = q.select_new(
        records,
        base_ex=pre_q,
        prior={"train": qrows},
        quota=sum(N.values()),
        size=qr["source"]["size"],
    )
    check(
        supply == left - q.N and len(excluded) == len(pre_q) + q.N,
        "incorrect U exclusions",
    )
    book = dict(
        schema="x49-u-native-train",
        spec=SPEC,
        q_rows_sha=result["data"]["rows_sha"],
        manifest_sha=qr["manifest_sha"],
        source=qr["source"],
        eligible_train=supply,
        excluded_unique=len(excluded),
        excluded_sha=fingerprint(sorted(excluded)),
        roles=dict(fit=rows[: N["fit"]], probe=rows[N["fit"] :]),
    )
    return book, records, excluded, w


def selection():
    return selection_context()[0]


def checked_roles(book):
    return role_rows(book, "x49-u-native-train", SPEC, N, MIN_OFFSET)


def role_rows(book, schema, spec, sizes, lo):
    roles = book["roles"]
    check(
        book["schema"] == schema
        and book["spec"] == spec
        and set(roles) == set(sizes)
        and all(len(rows) == sizes[k] for k, rows in roles.items()),
        "foreign FIT/PROBE reservation",
    )
    seen = set()
    docs = set()
    rank = []
    for r in roles["fit"] + roles["probe"]:
        h = r["sha256"]
        check(
            len(h) == 64
            and r["split"] == "train"
            and int(h, 16) % 100 >= 2
            and r["offset"] == r["doc"] >= lo
            and r["offset"] + r["length"] <= book["source"]["size"]
            and r["length"] >= T + 1
            and r["start"] == T * (int(h[16:32], 16) % ((r["length"] - 1) // T))
            and h not in seen
            and r["doc"] not in docs,
            "invalid or overlapping FIT/PROBE row",
        )
        seen.add(h)
        docs.add(r["doc"])
        rank.append((digest(("x49-live-v1:" + h).encode("ascii")), h, r["doc"]))
    check(rank == sorted(rank), "roles are not contiguous rank blocks")
    return roles


def prepare(*, study="U"):
    from .panel import cpu

    data, _, runner = phase(study)
    base, sizes = data.BASE, data.N
    cpu()
    ref = runner.checked_fixture_ref()
    os.umask(0o077)
    t0 = monotonic()
    c0 = process_time()
    base.mkdir(mode=0o700)
    book = data.selection()
    data.checked_roles(book)
    f = base / "reservation.json"
    write_json(f, book)
    flush_dir(base)
    rs = sha(f)
    ident = data.extract("fit")[0]
    check(ident["rows_sha"] == rs and sha(f) == rs, "changed membership at FIT prepare")
    z = dict(
        spec=data.SPEC,
        qualified_ref=ref,
        rows_sha=rs,
        fit_identity_sha=sha(base / "fit/identity.json"),
        documents=dict(sizes),
        excluded_unique=book["excluded_unique"],
        eligible_train=book["eligible_train"],
        fit_source_bytes=ident["source_bytes"],
        fit_window_bytes=sizes["fit"] * (T + 1),
        fit_targets=sizes["fit"] * T,
        probe_documents_opened=0,
        wall_s=monotonic() - t0,
        cpu_s=process_time() - c0,
    )
    write_json(base / "prepared.json", z)
    return z


def metadata(*, study="U"):
    data, _, _ = phase(study)
    data.checked_spec()
    f = data.BASE / "prepared.json"
    raw = f.read_bytes()
    z = json.loads(raw)
    book = bound(data.BASE / "reservation.json", z["rows_sha"])
    data.checked_roles(book)
    i = bound(data.BASE / "fit/identity.json", z["fit_identity_sha"])
    check(
        z["spec"] == i["spec"] == data.SPEC
        and i["role"] == "fit"
        and i["count"] == data.N["fit"]
        and i["targets"] == z["fit_targets"] == data.N["fit"] * T
        and i["rows_sha"] == z["rows_sha"]
        and z["documents"] == data.N
        and z["probe_documents_opened"] == 0,
        f"incorrect {study} preparation",
    )
    return dict(
        prepared_sha=digest(raw),
        prepared_source=z["qualified_ref"],
        rows_sha=z["rows_sha"],
        fit_identity_sha=z["fit_identity_sha"],
    )


def licensed(job, *, study="U"):
    _, p, runner = phase(study)
    from .panel import licensed_admitted_reader

    check(job == p.JOB, f"wrong {study} reader")
    d, ref, start = runner.admit(job)
    plan = licensed_admitted_reader(job, d, ref, start)
    check(
        d == p.ROOT / p.JOB.name and plan == p.plan(ref, hash_weights=True),
        f"foreign {study} plan or unbound checkpoint bytes",
    )
    return d, plan


def fit_windows(job, *, study="U"):
    data, _, _ = phase(study)
    d, plan = data.licensed(job)
    meta = plan["data"]
    rows = data.checked_roles(bound(data.BASE / "reservation.json", meta["rows_sha"]))[
        "fit"
    ]
    folder = data.BASE / "fit"
    ident = bound(folder / "identity.json", meta["fit_identity_sha"])
    check(
        ident["role"] == "fit"
        and ident["rows_sha"] == meta["rows_sha"]
        and not (d / "fit-frozen.json").exists(),
        "wrong FIT stream",
    )
    (d / "source-open-attempted").touch(mode=0o600, exist_ok=False)
    return q.verified_windows(
        rows,
        ident,
        bound(folder / "shifts.json", ident["shifts_sha"]),
        (folder / "windows.u8").read_bytes(),
        data.N["fit"],
    )


def extract(role, *, job=None, fit_sha=None, study="U"):
    """License inside the raw reader, before opening the source. No caller-supplied rows/offsets."""
    data, p, runner = phase(study)
    readouts = None
    if role == "fit":
        from .panel import cpu

        cpu()
        runner.checked_fixture_ref()
        check(
            job is None
            and fit_sha is None
            and not (data.BASE / "prepared.json").exists(),
            "FIT is prepare-only",
        )
        f = data.BASE / "reservation.json"
        rs = sha(f)
        book = bound(f, rs)
        check(book == data.selection(), "FIT reservation does not rederive")
        d = data.BASE
        folder = d / "fit"
    else:
        check(
            role == "probe"
            and job == p.JOB
            and isinstance(fit_sha, str)
            and len(fit_sha) == 64,
            "unlicensed PROBE request",
        )
        d, plan = data.licensed(job)
        readouts = p.checked_readouts(d, plan, fit_sha)
        rs = plan["data"]["rows_sha"]
        book = bound(data.BASE / "reservation.json", rs)
        folder = d / "probe"
    rows = data.checked_roles(book)[role]
    folder.mkdir(mode=0o700)
    (d / f"raw-{role}-open-attempted").touch(mode=0o600, exist_ok=False)
    flush_dir(d)
    t0 = monotonic()
    src = book["source"]
    shifts = []
    chunks = []
    size = 0

    def verify(f):
        st = os.fstat(f.fileno())
        check(
            (st.st_size, st.st_mtime_ns, st.st_ino)
            == tuple(src[k] for k in ("size", "mtime_ns", "inode")),
            f"changed {study} raw source",
        )

    with open(src["path"], "rb", buffering=0) as fh:
        verify(fh)
        for r in rows:
            raw = os.pread(fh.fileno(), r["length"], r["offset"])
            w = window(raw, r)
            check(len(w) == T + 1, f"incomplete {study} shift")
            chunks.append(w)
            size += len(raw)
            shifts.append(
                dict(
                    document_sha=r["sha256"],
                    start=r["start"],
                    window=digest(w),
                    inputs=digest(w[:-1]),
                    targets=digest(w[1:]),
                )
            )
        verify(fh)
    payload = b"".join(chunks)
    write_json(folder / "shifts.json", shifts)
    ident = dict(
        spec=data.SPEC,
        role=role,
        rows_sha=rs,
        count=len(rows),
        targets=len(rows) * T,
        source_bytes=size,
        raw_sha=digest(payload),
        shifts_sha=sha(folder / "shifts.json"),
        fit_freeze_sha=fit_sha,
        extraction_wall_s=monotonic() - t0,
    )
    wins = q.verified_windows(rows, ident, shifts, payload, len(rows))
    if role == "fit":
        with (folder / "windows.u8").open("xb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
    write_json(folder / "identity.json", ident)
    flush_dir(folder)
    return ident, wins, readouts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["prepare"])
    ap.parse_args()
    print(json.dumps(prepare(), sort_keys=True))
