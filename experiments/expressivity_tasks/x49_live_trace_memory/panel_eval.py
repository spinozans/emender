"""Inside an admitted stage only: open its frozen split once, score the complete endpoint map."""

from __future__ import annotations
from itertools import islice
import os
import resource
from time import perf_counter
import torch
from ..x43_fork_resolved.c14_documents import DocumentWindows
from ..x48_muon_continuation.data import load, sha, write_json
from ..x48_muon_continuation.train import cpu_seconds
from .coupled import cpu_origin
from .endpoint import score_window, sync
from .meter_run import fixture
from .panel_spec import ROOT, STEPS, T, binding, check, training
from .panel_train import append
from .panels import OUT, digest, fingerprint, reserve_data
from .train_stream import V2_IDENTITY


def checked_eval_window(raw, row, split):
    check(
        split in ("calibration", "final_test")
        and row["split"] == split
        and len(raw) == row["length"]
        and row["start"] % T == 0
        and 0 <= row["start"] <= len(raw) - T - 1,
        "wrong evaluation split or bounds",
    )
    # C14's public API calls BOTH kinds of held-out documents 'valid'. The caller
    # binds the native split, offset and content identity before the descriptor read.
    st = row["start"]
    w = next(
        islice(
            DocumentWindows(T).windows(
                raw,
                dict(source_index=row["doc"], split="valid", sha256=row["sha256"]),
                split="valid",
            ),
            st // T,
            None,
        )
    )
    check(
        w.document == row["doc"]
        and w.start == st
        and len(w.targets) == T
        and w.inputs == raw[st : st + T]
        and w.targets == raw[st + 1 : st + T + 1],
        "unbound evaluation shift",
    )
    return w.inputs + w.targets[-1:]


def checked_short_document(raw, row):
    check(
        row["split"] == "final_test"
        and row["start"] == 0
        and row["doc"] == row["offset"]
        and 2 <= len(raw) == row["length"] <= T
        and int(row["sha256"], 16) % 100 == 1,
        "not a registered short final document",
    )
    windows = DocumentWindows(T).windows(
        raw,
        dict(source_index=row["doc"], split="valid", sha256=row["sha256"]),
        split="valid",
    )
    w = next(windows)
    check(
        next(windows, None) is None
        and w.document == row["doc"]
        and w.start == 0
        and w.inputs == raw[:-1]
        and w.targets == raw[1:],
        "wrong short shift or truncated tail",
    )
    return raw


def _v2_descriptor(job, frozen):
    from .panel import licensed_split

    check(
        licensed_split(job) == frozen, "unlicensed or changed split before source open"
    )
    check(
        job.kind in ("calibration", "final_test")
        and frozen["split"] == job.kind
        and frozen["job"] == job.name
        and frozen["train_identity"] == V2_IDENTITY,
        "foreign frozen split",
    )
    base = OUT / "v2"
    idp = base / "train/identity.json"
    p = base / "reservation.json"
    check(
        sha(idp) == V2_IDENTITY and sha(p) == load(idp)["reservation_sha"],
        "foreign private reservation",
    )
    r = load(p)
    check(r == reserve_data("v2"), "reservation has changed")
    rows = r["panels"][job.kind]
    check(
        len(rows) == frozen["documents"]
        and all(
            x["split"] == job.kind
            and int(x["sha256"], 16) % 100
            == {"calibration": 0, "final_test": 1}[job.kind]
            for x in rows
        ),
        "refuse source open",
    )
    return ROOT / job.name, r["source"], rows


def _documents(job, frozen):
    """This is the only panel raw source open: accept no caller-selected rows or paths."""
    if job.name == "w-final":
        from .write_panel import fresh_descriptor

        folder, src, rows = fresh_descriptor(job, frozen)
    elif job.name == "r-final":
        from .relative_panel import fresh_descriptor

        folder, src, rows = fresh_descriptor(job, frozen)
    else:
        folder, src, rows = _v2_descriptor(job, frozen)
    split = job.kind
    windows = []
    nread = 0

    def verify(f):
        s = os.fstat(f.fileno())
        check(
            (s.st_ino, s.st_size, s.st_mtime_ns)
            == tuple(src[k] for k in ("inode", "size", "mtime_ns")),
            "changed evaluation source descriptor",
        )

    (folder / "source-open-attempted").touch(
        mode=0o600, exist_ok=False
    )  # single use, including refusal/partial
    with open(src["path"], "rb", buffering=0) as f:
        verify(f)
        for row in rows:
            raw = os.pread(f.fileno(), row["length"], row["offset"])
            w = (
                checked_short_document(raw, row)
                if job.name == "w-final"
                else checked_eval_window(raw, row, split)
            )
            windows.append((row["doc"], row["sha256"], w))
            nread += row["length"]
        verify(f)
    return windows, nread


@torch.inference_mode()
def decode_probe(host, head, arm, seed):
    """Separate synthetic diagnostic: one observed advance, full-prefix recompute; not a cached service."""
    word = fixture(4)[:-1]
    out = []
    dev = next(host.parameters()).device
    for n in [T - 1, T]:
        sync(dev)
        tic = perf_counter()
        w = word[:n]
        p = host(torch.tensor([list(w)], device=dev, dtype=torch.long)).to("cpu")
        h, z = p[0].split((256, 256), dim=-1)
        y = (
            head(w, h, z, seed=seed, document=4, visit=0, horizon=T).lp
            if arm in ("F", "C")
            else head(w, h, z, mode=arm, horizon=T)
        )
        check(y[-1].shape == (256,) and y[-1].isfinite().all(), "invalid one-byte read")
        sync(dev)
        out.append(perf_counter() - tic)
    return dict(
        prefix_inputs=T - 1,
        prefill_s=out[0],
        observed_increment=1,
        full_recompute_decode_s=out[1],
    )


def evaluate(job):
    from .panel import enter, progress

    def resolve(k, s, i, ref):
        return ROOT / training(k, s) / f"weights-{i:04d}.pt", binding(ref, k, s)

    return evaluate_admitted(job, enter(job), _documents, progress, resolve)


def evaluate_admitted(job, ctx, documents, progress, resolve, *, origin=None):
    d = ctx["folder"]
    f = d / "frozen.json"
    start = load(d / "start.json")
    check(sha(f) == start["freeze_sha"], "endpoint map changed before evaluation")
    plan = load(f)
    check(
        plan["ref"] == ctx["ref"]
        and plan["visit"] == 0
        and plan["horizons"] == [128, 1024, T],
        "foreign evaluation modes",
    )
    c0 = cpu_seconds()
    tic = perf_counter()
    windows, nread = documents(job, plan)
    prep = perf_counter() - tic
    write_json(
        d / "stream.json",
        dict(
            split=job.kind,
            windows=len(windows),
            full_document_source_bytes=nread,
            shift_receipt=fingerprint(
                [(h, digest(w[:-1]), digest(w[1:])) for _, h, w in windows]
            ),
            wall_s=prep,
        ),
    )
    n = 0
    total_targets = 0
    endpoint_receipts = {}
    for key, e in sorted(plan["endpoints"].items()):
        k, s, i = e["arm"], e["seed"], e["step"]
        p, expected = resolve(k, s, i, ctx["ref"])
        check(
            key == f"{k}-{s}@{i}"
            and 1 <= i <= STEPS
            and i % 32 == 0
            and e["binding"] == expected,
            "foreign analysis endpoint",
        )
        check(
            p.stat().st_size == e["bytes"] and sha(p) == e["sha256"],
            "weights changed since freeze",
        )
        host, head = (cpu_origin if origin is None else origin)(s, k)
        ck = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
        check(
            set(ck) == {"binding", "step", "host", "head"}
            and ck["binding"] == e["binding"]
            and ck["step"] == i,
            "unbound weights",
        )
        host.load_state_dict(ck["host"], strict=True)
        head.load_state_dict(ck["head"], strict=True)
        del ck
        host = host.eval().to("cuda")
        head.eval()
        torch.set_num_threads(4)
        # R changes training credit only. Its frozen prediction path is exactly C's;
        # checkpoint/source identities above and in the log key remain R, never C.
        inference_arm = "C" if k == "R" else k
        probe = decode_probe(host, head, inference_arm, s)
        log = d / (key + ".jsonl")
        log.touch(mode=0o600, exist_ok=False)
        for doc, h, w in windows:
            data = score_window(
                host,
                head,
                w,
                arm=inference_arm,
                seed=s,
                document=doc,
                stratify=True,
                **({"short_full_dose": i == STEPS} if job.name == "w-final" else {}),
            )
            append(log, dict(document_sha=h, **data))
            n += 1
            total_targets += len(w) - 1
            progress(d, n, total_targets)
        endpoint_receipts[key] = dict(
            scores_sha=sha(log),
            checkpoint_sha=e["sha256"],
            windows=len(windows),
            decode_probe=probe,
        )
        del host, head
        torch.cuda.empty_cache()
    check(
        n == plan["units"]
        and total_targets
        == (
            plan["targets_per_endpoint"] * len(endpoint_receipts)
            if job.name == "w-final"
            else n * T
        ),
        "incomplete frozen split",
    )
    return dict(
        kind=job.kind,
        units=n,
        targets=total_targets,
        freeze_sha=sha(f),
        stream_sha=sha(d / "stream.json"),
        endpoints=endpoint_receipts,
        gpu_peak_alloc=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved=torch.cuda.max_memory_reserved(),
        process_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_core_s=[y - x for x, y in zip(c0, cpu_seconds(), strict=True)],
    )
