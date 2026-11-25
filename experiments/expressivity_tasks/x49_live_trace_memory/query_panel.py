"""X49-Q: single registered frozen inference job, distinct CPU preparation and budget identities."""

from __future__ import annotations
from pathlib import Path
import torch
from ..x48_muon_continuation.data import check, load, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from .panels import SEEDS, T, OUT, fingerprint
from .panel_spec import Job
from . import query_data as data, write_panel as w

ROOT = OUT / "query-probe-1"
TOTAL = 1800
JOB = Job("q-probe", "diagnostic_train", TOTAL, "JEM")
JOBS = {JOB.name: JOB}


def checkpoint(seed, *, hash_weights):
    data.checked_spec()
    path, b = w.resolve("B", seed, 1024, data.W_REF)
    r = load(Path(__file__).with_name("receipts") / "write_result.json")
    dig = r["training"][w.name("B", seed)]["terminal_weights_sha"]
    check(
        r["qualified_source"] == data.W_REF and len(dig) == 64,
        "foreign W result lineage",
    )
    size = path.stat().st_size
    check(
        load(path.with_suffix(".json"))
        == dict(binding=b, bytes=size, file=path.name, sha256=dig, step=1024)
        and (not hash_weights or sha(path) == dig),
        "unbound W continued-B weights",
    )
    return path, dict(binding=b, bytes=size, sha256=dig, step=1024)


def module(seed, expected):
    from .coupled import cpu_origin

    path, e = checkpoint(seed, hash_weights=True)
    check(e == expected, "foreign frozen Q endpoint")
    ck = torch.load(path, mmap=True, weights_only=True, map_location="cpu")
    check(
        set(ck) == {"binding", "step", "host", "head"}
        and ck["binding"] == e["binding"]
        and ck["step"] == 1024,
        "wrong W state",
    )
    m, c = cpu_origin(seed, "B")
    m.load_state_dict(ck["host"], strict=True)
    c.load_state_dict(ck["head"], strict=True)
    return m.eval().requires_grad_(False), c.eval().requires_grad_(False)


def expected(ref):
    return dict(
        scope="x49-q-one-read-frozen-train-v1",
        qualified_ref=ref,
        spec=data.SPEC,
        w_result=data.W_RESULT,
        checkpoints={
            str(s): checkpoint(s, hash_weights=False)[1]["sha256"] for s in SEEDS
        },
        data=data.metadata(),
        max_acquisitions=1,
        total_card_seconds=TOTAL,
        job_caps={JOB.name: TOTAL},
        local_gpus=1,
        remote=False,
        retries=0,
        updates=0,
        documents=data.N,
        split="native_train_diagnostic",
        units=3 * data.N,
    )


def validate(a, ref):
    e = expected(ref)
    check(
        type(a) is dict
        and set(a) == set(e) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in e}) == fingerprint(e),
        "wrong or incomplete Q authority",
    )


def plan(ref, *, hash_weights):
    return dict(
        job=JOB.name,
        ref=ref,
        spec=data.SPEC,
        kind=JOB.kind,
        horizon=T,
        data=data.metadata(),
        documents=data.N,
        units=3 * data.N,
        targets_per_endpoint_cell=data.N * T,
        updates=0,
        endpoints={str(s): checkpoint(s, hash_weights=hash_weights)[1] for s in SEEDS},
    )


def freeze(job, ref):
    f = ROOT / JOB.name / "frozen.json"
    check(job == JOB and not f.exists(), "Q already frozen or wrong job")
    write_json(f, plan(ref, hash_weights=True))
    flush_dir(f.parent)
    return sha(f)


def finish(root):
    from .query_analysis import analyze

    check(root == ROOT, "wrong Q result root")
    write_json(root / "summary.json", analyze(root))
