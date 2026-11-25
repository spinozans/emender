"""Admission, initial endpoint freeze, and the all-seed FIT→PROBE boundary for X49-U."""

from __future__ import annotations
import json
import os
from time import monotonic, sleep
import numpy as np
from ..x48_muon_continuation.data import check, load, sha, write_json
from ..x48_muon_continuation.train import flush_dir
from . import surprise_data as data, query_panel as q
from .panels import SEEDS, T, OUT, fingerprint, digest
from .panel_spec import Job
from .surprise_fit import Moments, RIDGE
from .surprise import DIMS

ROOT = OUT / "surprise-probe-1"
TOTAL = 1800
JOB = Job("u-fit-probe", "fit_then_probe", TOTAL, "BNUS")
JOBS = {JOB.name: JOB}


def checkpoint(seed, *, hash_weights):
    result = data.checked_spec()
    path, e = q.checkpoint(seed, hash_weights=hash_weights)
    check(e["sha256"] == result["checkpoints"][str(seed)], "Q/W predictor mismatch")
    return path, e


def units():
    return len(SEEDS) * sum(data.N.values())


def prepared(ref):
    meta = data.metadata()
    check(meta["prepared_source"] == ref, "U preparation from a different source")
    return meta


def expected(ref):
    return dict(
        scope="x49-u-bound-arrival-mark-v1",
        spec=data.SPEC,
        qualified_ref=ref,
        q_result=data.Q_RESULT,
        checkpoints={
            str(s): checkpoint(s, hash_weights=False)[1]["sha256"] for s in SEEDS
        },
        data=prepared(ref),
        role_documents=dict(data.N),
        units=units(),
        fitted_scalars_per_seed=dict(DIMS),
        ridge=RIDGE,
        neural_updates=0,
        split="native_train_fit_and_disjoint_probe",
        total_card_seconds=TOTAL,
        job_caps={JOB.name: TOTAL},
        local_gpus=1,
        max_acquisitions=1,
        retries=0,
        remote=False,
    )


def validate(a, ref):
    e = expected(ref)
    check(
        type(a) is dict
        and set(a) == set(e) | {"operator_note"}
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip()
        and fingerprint({k: a[k] for k in e}) == fingerprint(e),
        "wrong or incomplete U authority",
    )


def plan(ref, *, hash_weights):
    return dict(
        job=JOB.name,
        kind=JOB.kind,
        ref=ref,
        spec=data.SPEC,
        horizon=T,
        data=prepared(ref),
        documents=dict(data.N),
        units=units(),
        ridge=RIDGE,
        neural_updates=0,
        endpoints={str(s): checkpoint(s, hash_weights=hash_weights)[1] for s in SEEDS},
    )


def freeze(job, ref):
    f = ROOT / JOB.name / "frozen.json"
    check(job == JOB and not f.exists(), "U plan already frozen or wrong stage")
    write_json(f, plan(ref, hash_weights=True))
    flush_dir(f.parent)
    return sha(f)


def journal(path, hs, *, sizes=None):
    counts = [T] * len(hs) if sizes is None else sizes
    raw = path.read_bytes()
    rows = list(map(json.loads, raw.splitlines()))
    check(
        [x["document_sha"] for x in rows] == hs
        and len(set(hs)) == len(hs)
        and len(counts) == len(hs)
        and all(
            x["host_prefills"] == 1 and x["targets"] == n
            for x, n in zip(rows, counts, strict=True)
        ),
        "U document order, exposure or prefill mismatch",
    )
    return digest(raw), rows


def checked_fit(state):
    check(
        state["documents"] == data.N["fit"]
        and state["targets"] == T * data.N["fit"]
        and state["ridge"] == RIDGE
        and set(state["gram"]) == set(state["rhs"]) == set(state["beta"]) == set(DIMS),
        "incomplete U fit",
    )
    a = Moments()
    a.documents = state["documents"]
    a.targets = state["targets"]
    for k, dim in DIMS.items():
        g = np.asarray(state["gram"][k], np.float64)
        r = np.asarray(state["rhs"][k], np.float64)
        b = np.asarray(state["beta"][k], np.float64)
        check(
            g.shape == (dim, dim)
            and r.shape == b.shape == (dim,)
            and np.isfinite(b).all(),
            "wrong fitted system",
        )
        a.gram[k] = g
        a.rhs[k] = r
    # Check the registered Newton solve from FIT moments only; do not refit/adapt to PROBE.
    val = a.freeze()
    check(
        all(
            np.allclose(val["beta"][k], state["beta"][k], rtol=1e-12, atol=1e-14)
            for k in DIMS
        ),
        "coefficients inconsistent with FIT moments",
    )


def readout_map(d, initial):
    f = d / "frozen.json"
    check(
        load(f) == initial and sha(f) == load(d / "start.json")["freeze_sha"],
        "foreign initial predictor freeze",
    )
    out = {}
    roles = data.checked_roles(
        data.bound(data.BASE / "reservation.json", initial["data"]["rows_sha"])
    )
    hs = [r["sha256"] for r in roles["fit"]]
    for s in SEEDS:
        fp = d / f"fit-{s}.json"
        raw = fp.read_bytes()
        z = json.loads(raw)
        dig, logs = journal(d / f"fit-{s}.jsonl", hs)
        check(
            z["seed"] == s
            and z["checkpoint_sha"] == initial["endpoints"][str(s)]["sha256"]
            and z["plan_sha"] == sha(f)
            and z["journal_sha"] == dig
            and z["fit_identity_sha"] == initial["data"]["fit_identity_sha"]
            and all(
                "nll_sums" not in x
                and 0 <= x["candidate_correct"] <= x["work"]["literal_hits"]
                for x in logs
            ),
            "foreign fit receipts",
        )
        checked_fit(z["state"])
        out[str(s)] = dict(
            receipt_sha=digest(raw),
            checkpoint_sha=z["checkpoint_sha"],
            beta=z["state"]["beta"],
        )
    return dict(
        schema="x49-u-fitted-readouts",
        spec=data.SPEC,
        plan_sha=sha(f),
        fit_identity_sha=initial["data"]["fit_identity_sha"],
        fit_units=len(SEEDS) * len(hs),
        seeds=out,
    )


def _fit_boundary(d):
    check(
        (d / "progress.json").is_file() and (d / "identity.json").is_file(),
        "no completed U FIT boundary",
    )
    n = len(SEEDS) * data.N["fit"]
    pr = load(d / "progress.json")
    ident = load(d / "identity.json")
    check(
        pr == dict(pid=os.getpid(), units=n, scored_targets=n * T)
        and ident["trainer"] == pr["pid"]
        and (d / "source-open-attempted").is_file()
        and not (d / "raw-probe-open-attempted").exists(),
        "incomplete or repeated U transition",
    )
    return n


def _witnessed(d, n):
    f = d / "survival.jsonl"
    if not f.is_file():
        return False
    # The independent observer may be writing the last line right now.
    raw = f.read_bytes().rpartition(b"\n")[0].splitlines()
    if not raw or not (d / "live.json").is_file():
        return False
    a = json.loads(raw[-1])
    live = load(d / "live.json")
    return (
        a["units"] == n
        and a["trainer"] == os.getpid()
        and a["ppid"] == live["ppid"] == 1
        and all(a[k] == live[k] for k in ("pid", "start", "trainer", "trainer_start"))
        and int(a["heartbeat"]) >= int(live["heartbeat"])
    )


def freeze_readouts(d, initial):
    f = d / "fit-frozen.json"
    n = _fit_boundary(d)
    check(not f.exists(), "fit map already frozen")
    out = readout_map(d, initial)
    limit = monotonic() + 10
    while not _witnessed(d, n):
        check(monotonic() < limit, "no independent terminal FIT witness")
        sleep(0.1)
    write_json(f, out)
    flush_dir(d)
    return sha(f)


def checked_readouts(d, initial, fit_sha):
    n = _fit_boundary(d)
    check(_witnessed(d, n), "PROBE lacks witnessed fit completion")
    z = data.bound(d / "fit-frozen.json", fit_sha)
    check(z == readout_map(d, initial), "changed readouts or incomplete fit freeze")
    return z


def finish(root):
    from .surprise_analysis import analyze

    check(root == ROOT, "wrong U closure root")
    write_json(root / "summary.json", analyze(root))
