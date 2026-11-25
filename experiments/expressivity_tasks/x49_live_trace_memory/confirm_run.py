"""X49-U frozen native final. Inference only, one local acquisition, no training/repair/retry phase."""

from __future__ import annotations
import os
import resource
import shutil
from time import monotonic
import torch
from ..x48_muon_continuation.data import check, load, sha, write_json
from ..x48_muon_continuation.train import cpu_seconds
from . import (
    confirm_data as data,
    confirm_panel as p,
    panel,
    followup_loop as loop,
    surprise_run as u,
)
from .meter import MODULE
from .panels import OUT, SEEDS, digest, fingerprint
from .panel_eval import _documents
from .panel_train import append
from .surprise_fit import short_frozen_document
from .short_phase import study as phase

MOD = MODULE + ".confirm_run"
AUTH = OUT / "confirm-authorization-1.json"
GATES = 100


def source():
    ref = u.source()
    p.endpoints(hash_weights=False)
    return ref


def qualification():
    return u.qualify_frozen(
        source,
        "confirm",
        data.SPEC,
        GATES,
        "Frozen U final: invented short documents, receipt boundaries, strict loads and bound pre-PROBE coefficients; no natural final opened",
    )


def checked_fixture_ref():
    return u.checked_gate(source, "confirm", data.SPEC, GATES)


def authority():
    check(
        AUTH.is_file() and shutil.disk_usage(OUT).free >= 32 << 30,
        "no separate short confirmation budget or insufficient disk",
    )
    ref = checked_fixture_ref()
    p.validate(load(AUTH), ref)
    return ref


def admit(job):
    check(job == p.JOB, "wrong confirmation stage")
    return loop.admit(job, p, authority)


def check_shapes(d, plan_sha, host, head, beta):
    f = d / "shape-check.json"
    check(
        not f.exists() and not (d / "source-open-attempted").exists(),
        "late or repeated shape check",
    )
    fixture = b"old:\0abc\xffabc\x1e " * 350
    for length in p.LENGTHS:
        x = short_frozen_document(host, head, fixture[: length + 1], beta=beta)
        check(
            x["targets"] == length
            and x["host_prefills"] == 1
            and x["work"]["mark_storage_bytes"] == length
            and set(x["nll_sums"]) == set("BNUS"),
            "short shape refusal",
        )
    write_json(
        f,
        dict(
            plan_sha=plan_sha,
            trainer=os.getpid(),
            trainer_start=load(d / "identity.json")["trainer_start"],
            seed=SEEDS[0],
            lengths=p.LENGTHS,
            natural_targets=0,
            synthetic_prefills=len(p.LENGTHS),
        ),
    )


def scorer(seed, e):
    return lambda host, head, word, rank: short_frozen_document(
        host, head, word, beta=e["beta"]
    )


def evaluate(job, ctx, *, study="UC"):
    data, p, runner, _ = phase(study)
    d = ctx["folder"]
    h = load(d / "start.json")["freeze_sha"]
    plan = data.u.bound(d / "frozen.json", h)
    check(
        job == p.JOB
        and plan["ref"] == ctx["ref"]
        and plan["new_fits"] == plan["neural_updates"] == 0
        and plan["fit_freeze_sha"] == data.FIT_SHA,
        "foreign final scope",
    )
    c0 = cpu_seconds()
    n = 0
    nt = 0
    cost = {}
    journals = {}
    for seed in SEEDS:
        k = str(seed)
        e = plan["endpoints"][k]
        tic = monotonic()
        host, head = u.model(seed, e["weights"])
        one = runner.scorer(seed, e)
        cost["load_" + k + "_s"] = monotonic() - tic
        if seed == SEEDS[0]:
            tic = monotonic()
            runner.check_shapes(d, h, host, head, e["beta"])
            cost["synthetic_short_shapes_s"] = monotonic() - tic
            tic = monotonic()
            wins, nread = _documents(job, plan)
            rows = [
                dict(
                    document_sha=doc,
                    inputs=digest(w[:-1]),
                    targets=digest(w[1:]),
                    target_count=len(w) - 1,
                )
                for _, doc, w in wins
            ]
            write_json(d / "shifts.json", rows)
            total = sum(x["target_count"] for x in rows)
            check(
                total == plan["targets_per_endpoint"]
                and nread == total + len(rows)
                and len(wins) * len(SEEDS) == plan["units"],
                "wrong short-final shifts",
            )
            write_json(
                d / "stream.json",
                dict(
                    windows=len(wins),
                    source_bytes=nread,
                    targets_per_endpoint=total,
                    shifts_sha=sha(d / "shifts.json"),
                    freeze_sha=h,
                ),
            )
            cost["final_stream_s"] = monotonic() - tic
        f = d / f"{seed}.jsonl"
        f.touch(mode=0o600, exist_ok=False)
        for i, (_, doc, w) in enumerate(wins):
            x = one(host, head, w, i)
            check(
                x["host_prefills"] == 1 and x["targets"] == len(w) - 1,
                "lost short input or target",
            )
            append(f, dict(document_sha=doc, **x))
            n += 1
            nt += x["targets"]
            panel.progress(d, n, nt, root=p.ROOT)
        journals[k] = dict(scores_sha=sha(f), endpoint_id=fingerprint(e))
        del host, head
        torch.cuda.empty_cache()
    check(
        n == plan["units"] and nt == len(SEEDS) * plan["targets_per_endpoint"],
        "incomplete natural final",
    )
    return dict(
        kind=job.kind,
        freeze_sha=h,
        shape_sha=sha(d / "shape-check.json"),
        stream_sha=sha(d / "stream.json"),
        new_fits=0,
        neural_updates=0,
        units=n,
        targets=nt,
        stage_seconds=cost,
        endpoints=journals,
        cpu_core_s=[b - a for a, b in zip(c0, cpu_seconds(), strict=True)],
        gpu_peak_alloc=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved=torch.cuda.max_memory_reserved(),
        process_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


def inside(job):
    ctx = panel.enter_admitted(job, *admit(job))
    loop.complete(ctx["folder"], evaluate(job, ctx))


if __name__ == "__main__":
    loop.cli(p, MOD, authority, qualification, inside, 32, targets_for=p.targets_for)
