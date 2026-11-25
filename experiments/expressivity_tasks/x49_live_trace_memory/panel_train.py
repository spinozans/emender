"""One full registered TRAIN trajectory, exclusively from the restored X48 direct endpoint."""

from __future__ import annotations
import json
import os
import resource
import shutil
from time import monotonic_ns
import torch
from ..x48_muon_continuation.train import cpu_seconds, flush_dir, save
from ..x48_muon_continuation.data import sha, write_json
from .coupled import cpu_origin, optimizers, step
from .endpoint import milestone, state_digest
from .panel_spec import STEPS, T, binding, check
from .train_stream import TrainStream


def append(path, event):
    with path.open("a") as f:
        f.write(json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


def train(job):
    from .panel import enter, progress

    return train_admitted(job, enter(job), progress, binding)


def train_admitted(
    job,
    ctx,
    progress,
    bind_fn,
    check_birth=None,
    check_first=None,
    *,
    stream_factory=None,
    origin=None,
):
    d, ref, t0 = ctx["folder"], ctx["ref"], ctx["t0"]
    check(job.kind == "train", "not a TRAIN job")
    arm, seed = job.arms, job.seed

    def wall():
        return (monotonic_ns() - t0) * 1e-9

    c0 = cpu_seconds()
    stream = (TrainStream if stream_factory is None else stream_factory)()
    host, head = (cpu_origin if origin is None else origin)(seed, arm)
    birth = dict(
        host=state_digest(host.state_dict()),
        literal=state_digest(getattr(head, "literal", head).state_dict()),
        head=state_digest(head.state_dict()),
    )
    if check_birth is not None:
        check_birth(birth)
    host = host.to("cuda")
    torch.set_num_threads(4)
    opts, routing = optimizers(host, head)
    check(all(not o.state and not o.aux.state for o in opts), "fresh moments required")
    bind = bind_fn(ref, arm, seed)
    write_json(
        d / "start-state.json",
        dict(binding=bind, birth=birth, groups=routing, setup_wall_s=wall()),
    )
    listed = []
    last = None
    work = {}
    loss_sum = 0.0
    for i in range(1, STEPS + 1):
        doc, w = stream.sample(seed, i)
        check(len(w) == T + 1, "not a full registered shift")
        r = step(host, head, opts, w, mode=arm, index=i, seed=seed, document=doc)
        torch.cuda.synchronize()
        check(r["targets"] == T and r["step"] == i, "wrong step")
        r["inclusive_wall_s"] = wall()
        loss_sum += r["preupdate_nll"] * T
        for k, v in r["work"].items():
            work[k] = work.get(k, 0) + v
        append(d / "steps.jsonl", r)
        if i == 1 and check_first is not None:
            check_first(r)
        if i % 32 == 0:
            check(
                shutil.disk_usage(d).free >= 4 << 30, "insufficient atomic write space"
            )
            if i % 128 == 0:
                last = save(d, bind, i, host, head, opts)
            m = milestone(d, bind, i, host, head)
            listed.append(
                dict(step=i, sha256=m["sha256"], bytes=m["bytes"], wall_s=wall())
            )
            append(d / "clock.jsonl", listed[-1])
            flush_dir(d)
        progress(d, i, i * T)
    check(
        last and last["step"] == STEPS and len(listed) == 32,
        "invalid terminal checkpoint",
    )
    return dict(
        kind="train",
        binding=bind,
        units=STEPS,
        targets=STEPS * T,
        preupdate_train_nll=loss_sum / (STEPS * T),
        work=work,
        steps_sha256=sha(d / "steps.jsonl"),
        clock_sha256=sha(d / "clock.jsonl"),
        final_recovery=last["sha256"],
        final_weights=listed[-1]["sha256"],
        gpu_peak_alloc=torch.cuda.max_memory_allocated(),
        gpu_peak_reserved=torch.cuda.max_memory_reserved(),
        process_RSS_peak_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_core_s=[y - x for x, y in zip(c0, cpu_seconds(), strict=True)],
    )
