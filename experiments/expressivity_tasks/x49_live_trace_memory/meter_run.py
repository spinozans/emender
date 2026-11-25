"""Meter the real 1x4096 X49 workload on fabricated bytes; never load the X49 natural stream."""

from __future__ import annotations
import gc
import json
import os
import platform
import random
import resource
import shutil
from time import perf_counter
import torch
from ..x48_muon_continuation.data import check, write_json
from ..x48_muon_continuation.train import cpu_seconds, restore, save
from .coupled import ORIGIN, cpu_origin, optimizers, step
from .endpoint import milestone, score_window, state_digest, sync

ARMS = ("A", "B", "F", "C")
SEED = 4901


def fixture(i):
    check(type(i) is int and 1 <= i <= 4, "four synthetic fixtures only")
    rng = random.Random(49100 + i)
    raw = bytearray(rng.randbytes(4097))
    pattern = b"\xffkey\x1e\x00synthetic.42=ab_"
    for k in range(9, 4097 - len(pattern), 43):
        raw[k : k + len(pattern)] = pattern
    return bytes(raw)


def arm_probe(folder, arm, host, head, opts, *, binding, report, width=256):
    """Three updates + one explicit replay. Timings and comparison are separate, no retry on mismatch."""
    dev = next(host.parameters()).device
    steps = []
    physical = 0
    journal = folder / "steps.jsonl"
    journal.touch(mode=0o600, exist_ok=False)

    def one(i):
        nonlocal physical
        sync(dev)
        c0 = cpu_seconds()
        tic = perf_counter()
        r = step(
            host,
            head,
            opts,
            fixture(i),
            mode=arm,
            seed=SEED,
            document=i,
            index=i,
            width=width,
        )
        sync(dev)
        wall = perf_counter() - tic
        c1 = cpu_seconds()
        physical += 1
        steps.append(
            dict(
                logical=i,
                physical=physical,
                wall_s=wall,
                cpu_s=[y - x for x, y in zip(c0, c1, strict=True)],
                **r,
            )
        )
        with journal.open("a") as f:
            f.write(json.dumps(steps[-1], allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        report(physical)
        return r

    def timed(fn, *args):
        sync(dev)
        start = perf_counter()
        v = fn(*args)
        sync(dev)
        return v, perf_counter() - start

    one(1)
    one(2)
    rec, save_s = timed(save, folder, binding, 2, host, head, opts)
    weights, weights_s = timed(milestone, folder, binding, 2, host, head)
    third = one(3)

    def identity():
        return state_digest(
            [host.state_dict(), head.state_dict(), *(o.state_dict() for o in opts)]
        )

    first, first_hash_s = timed(identity)
    restored, restore_s = timed(restore, folder, binding, 2, host, head, opts)
    check(restored == rec["sha256"], "foreign restore")
    again = one(3)
    second, second_hash_s = timed(identity)
    check(first == second and third == again, "replay differed at the actual successor")
    ev = score_window(
        host.eval(),
        head.eval(),
        fixture(4),
        arm=arm,
        seed=SEED,
        document=4,
        width=width,
    )
    return dict(
        arm=arm,
        updates=steps,
        checkpoint=rec,
        weights=weights,
        full_save_s=save_s,
        weights_save_s=weights_s,
        verify_restore_s=restore_s,
        state_hash_s=first_hash_s + second_hash_s,
        successor_state_sha=first,
        exact_successor=True,
        synthetic_inference=ev,
    )


def run():
    from .meter import authorize, RUN, fields, procstat, source_ref
    from ..x47_xsa_attention.workbench_run import check_gpu_owner

    # No device operations until admitted by the detached, matched broker owner.
    authorize()
    owner = check_gpu_owner()
    initial = fields(RUN / "initial")
    key = ("pid", "starttime", "gpu", "created")
    check(
        all(initial[k] == owner[k] for k in key)
        and procstat(int(owner["pid"]))[1] == "1",
        "meter lease mismatch",
    )
    check(
        os.getenv("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and os.getenv("OMP_NUM_THREADS") == "4"
        and all(
            os.getenv(k) == "expandable_segments:True"
            for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        ),
        "unregistered execution environment",
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    check(
        torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
        is True,
        "allocator not active",
    )
    torch.cuda.set_per_process_memory_fraction(0.80)
    import triton

    ref = source_ref()
    write_json(
        RUN / "identity.json",
        dict(
            ref=ref,
            origin=ORIGIN,
            pid=os.getpid(),
            trainer_start=procstat(os.getpid())[19],
            torch=str(torch.__version__),
            triton=triton.__version__,
            python=platform.python_version(),
            device=torch.cuda.get_device_name(),
            device_bytes=torch.cuda.get_device_properties(0).total_memory,
            owner=owner,
            natural_targets=0,
        ),
    )
    for rank, arm in enumerate(ARMS):
        check(
            shutil.disk_usage(RUN).free >= 4 << 30,
            "need 4 GiB free before arm artifacts",
        )
        folder = RUN / arm
        folder.mkdir()
        tic = perf_counter()
        c0 = cpu_seconds()
        torch.cuda.reset_peak_memory_stats()
        host, head = cpu_origin(SEED, arm)
        host = host.to("cuda")
        torch.set_num_threads(4)
        opts, _ = optimizers(host, head)
        sync(next(host.parameters()).device)
        setup = perf_counter() - tic
        check(
            all(not o.state and not o.aux.state for o in opts),
            "non-fresh optimizer moments",
        )

        def report(n):
            write_json(
                RUN / "progress.json",
                dict(
                    pid=os.getpid(),
                    arm=arm,
                    physical_updates=rank * 4 + n,
                    physical_synthetic_targets=(rank * 4 + n) * 4096,
                ),
            )

        bind = dict(
            schema="x49-synthetic-meter-v1", origin=ORIGIN, seed=SEED, ref=ref, arm=arm
        )
        receipt = arm_probe(folder, arm, host, head, opts, binding=bind, report=report)
        receipt.update(
            setup_s=setup,
            allocated_peak=torch.cuda.max_memory_allocated(),
            reserved_peak=torch.cuda.max_memory_reserved(),
            process_lifetime_peak_rss_KiB=resource.getrusage(
                resource.RUSAGE_SELF
            ).ru_maxrss,
            host_saved=dict(host.saved),
            arm_wall_s=perf_counter() - tic,
            arm_cpu_s=[y - x for x, y in zip(c0, cpu_seconds(), strict=True)],
        )
        write_json(folder / "completed.json", receipt)
        del host, head, opts
        gc.collect()
        torch.cuda.empty_cache()
    write_json(
        RUN / "complete.json",
        dict(
            arms=list(ARMS),
            physical_updates=16,
            natural_training_targets=0,
            natural_evaluation_targets=0,
        ),
    )
