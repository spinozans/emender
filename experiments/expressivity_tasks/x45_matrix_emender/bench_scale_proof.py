"""Scale proof: is the fused matrix emender fast at all three scales, honestly?

The K4 receipt flagged rank 512 as trajectory-bandwidth-bound and "still 2.2x over
eager"; this bench settles whether the fused path is fast at the scales that matter,
with a same-session comparison at every level and the measurement boundary stated on
every row.

Three scales, three boundaries (``RESULT_SCALE_PROOF.md`` is the narrative):

``--scale 1``  the block call.  Fused forward+backward (and forward-only) at rank
               64/128/512, each at b8/T2048 AND at a second batch/length
               (b16/T2048, b8/T4096), against the eager path **re-measured in the
               same session** (never the old receipts).  Two baselines with different
               boundaries, both stated: the scan-block call (the K4 boundary: state
               in, recorded schedule in, reads + four gradients out, fp32) and the
               whole mixer block (``bench_blocks``' boundary: projections + scan +
               backward of ``out.pow(2).sum()``, eager, no compile, no graph), the
               latter matched against the GDN-2 block at the same shapes -- the
               same-session GDN2 baseline gate D used, with its autotune pin
               recorded (``NDM_PIN_TRITON_AUTOTUNE=0``: the pinned config OOMs
               shared memory at the rank-128 shape).

``--scale 2``  the model training step.  The x45 model at its registered shape
               (width 576 / depth 24 / chunk 64 / ~150 M parameters, embedding +
               head + AdamW + clip included), fused ON vs OFF, on the same card,
               >=5 steady steps after warmup, per-step wall and device time.

``--scale 3``  training throughput.  A short matched-window run of the
               ``matrix_ops`` arm on the gate-D harness protocol (64 flagship-stream
               windows/step, bf16 compute, fused CE, AdamW), fused path enabled,
               against the same arm's registered path in the same session and the
               RECORDED gate-D receipts (gdn2 28,124 / matrix_ops 13,190 --
               labelled RECORDED, not same-session).

The fused path is the K4 flag wired into the model (``MatrixMixer(fused_backward=
True)`` -> ``routed_chunk_scan(trainable_fused=True)`` -> one
``trainable_block_scan(fused=True)`` autograd node per chunk): the routing loop,
the recorded schedule and every gradient path are unchanged -- the read weights and
payloads stay attached exactly as the registered path attaches them, so the fused
node replaces the chunk's forward+backward and nothing else.  Default remains off.

Usage (one process per scale/config so a blown budget cannot lose a receipt)::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    .venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.bench_scale_proof \\
        --scale 1 --out experiments/expressivity_tasks/x45_matrix_emender/receipts/scale_proof_blocks.json
    .venv/bin/python -m ...bench_scale_proof --scale 2 --config off \\
        --out .../receipts/scale_proof_model_step.json
    .venv/bin/python -m ...bench_scale_proof --scale 3 --config fused_mb4 \\
        --out .../receipts/scale_proof_training.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import empty_schedule
from experiments.expressivity_tasks.x45_matrix_emender.bench_blocks import (
    measure as measure_whole_block)
from experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free import (
    head_major, kernel_count, timed)
from experiments.expressivity_tasks.x45_matrix_emender.bench_k4_fused_backward import (
    attached, eager_step, fused_step)
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward import (
    eager_block_scan, persistent_scan_trajectory, trainable_block_scan)
from experiments.expressivity_tasks.x45_matrix_emender.model import (
    MatrixMixer, MatrixShape, matrix_config)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import chunk_scan

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
DEVICE = "cuda"
HEADS, VALUE_DIM, LANES = 2, 128, 2
ROUTE_CHUNK_TOKENS = 128
REPEATS = 5
# The registered gate-D tok/s receipts these runs are compared against (RECORDED,
# not same-session): receipts/cd_gdn2_c128_rerun.json and receipts/cd_matrix_*_c128.json.
RECORDED = {
    "gdn2_tokens_per_s": 28124.0,
    "matrix_ops_tokens_per_s": 13189.7,
    "matrix_append_tokens_per_s": 15855.9,
    "source": ("receipts/cd_gdn2_c128_rerun.json, receipts/cd_matrix_ops_c128.json, "
               "receipts/cd_matrix_append_c128.json -- RECORDED, not same-session"),
}


def session_meta() -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2**30, 2),
                torch=torch.__version__, triton=__import__("triton").__version__)


def dump(out_path: str, report: dict) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def accumulate(out_path: str, key: str, payload: dict) -> dict:
    """Merge one config's result into a shared receipt so partial progress survives."""
    path = Path(out_path)
    data = {}
    if path.exists():
        data = json.loads(path.read_text())
    data[key] = payload
    dump(out_path, data)
    return data


# --------------------------------------------------------------------- scale 1


def record_window(policy: RoutingPolicy, batch: int, tokens: int, rank: int,
                  seed: int = 3401):
    """The arm's own routing for one whole window, recorded once (real ``route_chunk``).

    ``bench_hostloop_free.recorded_block`` generalised to any batch/tokens/rank: the
    schedule and read weights are recorded against the eager chunk scan's states, so
    no measured path pays for routing inside its boundary and every path consumes
    one identical decision channel.
    """
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, HEADS, VALUE_DIM, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(batch, tokens, HEADS, VALUE_DIM, generator=generator).to(DEVICE)
    decay = torch.full((batch, tokens, HEADS), math.exp(-1.0 / 64.0), device=DEVICE)
    state = torch.zeros(batch, HEADS, rank, VALUE_DIM, device=DEVICE)
    live = torch.zeros(batch, HEADS, rank, dtype=torch.bool, device=DEVICE)
    schedule = empty_schedule(batch, tokens, HEADS, VALUE_DIM, lanes=LANES, device=DEVICE)
    weights = []
    for index, start in enumerate(range(0, tokens, ROUTE_CHUNK_TOKENS)):
        stop = min(start + ROUTE_CHUNK_TOKENS, tokens)
        piece, chunk_weights = route_chunk(
            index, start, state, live,
            query[:, start:stop].permute(0, 2, 1, 3),
            payload[:, start:stop].permute(0, 2, 1, 3), policy, rank)
        schedule.slot[:, start:stop] = piece.slot
        schedule.erase[:, start:stop] = piece.erase
        schedule.payload[:, start:stop] = piece.payload
        schedule.op[:, start:stop] = piece.op
        schedule.live_after[:, start:stop] = piece.live_after
        weights.append(chunk_weights)
        _, state, live = chunk_scan(state, live, chunk_weights,
                                     decay[:, start:stop].permute(0, 2, 1), piece,
                                     fused=False)
    entry_state = torch.zeros_like(state)
    entry_live = torch.zeros_like(live)
    return (entry_state, entry_live, torch.cat(weights, dim=2), head_major(decay),
            schedule)


def measure_scan_case(rank: int, batch: int, tokens: int, repeats: int,
                      seed: int = 3401) -> dict:
    """The K4 boundary at one shape: fused vs eager, forward and forward+backward."""
    state, live, weights, decay, schedule = record_window(
        RoutingPolicy(ops=True), batch, tokens, rank, seed)
    dreads = torch.randn(batch, HEADS, tokens, VALUE_DIM, device=DEVICE)
    dexit = torch.randn(batch, HEADS, rank, VALUE_DIM, device=DEVICE)
    label = f"r{rank}_b{batch}_T{tokens}"

    trajectory_bytes = batch * HEADS * tokens * rank * VALUE_DIM * 4
    row = dict(
        boundary=("one block call: state+liveness in, the recorded schedule and read "
                  "weights as inputs; reads and exit state out, and (for the step "
                  "rows) the four gradients back.  Routing outside, as in K4."),
        precision="fp32", session="same-session",
        shape=dict(batch=batch, tokens=tokens, heads=HEADS, rank=rank,
                   value_dim=VALUE_DIM, lanes=LANES),
        trajectory_bytes=trajectory_bytes,
        trajectory_gib=round(trajectory_bytes / 2**30, 2))

    s, w, d, p, sch = attached(state, live, weights, decay, schedule)
    fused_call = lambda: fused_step(s, live, w, d, sch, dreads, dexit)  # noqa: E731
    row["fused"] = dict(
        form="trajectory forward + reverse walk (fused_block_scan)",
        step=timed(fused_call, repeats, label=f"{label}/fused"),
        kernels=kernel_count(fused_call),
        forward_only=dict(
            kernels=kernel_count(lambda: torch.no_grad()(persistent_scan_trajectory)(
                state, live, weights, decay, schedule)),
            timing=timed(lambda: torch.no_grad()(persistent_scan_trajectory)(
                state, live, weights, decay, schedule), repeats,
                label=f"{label}/fused/fwd")))

    for chunk in (128, 64):
        es, ew, ed, ep, esch = attached(state, live, weights, decay, schedule)
        eager_call = (lambda chunk=chunk, es=es, ew=ew, ed=ed, esch=esch:
                      eager_step(es, live, ew, ed, esch, chunk, dreads, dexit))
        row[f"eager_C{chunk}"] = dict(
            form=f"autograd through the eager chunk scan, chunk {chunk}",
            step=timed(eager_call, repeats, label=f"{label}/eager_C{chunk}"),
            kernels=kernel_count(eager_call),
            forward_only=dict(
                kernels=kernel_count(
                    lambda chunk=chunk: torch.no_grad()(eager_block_scan)(
                        state, live, weights, decay, schedule, chunk_tokens=chunk)),
                timing=timed(
                    lambda chunk=chunk: torch.no_grad()(eager_block_scan)(
                        state, live, weights, decay, schedule, chunk_tokens=chunk),
                    repeats, label=f"{label}/eager_C{chunk}/fwd")))
    return row


def measure_whole_blocks(batch: int, tokens: int, ranks, reps: int, seed: int) -> dict:
    """The bench_blocks boundary at one shape: the matrix mixer block vs GDN-2's.

    Boundary (bench_blocks' own, unchanged): ONE whole mixer block, forward +
    backward of ``out.pow(2).sum()``, eager, no compile, no graph, fp32 weights;
    GDN-2 through its external Triton chunk kernel.  NOT the scan-block boundary of
    the rows above and NOT comparable to whole-arm tokens/s.
    """
    torch.manual_seed(seed)
    x = torch.randn(batch, tokens, matrix_config().width, device=DEVICE)
    rows = dict(boundary=("one whole mixer block (projections + scan + MLP-free mixer "
                          "arithmetic), forward+backward of out.pow(2).sum(), eager, "
                          "no torch.compile, no CUDA graph; GDN-2 external Triton "
                          "chunk kernel with NDM_PIN_TRITON_AUTOTUNE=0 (the pinned "
                          "config OOMs shared memory at the rank-128 shape)"),
        precision="fp32 (bench_blocks protocol)", session="same-session",
        shape=dict(batch=batch, tokens=tokens, width=matrix_config().width,
                   heads=HEADS, value_dim=VALUE_DIM))
    for rank in ranks:
        shape = MatrixShape(width=matrix_config().width, depth=24,
                            hidden=matrix_config().hidden, heads=HEADS,
                            value_dim=VALUE_DIM, rank=rank, rank_max=rank)
        for name, fused_backward in (("matrix_ops_block_eager", False),
                                     ("matrix_ops_block_fused", True)):
            block = MatrixMixer(shape, RoutingPolicy(ops=True), ROUTE_CHUNK_TOKENS,
                                compile_scan=False,
                                fused_backward=fused_backward).cuda()
            rows[f"{name}_r{rank}"] = measure_whole_block(block, x, reps)
            rows[f"{name}_r{rank}"]["fused_backward"] = fused_backward
            rows[f"{name}_r{rank}"]["state_per_head"] = [rank, VALUE_DIM]
            del block
            torch.cuda.empty_cache()
        try:
            from ndm.models.external_gdn2 import GDN2ExternalLayer
            # head_dim = the store's rank; expansion lifts GDN-2's value head dim to
            # 128 so its state is rank x d_v per head, the axis the registration
            # calls capacity (2.0 at rank 64, 1.0 at rank 128).
            expansion = round(VALUE_DIM / rank, 3) if rank <= VALUE_DIM else 1.0
            block = GDN2ExternalLayer(dim=matrix_config().width, expansion=expansion,
                                     head_dim=rank, num_heads=HEADS, use_conv=True,
                                     d_conv=4).cuda().train()
            measured = measure_whole_block(block, x, reps)
            measured["state_per_head"] = [rank, VALUE_DIM]
            measured["expansion"] = expansion
            measured["autotune_pin"] = "NDM_PIN_TRITON_AUTOTUNE=0 (required: the pin OOMs shared memory at this shape)"
            rows[f"gdn2_block_r{rank}"] = measured
            del block
            torch.cuda.empty_cache()
        except Exception as error:                              # pragma: no cover
            rows[f"gdn2_block_r{rank}"] = dict(
                status="NOT MEASURED", reason=repr(error)[:400])
    return rows


def scale1(args) -> dict:
    # Required before the GDN-2 import: the pinned autotune config OOMs shared
    # memory at the rank-128 shape (the gate-D block bench's own note).
    os.environ["NDM_PIN_TRITON_AUTOTUNE"] = "0"
    torch.set_num_threads(1)
    scan_rows = {}
    for rank in (64, 128, 512):
        for batch, tokens in ((8, 2048), (16, 2048), (8, 4096)):
            key = f"rank{rank}_b{batch}_T{tokens}"
            print(f"[scale1] measuring {key} ...", flush=True)
            scan_rows[key] = measure_scan_case(rank, batch, tokens, args.repeats,
                                               args.seed)
            dump_partial(args.out, "scan_block_calls", scan_rows)
    block_rows = {}
    for batch, tokens in ((8, 2048), (16, 2048), (8, 4096)):
        print(f"[scale1] whole-block baselines b{batch}/T{tokens} ...", flush=True)
        block_rows[f"b{batch}_T{tokens}"] = measure_whole_blocks(
            batch, tokens, (64, 128, 512), reps=3, seed=args.seed)
        dump_partial(args.out, "whole_block_baselines", block_rows)
    report = dict(
        gate="scale-proof 1: the block call",
        session=session_meta(),
        scan_block_calls=scan_rows,
        whole_block_baselines=block_rows,
        notes=[
            "every eager and GDN2 number in this receipt was measured in this "
            "session on this card; no recorded receipt is used as a comparison",
            "the matrix scan rows and the whole-block rows are DIFFERENT "
            "boundaries (stated per row group) and are not comparable to each "
            "other or to whole-arm tokens/s",
            "GDN-2's kernel refuses head_dim > 256, so it cannot express the "
            "rank-512 store at all; the rank-64 row runs at expansion 2.0 so its "
            "state is 64 x 128 per head like the matrix store's",
        ])
    dump(args.out, report)
    return report


def dump_partial(out_path: str, key: str, payload: dict) -> None:
    path = Path(out_path)
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            data = {}
    data.setdefault("_partial", {})[key] = payload
    dump(out_path, data)


# ------------------------------------------------------------------ scales 2 & 3


SCALE2_CONFIGS = {
    "off": dict(fused_backward=False, compile_scan=True, microbatch=8,
                description=("registered path: torch.compile(mode=default) on the one "
                             "shared scan, autograd through the compiled eager chunk "
                             "arithmetic")),
    "on_mb8": dict(fused_backward=True, compile_scan=False, microbatch=8,
                   description=("fused path, registered microbatch: one K4 autograd "
                                "node per chunk, routing uncompiled (torch.compile "
                                "cannot host the fused node: dynamo/inductor refuse, "
                                "receipted)")),
    "off_mb4": dict(fused_backward=False, compile_scan=True, microbatch=4,
                    description="registered path at microbatch 4 (the fused path's memory-matched twin)"),
    "on_mb4": dict(fused_backward=True, compile_scan=False, microbatch=4,
                   description="fused path at microbatch 4 (the largest power of two whose trajectory store fits the card)"),
    "off_mb2": dict(fused_backward=False, compile_scan=True, microbatch=2,
                    description="registered path at microbatch 2 (the largest microbatch at which the fused path's trajectory store fits the card)"),
    "on_mb2": dict(fused_backward=True, compile_scan=False, microbatch=2,
                   description="fused path at microbatch 2 -- the largest microbatch whose 24-layer trajectory store fits the card (mb 8 and 4 OOM, receipted)"),
}


def train_run(label: str, chunk_tokens: int, compile_scan: bool, fused_backward: bool,
              microbatch: int, warmup: int, steps: int, min_seconds: float | None,
              description: str = "", seed: int = 3401, profile_mode: str = "full",
              profile_steps: int = 1) -> dict:
    """One gate-D-protocol training run: the same loop, phases and stream as gates.py.

    ``profile_mode``: ``full`` is the gates/cd protocol (CPU+CUDA activities across
    every steady step, aggregated with ``key_averages``) -- honest but its
    post-processing is quadratic-ish in the event count, which the microbatch-2
    eager runs make enormous.  ``cuda`` profiles only the first ``profile_steps``
    steady steps with CUDA activity alone and aggregates with one pass over the
    event list (busy device ms, kernel count, top kernels) -- the step walls of
    the unprofiled steps stay clean.  ``off`` is walls only.
    """
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
        configure_production)
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        model_for as matrix_model_for, parameter_count)

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    shape = matrix_config()
    policy = RoutingPolicy(ops=True)
    model = matrix_model_for(shape, policy, chunk_tokens, compile_scan=compile_scan,
                             fused_backward=fused_backward).cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=5.42e-4, betas=(0.9, 0.9392004369596886),
                                  weight_decay=0.20059436743465126, fused=True)
    batches = iter(stream.batches("train"))

    def one_step(rows, timed):
        # gates.py's own phase protocol: prepare = host-side window grouping,
        # fwd_bwd = every microbatch's forward AND backward, opt = clip + AdamW;
        # the flagship-stream read is outside every phase.
        count = sum(len(r[3]) for r in rows)
        optimizer.zero_grad(set_to_none=True)
        phases = {"prepare": 0.0, "fwd_bwd": 0.0, "opt": 0.0}
        if timed:
            torch.cuda.synchronize()
            mark = time.monotonic()
        groups = list(grouped(rows, microbatch))
        if timed:
            torch.cuda.synchronize()
            phases["prepare"] = time.monotonic() - mark
            mark = time.monotonic()
        loss_sum = 0.0
        for x, y, valid in groups:
            torch.compiler.cudagraph_mark_step_begin()
            loss = model.loss(x, y)
            (loss * (valid / count)).backward()
            loss_sum += float(loss.detach()) * valid / count
        if timed:
            torch.cuda.synchronize()
            phases["fwd_bwd"] = time.monotonic() - mark
            mark = time.monotonic()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                       error_if_nonfinite=True)
        optimizer.step()
        if timed:
            torch.cuda.synchronize()
            phases["opt"] = time.monotonic() - mark
        return count, loss_sum, phases

    grad_probe = {}
    oom = None
    records = []
    profiled_steps = 0
    if profile_mode == "full":
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA,
                        torch.profiler.ProfilerActivity.CPU],
            record_shapes=False, profile_memory=False)
    elif profile_mode == "cuda":
        profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, profile_memory=False)
    else:
        profiler = None
    try:
        began_warmup = time.monotonic()
        for _ in range(warmup):
            one_step(next(batches), False)
        print(f"[{label}] warmup done in {time.monotonic() - began_warmup:.0f} s",
              flush=True)
        # Every parameter must carry a finite, non-zero gradient on the fused path
        # (the read weights and payloads stay attached, so nothing is silently cut).
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad = param.grad
                grad_probe[name] = dict(
                    present=grad is not None,
                    finite=bool(grad is None or torch.isfinite(grad).all()),
                    nonzero=bool(grad is None or float(grad.abs().max()) > 0.0))
        torch.cuda.reset_peak_memory_stats()
        if profiler is not None and profile_mode == "full":
            profiler.start()
        began = time.monotonic()
        while len(records) < steps and (min_seconds is None
                                        or time.monotonic() - began < min_seconds):
            # In cuda mode only the first `profile_steps` steady steps carry the
            # profiler, so the rest of the step walls stay clean.
            if profiler is not None and profile_mode == "cuda" and profiled_steps == 0:
                profiler.start()
            count, loss, phases = one_step(next(batches), True)
            records.append(dict(
                tokens=count, loss=round(loss, 6),
                wall_s=round(sum(phases.values()), 3),
                phases={key: round(value, 3) for key, value in phases.items()}))
            print(f"[{label}] step {len(records)}: {records[-1]['wall_s']:.1f} s "
                  f"(fwd_bwd {records[-1]['phases']['fwd_bwd']:.1f} s)", flush=True)
            if profiler is not None and profile_mode == "cuda" and profiled_steps == 0:
                profiler.stop()
                profiled_steps = 1
        if profiler is not None and profile_mode == "full":
            profiler.stop()
    except torch.cuda.OutOfMemoryError as error:
        oom = dict(error=repr(error)[:400],
                   note=("the fused path stores the per-chunk state trajectory for "
                         "the backward; 24 layers of it exceed the card at this "
                         "microbatch"))
        try:
            if profiler is not None:
                profiler.stop()
        except Exception:                                   # pragma: no cover
            pass

    total_s = sum(record["wall_s"] for record in records)
    tokens = sum(record["tokens"] for record in records)
    steady = records[2:] if len(records) > 3 else records
    steady_s = sum(record["wall_s"] for record in steady)
    steady_tokens = sum(record["tokens"] for record in steady)
    phases = {key: round(sum(record["phases"][key] for record in records), 3)
              for key in ("prepare", "fwd_bwd", "opt")}
    kernels = []
    device_busy_ms = None
    kernel_count_per_step = None
    if not oom and profiler is not None:
        if profile_mode == "full":
            for event in profiler.key_averages():
                device = getattr(event, "self_device_time_total", 0) or 0
                if device > 0:
                    kernels.append(dict(name=event.key[:110],
                                        cuda_ms=round(device / 1000, 3),
                                        calls=event.count))
        else:
            # One pass over the raw event list: the microbatch-2 eager runs produce
            # millions of events and `key_averages` does not finish in a session.
            busy_us = 0.0
            per_name: dict[str, list] = {}
            events = 0
            for event in profiler.events():
                if event.device_type != torch.autograd.DeviceType.CUDA:
                    continue
                device = getattr(event, "self_device_time_total", 0) or 0
                busy_us += device
                events += 1
                row = per_name.setdefault(event.name, [0.0, 0])
                row[0] += device
                row[1] += 1
            divisor = max(1, profiled_steps)
            device_busy_ms = round(busy_us / 1000.0 / divisor, 1)
            kernel_count_per_step = round(events / divisor, 1)
            kernels = [dict(name=name[:110], cuda_ms=round(busy / 1000.0 / divisor, 3),
                            calls=round(calls / divisor, 1))
                       for name, (busy, calls) in per_name.items()]
        kernels.sort(key=lambda row: -row["cuda_ms"])
        device_total = sum(row["cuda_ms"] for row in kernels) or 1.0
        for row in kernels:
            row["cuda_frac"] = round(row["cuda_ms"] / device_total, 4)
    forbidden = sorted({name for name in
                        ("replaybackward", "reverse_kernel", "replay", "scan_kernel")
                        if any(name in row["name"].lower() for row in kernels)})
    if profiler is None:
        forbidden = ["not checked: profile off (per-step walls only)"]
    report = dict(
        label=label,
        config=dict(fused_backward=fused_backward, compile_scan=compile_scan,
                    microbatch=microbatch, chunk_tokens=chunk_tokens,
                    description=description),
        boundary=("whole model, whole step: prepare + all microbatches' forward and "
                  "backward + clip/AdamW; flagship-stream read excluded; tokens are "
                  "measured targets over the same 64-window stream batches for every "
                  "arm (the gates.py/cd_* protocol)"),
        precision="bf16 compute (autocast), fp32 master weights, fused CE -- the registered regime",
        session="same-session (this card, this process)",
        parameters=parameters,
        summary=dict(shape=shape.__dict__, parameters=parameter_count(shape),
                     routing=policy.record(), chunk_tokens=chunk_tokens),
        microbatches=microbatch, windows_per_step=64, context=2048,
        warmup=warmup, steps=len(records), tokens=tokens,
        step_walls=[record["wall_s"] for record in records],
        phase_s=phases,
        phase_frac={key: (round(value / total_s, 4) if total_s else 0.0)
                    for key, value in phases.items()},
        total_s=round(total_s, 3),
        tokens_per_s=(round(tokens / total_s, 1) if total_s else None),
        tokens_per_s_steady=(round(steady_tokens / steady_s, 1) if steady_s else None),
        ms_per_step_p50=(round(1000 * statistics.median(
            [record["wall_s"] for record in records]), 1) if records else None),
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        losses=[record["loss"] for record in records],
        grad_probe=dict(finite=all(v["finite"] for v in grad_probe.values()),
                        nonzero=all(v["nonzero"] for v in grad_probe.values()),
                        parameters_probed=len(grad_probe)),
        forbidden_kernels_found=forbidden,
        replay_absent=(None if profiler is None else not forbidden),
        top_kernels=kernels[:20],
        cuda_total_ms=(round(sum(row["cuda_ms"] for row in kernels), 1)
                       if kernels else None),
        device_busy_ms_per_step=device_busy_ms,
        kernel_count_per_step=kernel_count_per_step,
        profile_mode=profile_mode,
        profiled_steps=profiled_steps,
        recorded_references=RECORDED,
        oom=oom,
        seed=seed)
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def scale2(args) -> dict:
    config = SCALE2_CONFIGS[args.config]
    print(f"[scale2] {args.config}: {config['description']}", flush=True)
    report = train_run(args.config, chunk_tokens=64, compile_scan=config["compile_scan"],
                       fused_backward=config["fused_backward"],
                       microbatch=config["microbatch"], warmup=args.warmup,
                       steps=args.steps, min_seconds=args.min_seconds,
                       description=config["description"], seed=args.seed,
                       profile_mode=args.profile_mode,
                       profile_steps=args.profile_steps)
    report["scale"] = 2
    report["shape_note"] = ("the registered shape: width 576, depth 24, chunk 64, "
                            "~150 M parameters, embedding + head + optimizer included")
    data = accumulate(args.out, args.config, report)
    return data


SCALE3_CONFIGS = {
    "registered_mb8": dict(fused_backward=False, compile_scan=True, microbatch=8,
                           description=("the registered matrix_ops arm on the cd_* "
                                       "protocol (chunk 128, torch.compile on the "
                                       "one shared scan)")),
    "registered_mb2": dict(fused_backward=False, compile_scan=True, microbatch=2,
                           description="the registered arm at microbatch 2 (the fused run's memory-matched twin)"),
    "fused_mb8": dict(fused_backward=True, compile_scan=False, microbatch=8,
                      description="the fused matrix_ops arm at the registered microbatch"),
    "fused_mb4": dict(fused_backward=True, compile_scan=False, microbatch=4,
                      description="the fused matrix_ops arm at microbatch 4"),
    "fused_mb2": dict(fused_backward=True, compile_scan=False, microbatch=2,
                      description="the fused matrix_ops arm at microbatch 2 -- the largest microbatch whose 24-layer trajectory store fits the card (mb 8 and 4 OOM, receipted)"),
}


def scale3(args) -> dict:
    config = SCALE3_CONFIGS[args.config]
    print(f"[scale3] {args.config}: {config['description']}", flush=True)
    report = train_run(args.config, chunk_tokens=128,
                       compile_scan=config["compile_scan"],
                       fused_backward=config["fused_backward"],
                       microbatch=config["microbatch"], warmup=args.warmup,
                       steps=args.steps, min_seconds=args.min_seconds,
                       description=config["description"], seed=args.seed,
                       profile_mode=args.profile_mode,
                       profile_steps=args.profile_steps)
    report["scale"] = 3
    report["shape_note"] = ("the cd_* protocol: chunk 128, 64 flagship windows/step, "
                            "microbatch as stated per config; compared against the "
                            "RECORDED gate-D receipts (gdn2 28,124 / matrix_ops "
                            "13,190) and the same-session runs in this receipt")
    data = accumulate(args.out, args.config, report)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--config", default=None,
                        help="scale 2: off|on_mb8|off_mb4|on_mb4; scale 3: registered_mb8|registered_mb4|fused_mb8|fused_mb4")
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=8,
                        help="steady steps (scale 2 default 8; scale 3 defaults 60)")
    parser.add_argument("--min-seconds", type=float, default=None,
                        help="also stop once the steady window has run this long")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--profile-mode", choices=("full", "cuda", "off"), default="full",
                        help="full = gates/cd protocol (CPU+CUDA, all steps, key_averages); "
                             "cuda = CUDA-only over the first --profile-steps steady step(s), "
                             "one-pass aggregation (for the huge microbatch-2 event volumes); "
                             "off = walls only")
    parser.add_argument("--profile-steps", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.scale == 1:
        report = scale1(arguments)
    elif arguments.scale == 2:
        if arguments.config not in SCALE2_CONFIGS:
            raise SystemExit(f"--config must be one of {sorted(SCALE2_CONFIGS)}")
        report = scale2(arguments)
    else:
        if arguments.config not in SCALE3_CONFIGS:
            raise SystemExit(f"--config must be one of {sorted(SCALE3_CONFIGS)}")
        if arguments.steps == 8:
            arguments.steps = 60
        report = scale3(arguments)
    print(json.dumps({key: value for key, value in report.items()
                      if not isinstance(value, (dict, list))}, indent=1)[:2000])


if __name__ == "__main__":
    main()
