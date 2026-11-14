"""K2 + K3 gates C and D: kernel count and timing of the fused read.

Gate C counts the CUDA kernels a chunk launches (``torch.profiler``, CUDA activity,
events attributed to the device) for four paths on the *same* inputs and the same
recorded schedule:

``eager``      ``scan.chunk_scan(..., fused=False)`` -- the reference: 87 kernels;
``k1``         ``k1_state_scan`` alone -- one launch, the state walk;
``k1_chunk``   stage K1 + the *torch* read reduction (``k1_schedule_scan.k1_chunk``) --
               the post-K1 drop-in, 24 kernels;
``k23_chunk``  the fused drop-in: K1 + K3 (inter) + K2 (intra), 3 kernels.

Gate D times one chunk and one whole 2048-token block call, device and wall, with the
boundary stated in the receipt.  The wall/device ratio is the number this stage is
about: the eager chunk is host-launch bound (5.2x in ``RESULT_MATRIX_EMENDER.md`` §6),
and the read was the last host-side reduction in the loop.

Two real routed schedules are measured, because they are the two regimes the arm can be
in and the kernels' cost depends on writes:

``matrix_ops``     the registered routing (CORRECT / MERGE / APPEND / POP / NONE);
                   the store saturates after one chunk of appends and later chunks are
                   write-sparse, so this is the cheap case;
``matrix_append``  the append-only control, a write on every token, so every step
                   carries a payload and an injection -- the expensive case.

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x45_matrix_emender.bench_k23_read \
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k23_read.json
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import time

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    empty_schedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.bench_k1_scan import (
    BATCH, CHUNK, DECAY, HEADS, RANK, TOKENS, VALUE_DIM, inputs)
from experiments.expressivity_tasks.x45_matrix_emender.k1_schedule_scan import (
    k1_chunk, k1_state_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k23_read import k23_chunk
from experiments.expressivity_tasks.x45_matrix_emender.scan import chunk_scan
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)

# The timed chunk is a mid-sequence one, so the store is populated and the read is not
# the certified zero of an empty store (the first chunk's read is exactly zero -- nothing
# is live yet -- and would be a flattering, empty measurement).
MEASURED_CHUNK = 4
DEVICE = "cuda"


def recorded(policy, query, payload, decay, pieces):
    """The arm's own routing, chunk by chunk, recorded once and replayed to every path.

    Routing runs on the reference trajectory (``fused=False``) and the read weights come
    out of the same call, so every measured path consumes one identical decision channel
    and none of them pays for routing inside the measurement.
    """
    state = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, device=DEVICE)
    live = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool, device=DEVICE)
    schedule = empty_schedule(BATCH, TOKENS, HEADS, VALUE_DIM, lanes=2, device=DEVICE)
    weights, states, lives = [], [], []
    for index, (start, stop) in enumerate(pieces):
        piece, chunk_weights = route_chunk(
            index, start, state, live, query[:, start:stop].permute(0, 2, 1, 3),
            payload[:, start:stop].permute(0, 2, 1, 3), policy, RANK)
        schedule.slot[:, start:stop] = piece.slot
        schedule.erase[:, start:stop] = piece.erase
        schedule.payload[:, start:stop] = piece.payload
        schedule.op[:, start:stop] = piece.op
        schedule.live_after[:, start:stop] = piece.live_after
        weights.append(chunk_weights)
        states.append(state)
        lives.append(live)
        _, state, live = chunk_scan(state, live, chunk_weights,
                                    decay[:, start:stop].permute(0, 2, 1), piece,
                                    fused=False)
    return schedule, weights, states, lives


def kernel_count(fn, repeats: int = 3) -> dict:
    """CUDA kernels and device ms per call, averaged over ``repeats`` calls."""
    for _ in range(2):                       # Triton JIT + allocator warm-up
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False) as profiler:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    events = [event for event in profiler.events()
              if event.device_type == torch.autograd.DeviceType.CUDA]
    device_us = sum(event.self_device_time_total for event in events)
    names = [event.name for event in events[:6]]
    return dict(kernels=len(events) / repeats, device_ms=device_us / 1000.0 / repeats,
                first_kernels=names)


def timed(fn, repeats: int = 5, warmup: int = 2) -> dict:
    """Wall, bracketed device and kernel-occupancy time, mean and p50 over ``repeats``.

    Three numbers, because they answer three different questions and only one of them is
    comparable to the 5.2x in ``RESULT_MATRIX_EMENDER.md`` §6:

    ``wall_ms``        ``perf_counter`` around the call plus a synchronise -- what a step
                       actually costs;
    ``device_ms``      CUDA events bracketing the call -- for a launch-bound call this
                       contains the host dispatch latency, so it tracks ``wall_ms``;
    ``occupancy_ms``   the profiler's summed kernel execution time -- real device work,
                       and the denominator of the wall/occupancy ratio (5.2x eager, 1.5x
                       for GDN-2's fused kernels).  This is the ratio this stage moves.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    wall, device = [], []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        fn()
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        end_event.record()
        torch.cuda.synchronize()
        wall.append(wall_ms)
        device.append(start_event.elapsed_time(end_event))
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False) as profiler:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    occupancy = sum(event.self_device_time_total for event in profiler.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA) / 1000.0
    return dict(wall_ms_mean=statistics.mean(wall), wall_ms_p50=statistics.median(wall),
                wall_ms_min=min(wall), wall_ms_max=max(wall),
                device_ms_mean=statistics.mean(device),
                device_ms_p50=statistics.median(device),
                occupancy_ms=occupancy / repeats,
                wall_over_device=statistics.mean(wall) / statistics.mean(device),
                wall_over_occupancy=statistics.median(wall) / (occupancy / repeats),
                repeats=repeats)


def measure_case(label: str, policy, query, payload, decay, pieces, repeats: int) -> dict:
    schedule, weights, states, lives = recorded(policy, query, payload, decay, pieces)
    index = min(MEASURED_CHUNK, len(pieces) - 1)
    start, stop = pieces[index]
    piece = schedule.chunk(start, stop)
    entry_state, entry_live = states[index], lives[index]
    chunk_decay = decay[:, start:stop].permute(0, 2, 1)
    chunk_weights = weights[index]
    shape = dict(batch=BATCH, tokens_per_chunk=stop - start, heads=HEADS, rank=RANK,
                 value_dim=VALUE_DIM, lanes=int(piece.lanes),
                 steps=int((stop - start) * piece.lanes))

    def eager():
        return chunk_scan(entry_state, entry_live, chunk_weights, chunk_decay, piece,
                          fused=False)

    def k1():
        return k1_state_scan(entry_state, entry_live, decay[:, start:stop], piece)

    def k1_only():
        return k1_chunk(entry_state, entry_live, chunk_weights, chunk_decay, piece)

    def k23():
        return k23_chunk(entry_state, entry_live, chunk_weights, chunk_decay, piece)

    gate_c = dict(
        boundary=f"one chunk of {stop - start} tokens = {shape['steps']} steps, forward "
                 f"only, fp32; the schedule and the read weights are pre-built inputs "
                 f"and the chunk-entry state is the reference trajectory's state at "
                 f"chunk {index}, so no path pays for routing here",
        shape=shape, ops=op_histogram(piece),
        live_fraction=float(entry_live.float().mean()),
        entry_state_scale=float(entry_state.abs().max()),
        read_scale=float(eager()[0].abs().max()),
        eager=kernel_count(eager), k1=kernel_count(k1), k1_chunk=kernel_count(k1_only),
        k23_chunk=kernel_count(k23))

    gate_d = dict(
        boundary=f"one chunk ({stop - start} tokens, {shape['steps']} steps) at the "
                 f"populated chunk-entry state, forward only, fp32; wall = perf_counter "
                 f"around the call plus synchronize, device = CUDA events bracketing it, "
                 f"occupancy = the profiler's summed kernel time; 2 warm-up + {repeats} "
                 f"measured reps",
        eager=timed(eager, repeats), k1=timed(k1, repeats),
        k1_chunk=timed(k1_only, repeats), k23_chunk=timed(k23, repeats))

    def eager_step(state_now, live_now, step_weights, step_decay, step_schedule):
        return chunk_scan(state_now, live_now, step_weights, step_decay, step_schedule,
                          fused=False)

    def k23_step(state_now, live_now, step_weights, step_decay, step_schedule):
        return k23_chunk(state_now, live_now, step_weights, step_decay, step_schedule)

    def sequence(impl):
        def run():
            state_now, live_now = states[0], lives[0]
            for chunk_index, (a, b) in enumerate(pieces):
                _, state_now, live_now = impl(state_now, live_now, weights[chunk_index],
                                              decay[:, a:b].permute(0, 2, 1),
                                              schedule.chunk(a, b))
            return state_now
        return run

    block = dict(
        boundary="the whole 2048-token block call = 16 chunk calls carrying the state, "
                 "forward only, fp32 (bench_blocks.py's block row adds the backward and "
                 "is therefore a different boundary)",
        eager=timed(sequence(eager_step), repeats=3),
        k23_chunk=timed(sequence(k23_step), repeats=3),
        eager_kernels=kernel_count(sequence(eager_step), repeats=2),
        k23_kernels=kernel_count(sequence(k23_step), repeats=2))

    eager_reads, eager_state, _ = eager()
    fused_reads, fused_state, _ = k23()
    read_deviation = float((eager_reads - fused_reads).abs().max())
    state_deviation = float((eager_state - fused_state).abs().max())
    read_scale = max(1e-12, float(eager_reads.abs().max()))
    state_scale = max(1e-12, float(eager_state.abs().max()))
    parity = dict(read_deviation=read_deviation, state_deviation=state_deviation,
                  read_scale=float(eager_reads.abs().max()),
                  state_scale=float(eager_state.abs().max()),
                  read_relative=read_deviation / read_scale,
                  state_relative=state_deviation / state_scale)
    return dict(label=label, policy=policy.record(), gate_c=gate_c, gate_d=gate_d,
                block=block, parity=parity)


def measure(out_path: str, seed: int = 3401, repeats: int = 5) -> dict:
    query, payload, decay, pieces = inputs(seed)
    cases = {}
    for label, policy in (("matrix_ops", RoutingPolicy(ops=True)),
                          ("matrix_append", RoutingPolicy(ops=False))):
        cases[label] = measure_case(label, policy, query, payload, decay, pieces, repeats)
    report = dict(seed=seed, shape=dict(batch=BATCH, tokens=TOKENS, heads=HEADS,
                                        rank=RANK, value_dim=VALUE_DIM, chunk=CHUNK),
                  decay=DECAY, measured_chunk=MEASURED_CHUNK, cases=cases,
                  device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  triton=__import__("triton").__version__)
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    print(json.dumps(measure(arguments.out, arguments.seed, arguments.repeats), indent=2))


if __name__ == "__main__":
    main()
