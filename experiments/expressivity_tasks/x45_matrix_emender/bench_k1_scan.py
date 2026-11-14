"""K1 gates C and D: kernel count and isolated timing of the fused state scan.

Gate C counts the CUDA kernels a chunk launches (``torch.profiler``, CUDA activity,
events attributed to the device) for three paths on the *same* inputs and the same
recorded schedule:

``eager``     ``scan.chunk_scan``, the reference this lane has been running;
``kernel``    ``k1_schedule_scan.k1_state_scan`` -- one launch: the state walk, the
              injection list and the exit channels;
``k1_chunk``  the same kernel plus the read reduction over the emitted injection
              list, i.e. ``scan.chunk_scan(..., fused=True)``, the drop-in path.

Gate D times one chunk and one whole 2048-token block call, device and wall, with
the boundary stated in the receipt.  The shape is the brief's: batch 8, heads 2,
rank 128, ``d_v`` 128, chunk 128 tokens, fp32, forward only.

Two real routed schedules are measured, because they are the two regimes the arm can
be in and K1's cost is dominated by writes:

``matrix_ops``     the registered routing (CORRECT / MERGE / APPEND / POP / NONE);
                   the store saturates after one chunk of appends and later chunks are
                   write-sparse, so this is the cheap case;
``matrix_append``  the append-only control, a write on every token, so every step
                   loads a payload and stores an injection -- the expensive case.

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x45_matrix_emender.bench_k1_scan \
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k1_schedule_scan.json
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
from experiments.expressivity_tasks.x45_matrix_emender.k1_schedule_scan import (
    k1_state_scan)
from experiments.expressivity_tasks.x45_matrix_emender.scan import chunk_scan
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)

BATCH, TOKENS, HEADS, RANK, VALUE_DIM, CHUNK = 8, 2048, 2, 128, 128, 128
# The arm's initialised DECAY gate: sigmoid(bias) with a 64-token horizon.
DECAY = math.exp(-1.0 / 64.0)
# The timed chunk is a mid-sequence one, so the store is populated and the read is
# not the certified zero of an empty store (the first chunk's read is exactly zero --
# nothing is live yet -- and would be a flattering, empty measurement).
MEASURED_CHUNK = 4
DEVICE = "cuda"


def inputs(seed: int = 3401):
    """Real value/query streams and a DECAY gate at the brief's block shape."""
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(BATCH, TOKENS, HEADS, VALUE_DIM, generator=generator).to(DEVICE)
    decay = torch.full((BATCH, TOKENS, HEADS), DECAY, device=DEVICE)
    pieces = [(start, min(start + CHUNK, TOKENS)) for start in range(0, TOKENS, CHUNK)]
    return query, payload, decay, pieces


def recorded(policy, query, payload, decay, pieces):
    """The arm's own routing, chunk by chunk, recorded once and replayed to every path.

    Routing runs on the reference trajectory and the read weights come out of the
    same call, so every measured path consumes one identical decision channel and none
    of them pays for routing inside the measurement.
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
                                    decay[:, start:stop].permute(0, 2, 1), piece)
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
    return dict(kernels=len(events) / repeats, device_ms=device_us / 1000.0 / repeats)


def timed(fn, repeats: int = 5, warmup: int = 2) -> dict:
    """``(wall_ms, device_ms)`` mean and p50 over ``repeats`` calls."""
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
    return dict(wall_ms_mean=statistics.mean(wall), wall_ms_p50=statistics.median(wall),
                device_ms_mean=statistics.mean(device),
                device_ms_p50=statistics.median(device), repeats=repeats)


def fused_chunk(state, live, weights, decay, piece):
    return chunk_scan(state, live, weights, decay, piece, fused=True)


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
        return chunk_scan(entry_state, entry_live, chunk_weights, chunk_decay, piece)

    def kernel():
        return k1_state_scan(entry_state, entry_live, decay[:, start:stop], piece)

    def fused():
        return fused_chunk(entry_state, entry_live, chunk_weights, chunk_decay, piece)

    gate_c = dict(
        boundary=f"one chunk of {stop - start} tokens = {shape['steps']} steps, forward "
                 f"only, fp32; the schedule and the read weights are pre-built inputs "
                 f"and the chunk-entry state is the reference trajectory's state at "
                 f"chunk {index}, so no path pays for routing here",
        shape=shape, ops=op_histogram(piece),
        live_fraction=float(entry_live.float().mean()),
        entry_state_scale=float(entry_state.abs().max()),
        eager=kernel_count(eager), kernel=kernel_count(kernel),
        k1_chunk=kernel_count(fused))

    gate_d = dict(
        boundary=f"one chunk ({stop - start} tokens, {shape['steps']} steps) at the "
                 f"populated chunk-entry state, forward only, fp32; device = CUDA "
                 f"events around the call, wall = perf_counter around the call plus "
                 f"synchronize; 2 warm-up + {repeats} measured reps",
        eager=timed(eager, repeats), kernel=timed(kernel, repeats),
        k1_chunk=timed(fused, repeats))

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
        eager=timed(sequence(chunk_scan), repeats=3),
        k1_chunk=timed(sequence(fused_chunk), repeats=3),
        eager_kernels=kernel_count(sequence(chunk_scan), repeats=2),
        k1_kernels=kernel_count(sequence(fused_chunk), repeats=2))

    eager_reads, eager_state, _ = eager()
    fused_reads, fused_state, _ = fused()
    parity = dict(read_deviation=float((eager_reads - fused_reads).abs().max()),
                  state_deviation=float((eager_state - fused_state).abs().max()),
                  read_scale=float(eager_reads.abs().max()),
                  state_scale=float(eager_state.abs().max()))
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
