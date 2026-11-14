"""Gates C and D for the no-host-loop paths: launches per block call, and the wall.

Gate C counts what a 2048-token block call actually launches, and gate D times it, for
every path in ``hostloop_free``:

``chunked_C<C>``   ``chunked_block_scan`` -- the K2+K3 pipeline, ``3*ceil(T/C)`` launches
                   per block call.  **Option C** is this row at C = 256/512.
``graph_C<C>``     ``BlockGraph`` -- the same pipeline, captured once, one replay per
                   block call (**Option B**).
``persistent``     ``persistent_scan`` -- one kernel for the whole call (**Option A**).
``persistent_*``   the same kernel at another value block / warp count, so the tuning
                   sweep in ``NO_HOST_LOOP.md`` §4 is in the receipt rather than a claim.

Two timing columns, and the difference between them is the whole point of the lane:

``latency_ms``    ``perf_counter`` around the call **plus a synchronise** -- what a step
                  actually waits for;
``dispatch_ms``   ``perf_counter`` around the call alone -- the host's share.  For the
                  eager loop these two are the same number because the host is the
                  bottleneck; for a single launch the dispatch is ~0.12 ms and the
                  latency is the kernel, which is exactly the ratio this lane moves.
``occupancy_ms``  the profiler's summed kernel time -- real device work, and the
                  denominator of the ratio the receipts have been tracking (5.16x eager,
                  5.69x for K2+K3, 1.5x for GDN-2's fused kernels).

The block shape is the brief's: batch 8, 2048 tokens, 2 heads, rank 128, ``d_v`` 128,
lanes 2, fp32, forward only.  Rank 64 and rank 512 repeat the sweep at the small and
large ends of the persistent kernel's range (that is where its register tile changes), and
rank 2048 is measured only through the chunked paths because one program cannot hold it.

Boundary of the block call: the state and liveness go in, the *recorded* schedule and read
weights are inputs, the tokens' reads and the exit state come out -- the same boundary
``K23_READ.md`` §5.3 used, so the numbers are comparable to its 48 launches / 6.57 ms.
Routing is NOT inside this measurement (it is the arm's per-chunk decision step, and it
needs the state, so it is a separate boundary; ``--routed`` measures it as well).

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x45_matrix_emender.bench_hostloop_free \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/hostloop_free.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.bench_k1_scan import (
    BATCH, CHUNK, HEADS, RANK, TOKENS, VALUE_DIM, inputs, recorded)
from experiments.expressivity_tasks.x45_matrix_emender.hostloop_free import (
    BlockGraph, chunked_block_scan, persistent_scan)
from experiments.expressivity_tasks.x45_matrix_emender.scan import chunk_scan
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    random_schedule, sequential_noise_floor)

DEVICE = "cuda"
LANES = 2
# The chunk lengths Option C is measured at, and the two the graph is captured for.
CHUNKS = (16, 64, 128, 256, 512)
GRAPH_CHUNKS = (128, 256, 512)
# The chunk lengths the eager reference is built at: 128 is the receipt's boundary,
# 256/512 say what the chunk length by itself moves, and each chunked/graphed path is
# compared to its own twin so its parity column is a kernel comparison.  16 is there for
# the persistent kernel at a rank where no per-token oracle is measured: it is the finest
# chunked grouping available, i.e. the closest twin a chunked path can offer.
EAGER_CHUNKS = (16, 128, 256, 512)
# The value blocks / warp counts the persistent kernel's default came from (§4).
SWEEP = ((16, None), (32, 4), (64, None), (128, 4))


def head_major(decay):
    return decay.permute(0, 2, 1).contiguous()


def recorded_block(policy, query, payload, decay, pieces):
    """The arm's own routing for the whole block, recorded once and replayed to every path.

    Identical to ``bench_k1_scan.recorded`` except that it also concatenates the per-chunk
    weights into one ``(batch, heads, tokens, rank)`` channel, which is the form the
    block-level paths take -- so no path pays for routing inside the measurement.
    """
    state = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, device=DEVICE)
    live = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool, device=DEVICE)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import empty_schedule
    schedule = empty_schedule(BATCH, TOKENS, HEADS, VALUE_DIM, lanes=LANES, device=DEVICE)
    weights = []
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
        _, state, live = chunk_scan(state, live, chunk_weights,
                                    decay[:, start:stop].permute(0, 2, 1), piece,
                                    fused=False)
    entry_state = torch.zeros(BATCH, HEADS, RANK, VALUE_DIM, device=DEVICE)
    entry_live = torch.zeros(BATCH, HEADS, RANK, dtype=torch.bool, device=DEVICE)
    return (entry_state, entry_live, torch.cat(weights, dim=2),
            head_major(decay), schedule)


def synthetic_block(rank, seed):
    """A block at another rank: a random schedule, its read weights and a fixed decay."""
    schedule, query, _, decay = random_schedule(BATCH, TOKENS, HEADS, rank, VALUE_DIM,
                                                lanes=LANES, seed=seed)
    state = torch.randn(BATCH, HEADS, rank, VALUE_DIM, device=DEVICE) * 0.1
    live = torch.rand(BATCH, HEADS, rank, device=DEVICE) < 0.7
    return state, live, query.permute(0, 2, 1, 3).contiguous(), head_major(decay), schedule


def kernel_count(fn, repeats: int = 3) -> dict:
    """CUDA kernels and device ms per call, averaged over ``repeats`` calls."""
    for _ in range(2):
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
    counts: dict[str, int] = {}
    for event in events:
        counts[event.name] = counts.get(event.name, 0) + 1
    return dict(kernels=len(events) / repeats, device_ms=device_us / 1000.0 / repeats,
                distinct_kernels=len(counts),
                top_kernels=sorted(counts, key=lambda name: -counts[name])[:4])


def timed(fn, repeats: int = 5, warmup: int = 2, label: str = "?") -> dict:
    """Latency, host dispatch, bracketed device time and kernel occupancy.

    ``latency_ms`` synchronises inside the timed region (what a step waits for);
    ``dispatch_ms`` stops at the call (the host's share).  Both are reported with mean,
    p50 and the min-max spread, because a single-launch path's host time is small enough
    that the spread says more than the mean.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    latency, dispatch, device = [], [], []
    for _ in range(repeats):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        began = time.perf_counter()
        fn()
        dispatch.append((time.perf_counter() - began) * 1000.0)
        torch.cuda.synchronize()
        latency.append((time.perf_counter() - began) * 1000.0)
        end_event.record()
        torch.cuda.synchronize()
        device.append(start_event.elapsed_time(end_event))
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=False, with_stack=False) as profiler:
        for _ in range(repeats):
            fn()
        torch.cuda.synchronize()
    occupancy = sum(event.self_device_time_total for event in profiler.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA) / 1000.0
    occupancy /= repeats
    if occupancy <= 0.0:
        raise RuntimeError(f"the profiler attributed no device time to {label!r}")
    return dict(
        latency_ms_mean=statistics.mean(latency),
        latency_ms_p50=statistics.median(latency),
        latency_ms_min=min(latency), latency_ms_max=max(latency),
        dispatch_ms_mean=statistics.mean(dispatch),
        dispatch_ms_p50=statistics.median(dispatch),
        device_ms_mean=statistics.mean(device),
        device_ms_p50=statistics.median(device),
        occupancy_ms=occupancy,
        latency_over_occupancy=statistics.median(latency) / occupancy,
        latency_over_device=statistics.mean(latency) / statistics.mean(device),
        dispatch_over_occupancy=statistics.median(dispatch) / occupancy,
        repeats=repeats)


def parity(reads, state, live, reference) -> dict:
    """Deviation from a reference triple, per token and in relative terms."""
    ref_reads, ref_state, ref_live = reference
    deviation = (reads - ref_reads).abs()
    per_token = deviation.amax(dim=(0, 3))
    read_scale = max(1e-12, float(ref_reads.abs().max()))
    state_scale = max(1e-12, float(ref_state.abs().max()))
    return dict(read_max=float(deviation.max()),
                read_median=float(per_token.median()),
                read_scale=float(ref_reads.abs().max()),
                read_relative=float(deviation.max()) / read_scale,
                state_max=float((state - ref_state).abs().max()),
                state_scale=float(ref_state.abs().max()),
                state_relative=float((state - ref_state).abs().max()) / state_scale,
                live_equal=(None if ref_live is None
                            else bool(torch.equal(live, ref_live))))


def measure_block(label: str, state, live, weights, decay, schedule, repeats: int,
                  oracle: dict | None) -> dict:
    """Every path on one block, with gate C, gate D and parity for each."""
    batch, heads, tokens, _ = weights.shape

    def eager_reference(chunk: int = CHUNK):
        # The chunked *eager* path, chunk by chunk: the reference the receipts compare to.
        state_now, live_now = state.clone(), live.clone()
        reads = torch.empty(batch, heads, tokens, state.shape[3], device=DEVICE)
        for start in range(0, tokens, chunk):
            stop = min(start + chunk, tokens)
            chunk_reads, state_now, live_now = chunk_scan(
                state_now, live_now, weights[:, :, start:stop], decay[:, :, start:stop],
                schedule.chunk(start, stop), fused=False)
            reads[:, :, start:stop] = chunk_reads
        return reads, state_now, live_now

    # The eager chunked path at three chunk lengths: the twin of every path that has a
    # chunk (so its parity is a kernel comparison, not a chunk-length comparison), and the
    # measurement of what the chunk length *alone* moves.
    references = {chunk: eager_reference() for chunk in EAGER_CHUNKS}
    reference = references[CHUNK]
    paths = {}
    for chunk in CHUNKS:
        paths[f"chunked_C{chunk}"] = (
            dict(host_launches=3 * -(-tokens // chunk), form="host loop",
                 twin_chunk=chunk if chunk in references else None),
            (lambda chunk=chunk: chunked_block_scan(state, live, weights, decay, schedule,
                                                    chunk)))
    for chunk in GRAPH_CHUNKS:
        graph = BlockGraph(batch, heads, tokens, weights.shape[3], state.shape[3],
                           schedule.lanes, chunk_tokens=chunk, device=DEVICE).capture()
        paths[f"graph_C{chunk}"] = (
            dict(host_launches=1, form="one replay", twin_chunk=chunk,
                 kernels_note="the pipeline's launches plus the copies the graph captures"),
            (lambda graph=graph: graph.run(state, live, weights, decay, schedule)))
    per_token_twin = dict(host_launches=1, form="one kernel",
                          twin_chunk=None if oracle is not None else min(EAGER_CHUNKS),
                          twin_note=("the per-token oracle, not a chunked path: the "
                                     "persistent kernel walks tokens, not chunks"))
    paths["persistent"] = (per_token_twin,
                           lambda: persistent_scan(state, live, weights, decay, schedule))
    for block, warps in SWEEP:
        paths[f"persistent_bv{block}"] = (
            dict(host_launches=1, form="one kernel", value_block=block,
                 warps=warps or "default"),
            (lambda block=block, warps=warps: persistent_scan(
                state, live, weights, decay, schedule, block=block, warps=warps)))

    rows = {}
    for name, (description, fn) in paths.items():
        reads, exit_state, exit_live = fn()
        twin = description.get("twin_chunk")
        rows[name] = dict(
            description=description, gate_c=kernel_count(fn),
            gate_d=timed(fn, repeats, label=name),
            parity_vs_eager=parity(reads, exit_state, exit_live, reference),
            parity_vs_eager_same_chunk=(None if twin is None
                                        else parity(reads, exit_state, exit_live,
                                                    references[twin])),
            parity_vs_oracle=(None if oracle is None
                              else parity(reads, exit_state, exit_live, oracle)))
        rows[name]["state_bit_equal_to_eager"] = bool(torch.equal(exit_state,
                                                                 reference[1]))
    return dict(label=label,
                boundary=("one whole block call: batch 8, 2048 tokens, 2 heads, lanes 2, "
                          "fp32, forward only; entry state and liveness in, the recorded "
                          "schedule and read weights are inputs, the tokens' reads and the "
                          "exit state out -- the same boundary as K23_READ.md §5.3"),
                shape=dict(batch=batch, tokens=tokens, heads=heads,
                           rank=weights.shape[3], value_dim=state.shape[3],
                           lanes=schedule.lanes),
                eager_reference=dict(
                    gate_c=kernel_count(eager_reference),
                    gate_d=timed(eager_reference, repeats=3, label=f"{label}-eager"),
                    # What the chunk length alone moves: the eager path at the other
                    # chunk lengths against the eager path at C = 128.
                    chunk_length_effect={
                        str(chunk): parity(*references[chunk], reference)
                        for chunk in references if chunk != CHUNK}),
                paths=rows)


def measure(out_path: str, seed: int = 3401, repeats: int = 5, ranks=(64, 128, 512)) -> dict:
    query, payload, decay, pieces = inputs(seed)
    cases = {}
    for label, policy in (("matrix_ops", RoutingPolicy(ops=True)),
                          ("matrix_append", RoutingPolicy(ops=False))):
        state, live, weights, head_first_decay, schedule = recorded_block(
            policy, query, payload, decay, pieces)
        oracle = None
        if RANK in ranks:
            # The per-token oracle and its measured floor, at the bench shape. This is the
            # semantic reference (sequential_matrix_scan), not the chunked eager path.
            floor, oracle_state, oracle_reads = sequential_noise_floor(
                weights.permute(0, 2, 1, 3), head_first_decay.permute(0, 2, 1), schedule)
            # ``sequential_matrix_scan`` does not report its exit liveness through this
            # helper, so the oracle row compares content only (the liveness channel is
            # compared against the eager path, which carries it).
            oracle = dict(triple=(oracle_reads.permute(0, 2, 1, 3), oracle_state, None),
                          floor=floor)
        block = measure_block(label, state, live, weights, head_first_decay, schedule,
                             repeats, oracle["triple"] if oracle else None)
        block["sequence_noise_floor"] = oracle["floor"] if oracle else None
        cases[label] = block
        for rank in ranks:
            if rank == RANK:
                continue
            synthetic = synthetic_block(rank, seed=rank)
            cases[f"synthetic_rank{rank}"] = measure_block(
                f"synthetic_rank{rank}", *synthetic, repeats=repeats, oracle=None)
    report = dict(seed=seed, repeats=repeats,
                  shape=dict(batch=BATCH, tokens=TOKENS, heads=HEADS, rank=RANK,
                             value_dim=VALUE_DIM, chunk=CHUNK, lanes=LANES),
                  device=torch.cuda.get_device_name(0), torch=torch.__version__,
                  triton=__import__("triton").__version__,
                  cases=cases)
    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--ranks", default="64,128,512")
    arguments = parser.parse_args()
    ranks = tuple(int(value) for value in arguments.ranks.split(","))
    report = measure(arguments.out, arguments.seed, arguments.repeats, ranks)
    for label, case in report["cases"].items():
        print(f"== {label} {case['shape']}")
        for name, row in case["paths"].items():
            gate_c, gate_d = row["gate_c"], row["gate_d"]
            print(f"  {name:16s} launches {row['description']['host_launches']:4d} "
                  f"kernels {gate_c['kernels']:5.0f} "
                  f"lat {gate_d['latency_ms_mean']:7.3f}/{gate_d['latency_ms_p50']:7.3f} "
                  f"disp {gate_d['dispatch_ms_mean']:6.3f} "
                  f"occ {gate_d['occupancy_ms']:6.3f} "
                  f"lat/occ {gate_d['latency_over_occupancy']:5.2f} "
                  f"read {row['parity_vs_eager']['read_max']:.2e}")


if __name__ == "__main__":
    main()
