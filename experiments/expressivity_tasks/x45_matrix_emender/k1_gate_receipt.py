"""K1 gates A and B as a receipt: bit-exactness counters and the measured floors.

The pytest file asserts these statements; this script measures them and writes the
numbers down, so ``K1_SCHEDULE_SCAN.md`` quotes a reproducible run rather than a
memory of one.  Nothing here is a tolerance decision -- the floors are measured on the
same fixtures by ``test_matrix_emender.sequential_noise_floor`` and the fused path is
held to ``NOISE_FACTOR`` times them.

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x45_matrix_emender.k1_gate_receipt \
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k1_gates.json
"""
from __future__ import annotations

import argparse
import json

import torch

from experiments.expressivity_tasks.x45_matrix_emender import read as read_module
from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    op_histogram, sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k1_schedule_scan import (
    k1_state_scan)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    chunked_scan, routed_chunk_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, sequential_noise_floor)

DEVICE = "cuda"


def routed(batch=2, tokens=64, heads=2, rank=5, value_dim=8, chunk=16, seed=23,
           distinct=6, policy=None):
    """The arm's own routing on real state, capped to a few value directions.

    ``distinct`` is what makes MERGE reachable at all: two live rows must be similar,
    and a uniform random value stream never is.  The default settings produce every
    one of the five ops.
    """
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE) * 0.5
    directions = torch.randn(distinct, value_dim, generator=generator).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    policy = RoutingPolicy(ops=True) if policy is None else policy
    result = routed_chunk_scan(query, payload, decay, router=route_chunk, policy=policy,
                              rank=rank, rank_max=rank, chunk_tokens=chunk, collect=True,
                              fused=False)
    return result, decay, query, payload, policy, rank


def entry_state(batch, heads, rank, value_dim, seed=0, live_fraction=0.5):
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(batch, heads, rank, value_dim, generator=generator).to(DEVICE)
    live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) < live_fraction
    return state * live.unsqueeze(-1), live


def gate_a(result, decay, seed=3):
    """Walk the whole sequence chunk by chunk: is every scalar channel bit-identical?"""
    tokens, chunk = result.schedule.tokens, 16
    weights = result.weights
    state, live = entry_state(weights.shape[0], weights.shape[2], weights.shape[3],
                              result.reads.shape[-1], seed=seed)
    boundaries = 0
    state_exact = live_exact = True
    strength_deviation = 0.0
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        piece = result.schedule.chunk(start, stop)
        _, expected_state, _, trajectory = sequential_matrix_scan(
            weights[:, start:stop], decay[:, start:stop], piece,
            initial_state=state.clone(), initial_live=live, return_liveness=True)
        actual = k1_state_scan(state, live, decay[:, start:stop], piece)
        state_exact &= torch.equal(actual.state, expected_state)
        live_exact &= torch.equal(actual.live, trajectory[-1])
        strength_deviation = max(strength_deviation, float(
            (actual.strength - read_module.row_strength(actual.state, actual.live)
             ).abs().max()))
        boundaries += 1
        state, live = actual.state, actual.live
    return dict(lanes=int(result.schedule.lanes), chunk=chunk, boundaries=boundaries,
                ops=op_histogram(result.schedule), state_bit_exact=bool(state_exact),
                liveness_bit_exact=bool(live_exact),
                strength_deviation=float(strength_deviation))


def gate_b(result, decay, chunks=(1, 2, 3, 4, 16)):
    """Reads and content against the measured sequential floor, per chunk length."""
    floor, reference_state, reference_reads = sequential_noise_floor(
        result.weights, decay, result.schedule)
    rows = []
    for chunk in chunks:
        reads, state, live = chunked_scan(result.weights, decay, result.schedule,
                                         chunk_tokens=chunk, fused=True)
        eager_reads, eager_state, eager_live = chunked_scan(
            result.weights, decay, result.schedule, chunk_tokens=chunk, fused=False)
        rows.append(dict(
            chunk=chunk,
            read_deviation=float((reads - reference_reads).abs().max()),
            state_deviation=float((state - reference_state).abs().max()),
            read_deviation_vs_chunked=float((reads - eager_reads).abs().max()),
            state_deviation_vs_chunked=float((state - eager_state).abs().max()),
            liveness_equal=bool(torch.equal(live, eager_live))))
    for row in rows:
        row['read_ratio_of_floor'] = row['read_deviation'] / floor
        row['state_ratio_of_floor'] = row['state_deviation'] / floor
    return dict(floor=floor, allowed=NOISE_FACTOR * floor, rows=rows)


def injection_receipt(result, decay, seed=7):
    """The emitted injection list must rebuild the exit state."""
    state, live = entry_state(result.weights.shape[0], result.weights.shape[2],
                              result.weights.shape[3], result.reads.shape[-1], seed=seed)
    piece = result.schedule.chunk(0, 40)
    chunk = k1_state_scan(state, live, decay[:, :40], piece)
    steps, lane_count = chunk.steps, piece.lanes
    batch, heads = state.shape[0], state.shape[1]
    slot = piece.slot.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    weight = torch.exp(chunk.log_gate[..., -1:] - chunk.log_gate).unsqueeze(-1) * chunk.delta
    injection = torch.zeros_like(state).scatter_add_(
        2, slot.unsqueeze(-1).expand(batch, heads, steps, state.shape[-1]), weight)
    rebuilt = torch.exp(chunk.log_gate[..., -1])[:, :, None, None] * state + injection
    return dict(steps=int(steps),
                deviation=float((rebuilt - chunk.state).abs().max()),
                state_scale=float(chunk.state.abs().max()))


def rerouting_drift(result, decay, query, payload, policy, rank, chunk=16):
    """How many decisions move when the router sees the fused trajectory instead."""
    rerouted = routed_chunk_scan(query, payload, decay, router=route_chunk, policy=policy,
                                rank=rank, rank_max=rank, chunk_tokens=chunk,
                                collect=True, fused=True)
    difference = (rerouted.reads - result.reads).abs()
    return dict(lanes=int(result.schedule.op.numel()),
                moved=int((rerouted.schedule.op != result.schedule.op).sum()),
                reads_deviation=float(difference.max()),
                reads_scale=float(result.reads.abs().max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    result, decay, query, payload, policy, rank = routed()
    report = dict(
        fixture=dict(tokens=int(result.schedule.tokens), rank=rank,
                     chunk=16, seed=23, decay=0.95),
        gate_a=gate_a(result, decay), gate_b=gate_b(result, decay),
        injection_list=injection_receipt(result, decay),
        rerouting=rerouting_drift(result, decay, query, payload, policy, rank),
        device=torch.cuda.get_device_name(0), torch=torch.__version__)
    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
