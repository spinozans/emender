"""K2 + K3 gates A and B as a receipt: read parity and the four normalisation fixtures.

``test_k23_read.py`` asserts these statements; this script measures them and writes the
numbers down, so ``K23_READ.md`` quotes a reproducible run rather than a memory of one.
Nothing here is a tolerance decision: the floor comes from
``test_matrix_emender.sequential_noise_floor`` on the same fixture and the fused path is
held to ``NOISE_FACTOR`` times it.

Three blocks:

``gate_a``        read parity vs the eager chunked path *and* the per-token oracle, at
                  every token, per chunk length -- with the floor and the median, not
                  just a max;
``gate_b``        the four fixtures the read's coefficients are made of: no writes, a
                  write on every token, a slot that dies mid-chunk, and a token whose
                  only support is an in-chunk injection;
``union_read``    the diagnostic: how far the registered chunk-entry-normalised read is
                  from a per-token read that recomputes the softmax over the slots live
                  at that token.  Behaviour-neutral; nothing on the runtime path uses it.

Usage::

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x45_matrix_emender.k23_gate_receipt \
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k23_gates.json
"""
from __future__ import annotations

import argparse
import json
import math

import torch
import triton
import triton.language as tl

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, NONE, POP, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k1_schedule_scan import (
    k1_state_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k23_read import (
    k23_read, union_normalised_reads)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    chunk_scan, chunked_scan, routed_chunk_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    entry_state, routed_fixture, scripted)
from experiments.expressivity_tasks.x45_matrix_emender.test_k23_read import (
    per_token_deviation, terms)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, sequential_noise_floor)

DEVICE = "cuda"


@triton.jit
def _precision_probe_kernel(a_ptr, b_ptr, o_ptr, K, PREC: tl.constexpr,
                            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """One fp32 ``tl.dot`` at a rank-K shape, at a chosen input precision."""
    rows = tl.arange(0, BM)
    lanes = tl.arange(0, BK)
    cols = tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        ranks = start + lanes
        left = tl.load(a_ptr + rows[:, None] * K + ranks[None, :])
        right = tl.load(b_ptr + ranks[:, None] * BN + cols[None, :])
        acc = tl.dot(left, right, acc, input_precision=PREC)
    tl.store(o_ptr + rows[:, None] * BN + cols[None, :], acc)


def precision_probe(rank: int = 128, rows: int = 64, cols: int = 64) -> dict:
    """What ``input_precision`` is worth, in the units of the gates.

    Triton's default for an fp32 ``tl.dot`` is TF32.  This measures both settings against
    torch's fp32 matmul (which is true fp32 here: ``allow_tf32`` is False), so the choice
    in the kernels above is a receipted number rather than a preference.
    """
    generator = torch.Generator().manual_seed(3401)
    left = torch.randn(rows, rank, generator=generator).to(DEVICE)
    right = torch.randn(rank, cols, generator=generator).to(DEVICE)
    reference = left @ right
    scale = float(reference.abs().max())
    out = {}
    for precision in ("ieee", "tf32"):
        result = torch.empty(rows, cols, dtype=torch.float32, device=DEVICE)
        _precision_probe_kernel[(1,)](left, right, result, rank, PREC=precision,
                                      BM=rows, BN=cols, BK=32)
        deviation = float((result - reference).abs().max())
        out[precision] = dict(deviation=deviation, relative=deviation / scale)
    out['scale'] = scale
    out['rank'] = rank
    return out


# --------------------------------------------------------------------------- gate A

def gate_a(chunks=(1, 2, 3, 4, 16)):
    """Every token, on the arm's own routing, with all five ops present."""
    fixture = routed_fixture()
    weights, decay, schedule = fixture.result.weights, fixture.decay, fixture.schedule
    floor, oracle_state, oracle_reads = sequential_noise_floor(weights, decay, schedule)
    rows = []
    for chunk in chunks:
        eager = chunked_scan(weights, decay, schedule, chunk_tokens=chunk, fused=False)
        fused = chunked_scan(weights, decay, schedule, chunk_tokens=chunk, fused=True)
        vs_eager = per_token_deviation(fused[0], eager[0])
        vs_oracle = per_token_deviation(fused[0], oracle_reads)
        rows.append(dict(
            chunk=chunk,
            read_max_vs_eager=vs_eager[0], read_median_vs_eager=vs_eager[1],
            read_max_vs_oracle=vs_oracle[0], read_median_vs_oracle=vs_oracle[1],
            state_max_vs_oracle=float((fused[1] - oracle_state).abs().max()),
            liveness_equal=bool(torch.equal(fused[2], eager[2]))))
    for row in rows:
        row['read_ratio_of_floor'] = row['read_max_vs_oracle'] / floor
    return dict(fixture=dict(tokens=int(schedule.tokens), rank=int(weights.shape[-1]),
                             heads=int(weights.shape[2]), batch=int(weights.shape[0]),
                             chunk_tokens=int(fixture.chunk_tokens), seed=23,
                             lanes=int(schedule.lanes)),
                ops=op_histogram(schedule), floor=float(floor),
                allowed=float(NOISE_FACTOR * floor),
                read_scale=float(oracle_reads.abs().max()), rows=rows)


# --------------------------------------------------------------------------- gate B

def gate_b():
    """The four fixtures, each isolating one failure mode (see ``test_k23_read``)."""
    cases = {}

    # (i) no writes at all: the read is the entry term and nothing else.
    tokens, rank, value_dim = 8, 5, 4
    shape = (1, tokens, 1, 1)
    silent = MatrixSchedule(
        slot=torch.zeros(shape, dtype=torch.int64, device=DEVICE),
        erase=torch.zeros(shape, dtype=torch.bool, device=DEVICE),
        payload=torch.zeros(1, tokens, 1, 1, value_dim, device=DEVICE),
        op=torch.full(shape, NONE, dtype=torch.int64, device=DEVICE),
        live_after=torch.zeros(shape, dtype=torch.bool, device=DEVICE))
    generator = torch.Generator().manual_seed(17)
    state = torch.randn(1, 1, rank, value_dim, generator=generator).to(DEVICE)
    live = torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE)
    weights = torch.rand(1, 1, tokens, rank, generator=generator).to(DEVICE)
    decay = torch.full((1, tokens, 1), 0.75, device=DEVICE)
    chunk = k1_state_scan(state, live, decay, silent)
    fused = k23_read(weights, state, chunk, silent)
    inter, intra = terms(weights, state, chunk, silent)
    cases['no_writes'] = dict(
        injection_scale=float(chunk.delta.abs().max()),
        intra_scale=float(intra.abs().max()),
        read_scale=float(fused.abs().max()),
        deviation_vs_inter_term=float((fused - inter).abs().max()))

    # (ii) a write on every lane, from an empty store: the intra block alone.
    tokens, rank, value_dim = 9, 6, 4
    shape = (1, tokens, 2, 1)
    generator = torch.Generator().manual_seed(19)
    slot = torch.randint(0, rank, shape, generator=generator)
    payload = torch.randn(1, tokens, 2, 1, value_dim, generator=generator)
    writing, decay_w, _ = scripted(slot, torch.full(shape, APPEND), payload,
                                   torch.full((1, tokens, 2), 0.9),
                                   torch.zeros(1, tokens, 2, rank))
    empty = torch.zeros(1, 2, rank, value_dim, device=DEVICE)
    dark = torch.zeros(1, 2, rank, dtype=torch.bool, device=DEVICE)
    weights = torch.rand(1, 2, tokens, rank, generator=generator).to(DEVICE)
    chunk = k1_state_scan(empty, dark, decay_w, writing)
    fused = k23_read(weights, empty, chunk, writing)
    inter, intra = terms(weights, empty, chunk, writing)
    cases['write_every_token'] = dict(
        injection_scale=float(chunk.delta.abs().max()),
        inter_scale=float(inter.abs().max()), intra_scale=float(intra.abs().max()),
        read_scale=float(fused.abs().max()),
        deviation_vs_intra_term=float((fused - intra).abs().max()))

    # (iii) a slot dies mid-chunk: its contribution must vanish from that token on.
    tokens, rank, value_dim, pop_at = 8, 4, 3, 4
    shape = (1, tokens, 1, 1)
    slot = torch.full(shape, 2, dtype=torch.int64)
    op = torch.where(torch.arange(tokens).view(1, tokens, 1, 1) == pop_at,
                     torch.full(shape, POP), torch.full(shape, NONE))
    dying, decay_d, _ = scripted(slot, op, torch.zeros(1, tokens, 1, 1, value_dim),
                                 torch.full((1, tokens, 1), 0.8),
                                 torch.zeros(1, tokens, 1, rank))
    content = torch.zeros(1, 1, rank, value_dim, device=DEVICE)
    content[0, 0, 2] = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    live = torch.zeros(1, 1, rank, dtype=torch.bool, device=DEVICE)
    live[0, 0, 2] = True
    weights = torch.zeros(1, 1, tokens, rank, device=DEVICE)
    weights[..., 2] = 0.5
    chunk = k1_state_scan(content, live, decay_d, dying)
    fused = k23_read(weights, content, chunk, dying)
    before = float(fused[0, 0, :pop_at + 1].abs().max())
    cases['slot_dies_mid_chunk'] = dict(
        pop_at=pop_at, read_scale_before_pop=before,
        residue_after_pop=float(fused[0, 0, pop_at + 1:].abs().max()),
        residue_ratio=float(fused[0, 0, pop_at + 1:].abs().max()) / before,
        slot3_live_at_exit=bool(chunk.live[0, 0, 2]))

    # (iv) a token whose only support is an in-chunk injection.
    tokens, rank, value_dim = 4, 4, 3
    shape = (1, tokens, 1, 1)
    payload = torch.zeros(1, tokens, 1, 1, value_dim, device=DEVICE)
    payload[0, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    injecting, decay_i, _ = scripted(torch.zeros(shape, dtype=torch.int64),
                                     torch.tensor([APPEND, NONE, NONE, NONE]).view(shape),
                                     payload, torch.full((1, tokens, 1), 0.9),
                                     torch.zeros(1, tokens, 1, rank))
    hollow = torch.zeros(1, 1, rank, value_dim, device=DEVICE)
    live = torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE)
    weights = torch.zeros(1, 1, tokens, rank, device=DEVICE)
    weights[..., 0] = 0.75
    chunk = k1_state_scan(hollow, live, decay_i, injecting)
    fused = k23_read(weights, hollow, chunk, injecting)
    inter, _ = terms(weights, hollow, chunk, injecting)
    closed_form = 0.75 * torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    cases['only_an_injection'] = dict(
        entry_term_scale=float(inter.abs().max()),
        token0_read=float(fused[0, 0, 0].abs().max()),
        token1_read=fused[0, 0, 1].tolist(),
        token1_closed_form=closed_form.tolist(),
        token1_deviation=float((fused[0, 0, 1] - closed_form).abs().max()),
        token2_read=fused[0, 0, 2].tolist(),
        token2_deviation=float((fused[0, 0, 2] - 0.9 * closed_form).abs().max()))
    return cases


# ----------------------------------------------------------------- union diagnostic

def _union_row(weights, decay, schedule, query, entry_state, entry_live, chunk, scale):
    """One row of the diagnostic: registered chunk-entry read vs the per-token read."""
    registered = chunked_scan(weights, decay, schedule, initial_state=entry_state,
                              initial_live=entry_live, chunk_tokens=chunk, fused=True)[0]
    union = union_normalised_reads(query, decay, schedule, entry_state, entry_live,
                                   scale)
    maximum, median = per_token_deviation(union, registered)
    overall = max(1.0, float(registered.abs().max()))
    return dict(read_scale=float(registered.abs().max()),
                max_delta=maximum, median_delta=median,
                max_relative=maximum / overall, median_relative=median / overall)


def union_read(fixture):
    """Registered chunk-entry read vs a per-token softmax over the live slots at t.

    Two rows, because the first one is a start-up artefact rather than a quality
    question: from an empty store the registered read of chunk 0 is *exactly zero* (no
    entry slot is live, so every weight is zero and the in-chunk writes are invisible to
    that chunk), while a per-token read sees them.  The populated row starts at chunk 1's
    entry state, which is the regime the arm actually runs in.
    """
    result, decay, query = fixture.result, fixture.decay, fixture.query
    weights, schedule = result.weights, result.schedule
    chunk = fixture.chunk_tokens
    tokens, heads = weights.shape[1], weights.shape[2]
    rank, value_dim = weights.shape[3], result.reads.shape[-1]
    batch = weights.shape[0]
    empty_state = torch.zeros(batch, heads, rank, value_dim, device=DEVICE)
    empty_live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    rows = {'from_empty_store': _union_row(weights, decay, schedule, query, empty_state,
                                           empty_live, chunk, 2.0)}
    # The reference trajectory's state and liveness at the second chunk boundary.
    first_state, first_live = empty_state, empty_live
    for start in range(0, min(chunk, tokens), chunk):
        stop = min(start + chunk, tokens)
        _, first_state, first_live = chunk_scan(
            first_state, first_live, weights[:, start:stop].permute(0, 2, 1, 3),
            decay[:, start:stop].permute(0, 2, 1), schedule.chunk(start, stop),
            fused=False)
    rows['from_populated_store'] = _union_row(
        weights[:, chunk:], decay[:, chunk:], schedule.chunk(chunk, tokens),
        query[:, chunk:], first_state, first_live, chunk, 2.0)
    return dict(boundary="the routed fixture; registered weights (chunk-entry softmax) vs "
                         "read_weights recomputed from the state live at each token; "
                         "delta is normalised by the row's own read scale",
                rows=rows)


def arm_shape_fixture(tokens=256, batch=2, heads=2, rank=128, value_dim=128,
                      chunk=128, seed=3401):
    """The arm's own shape, routed on real state: the regime the drift number is about."""
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy, route_chunk)

    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE)
    decay = torch.full((batch, tokens, heads), math.exp(-1.0 / 64.0), device=DEVICE)
    result = routed_chunk_scan(query, payload, decay, router=route_chunk,
                               policy=RoutingPolicy(ops=True), rank=rank, rank_max=rank,
                               chunk_tokens=chunk, collect=True, fused=False)

    class Fixture:
        pass

    fixture = Fixture()
    fixture.result, fixture.decay, fixture.query = result, decay, query
    fixture.chunk_tokens = chunk
    return fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    report = dict(gate_a=gate_a(), gate_b=gate_b(),
                  union_read=dict(
                      routed_fixture=union_read(routed_fixture()),
                      arm_shape=union_read(arm_shape_fixture())),
                  precision_probe=precision_probe(),
                  noise_factor=NOISE_FACTOR,
                  device=torch.cuda.get_device_name(0), torch=torch.__version__)
    with open(arguments.out, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
