"""K5B -- the whole-scan op pair, gated.

The parity discipline is the settled ruling, applied at the new granularity:

* **Bit-equality by construction is the headline gate**: the whole-scan body
  is the per-chunk K5A arm's kernel sequence walked inside one op (the eager
  preprocessing the decision semantics demand, the K5A prep/emit pair, the
  K4 persistent scan, and their two adjoints chained as autograd chains them),
  so the recorded schedule, weights, reads, state, liveness AND the three
  gradients must be **bit-identical to that arm** -- asserted here on every
  fixture, eager and captured and compiled.  A difference would be a bug, not
  a floor.
* **The mandatory decision-flip test** (A-phase's bar, matched): on real
  routed fixtures at the registered shape under bf16 autocast, the router the
  whole-scan walk runs is compared against the registered ``route_chunk`` on
  the same pinned chunk-entry states -- zero flips in 8.65M lanes.
* **The read-weights floor is re-measured first** (the K23 discipline) and the
  whole-scan weights sit inside ``NOISE_FACTOR`` x it -- they are K5A's
  weights, recomputed by the same kernels, so this is the same floor story.
* **The graph discipline**: capture and replay are bit-identical to the eager
  body, two replays are bit-identical (determinism, no atomics anywhere), and
  the refusals (append-only, growth, rank 512, explicit initial state) take
  the per-chunk stack verbatim.
* **Compile hosting**: ``torch.compile`` traces the wrapper with the op pair
  opaque (no ``InductorError``), and the compiled path's outputs and gradients
  are bit-identical to the uncompiled path's and to the arm's.

Every test needs a CUDA device (the kernels are Triton, the graphs are CUDA).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.k5a_routing_diff import (
    K5A_MAX_RANK, routing_diff_route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan import (
    DISPATCH, compiled_whole_scan, next_slot_id, reset_slots, whole_scan,
    whole_scan_record)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    routed_chunk_scan)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

REG_BATCH, REG_TOKENS, REG_HEADS, REG_RANK, REG_VALUE_DIM = 8, 2048, 2, 128, 128
REG_CHUNK = 64
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after", "payload")


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed,
                    device=DEVICE):
    """The lane's canonical fixture (``test_k5a_routing_diff``'s)."""
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).to(device) * 0.5
    directions = torch.randn(distinct, value_dim, generator=generator).to(device)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.rand(batch, tokens, heads, generator=generator).to(device) \
        .clamp(0.9, 1.0)
    return query, payload, decay


def k5a_arm_scan(query, payload, decay, policy, rank, rank_max, chunk,
                 collect=False, upstream=None):
    """The per-chunk K5A+K4 stack (the arm K5B subsumes), fwd (+bwd)."""
    q = query.detach().clone().requires_grad_(True)
    p = payload.detach().clone().requires_grad_(True)
    d = decay.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = routed_chunk_scan(
            q, p, d, router=routing_diff_route_chunk, policy=policy, rank=rank,
            rank_max=rank_max, chunk_tokens=chunk, fused=False,
            trainable_fused=True, collect=collect)
    if upstream is not None:
        (result.reads * upstream).sum().backward()
    return (q, p, d), result


def assert_record_bit_equal_to_arm(query, payload, decay, policy, rank,
                                   rank_max, chunk, label, upstream=None):
    """The construction claim: every recorded channel bit-equal to the arm."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        record = whole_scan_record(query, payload, decay, policy=policy,
                                   rank=rank, rank_max=rank_max,
                                   chunk_tokens=chunk)
    inputs, arm = k5a_arm_scan(query, payload, decay, policy, rank, rank_max,
                               chunk, collect=True, upstream=upstream)
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(arm.schedule, name),
                           getattr(record.schedule, name)), (label, name)
    assert torch.equal(arm.weights, record.weights), (label, "weights")
    assert torch.equal(arm.reads, record.reads), (label, "reads")
    assert torch.equal(arm.state, record.state), (label, "state")
    assert torch.equal(arm.live, record.live), (label, "live")
    return inputs, arm, record


def assert_graph_bit_equal_to_arm(query, payload, decay, policy, rank,
                                  rank_max, chunk, label, upstream):
    """The op-pair claim: eager, captured and replayed calls all bit-equal."""
    inputs, arm = k5a_arm_scan(query, payload, decay, policy, rank, rank_max,
                               chunk, upstream=upstream)
    reset_slots()
    slot = next_slot_id()
    outputs = []
    for call in range(3):      # eager, capture+replay, replay
        q = query.detach().clone().requires_grad_(True)
        p = payload.detach().clone().requires_grad_(True)
        d = decay.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = whole_scan(q, p, d, policy=policy, rank=rank,
                               rank_max=rank_max, chunk_tokens=chunk,
                               slot_id=slot)
        (result.reads * upstream).sum().backward()
        outputs.append((result, q.grad, p.grad, d.grad))
    for call, (result, dq, dp, dd) in enumerate(outputs):
        assert torch.equal(result.reads, arm.reads), (label, call, "reads")
        assert torch.equal(result.state, arm.state), (label, call, "state")
        assert torch.equal(result.live, arm.live), (label, call, "live")
        assert torch.equal(dq, inputs[0].grad), (label, call, "dquery")
        assert torch.equal(dp, inputs[1].grad), (label, call, "dpayload")
        assert torch.equal(dd, inputs[2].grad), (label, call, "ddecay")
    # determinism: two replays bit-identical on every channel
    first, second = outputs[1], outputs[2]
    assert torch.equal(first[0].reads, second[0].reads), (label, "replay reads")
    assert torch.equal(first[0].state, second[0].state), (label, "replay state")
    assert torch.equal(first[1], second[1]), (label, "replay dquery")
    assert torch.equal(first[2], second[2]), (label, "replay dpayload")
    assert torch.equal(first[3], second[3]), (label, "replay ddecay")
    ledger = dict(DISPATCH)
    assert ledger["fwd_graph_replays"] >= 2 and ledger["bwd_graph_replays"] >= 2, \
        (label, ledger)
    return outputs


# ----------------------------------------------------------------------- gate A


@needs_cuda
def test_gate_a_record_bit_equal_to_the_k5a_arm_on_all_five_ops():
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, 64, 2, 8, generator=generator).to(DEVICE)
    inputs, arm, record = assert_record_bit_equal_to_arm(
        query, payload, decay, policy, 5, 5, 16, "five-ops", upstream)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        op_histogram)
    histogram = op_histogram(arm.schedule)
    assert all(histogram[name] > 0 for name in
               ("append", "correct", "pop", "merge")), histogram


@needs_cuda
@pytest.mark.parametrize("tokens,chunk", [(100, 64), (65, 64), (33, 16), (7, 3)])
def test_gate_a_ragged_and_short_chunks_agree_bit_for_bit(tokens, chunk):
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, tokens, 2, 7, 8, tokens)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, tokens, 2, 7, generator=generator).to(DEVICE)
    assert_record_bit_equal_to_arm(query, payload, decay, policy, 7, 7, chunk,
                                   f"ragged {tokens}/{chunk}", upstream)


@needs_cuda
def test_gate_a_a_rank_prefix_agrees_bit_for_bit():
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 10, 3)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, 128, 2, 8, generator=generator).to(DEVICE)
    assert_record_bit_equal_to_arm(query, payload, decay, policy, 6, 16, 32,
                                   "rank-prefix", upstream)
    inputs, arm = k5a_arm_scan(query, payload, decay, policy, 6, 16, 32,
                               upstream=upstream)
    reset_slots()
    slot = next_slot_id()
    for _ in range(3):
        q = query.detach().clone().requires_grad_(True)
        p = payload.detach().clone().requires_grad_(True)
        d = decay.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = whole_scan(q, p, d, policy=policy, rank=6, rank_max=16,
                                chunk_tokens=32, slot_id=slot)
        (result.reads * upstream).sum().backward()
        assert torch.equal(result.reads, arm.reads), "prefix graph reads"
        assert torch.equal(q.grad, inputs[0].grad), "prefix graph dquery"
        assert torch.equal(d.grad, inputs[2].grad), "prefix graph ddecay"


@needs_cuda
def test_gate_a_the_registered_shape_under_bf16_autocast_agrees_bit_for_bit():
    """The model's own regime, the whole 2048-token window in one op."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(REG_BATCH, REG_TOKENS, REG_HEADS,
                                            REG_VALUE_DIM, 512, 77)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(REG_BATCH, REG_TOKENS, REG_HEADS, REG_VALUE_DIM,
                           generator=generator).to(DEVICE)
    assert_record_bit_equal_to_arm(query, payload, decay, policy, REG_RANK,
                                   REG_RANK, REG_CHUNK, "registered bf16",
                                   upstream)


@needs_cuda
def test_gate_a_the_decision_flip_test_on_real_routed_fixtures():
    """THE MANDATORY flip test, A-phase's bar matched: the router the
    whole-scan walk runs, against the registered ``route_chunk`` on real
    chunk-entry states from a real routed scan at the registered shape under
    bf16 autocast -- every lane of every schedule field, zero flips."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(REG_BATCH, REG_TOKENS, REG_HEADS,
                                            REG_VALUE_DIM, 512, 77)
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=REG_RANK, rank_max=REG_RANK,
                          chunk_tokens=REG_CHUNK, fused=False)
    flips = 0
    lanes = 0
    for index, (state, live, qc, pc) in enumerate(records):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager = route_chunk(index, index * REG_CHUNK, state, live, qc, pc,
                                policy, REG_RANK)
            fused = routing_diff_route_chunk(index, index * REG_CHUNK, state,
                                             live, qc, pc, policy, REG_RANK)
        for name in SCHEDULE_FIELDS:
            a, b = getattr(eager[0], name), getattr(fused[0], name)
            flips += int((a != b).sum())
            lanes += a.numel()
    assert flips == 0, f"{flips} decision flips in {lanes} lanes"
    print(f"flip test: 0 flips in {lanes} lanes over {len(records)} chunks")


# ----------------------------------------------------------------------- gate B


@needs_cuda
def test_gate_b_weights_within_the_remeasured_floor():
    """The K23 discipline, floors re-measured first: the eager router's
    weights against two orderings of the same math plus the fp64 oracle, then
    the weights of the router the whole-scan walk runs (K5A's kernels, the
    values the record is bit-equal to) inside NOISE_FACTOR x that floor, on
    the same pinned chunk-entry states."""
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender \
        import NOISE_FACTOR
    from experiments.expressivity_tasks.x45_matrix_emender.test_k5a_routing_diff \
        import recorded_chunk_states, weights_floor
    policy = RoutingPolicy(ops=True)
    for label, stream, rank, autocast in (
        ("fp32", (2, 128, 2, 8, 10, 4), 8, False),
        ("registered_bf16", (4, 512, 2, 64, 512, 77), 64, True)):
        query, payload, decay = distinct_stream(*stream[:5], seed=stream[5])
        chunk = 64
        records = recorded_chunk_states(query, payload, decay, policy, rank,
                                        chunk, autocast=autocast)
        worst_floor, worst_dev, worst_median = 0.0, 0.0, 0.0
        for state, live, qc, pc in records[::max(1, len(records) // 4)]:
            floor = weights_floor(qc, state[:, :, :rank], live[:, :, :rank],
                                  policy.read_scale, autocast=autocast)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                eager = route_chunk(0, 0, state, live, qc, pc, policy, rank)
                walked = routing_diff_route_chunk(0, 0, state, live, qc, pc,
                                                 policy, rank)
            delta = (eager[1].double() - walked[1].double()).abs()
            worst_floor = max(worst_floor, floor)
            worst_dev = max(worst_dev, float(delta.max()))
            worst_median = max(worst_median, float(delta.median()))
        print(f"{label}: floor={worst_floor:.3e} max_dev={worst_dev:.3e} "
              f"median_dev={worst_median:.3e}")
        assert worst_dev <= NOISE_FACTOR * max(worst_floor, 1e-30), \
            (label, worst_dev, worst_floor)


@needs_cuda
def test_gate_b_gradients_bit_equal_to_the_k5a_arm():
    """The K4 grad gates, unchanged and stronger: the whole-scan adjoint is
    bit-equal to the per-chunk arm's (same kernels, same order), which the
    K5A receipts already hold inside the measured floors."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 512, 2, 16, 32, 31)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, 512, 2, 16, generator=generator).to(DEVICE)
    inputs, arm = k5a_arm_scan(query, payload, decay, policy, 16, 16, 64,
                               upstream=upstream)
    reset_slots()
    slot = next_slot_id()
    q = query.detach().clone().requires_grad_(True)
    p = payload.detach().clone().requires_grad_(True)
    d = decay.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = whole_scan(q, p, d, policy=policy, rank=16, rank_max=16,
                            chunk_tokens=64, slot_id=slot)
    (result.reads * upstream).sum().backward()
    assert torch.equal(q.grad, inputs[0].grad), "dquery"
    assert torch.equal(p.grad, inputs[1].grad), "dpayload"
    assert torch.equal(d.grad, inputs[2].grad), "ddecay"


# ----------------------------------------------------------------------- gate C


@needs_cuda
def test_gate_c_the_graph_replays_are_bit_equal_to_the_arm():
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 256, 2, 8, 24, 5)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, 256, 2, 8, generator=generator).to(DEVICE)
    assert_graph_bit_equal_to_arm(query, payload, decay, policy, 8, 8, 64,
                                  "graph", upstream)


@needs_cuda
def test_gate_c_compile_hosts_the_op_pair_and_decisions_match():
    """No InductorError, and the compiled wrapper's outputs and gradients are
    bit-identical to the uncompiled path's and to the arm's."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 512, 2, 16, 32, 31)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(2, 512, 2, 16, generator=generator).to(DEVICE)
    inputs, arm = k5a_arm_scan(query, payload, decay, policy, 16, 16, 64,
                              upstream=upstream)
    reset_slots()
    compiled = compiled_whole_scan("default")
    slot = next_slot_id()
    for call in range(3):
        q = query.detach().clone().requires_grad_(True)
        p = payload.detach().clone().requires_grad_(True)
        d = decay.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = compiled(q, p, d, policy=policy, rank=16, rank_max=16,
                              chunk_tokens=64, slot_id=slot)
        (result.reads * upstream).sum().backward()
        assert torch.equal(result.reads, arm.reads), (call, "reads")
        assert torch.equal(q.grad, inputs[0].grad), (call, "dquery")
        assert torch.equal(p.grad, inputs[1].grad), (call, "dpayload")
        assert torch.equal(d.grad, inputs[2].grad), (call, "ddecay")


# ----------------------------------------------------------------------- gate E


@needs_cuda
def test_gate_e_refusals_take_the_per_chunk_stack_verbatim():
    """Growth, the append-only control, rank 512 and an explicit initial state
    are refused by name to the per-chunk stack, and the refusal is bit-equal
    to calling that stack directly."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 12, 9)
    chunk = 32

    def reference(pol, rank=8, rank_max=8, initial_state=None,
                 initial_live=None):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return routed_chunk_scan(
                query, payload, decay, router=routing_diff_route_chunk,
                policy=pol, rank=rank, rank_max=rank_max, chunk_tokens=chunk,
                initial_state=initial_state, initial_live=initial_live,
                fused=False, trainable_fused=True)

    reset_slots()
    growing = RoutingPolicy(ops=True, grow=True, grow_window=2,
                            grow_trigger=0.9)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refused = whole_scan(query, payload, decay, policy=growing, rank=8,
                             rank_max=16, chunk_tokens=chunk)
    honest = reference(growing, rank=8, rank_max=16)
    assert torch.equal(refused.reads, honest.reads), "growth refusal"
    assert DISPATCH["fallbacks"] >= 1

    append_only = RoutingPolicy(ops=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refused = whole_scan(query, payload, decay, policy=append_only,
                              rank=8, rank_max=8, chunk_tokens=chunk)
    honest = reference(append_only)
    assert torch.equal(refused.reads, honest.reads), "append-only refusal"

    state0 = torch.randn(2, 2, 8, 8, device=DEVICE)
    live0 = torch.ones(2, 2, 8, dtype=torch.bool, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refused = whole_scan(query, payload, decay, policy=policy, rank=8,
                             rank_max=8, chunk_tokens=chunk,
                             initial_state=state0, initial_live=live0)
    honest = reference(policy, initial_state=state0, initial_live=live0)
    assert torch.equal(refused.reads, honest.reads), "initial-state refusal"


@needs_cuda
def test_gate_e_rank_512_is_refused_by_name():
    """The pathology shape: the whole-scan path refuses rank 512 (K5A's
    register-tile budget, and the persistent kernel's), and the refusal is
    the registered per-chunk path verbatim."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 512, 9)
    rank = 512
    reset_slots()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        refused = whole_scan(query, payload, decay, policy=policy, rank=rank,
                             rank_max=rank, chunk_tokens=64)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        honest = routed_chunk_scan(query, payload, decay,
                                   router=routing_diff_route_chunk,
                                   policy=policy, rank=rank, rank_max=rank,
                                   chunk_tokens=64, fused=False,
                                   trainable_fused=True)
    assert torch.equal(refused.reads, honest.reads)
    assert torch.equal(refused.state, honest.state)
