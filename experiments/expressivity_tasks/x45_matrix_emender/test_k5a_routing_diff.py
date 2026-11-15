"""K5A -- the routing region's differentiable channels fused, with its adjoint.

The parity discipline is the settled ruling, split by channel:

* **Decisions, slots, op codes, liveness and payloads are bit-exact** -- no
  noise floor, no allowance.  The fused router consumes the eager
  preprocessing's own ``similarity`` and ``strength`` tensors (their reduction
  orders are the decision semantics; the probe in ``receipts/k5a_routing_diff.json``
  shows they cannot be reproduced from Triton), so the discrete channel and the
  payload lanes are bit-identical to ``schedule.route_chunk`` **by construction**
  -- and the mandatory **decision-flip test** runs the fused path against the
  registered path on real routed fixtures and asserts zero flips.
* **The read WEIGHTS are values, not decisions**: they are the one recomputed
  channel and are held to the K23 floor discipline -- a fp32 floor is measured
  FIRST on the same fixture (two orderings of the same math, and the fp64
  oracle), and the fused deviation must stay within ``NOISE_FACTOR`` times it,
  max and median both reported.
* **The routing gradients** (the adjoint of the read weights and the payload
  lanes into ``dquery``/``dstate``/``dpayload``) follow the same floor
  discipline against the eager router's autograd, with the fp64 adjoint as the
  oracle.  ``dpayload`` is asserted bit-exact: it is a masked passthrough of
  the incoming gradient, so any difference would be a bug, not a floor.

What is tested, per section:

* **A** -- bit-exact parity on pinned chunk-entry states, through whole routed
  scans (all five ops, pop-saturated, ragged chunks, rank prefix, growth, the
  registered bf16 shape), on the duplicate-row tie fixture, and the
  **decision-flip test** on real routed fixtures (eager and compiled regimes).
* **B** -- the measured floors and the fused deviations within them, forward
  weights and the three gradients.
* **C** -- compile hosting: ``torch.compile`` traces the op pair without an
  ``InductorError``, the compiled fused router's decisions are bit-identical
  to the compiled eager router's, and the backward runs through the compiled
  region.
* **E** -- determinism (two runs bit-identical), the rank-512 pathology
  (refused to the eager router by name, still correct), and the append-only
  control.

Every test needs a CUDA device (the kernels are Triton).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k5a_routing_diff import (
    K5A_MAX_RANK, routing_diff_route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    routed_chunk_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    EPS, RoutingPolicy, route_chunk)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

REG_BATCH, REG_TOKENS, REG_HEADS, REG_RANK, REG_VALUE_DIM = 8, 2048, 2, 128, 128
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after", "payload")


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed,
                    device=DEVICE):
    """The lane's canonical fixture: recurring directions make CORRECT and MERGE
    fire; a pool wider than the rank drives APPEND to exhaust and POP to appear."""
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).to(device) * 0.5
    directions = torch.randn(distinct, value_dim, generator=generator).to(device)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), 0.95, device=device)
    return query, payload, decay


def recorded_chunk_states(query, payload, decay, policy, rank, chunk_tokens,
                          autocast=False):
    """Pin every chunk's routing inputs so router parity is deterministic."""
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                         enabled=autocast):
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=rank, rank_max=rank, chunk_tokens=chunk_tokens,
                          fused=False)
    return records


def assert_schedule_bit_exact(eager, fused, label):
    """The discrete channel and the payloads: bit-exact or the gate fails."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(eager[0], name), getattr(fused[0], name)), \
            (label, name)


def assert_routers_agree(query, payload, decay, policy, rank, rank_max=None,
                         chunk_tokens=16, autocast=False, label="",
                         midchunk_floor=True):
    """Whole-scan parity: schedule bit-equal always; the continuous channels
    (weights, reads, state) against the eager arm's own run-to-run noise
    (the scan's duplicate-slot ``scatter_add_`` is atomics-nondeterministic,
    both arms alike) AND the measured read-weights floor -- the fused weights
    are floor-class, and the reads/state inherit that class through the scan,
    so the honest bound is NOISE_FACTOR x the floor measured on the same
    fixture first, never a fitted constant."""
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender \
        import NOISE_FACTOR
    rank_max = rank if rank_max is None else rank_max

    def run(router):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return routed_chunk_scan(
                query, payload, decay, router=router, policy=policy, rank=rank,
                rank_max=rank_max, chunk_tokens=chunk_tokens, collect=True)

    eager, eager_again, fused = run(route_chunk), run(route_chunk), \
        run(routing_diff_route_chunk)
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(eager.schedule, name),
                           getattr(fused.schedule, name)), (label, name)
    floor = 0.0
    if midchunk_floor:
        records = recorded_chunk_states(query, payload, decay, policy, rank,
                                         chunk_tokens, autocast=autocast)
        state, live, qc, pc = records[len(records) // 2]
        floor = weights_floor(qc, state[:, :, :rank], live[:, :, :rank],
                              policy.read_scale, autocast=autocast)
    for name in ("weights", "reads", "state"):
        first, second = getattr(eager, name), getattr(eager_again, name)
        observed = getattr(fused, name)
        if torch.equal(first, second):
            noise = 0.0
        else:
            noise = float((first.double() - second.double()).abs().max())
        deviation = float((first.double() - observed.double()).abs().max())
        assert deviation <= NOISE_FACTOR * max(noise, floor, 1e-30), \
            (label, name, deviation, noise, floor)
    return op_histogram(eager.schedule)


# ------------------------------------------------------------------ the floors


def weights_floor(query, rows, live, scale, autocast):
    """The K23 discipline for the read-weights channel: the max deviation
    between two orderings of the same math and the fp64 oracle, measured on
    the same fixture BEFORE the fused path is judged against it."""
    from experiments.expressivity_tasks.x45_matrix_emender.read import (
        read_weights)

    def eager_weights():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return read_weights(query, rows, live, scale)

    def explicit_weights():
        # a second ordering of the same math: the softmax written out
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            uq = query / query.norm(dim=-1, keepdim=True).clamp_min(EPS)
            ur = rows / rows.norm(dim=-1, keepdim=True).clamp_min(EPS)
            scores = scale * (uq @ ur.transpose(-1, -2))
            scores = scores.masked_fill(~live.unsqueeze(-2), float("-inf"))
            all_dead = (~live.any(-1)).unsqueeze(-1).unsqueeze(-1)
            safe = torch.where(all_dead, torch.zeros_like(scores), scores)
            shifted = safe - safe.max(dim=-1, keepdim=True).values
            probs = torch.where(all_dead, torch.zeros_like(safe),
                                torch.exp(shifted)
                                / torch.exp(shifted).sum(dim=-1, keepdim=True))
        strength = rows.norm(dim=-1) * live.to(rows.dtype)
        return probs * strength.unsqueeze(-2)

    def oracle_weights():
        q = query.double()
        r = rows.double()
        l = live.double()
        uq = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
        ur = r / r.norm(dim=-1, keepdim=True).clamp_min(EPS)
        scores = scale * (uq @ ur.transpose(-1, -2))
        scores = scores.masked_fill(~l.unsqueeze(-2).bool(), float("-inf"))
        all_dead = (~l.any(-1)).unsqueeze(-1).unsqueeze(-1)
        safe = torch.where(all_dead, torch.zeros_like(scores), scores)
        probs = torch.softmax(safe, dim=-1)
        probs = torch.where(all_dead, torch.zeros_like(probs), probs)
        strength = r.norm(dim=-1) * l
        return (probs * strength.unsqueeze(-2)).to(query.dtype)

    first = eager_weights().double()
    order = float((first - explicit_weights().double()).abs().max())
    exact = float((first - oracle_weights().double()).abs().max())
    return max(order, exact)


def gradient_floor(query, payload, state, live, policy, rank, dweights, dpay,
                   autocast):
    """The adjoint floor: the eager router's autograd gradients against the
    fp64 adjoint oracle of the same math (the discrete channel pinned from the
    eager schedule, as the recorded-schedule doctrine demands)."""
    active = live[:, :, :rank]
    rows = state[:, :, :rank]

    def eager_grads():
        q = query.detach().clone().requires_grad_(True)
        p = payload.detach().clone().requires_grad_(True)
        r = state.detach().clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            schedule, weights = route_chunk(0, 0, r, live, q, p, policy, rank)
        loss = (weights * dweights).sum() + (schedule.payload * dpay).sum()
        loss.backward()
        return q.grad, p.grad, r.grad

    def oracle_grads():
        from experiments.expressivity_tasks.x45_matrix_emender.read import (
            read_weights)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=autocast):
            schedule, _ = route_chunk(0, 0, state, live, query, payload, policy,
                                      rank)
        carries = ((schedule.op[..., 0] == APPEND) | (
            schedule.op[..., 0] == CORRECT)).permute(0, 2, 1)
        q = query.detach().double().requires_grad_(True)
        p = payload.detach().double().requires_grad_(True)
        r = state.detach().double().requires_grad_(True)
        act = active.double()
        uq = q / q.norm(dim=-1, keepdim=True).clamp_min(EPS)
        ur = r / r.norm(dim=-1, keepdim=True).clamp_min(EPS)
        scores = policy.read_scale * (uq @ ur.transpose(-1, -2))
        scores = scores.masked_fill(~act.unsqueeze(-2).bool(), float("-inf"))
        all_dead = (~act.any(-1)).unsqueeze(-1).unsqueeze(-1)
        safe = torch.where(all_dead, torch.zeros_like(scores), scores)
        probs = torch.softmax(safe, dim=-1)
        probs = torch.where(all_dead, torch.zeros_like(probs), probs)
        strength = r.norm(dim=-1) * act
        weights = probs * strength.unsqueeze(-2)
        payload_lane = torch.where(carries.unsqueeze(-1), p,
                                   torch.zeros_like(p))
        weights.backward(dweights.double())
        payload_lane.backward(dpay[..., 0, :].permute(0, 2, 1, 3).double(),
                              inputs=[q, p, r])
        return q.grad, p.grad, r.grad

    qe, pe, re = eager_grads()
    qo, po, ro = oracle_grads()
    floors = []
    for e, o in ((qe, qo), (pe, po), (re, ro)):
        floors.append(float((e.double() - o.double()).abs().max()))
    return max(floors), (qe, pe, re)


def assert_grads_within_floor(query, payload, state, live, policy, rank,
                              autocast, label):
    """The fused adjoint vs the eager autograd, both against the fp64 oracle:
    the fused deviation must stay within NOISE_FACTOR times the eager arm's own
    floor, and dpayload (a masked passthrough) must be bit-exact."""
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender \
        import NOISE_FACTOR

    def upstream(router_inputs):
        generator = torch.Generator().manual_seed(1234)
        q, p, r = router_inputs
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                             enabled=autocast):
            schedule, weights = route_chunk(0, 0, r, live, q, p, policy, rank)
        dweights = torch.randn(weights.shape, generator=generator).to(DEVICE)
        dpay = torch.randn(schedule.payload.shape, generator=generator).to(DEVICE)
        return dweights, dpay

    dweights, dpay = upstream((query, payload, state))
    floor, (qe, pe, re) = gradient_floor(query, payload, state, live, policy,
                                         rank, dweights, dpay, autocast)

    q = query.detach().clone().requires_grad_(True)
    p = payload.detach().clone().requires_grad_(True)
    r = state.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        schedule, weights = routing_diff_route_chunk(0, 0, r, live, q, p, policy,
                                                     rank)
    loss = (weights * dweights).sum() + (schedule.payload * dpay).sum()
    loss.backward()
    for name, eager, fused in (("dquery", qe, q.grad), ("dpayload", pe, p.grad),
                               ("dstate", re, r.grad)):
        deviation = float((eager.double() - fused.double()).abs().max())
        print(f"{label} {name}: floor={floor:.3e} deviation={deviation:.3e}")
        if name == "dpayload":
            assert torch.equal(eager, fused), (label, name, deviation)
        else:
            assert deviation <= NOISE_FACTOR * max(floor, 1e-30), \
                (label, name, deviation, floor)


# ----------------------------------------------------------------------- gate A


@needs_cuda
def test_gate_a_router_outputs_bit_identical_on_pinned_chunk_states():
    """The core parity claim on identical inputs: every discrete field and the
    payload bit-exact, the weights inside the measured floor."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 10, 4)
    records = recorded_chunk_states(query, payload, decay, policy, 8, 32)
    assert len(records) == 4
    for index, (state, live, qc, pc) in enumerate(records):
        eager = route_chunk(index, index * 32, state, live, qc, pc, policy, 8)
        fused = routing_diff_route_chunk(index, index * 32, state, live, qc, pc,
                                         policy, 8)
        assert_schedule_bit_exact(eager, fused, f"chunk {index}")
        floor = weights_floor(qc, state[:, :, :8], live[:, :, :8],
                              policy.read_scale, autocast=False)
        deviation = float((eager[1].double() - fused[1].double()).abs().max())
        median = float((eager[1].double() - fused[1].double()).abs().median())
        assert deviation <= 8.0 * max(floor, 1e-30), (index, deviation, floor)
        assert median <= deviation


@needs_cuda
def test_gate_a_decisions_bit_identical_with_all_five_ops():
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    histogram = assert_routers_agree(query, payload, decay, policy, 5,
                                      chunk_tokens=16, label="five-ops")
    assert all(histogram[name] > 0 for name in
               ("append", "correct", "pop", "merge")), histogram
    assert histogram["none"] > 0, histogram


@needs_cuda
def test_gate_a_a_pop_saturated_store_agrees_bit_for_bit():
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 256, 2, 8, 24, 5)
    histogram = assert_routers_agree(query, payload, decay, policy, 8,
                                      chunk_tokens=64, label="pop-saturated")
    assert histogram["pop"] > 0 and histogram["append"] > 0, histogram


@needs_cuda
@pytest.mark.parametrize("tokens,chunk", [(7, 3), (100, 64), (65, 64), (33, 16)])
def test_gate_a_ragged_and_short_chunk_lengths_agree(tokens, chunk):
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, tokens, 2, 7, 8, tokens)
    assert_routers_agree(query, payload, decay, policy, 7, chunk_tokens=chunk,
                         label=f"ragged {tokens}/{chunk}")


@needs_cuda
def test_gate_a_a_rank_prefix_and_a_growth_policy_agree():
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 10, 3)
    assert_routers_agree(query, payload, decay, policy, 6, rank_max=16,
                         chunk_tokens=32, label="rank-prefix")
    growing = RoutingPolicy(ops=True, grow=True, grow_window=2, grow_trigger=0.9)
    query, payload, decay = distinct_stream(2, 256, 2, 8, 12, 9)
    histogram = assert_routers_agree(query, payload, decay, growing, 8,
                                      rank_max=32, chunk_tokens=64, label="grow")
    assert histogram["append"] > 0, histogram


@needs_cuda
def test_gate_a_the_registered_shape_under_bf16_autocast_agrees_bit_for_bit():
    """The model's own regime; the fixture where a threshold rounded one dtype
    wrong, or a similarity recomputed one ulp off, would flip decisions."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(REG_BATCH, REG_TOKENS, REG_HEADS,
                                            REG_VALUE_DIM, 512, 77)
    histogram = assert_routers_agree(query, payload, decay, policy, REG_RANK,
                                      chunk_tokens=64, autocast=True,
                                      label="registered bf16")
    assert histogram["pop"] > 0 and histogram["merge"] > 0, histogram


@needs_cuda
def test_gate_a_exact_similarity_ties_break_to_the_lower_index():
    batch, heads, tokens, rank, value_dim = 2, 2, 64, 8, 8
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).to(DEVICE) * 0.5
    directions = torch.randn(4, value_dim, generator=generator).to(DEVICE)
    choice = torch.randint(0, 4, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    live = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    state = torch.zeros(batch, heads, rank, value_dim, device=DEVICE)
    state[:] = directions[torch.arange(rank, device=DEVICE) // 2] \
        .view(1, 1, rank, value_dim)
    for router, name in ((route_chunk, "eager"),
                         (routing_diff_route_chunk, "fused")):
        schedule, _ = router(0, 0, state, live,
                             query[:, :64].permute(0, 2, 1, 3).contiguous(),
                             payload[:, :64].permute(0, 2, 1, 3).contiguous(),
                             RoutingPolicy(ops=True), rank)
        if name == "eager":
            reference = schedule
        else:
            for field in SCHEDULE_FIELDS:
                assert torch.equal(getattr(reference, field),
                                   getattr(schedule, field)), field
    assert bool(((reference.slot[..., 0] % 2) == 0).any())


@needs_cuda
def test_gate_a_the_decision_flip_test_on_real_routed_fixtures():
    """The MANDATORY flip test of the settled ruling: on real routed fixtures
    (real chunk-entry states from a real routed scan, the registered shape,
    bf16 autocast), the fused path's decisions must show ZERO flips against
    the registered path -- every lane of every schedule field, every chunk.
    The fused router consumes the registered similarity, so this is bit-equality
    by construction; the test is the proof, run where a flip would hide."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(REG_BATCH, REG_TOKENS, REG_HEADS,
                                            REG_VALUE_DIM, 512, 77)
    records = recorded_chunk_states(query, payload, decay, policy, REG_RANK,
                                    64, autocast=True)
    flips = 0
    lanes = 0
    for index, (state, live, qc, pc) in enumerate(records):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager = route_chunk(index, index * 64, state, live, qc, pc,
                                policy, REG_RANK)
            fused = routing_diff_route_chunk(index, index * 64, state, live,
                                             qc, pc, policy, REG_RANK)
        for name in SCHEDULE_FIELDS:
            a, b = getattr(eager[0], name), getattr(fused[0], name)
            flips += int((a != b).sum())
            lanes += a.numel()
    assert flips == 0, f"{flips} decision flips in {lanes} lanes"
    print(f"flip test: 0 flips in {lanes} lanes over {len(records)} chunks")


# ----------------------------------------------------------------------- gate B


@needs_cuda
def test_gate_b_read_weights_within_the_measured_floor():
    """The K23 discipline on the one recomputed channel: floor first, then the
    fused deviation inside NOISE_FACTOR x it -- max and median both reported."""
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender \
        import NOISE_FACTOR
    policy = RoutingPolicy(ops=True)
    for label, stream, rank, autocast in (
        ("fp32", (2, 128, 2, 8, 10, 4), 8, False),
        ("registered_bf16", (4, 512, 2, 64, 512, 77), 64, True)):
        query, payload, decay = distinct_stream(*stream[:5], seed=stream[5])
        records = recorded_chunk_states(query, payload, decay, policy, rank, 64,
                                        autocast=autocast)
        worst_floor, worst_dev, worst_median = 0.0, 0.0, 0.0
        for state, live, qc, pc in records[::max(1, len(records) // 4)]:
            floor = weights_floor(qc, state[:, :, :rank], live[:, :, :rank],
                                  policy.read_scale, autocast=autocast)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=autocast):
                eager = route_chunk(0, 0, state, live, qc, pc, policy, rank)
                fused = routing_diff_route_chunk(0, 0, state, live, qc, pc,
                                                 policy, rank)
            delta = (eager[1].double() - fused[1].double()).abs()
            worst_floor = max(worst_floor, floor)
            worst_dev = max(worst_dev, float(delta.max()))
            worst_median = max(worst_median, float(delta.median()))
        print(f"{label}: floor={worst_floor:.3e} max_dev={worst_dev:.3e} "
              f"median_dev={worst_median:.3e}")
        assert worst_dev <= NOISE_FACTOR * max(worst_floor, 1e-30), \
            (label, worst_dev, worst_floor)


@needs_cuda
def test_gate_b_gradients_within_the_measured_floor():
    """The adjoint vs the eager autograd, inside NOISE_FACTOR x the eager arm's
    own floor against the fp64 adjoint oracle; dpayload bit-exact."""
    policy = RoutingPolicy(ops=True)
    for label, stream, rank, autocast in (
        ("fp32", (2, 64, 2, 8, 6, 23), 8, False),
        ("bf16", (2, 256, 2, 16, 24, 5), 16, True)):
        query, payload, decay = distinct_stream(*stream[:5], seed=stream[5])
        records = recorded_chunk_states(query, payload, decay, policy, rank, 64,
                                        autocast=autocast)
        state, live, qc, pc = records[len(records) // 2]
        assert_grads_within_floor(qc, pc, state, live, policy, rank, autocast,
                                  label)


# ----------------------------------------------------------------------- gate C


@needs_cuda
def test_gate_c_compile_hosts_the_op_pair_and_decisions_match():
    """No InductorError (the K4 blocker class), and the compiled fused
    router's decisions are bit-identical to the compiled EAGER router's -- the
    registered arm's own compiled decision semantics, unchanged."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 512, 2, 16, 32, 31)
    compiled = torch.compile(routed_chunk_scan, dynamic=False)
    results = []
    for router in (route_chunk, routing_diff_route_chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            results.append(compiled(query, payload, decay, router=router,
                                    policy=policy, rank=16, rank_max=16,
                                    chunk_tokens=64, fused=False, collect=True))
    eager_compiled, fused_compiled = results
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(eager_compiled.schedule, name),
                           getattr(fused_compiled.schedule, name)), name
    fused_c32 = compiled(query, payload, decay, router=routing_diff_route_chunk,
                         policy=policy, rank=16, rank_max=16, chunk_tokens=64,
                         fused=False, collect=True)
    eager32 = routed_chunk_scan(query, payload, decay, router=route_chunk,
                                 policy=policy, rank=16, rank_max=16,
                                 chunk_tokens=64, fused=False, collect=True)
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(eager32.schedule, name),
                           getattr(fused_c32.schedule, name)), name


@needs_cuda
def test_gate_c_backward_runs_through_the_compiled_region():
    """The register_autograd backward is traced into the joint graph; the
    reverse op is opaque, the Triton launches stay out of the trace, and the
    routing gradients actually flow."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 512, 2, 16, 32, 31)
    compiled = torch.compile(routed_chunk_scan, dynamic=False)
    inputs = [tensor.detach().clone().requires_grad_(True)
              for tensor in (query, payload, decay)]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = compiled(*inputs, router=routing_diff_route_chunk,
                          policy=policy, rank=16, rank_max=16, chunk_tokens=64,
                          fused=False)
    (result.reads.sum() + result.state.sum()).backward()
    for tensor, name in zip(inputs, ("query", "payload", "decay")):
        assert tensor.grad is not None and float(tensor.grad.abs().sum()) > 0, name


# ----------------------------------------------------------------------- gate E


@needs_cuda
def test_gate_e_deterministic_rerun():
    """Two fused runs on the same pinned inputs are bit-identical, every
    channel (no atomics anywhere in the pair)."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 256, 2, 8, 24, 5)
    records = recorded_chunk_states(query, payload, decay, policy, 8, 64)
    state, live, qc, pc = records[len(records) // 2]
    runs = []
    for _ in range(2):
        schedule, weights = routing_diff_route_chunk(0, 0, state, live, qc, pc,
                                                     policy, 8)
        runs.append((schedule, weights))
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(runs[0][0], name),
                           getattr(runs[1][0], name)), name
    assert torch.equal(runs[0][1], runs[1][1])


@needs_cuda
def test_gate_e_rank_512_is_refused_to_the_eager_router_by_name():
    """The pathology shape: the fused pair holds the whole rank axis in one
    register tile, so rank 512 is refused -- and the refusal is the registered
    eager router verbatim, bit-identical by construction."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 512, 9)
    records = recorded_chunk_states(query, payload, decay, policy, 64, 64)
    state, live, qc, pc = records[1]
    rank = 512
    state = torch.zeros(2, 2, rank, 8, device=DEVICE)
    state[:] = records[1][0][:, :, :64, :].repeat(1, 1, rank // 64, 1)
    live = torch.zeros(2, 2, rank, dtype=torch.bool, device=DEVICE)
    live[:, :, :64] = records[1][1][:, :, :64]
    assert rank > K5A_MAX_RANK
    eager = route_chunk(1, 64, state, live, qc, pc, policy, rank)
    fused = routing_diff_route_chunk(1, 64, state, live, qc, pc, policy, rank)
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(eager[0], name), getattr(fused[0], name)), name
    assert torch.equal(eager[1], fused[1])


@needs_cuda
def test_the_append_only_control_is_the_eager_branch_verbatim():
    policy = RoutingPolicy(ops=False)
    query, payload, decay = distinct_stream(2, 96, 2, 8, 6, 17)
    assert_routers_agree(query, payload, decay, policy, 8, chunk_tokens=32,
                         label="append-only")
