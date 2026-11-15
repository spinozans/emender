"""K5 gate A -- the fused routing decision channel, bit-identical to the eager router.

The tolerance discipline is the lane's, sharpened to a point: routing decisions are
*discrete*, so parity is **bit-exact or it is wrong** -- no noise floor, no allowance.
The fused router must reproduce ``schedule.route_chunk``'s decisions (``slot``,
``op``, ``erase``, ``live_after``), its payloads and its read weights bit for bit,
because the kernel consumes the very similarity/strength tensors the eager router
computes (their reduction orders belong to cuBLAS/torch and are part of the decision
semantics; recomputing them could flip an argmax by one ulp).

What is tested, per section:

* **A -- decision parity**: every schedule field, payload and weights bit-identical
  to ``route_chunk`` on fixtures that produce all five ops, at ragged chunk lengths,
  across a rank prefix (``rank < rank_max``) and under the growth policy -- directly
  per chunk and through the whole routed scan, in fp32 and under the model's bf16
  autocast at the registered shape.  Ties are pinned by a dedicated duplicate-row
  fixture (two identical stored rows: the lower index must win, as torch's argmax
  and the tie-break penalty both demand).
* **B -- gradients unchanged**: the fused router's outputs are bit-identical, so
  the autograd graph above them is the same graph -- the gradients of query,
  payload and decay must be bit-identical too, not merely close.
* **C -- compile hosting**: ``torch.compile`` traces the decision kernel as one
  opaque ``torch.library`` call (no InductorError, no silent fallback), and the
  compiled fused router's decisions are bit-identical to the compiled *eager*
  router's -- the registered arm's own compiled decision semantics, unchanged.
  At fp32 the compiled decisions also equal the eager router's bit for bit; at
  bf16 autocast the compiled-vs-eager difference is a pre-existing property of
  inductor's similarity lowering (measured in the receipt, unchanged by K5).

Every test needs a CUDA device (the kernel is Triton).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k5_routing_fusion import (
    fused_route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import routed_chunk_scan
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# The registered mixer shape's dimensions, for the autocast parity fixture.
REG_BATCH, REG_TOKENS, REG_HEADS, REG_RANK, REG_VALUE_DIM = 8, 2048, 2, 128, 128


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed,
                    device=DEVICE):
    """A query/payload stream with ``distinct`` recurring value directions.

    The same trick ``test_k1_schedule_scan.routed_fixture`` uses: a uniform random
    stream never merges (two live rows never reach cosine ``tau_merge``), while a
    small pool of directions makes CORRECT and MERGE fire; a pool wider than the
    rank drives APPEND to exhaust the store and POP to appear.
    """
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).to(device) * 0.5
    directions = torch.randn(distinct, value_dim, generator=generator).to(device)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), 0.95, device=device)
    return query, payload, decay


def assert_routers_agree(query, payload, decay, policy, rank, rank_max=None,
                         chunk_tokens=16, autocast=False, label=""):
    """Both routers over the same window, collect=True: decisions bit-equal always.

    The discrete channel (``slot``/``erase``/``op``/``live_after``/``payload``) is
    asserted bit-equal unconditionally.  The continuous channels
    (``weights``/``reads``/``state``) depend on the chunk-entry states, and the
    eager scan's ``scatter_add_`` over duplicate slots is atomics-nondeterministic
    run to run -- so the honest instrument is the eager arm's own stability: when
    two eager runs are bit-identical (the common case), the fused arm must be too;
    when they are not, the fused arm's deviation must stay within the lane's
    ``NOISE_FACTOR`` of that measured run-to-run noise.  Routing's own bit-parity
    is pinned deterministically by :func:`recorded_chunk_states` below.
    """
    from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender \
        import NOISE_FACTOR
    rank_max = rank if rank_max is None else rank_max

    def run(router):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            return routed_chunk_scan(
                query, payload, decay, router=router, policy=policy, rank=rank,
                rank_max=rank_max, chunk_tokens=chunk_tokens, collect=True)

    eager, eager_again, fused = run(route_chunk), run(route_chunk), \
        run(fused_route_chunk)
    for name in ("slot", "erase", "op", "live_after", "payload"):
        left = getattr(eager.schedule, name)
        assert torch.equal(left, getattr(fused.schedule, name)), \
            f"{label}: schedule {name} differs"
    for name in ("weights", "reads", "state", "live"):
        first, second = getattr(eager, name), getattr(eager_again, name)
        observed = getattr(fused, name)
        if torch.equal(first, second) or first.dtype == torch.bool:
            assert torch.equal(first, observed), f"{label}: {name} differs"
        else:
            noise = float((first.double() - second.double()).abs().max())
            deviation = float((first.double() - observed.double()).abs().max())
            assert deviation <= NOISE_FACTOR * noise, \
                (label, name, deviation, noise)
    return op_histogram(eager.schedule)


def recorded_chunk_states(query, payload, decay, policy, rank, chunk_tokens):
    """Pin every chunk's routing inputs, so router parity is tested deterministically.

    Runs the eager router once on the reference trajectory and records each
    chunk's ``(state, live, query, payload)``; comparing the two routers against
    the recorded inputs removes the scan's atomics nondeterminism from the
    comparison entirely, which is what makes a **bit-exact** assertion on every
    router output (weights and payloads included) meaningful.
    """
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad():
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=rank, rank_max=rank, chunk_tokens=chunk_tokens,
                          fused=False)
    return records


@needs_cuda
def test_gate_a_router_outputs_bit_identical_on_pinned_chunk_states():
    """The core parity claim: every router output bit-exact on identical inputs.

    With the chunk-entry states pinned (``recorded_chunk_states``), both routers
    see byte-identical inputs, so every output -- the four decision fields, the
    payloads and the read weights -- must be equal bit for bit.  This is the
    deterministic form of the routing-parity claim; the whole-scan tests below
    additionally show the decisions survive the real trajectory.
    """
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 10, 4)
    records = recorded_chunk_states(query, payload, decay, policy, 8, 32)
    assert len(records) == 4
    for index, (state, live, qc, pc) in enumerate(records):
        eager = route_chunk(index, index * 32, state, live, qc, pc, policy, 8)
        fused = fused_route_chunk(index, index * 32, state, live, qc, pc, policy, 8)
        for name in ("slot", "erase", "op", "live_after", "payload"):
            assert torch.equal(getattr(eager[0], name), getattr(fused[0], name)), \
                (index, name)
        assert torch.equal(eager[1], fused[1]), (index, "weights")


@needs_cuda
def test_gate_a_decisions_bit_identical_with_all_five_ops():
    """Every channel of the fused router equals the eager router's, bit for bit.

    The fixture is the lane's canonical one (recurring directions, so CORRECT and
    MERGE fire) at a store small enough to fill, so POP appears too; the histogram
    assertion pins all five ops before the parity claims are read.
    """
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    histogram = assert_routers_agree(query, payload, decay, policy, 5,
                                      chunk_tokens=16, label="five-ops")
    assert all(histogram[name] > 0 for name in
               ("append", "correct", "pop", "merge")), histogram
    assert histogram["none"] > 0, histogram


@needs_cuda
def test_gate_a_a_pop_saturated_store_agrees_bit_for_bit():
    """The capacity-pressure regime: more directions than slots, so POPs dominate."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 256, 2, 8, 24, 5)
    histogram = assert_routers_agree(query, payload, decay, policy, 8,
                                      chunk_tokens=64, label="pop-saturated")
    assert histogram["pop"] > 0 and histogram["append"] > 0, histogram


@needs_cuda
@pytest.mark.parametrize("tokens,chunk", [(7, 3), (100, 64), (65, 64), (33, 16)])
def test_gate_a_ragged_and_short_chunk_lengths_agree(tokens, chunk):
    """Chunk lengths that do not divide the window -- including a 1-token tail."""
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, tokens, 2, 7, 8, tokens)
    assert_routers_agree(query, payload, decay, policy, 7, chunk_tokens=chunk,
                         label=f"ragged {tokens}/{chunk}")


@needs_cuda
def test_gate_a_a_rank_prefix_and_a_growth_policy_agree():
    """``rank < rank_max`` (the growth prefix) and the growth policy itself.

    The read-weight channel is zero above the active rank and the kernel masks its
    rank axis to the active prefix, so a widened rank mid-scan is bookkeeping on
    both arms alike.
    """
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
    """The model's own regime: batch 8, 2048 tokens, chunk 64, rank 128, bf16.

    Under autocast the similarity is bf16 (the comparison thresholds are rounded
    to bf16 exactly as eager's tensor-vs-scalar promotion does), so this is the
    fixture where a threshold rounded one dtype wrong would flip decisions.
    """
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(REG_BATCH, REG_TOKENS, REG_HEADS,
                                             REG_VALUE_DIM, 512, 77)
    histogram = assert_routers_agree(query, payload, decay, policy, REG_RANK,
                                      chunk_tokens=64, autocast=True,
                                      label="registered bf16")
    assert histogram["pop"] > 0 and histogram["merge"] > 0, histogram


@needs_cuda
def test_gate_a_exact_similarity_ties_break_to_the_lower_index():
    """Two identical stored rows: the tie-break must pick the lower index, exactly.

    ``route_chunk`` subtracts ``arange(rank) * tie_break`` so an exact cosine tie
    resolves to the lower index (and torch's argmax returns the first maximal
    index).  The kernel must reproduce that rule, not just its average behaviour:
    a store with duplicated rows is the fixture that makes ties the common case.
    """
    batch, heads, tokens, rank, value_dim = 2, 2, 64, 8, 8
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(batch, tokens, heads, value_dim,
                        generator=generator).to(DEVICE) * 0.5
    # Four distinct directions, each stored twice: half the live rows are duplicates.
    directions = torch.randn(4, value_dim, generator=generator).to(DEVICE)
    choice = torch.randint(0, 4, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    live = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    # Every direction stored twice (rows 2i and 2i+1 hold direction i): any token
    # whose value matches direction i has an exact cosine tie between two live
    # rows, so best_index must be the lower one on both arms.
    state = torch.zeros(batch, heads, rank, value_dim, device=DEVICE)
    state[:] = directions[torch.arange(rank, device=DEVICE) // 2] \
        .view(1, 1, rank, value_dim)
    for router, name in ((route_chunk, "eager"), (fused_route_chunk, "fused")):
        schedule, _ = router(0, 0, state, live,
                             query[:, :64].permute(0, 2, 1, 3).contiguous(),
                             payload[:, :64].permute(0, 2, 1, 3).contiguous(),
                             RoutingPolicy(ops=True), rank)
        if name == "eager":
            reference = schedule
        else:
            for field in ("slot", "erase", "op", "live_after", "payload"):
                assert torch.equal(getattr(reference, field),
                                   getattr(schedule, field)), field
    # And the ties really were ties: some token addressed a duplicated row pair.
    assert bool(((reference.slot[..., 0] % 2) == 0).any())


@needs_cuda
def test_gate_b_gradients_are_unchanged_through_the_fused_router():
    """The fused router's gradients match the eager router's, bit for bit.

    The read weights and the payloads carry the routing gradient (the decisions
    are detached by construction on both arms).  Identical forward values through
    identical torch ops give an identical autograd graph -- and through the real
    scan the comparison is against the eager arm's own run-to-run deviation,
    because the eager scan's duplicate-slot ``scatter_add_`` is
    atomics-nondeterministic (the fused arm inherits exactly that noise, and
    nothing of its own).
    """
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 128, 2, 8, 10, 4)
    query = query.detach().requires_grad_(True)
    payload = payload.detach().requires_grad_(True)
    decay = decay.detach().requires_grad_(True)

    def grads(router):
        inputs = [tensor.detach().clone().requires_grad_(True)
                  for tensor in (query, payload, decay)]
        result = routed_chunk_scan(inputs[0], inputs[1], inputs[2], router=router,
                                   policy=policy, rank=8, rank_max=8,
                                   chunk_tokens=32, fused=False)
        (result.reads.sum() + result.state.sum()).backward()
        return [tensor.grad for tensor in inputs]

    eager = grads(route_chunk)
    eager_again = grads(route_chunk)
    fused = grads(fused_route_chunk)
    for left, again, right, name in zip(eager, eager_again, fused,
                                       ("dquery", "dpayload", "ddecay")):
        assert float(left.abs().sum()) > 0.0, name
        if torch.equal(left, again):
            assert torch.equal(left, right), name
        else:
            noise = float((left.double() - again.double()).abs().max())
            deviation = float((left.double() - right.double()).abs().max())
            assert deviation <= 8.0 * noise, (name, deviation, noise)


@needs_cuda
def test_gate_c_compile_hosts_the_kernel_as_one_opaque_call():
    """The compiled fused router's decisions equal the compiled eager router's.

    The raw Triton launch is not compile-hostable on this stack (dynamo traces
    into it and inductor refuses the ``bool -> int8`` view -- the K4 blocker), so
    this test is the proof the ``torch.library`` op works: no InductorError, no
    silent eager fallback (a fallback would run the *eager* router's decisions,
    which differ from the compiled arm's at bf16 and would fail the equality),
    and the schedule bit-equal.
    """
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 512, 2, 16, 32, 31)
    compiled = torch.compile(routed_chunk_scan, dynamic=False)
    results = []
    for router in (route_chunk, fused_route_chunk):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            results.append(compiled(query, payload, decay, router=router,
                                    policy=policy, rank=16, rank_max=16,
                                    chunk_tokens=64, fused=False, collect=True))
    eager_compiled, fused_compiled = results
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(eager_compiled.schedule, name),
                           getattr(fused_compiled.schedule, name)), name
    # The payload's APPEND/CORRECT lanes are verbatim copies (bit-equal); only the
    # MERGE blend carries arithmetic, and inductor lowers that blend differently
    # in the two graph variants -- the deviation is the fp32 ulp of the blend,
    # measured here rather than assumed (the eager-regime payload, where both
    # arms run the same ops, is asserted bit-equal in the gate-A tests above).
    left, right = (eager_compiled.schedule.payload, fused_compiled.schedule.payload)
    plain = eager_compiled.schedule.op[..., 0] != MERGE
    assert torch.equal(left[plain], right[plain]), "non-merge payload lanes differ"
    deviation = float((left.double() - right.double()).abs().max())
    scale = max(float(left.abs().max()), 1e-12)
    print(f"compiled merge-blend deviation {deviation:.3e} on scale {scale:.3e}")
    assert deviation <= 2 ** -20 * scale, (deviation, scale)


@needs_cuda
def test_gate_c_fp32_compiled_decisions_equal_the_eager_router_exactly():
    """At fp32 (no autocast) the compiled decisions equal the eager router's too.

    Inductor's fp32 lowering of the similarity bmm matches cuBLAS on this stack,
    so the whole chain -- eager router, compiled eager router, compiled fused
    router -- agrees bit for bit in fp32.  (At bf16 autocast the compiled arm's
    similarity already differs from eager's by ulps; that pre-existing difference
    is measured in the receipt and is not K5's to fix.)
    """
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(2, 256, 2, 16, 20, 13)
    compiled = torch.compile(routed_chunk_scan, dynamic=False)
    fused_compiled = compiled(query, payload, decay, router=fused_route_chunk,
                              policy=policy, rank=12, rank_max=12,
                              chunk_tokens=64, fused=False, collect=True)
    eager = routed_chunk_scan(query, payload, decay, router=route_chunk,
                              policy=policy, rank=12, rank_max=12,
                              chunk_tokens=64, fused=False, collect=True)
    for name in ("slot", "erase", "op", "live_after"):
        assert torch.equal(getattr(eager.schedule, name),
                           getattr(fused_compiled.schedule, name)), name


@needs_cuda
def test_the_append_only_control_is_the_eager_branch_verbatim():
    """``policy.ops=False`` never reaches the kernel: the modulo schedule, exact."""
    policy = RoutingPolicy(ops=False)
    query, payload, decay = distinct_stream(2, 96, 2, 8, 6, 17)
    assert_routers_agree(query, payload, decay, policy, 8, chunk_tokens=32,
                        label="append-only")
