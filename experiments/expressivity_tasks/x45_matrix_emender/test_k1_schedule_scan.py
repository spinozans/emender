"""K1 gates A, B and E for the fused schedule-decoded state scan.

Gate discipline is ``test_matrix_emender.py``'s and ``x44``'s:

* **A** — the scalar channels are not "close": the fused walk must reproduce the
  sequential oracle's **state, liveness and therefore every death boundary
  bit-for-bit** on real routed schedules.  A 1-ulp drift is exactly what moved a
  death boundary in X44 (``RESULT_CHUNK_SCAN.md`` §6.2), so a tolerance would be the
  wrong instrument here.
* **B** — content against the eager *chunked* reference stays within
  ``NOISE_FACTOR`` times a **measured** sequential-vs-sequential floor; the floor is
  measured on the same fixtures and never fitted to the result.
* **E** — the pathologies: a slot that dies and is re-allocated inside one chunk, a
  whole chunk of POPs, a chunk with no writes, a chunk length that does not divide
  the token count, and the rank at both ends of the precompiled set.

Every test needs a CUDA device (the kernel is Triton) and is skipped without one.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender import read as read_module
from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, NONE, POP, MatrixSchedule, op_histogram, sequential_matrix_scan,
    update_liveness)
from experiments.expressivity_tasks.x45_matrix_emender.k1_schedule_scan import (
    FUSED_DEFAULT, MAX_TILE_VALUE_DIM, k1_state_scan)
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    chunk_scan, chunked_scan, routed_chunk_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, random_schedule, sequential_noise_floor)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------- fixtures

class Routed:
    """A real routed schedule plus the inputs that produced it."""

    def __init__(self, result, decay, query, payload, rank, policy, chunk_tokens):
        self.result = result
        self.decay = decay
        self.query = query
        self.payload = payload
        self.rank = rank
        self.policy = policy
        self.chunk_tokens = chunk_tokens

    @property
    def schedule(self):
        return self.result.schedule

    @property
    def weights(self):
        """``(batch, heads, tokens, rank)``, the layout the scan consumes."""
        return self.result.weights.permute(0, 2, 1, 3)


def routed_fixture(tokens=64, batch=2, heads=2, rank=5, value_dim=8, chunk_tokens=16,
                   seed=23, distinct=6, policy=None):
    """The arm's own routing on real state.

    ``distinct`` caps the number of distinct value directions in the stream, which
    is what lets two live rows become similar enough for the routing to take a
    MERGE: a uniform random value stream never merges, and gate A has to cover all
    five ops.  The defaults are measured to produce every one of them on a store of
    ``rank`` slots (append 26 / correct 125 / merge 34 / pop 32 / none 295 of 512
    lanes at seed 23).
    """
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE) * 0.5
    directions = torch.randn(distinct, value_dim, generator=generator).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=generator)
    payload = directions[choice]
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    policy = RoutingPolicy(ops=True) if policy is None else policy
    result = routed_chunk_scan(query, payload, decay, router=route_chunk, policy=policy,
                              rank=rank, rank_max=rank, chunk_tokens=chunk_tokens,
                              collect=True)
    return Routed(result, decay, query, payload, rank, policy, chunk_tokens)


def entry_state(batch, heads, rank, value_dim, seed=0, live_fraction=0.5):
    """An entry store with a mixed liveness pattern and *zero* content where dead.

    A dead row is exactly zero in this algebra -- a POP writes zero -- so the
    fixture keeps that invariant and the tests can assert it.
    """
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(batch, heads, rank, value_dim, generator=generator).to(DEVICE)
    live = torch.rand(batch, heads, rank, generator=generator).to(DEVICE) < live_fraction
    return state * live.unsqueeze(-1), live


def oracle(state, live, weights, decay, schedule):
    """The per-token oracle over one chunk: ``(state, exit live)``."""
    _, new_state, _, trajectory = sequential_matrix_scan(
        weights, decay, schedule, initial_state=state, initial_live=live,
        return_liveness=True)
    return new_state, trajectory[-1]


def oracle_sequence(state, live, weights, decay, schedule):
    """The per-token oracle over the whole sequence: ``(reads, state, exit live)``.

    ``state`` is mutated in place by the oracle, so it is cloned here.
    """
    reads, new_state, _, trajectory = sequential_matrix_scan(
        weights, decay, schedule, initial_state=state.clone(), initial_live=live,
        return_liveness=True)
    return reads, new_state, trajectory[-1]


def schedule_liveness(live, schedule, start, stop):
    """The reference's own liveness transition, driven by the schedule alone."""
    expected = live.clone()
    for t in range(start, stop):
        for lane in range(schedule.lanes):
            expected = update_liveness(expected, schedule.slot[:, t, :, lane],
                                       schedule.op[:, t, :, lane],
                                       schedule.live_after[:, t, :, lane])
    return expected


# --------------------------------------------------------------------------- gate A

@needs_cuda
def test_k1_state_and_liveness_are_bit_exact_against_the_oracle():
    """Real routed schedule, every op present: state and liveness bit-for-bit."""
    fixture = routed_fixture()
    histogram = op_histogram(fixture.schedule)
    assert histogram["merge"] > 0, histogram          # the two-lane op is exercised
    assert histogram["pop"] > 0 and histogram["append"] > 0, histogram
    tokens, chunk = fixture.schedule.tokens, fixture.chunk_tokens
    state, live = entry_state(fixture.result.reads.shape[0],
                              fixture.result.reads.shape[2],
                              fixture.result.weights.shape[-1],
                              fixture.query.shape[-1], seed=3)
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        piece = fixture.schedule.chunk(start, stop)
        expected_state, expected_live = oracle(state, live,
                                               fixture.result.weights[:, start:stop],
                                               fixture.decay[:, start:stop], piece)
        actual = k1_state_scan(state, live, fixture.decay[:, start:stop], piece)
        assert torch.equal(actual.state, expected_state), (start, stop)
        assert torch.equal(actual.live, expected_live), (start, stop)
        state, live = actual.state, actual.live


@needs_cuda
def test_k1_liveness_is_bit_exact_at_every_token():
    """One token per call, so the *per-token* liveness trajectory is compared."""
    schedule, query, _, decay = random_schedule(2, 24, 2, 9, 5, lanes=2, seed=13)
    state, live = entry_state(2, 2, 9, 5, seed=4)
    _, _, _, trajectory = sequential_matrix_scan(query, decay, schedule,
                                                 initial_state=state.clone(),
                                                 initial_live=live,
                                                 return_liveness=True)
    for t in range(schedule.tokens):
        piece = schedule.chunk(t, t + 1)
        actual = k1_state_scan(state, live, decay[:, t:t + 1], piece)
        assert torch.equal(actual.live, trajectory[t]), t
        state, live = actual.state, actual.live


@needs_cuda
def test_k1_death_codes_follow_the_schedule_on_a_pop_heavy_stream():
    """A slot is live at the exit iff its last touching lane said ``live_after``.

    On the matrix store the death code *is* the schedule's ``op``/``live_after``
    channel -- there is no strength threshold in this algebra -- so the check is
    that the walk's exit liveness equals the schedule-driven transition *and* the
    per-token oracle, and that a POP is the only op that clears a slot.
    """
    fixture = routed_fixture(tokens=48, rank=5, value_dim=6, chunk_tokens=16,
                             policy=RoutingPolicy(ops=True, tau_match=0.7, tau_pop=0.4))
    histogram = op_histogram(fixture.schedule)
    assert histogram["pop"] > 0 and histogram["append"] > 0, histogram
    schedule, tokens, chunk = fixture.schedule, fixture.schedule.tokens, 16
    state, live = entry_state(fixture.result.reads.shape[0],
                              fixture.result.reads.shape[2],
                              fixture.result.weights.shape[-1],
                              fixture.query.shape[-1], seed=5)
    for start in range(0, tokens, chunk):
        stop = min(start + chunk, tokens)
        piece = schedule.chunk(start, stop)
        expected_state, _ = oracle(state, live, fixture.result.weights[:, start:stop],
                                   fixture.decay[:, start:stop], piece)
        actual = k1_state_scan(state, live, fixture.decay[:, start:stop], piece)
        assert torch.equal(actual.state, expected_state), (start, stop)
        expected_live = schedule_liveness(live, schedule, start, stop)
        assert torch.equal(actual.live, expected_live), (start, stop)
        # A POP writes exactly zero, so a dead slot's row is exactly zero at every
        # chunk boundary -- the content and liveness channels agree.
        assert torch.equal(actual.state[~actual.live],
                           torch.zeros_like(actual.state[~actual.live]))
        state, live = actual.state, actual.live


@needs_cuda
def test_k1_strength_channel_is_the_row_norm_of_the_bit_exact_rows():
    """The matrix store's strength is ``||S[j]|| * live`` (``read.row_strength``)."""
    schedule, query, _, decay = random_schedule(1, 8, 2, 7, 5, lanes=2, seed=9)
    state, live = entry_state(1, 2, 7, 5, seed=6)
    actual = k1_state_scan(state, live, decay, schedule)
    reference_state, reference_live = oracle(state, live, query, decay, schedule)
    assert torch.equal(actual.state, reference_state)
    assert torch.equal(actual.live, reference_live)
    expected = read_module.row_strength(actual.state, actual.live)
    deviation = float((actual.strength - expected).abs().max())
    assert deviation <= 4e-6 * max(1.0, float(expected.abs().max())), deviation


# --------------------------------------------------------------------------- gate B

@needs_cuda
@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 16])
def test_k1_matches_the_chunked_reference_within_the_measured_noise_floor(chunk):
    """The fused path against ``chunk_scan``, at the measured sequential floor."""
    schedule, query, _, decay = random_schedule(2, 48, 2, 9, 5, lanes=2, seed=7, decay=0.9)
    floor, reference_state, reference_reads = sequential_noise_floor(query, decay, schedule)
    assert floor > 0.0
    reads, state, live = chunked_scan(query, decay, schedule, chunk_tokens=chunk, fused=True)
    eager_reads, eager_state, eager_live = chunked_scan(query, decay, schedule,
                                                        chunk_tokens=chunk)
    read_deviation = float((reads - reference_reads).abs().max())
    state_deviation = float((state - reference_state).abs().max())
    assert read_deviation <= NOISE_FACTOR * floor, (chunk, read_deviation, floor)
    assert state_deviation <= NOISE_FACTOR * floor, (chunk, state_deviation, floor)
    assert torch.equal(live, eager_live), "final liveness differs from the reference"
    assert float((reads - eager_reads).abs().max()) <= NOISE_FACTOR * floor, chunk
    assert float((state - eager_state).abs().max()) <= NOISE_FACTOR * floor, chunk


@needs_cuda
def test_the_injection_list_reconstructs_the_exit_state():
    """``S = exp(gc_last) H + sum_u exp(gc_last-gc_u) k_{j_u} d_u`` from the emission.

    This is the emission contract K2 will consume: the exit state must be
    recoverable from the chunk-entry state plus the emitted injection list, through
    the reference's own expression.
    """
    schedule, query, _, decay = random_schedule(2, 40, 2, 9, 5, lanes=2, seed=11)
    state, live = entry_state(2, 2, 9, 5, seed=7)
    chunk = k1_state_scan(state, live, decay, schedule)
    steps, lanes = chunk.steps, schedule.lanes
    batch, heads = state.shape[0], state.shape[1]
    slot = schedule.slot.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    weight = torch.exp(chunk.gate[..., -1:] - chunk.gate).unsqueeze(-1) * chunk.delta
    injection = torch.zeros_like(state).scatter_add_(
        2, slot.unsqueeze(-1).expand(batch, heads, steps, state.shape[-1]), weight)
    reconstructed = torch.exp(chunk.gate[..., -1])[:, :, None, None] * state + injection
    deviation = float((reconstructed - chunk.state).abs().max())
    assert deviation <= 1e-5 * max(1.0, float(chunk.state.abs().max())), deviation


@needs_cuda
def test_k1_and_the_reference_agree_on_a_real_routed_schedule():
    """Full sequence on recorded routing, plus the measured re-routing drift.

    Re-running the *router* on the fused trajectory is a different statement: the
    router reads the chunk-entry state, which differs from the reference's by fp
    noise, so a decision on a threshold can move.  That is measured here rather
    than assumed away, and it is reported in ``K1_SCHEDULE_SCAN.md``.
    """
    fixture = routed_fixture()
    chunk = fixture.chunk_tokens
    weights = fixture.result.weights                      # (batch, tokens, heads, rank)
    eager = chunked_scan(weights, fixture.decay, fixture.schedule, chunk_tokens=chunk)
    fused = chunked_scan(weights, fixture.decay, fixture.schedule, chunk_tokens=chunk,
                        fused=True)
    floor, reference_state, reference_reads = sequential_noise_floor(
        weights, fixture.decay, fixture.schedule)
    assert torch.equal(fused[2], eager[2])
    for name, observed, expected in (("reads", fused[0], reference_reads),
                                     ("state", fused[1], reference_state)):
        deviation = float((observed - expected).abs().max())
        assert deviation <= NOISE_FACTOR * floor, (name, deviation, floor)
    assert float((fused[0] - eager[0]).abs().max()) <= NOISE_FACTOR * floor

    rerouted = routed_chunk_scan(fixture.query, fixture.payload, fixture.decay,
                                router=route_chunk, policy=fixture.policy,
                                rank=fixture.rank, rank_max=fixture.rank,
                                chunk_tokens=chunk, collect=True, fused=True)
    moved = int((rerouted.schedule.op != fixture.schedule.op).sum())
    print(f"re-routed decisions moved: {moved} of {fixture.schedule.op.numel()}")
    assert rerouted.reads.shape == fixture.result.reads.shape


# --------------------------------------------------------------------------- gate E

def scripted(slot, op, payload, decay, query):
    """Build a schedule from explicit ops, with the schedule's own bookkeeping."""
    shape = slot.shape
    op = op.to(torch.int64)
    schedule = MatrixSchedule(slot=slot.to(torch.int64).to(DEVICE),
                              erase=((op == CORRECT) | (op == POP)).to(DEVICE),
                              payload=payload.to(DEVICE), op=op.to(DEVICE),
                              live_after=((op == APPEND) | (op == CORRECT)).to(DEVICE))
    return schedule, decay.to(DEVICE), query.to(DEVICE)


@needs_cuda
def test_a_slot_that_dies_and_is_reallocated_inside_one_chunk():
    """POP then APPEND back into the freed slot, inside the same chunk."""
    rank, tokens = 3, 6
    shape = (1, tokens, 1, 1)
    slot = torch.zeros(shape, dtype=torch.int64)
    op = torch.tensor([APPEND, APPEND, POP, APPEND, CORRECT, NONE]).view(*shape)
    payload = torch.zeros(1, tokens, 1, 1, 4)
    payload[:, 0, 0, 0] = torch.tensor([1.0, 0, 0, 0])
    payload[:, 1, 0, 0] = torch.tensor([0, 2.0, 0, 0])
    payload[:, 3, 0, 0] = torch.tensor([0, 0, 3.0, 0])
    payload[:, 4, 0, 0] = torch.tensor([0, 0, 0, 4.0])
    schedule, decay, query = scripted(slot, op, payload, torch.ones(1, tokens, 1),
                                      torch.ones(1, tokens, 1, rank) / 4)
    for chunk in (1, 2, 3, 6):
        reads, final, final_live = chunked_scan(query, decay, schedule,
                                               chunk_tokens=chunk, fused=True)
        expected_reads, expected_state, expected_live = oracle_sequence(
            torch.zeros(1, 1, rank, 4, device=DEVICE),
            torch.zeros(1, 1, rank, dtype=torch.bool, device=DEVICE),
            query, decay, schedule)
        assert torch.equal(final, expected_state), chunk
        assert torch.equal(final_live, expected_live), chunk
        assert float((reads - expected_reads).abs().max()) == 0.0, chunk
        assert torch.equal(final[:, :, 2:], torch.zeros(1, 1, rank - 2, 4, device=DEVICE))
        assert bool(final_live[0, 0, 0]), "the reallocated slot must be live at the exit"
        assert not bool(final_live[0, 0, 1:].any()), "no other slot was ever written"


@needs_cuda
def test_a_whole_chunk_of_pops():
    """Every step of the chunk is a POP: the store empties and nothing comes back."""
    rank, tokens = 6, 8
    shape = (1, tokens, 2, 1)
    slot = torch.arange(tokens, device=DEVICE).remainder(rank).view(1, tokens, 1, 1).expand(shape)
    op = torch.full(shape, POP)
    payload = torch.zeros(1, tokens, 2, 1, 4)
    decay = torch.full((1, tokens, 2), 0.9, device=DEVICE)
    generator = torch.Generator().manual_seed(1)
    state = torch.randn(1, 2, rank, 4, generator=generator).to(DEVICE)
    live = torch.ones(1, 2, rank, dtype=torch.bool, device=DEVICE)
    query = torch.randn(1, tokens, 2, rank, generator=generator).to(DEVICE)
    schedule = MatrixSchedule(
        slot=slot.contiguous(), erase=torch.ones(shape, dtype=torch.bool, device=DEVICE),
        payload=payload.to(DEVICE), op=op.to(DEVICE),
        live_after=torch.zeros(shape, dtype=torch.bool, device=DEVICE))
    reads, final, final_live = chunked_scan(query, decay, schedule, chunk_tokens=tokens,
                                           initial_state=state, initial_live=live,
                                           fused=True)
    expected_reads, expected_state, expected_live = oracle_sequence(
        state, live, query, decay, schedule)
    assert torch.equal(final, expected_state)
    assert torch.equal(final_live, expected_live)
    assert not bool(final_live.any()), "a chunk of POPs must leave nothing live"
    # A POP writes exactly zero, so the walk lands on exact zeros.  The chunked
    # reference reaches the same place by cancelling exponentials and leaves
    # ~1e-8 of rounding behind; that difference is the fp noise floor, not a
    # semantic one, and it is why the bit-exact oracle is the gate here.
    assert torch.equal(final, torch.zeros_like(final))
    scale = max(1.0, float(expected_reads.abs().max()))
    assert float((reads - expected_reads).abs().max()) <= 1e-6 * scale
    reference_reads, reference_state, _ = chunked_scan(
        query, decay, schedule, chunk_tokens=tokens, initial_state=state,
        initial_live=live.clone())
    assert float((final - reference_state).abs().max()) <= 1e-6


@needs_cuda
def test_a_chunk_with_no_writes():
    """Pure decay: the store scales by the gate and the read of zero weights is zero."""
    tokens, rank, value_dim = 9, 5, 3
    shape = (1, tokens, 2, 1)
    schedule = MatrixSchedule(
        slot=torch.zeros(shape, dtype=torch.int64, device=DEVICE),
        erase=torch.zeros(shape, dtype=torch.bool, device=DEVICE),
        payload=torch.zeros(1, tokens, 2, 1, value_dim, device=DEVICE),
        op=torch.zeros(shape, dtype=torch.int64, device=DEVICE),
        live_after=torch.zeros(shape, dtype=torch.bool, device=DEVICE))
    decay = torch.full((1, tokens, 2), 0.5, device=DEVICE)
    query = torch.zeros(1, tokens, 2, rank, device=DEVICE)
    state = torch.randn(1, 2, rank, value_dim,
                        generator=torch.Generator().manual_seed(2)).to(DEVICE)
    live = torch.zeros(1, 2, rank, dtype=torch.bool, device=DEVICE)
    for chunk in (4, 9):
        reads, final, final_live = chunked_scan(query, decay, schedule,
                                               chunk_tokens=chunk, initial_state=state,
                                               initial_live=live, fused=True)
        expected_reads, expected_state, expected_live = oracle_sequence(
            state, live, query, decay, schedule)
        assert torch.equal(final, expected_state), chunk
        assert torch.equal(final_live, expected_live), chunk
        assert float(reads.abs().max()) == 0.0
        assert float((final - state * 0.5 ** tokens).abs().max()) <= 1e-6
        # No write ever fires, so the injection list is exactly zero.
        fused_chunk = k1_state_scan(state, live, decay[:, :chunk], schedule.chunk(0, chunk))
        assert float(fused_chunk.delta.abs().max()) == 0.0


@needs_cuda
def test_a_chunk_length_that_does_not_divide_the_token_count():
    tokens, rank, value_dim = 13, 6, 4
    shape = (1, tokens, 2, 1)
    generator = torch.Generator().manual_seed(5)
    slot = torch.randint(0, rank, shape, generator=generator)
    op = torch.randint(0, 5, shape, generator=generator)
    live_after = (op == APPEND) | (op == CORRECT)
    op = torch.where(live_after | (op == POP), op, torch.full_like(op, NONE))
    erase = (op == CORRECT) | (op == POP)
    payload = torch.randn(1, tokens, 2, 1, value_dim, generator=generator)
    carried = ((op == APPEND) | (op == CORRECT)).unsqueeze(-1)
    payload = torch.where(carried, payload, torch.zeros_like(payload))
    schedule = MatrixSchedule(slot=slot.to(DEVICE), erase=erase.to(DEVICE),
                              payload=payload.to(DEVICE), op=op.to(DEVICE),
                              live_after=live_after.to(DEVICE))
    query = torch.randn(1, tokens, 2, rank, generator=generator).to(DEVICE) * 0.3
    decay = torch.full((1, tokens, 2), 0.9, device=DEVICE)
    floor, reference_state, reference_reads = sequential_noise_floor(query, decay, schedule)
    state0 = torch.zeros(1, 2, rank, value_dim, device=DEVICE)
    live0 = torch.zeros(1, 2, rank, dtype=torch.bool, device=DEVICE)
    for chunk in (2, 3, 4, 5, 13):
        reads, final, final_live = chunked_scan(query, decay, schedule,
                                               chunk_tokens=chunk, fused=True)
        expected_reads, expected_state, expected_live = oracle_sequence(
            state0, live0, query, decay, schedule)
        assert torch.equal(final, expected_state), chunk
        assert torch.equal(final_live, expected_live), chunk
        assert float((reads - expected_reads).abs().max()) <= 1e-5, chunk
        assert float((reads - reference_reads).abs().max()) <= NOISE_FACTOR * floor, chunk


@needs_cuda
@pytest.mark.parametrize("rank", [64, 2048])
def test_rank_at_both_ends_of_the_precompiled_set(rank):
    """The smallest and largest precompiled rank, at a partial and a full block."""
    tokens, value_dim = 10, 4
    shape = (1, tokens, 1, 1)
    generator = torch.Generator().manual_seed(31)
    slot = torch.randint(0, rank, shape, generator=generator)
    payload = torch.randn(1, tokens, 1, 1, value_dim, generator=generator)
    query = torch.randn(1, tokens, 1, rank, generator=generator).to(DEVICE) * 0.3
    schedule, decay, query = scripted(slot, torch.full(shape, APPEND), payload,
                                      torch.full((1, tokens, 1), 0.9), query)
    state = torch.randn(1, 1, rank, value_dim, generator=generator).to(DEVICE)
    live = torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE)
    for chunk in (4, 10):
        reads, final, final_live = chunked_scan(query, decay, schedule,
                                               chunk_tokens=chunk, initial_state=state,
                                               initial_live=live, fused=True)
        expected_reads, expected_state, expected_live = oracle_sequence(
            state, live, query, decay, schedule)
        assert torch.equal(final, expected_state), (rank, chunk)
        assert torch.equal(final_live, expected_live), (rank, chunk)
        scale = max(1.0, float(expected_reads.abs().max()))
        assert float((reads - expected_reads).abs().max()) <= 1e-5 * scale, (rank, chunk)


@needs_cuda
def test_the_fused_path_is_bit_deterministic_across_slot_blocks():
    """Two runs are bit-identical, on a rank that spans several slot blocks.

    Every emitted channel has exactly one writer: an addressed slot's injection is
    written by the block that owns the slot, and only block 0 writes the shared gate
    channel.  Without that ownership the blocks race with zeros and the reads become
    run-dependent, which is what this test would catch.
    """
    fixture = routed_fixture(tokens=32, rank=64, value_dim=8, chunk_tokens=8)
    weights = fixture.result.weights
    first = chunked_scan(weights, fixture.decay, fixture.schedule, chunk_tokens=8,
                         fused=True)
    second = chunked_scan(weights, fixture.decay, fixture.schedule, chunk_tokens=8,
                          fused=True)
    for left, right in zip(first, second):
        assert torch.equal(left, right)
    state, live = entry_state(2, 2, 64, 8, seed=8)
    one = k1_state_scan(state, live, fixture.decay[:, :8], fixture.schedule.chunk(0, 8))
    two = k1_state_scan(state, live, fixture.decay[:, :8], fixture.schedule.chunk(0, 8))
    for left, right in zip((one.state, one.live, one.strength, one.delta, one.gate),
                           (two.state, two.live, two.strength, two.delta, two.gate)):
        assert torch.equal(left, right)
    assert float(one.delta.abs().max()) > 0.0, "the chunk must write something"


# ------------------------------------------------------------------- the flag itself

@needs_cuda
def test_the_fused_path_is_default_off_and_the_reference_is_unchanged():
    assert FUSED_DEFAULT is False
    schedule, query, _, decay = random_schedule(1, 12, 1, 5, 3, lanes=1, seed=19)
    default = chunked_scan(query, decay, schedule, chunk_tokens=4)
    explicit = chunked_scan(query, decay, schedule, chunk_tokens=4, fused=False)
    for left, right in zip(default, explicit):
        assert torch.equal(left, right)
    state = torch.zeros(1, 1, 5, 3, device=DEVICE)
    live = torch.zeros(1, 1, 5, dtype=torch.bool, device=DEVICE)
    weights = query.permute(0, 2, 1, 3)
    chunk_decay = decay.permute(0, 2, 1)
    reference = chunk_scan(state, live, weights, chunk_decay, schedule)
    fused = chunk_scan(state, live, weights, chunk_decay, schedule, fused=True)
    assert torch.equal(fused[2], reference[2])


@needs_cuda
def test_the_kernel_rejects_a_value_dim_it_is_not_compiled_for():
    schedule, query, _, decay = random_schedule(1, 4, 1, 5, 3, lanes=1, seed=21)
    wide = MAX_TILE_VALUE_DIM * 2
    state = torch.zeros(1, 1, 5, wide, device=DEVICE)
    live = torch.zeros(1, 1, 5, dtype=torch.bool, device=DEVICE)
    oversized = MatrixSchedule(slot=schedule.slot, erase=schedule.erase,
                               payload=torch.zeros(1, 4, 1, 1, wide, device=DEVICE),
                               op=schedule.op, live_after=schedule.live_after)
    with pytest.raises(ValueError, match="value dims up to"):
        k1_state_scan(state, live, decay, oversized)
