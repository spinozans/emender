"""K2 + K3 gates A, B and E for the fused read.

Gate discipline is the lane's: **A** — the fused read must reproduce the eager
reduction at *every token* (not just at chunk ends), within ``NOISE_FACTOR`` times a
**measured** fp32 sequential floor that is taken on the same fixture first; **B** — the
four fixtures the read's normalisation and coefficients are made of (no writes, a write
on every token, a slot that dies mid-chunk, a token supported only by an injection);
**E** — rank, chunk length, a chunk that does not divide the token count, and
determinism.

The oracle is the eager path, never a fitted constant: ``scan.chunk_scan(fused=False)``
for the drop-in comparison and ``sequential_matrix_scan`` (through
``sequential_noise_floor``) for the semantic one.  Every test needs a CUDA device.

What the four gate-B fixtures each catch, stated so the test names are checkable:

============================  ==================================================
fixture                       catches
============================  ==================================================
no writes (pure inter)        a stray intra term, or a coefficient that fires on a
                              zero injection; the read must be ``exp(gc) w @ H``
a write on every token        a missing or mis-scaled intra block (the term is
                              large here and nil in the previous fixture)
a slot dying mid-chunk        a wrong decay weight or a double-counted slot: the
                              entry term and the POP injection must cancel, so the
                              slot's contribution vanishes from that token on
only support is an injection  a wrong coefficient: the read is exactly ``w[j] * d``
                              with nothing else to hide behind
============================  ==================================================
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, NONE, POP, MatrixSchedule, apply_lane, sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k1_schedule_scan import (
    K1Chunk, _reads_from_injections, k1_state_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k23_read import (
    MAX_VALUE_DIM, k23_chunk, k23_read, union_normalised_reads)
from experiments.expressivity_tasks.x45_matrix_emender.read import read_weights
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    FUSED_SCAN_DEFAULT, chunk_scan, chunked_scan)
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    entry_state, routed_fixture, scripted)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, random_schedule, sequential_noise_floor)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------- helpers

def read_index(tokens, lanes, device=DEVICE):
    """The step each token reads from: the last lane of the previous token, or -1."""
    return torch.arange(tokens, device=device) * lanes - 1


def terms(weights, entry_state, chunk, schedule):
    """The eager read's two terms, in the reference's own expression (torch)."""
    batch, heads, tokens, _ = weights.shape
    index = read_index(tokens, schedule.lanes)
    inside = index >= 0
    zero = torch.zeros((), dtype=torch.float32, device=weights.device)
    gc_read = torch.where(inside, chunk.log_gate[..., index.clamp(min=0)], zero)
    inter = (torch.exp(gc_read).unsqueeze(-1) * weights) @ entry_state
    steps = chunk.steps
    slot = schedule.slot.permute(0, 2, 1, 3).reshape(batch, heads, steps)
    addressed = weights.gather(3, slot.unsqueeze(2).expand(batch, heads, tokens, steps))
    order = torch.arange(steps, device=weights.device)
    before = torch.arange(tokens, device=weights.device).unsqueeze(-1) * schedule.lanes \
        > order.unsqueeze(0)
    coefficients = torch.where(before,
                              torch.exp(gc_read.unsqueeze(-1) - chunk.log_gate.unsqueeze(-2)),
                              zero)
    intra = (coefficients * addressed) @ chunk.delta
    return inter, intra


def per_token_deviation(observed, expected):
    """``(max, median)`` of the per-token max deviation over ``(batch, heads, value)``."""
    batch, tokens = observed.shape[0], observed.shape[1]
    flat = (observed - expected).abs().reshape(batch, tokens, -1).amax(dim=(0, 2))
    return float(flat.max()), float(flat.median())


# --------------------------------------------------------------------------- gate A

@needs_cuda
def test_k23_read_parity_at_every_token_on_a_real_routed_schedule():
    """All five ops, real routing, whole sequence: read parity at every token.

    The floor is measured first (two orderings of the sequential math plus fp64 on the
    same fixture); the fused path is then required to stay inside ``NOISE_FACTOR`` of it
    against the per-token oracle, and inside the same allowance against the eager chunked
    path, per token.  Max and median are printed because a max alone hides a drift that
    only a few tokens see.
    """
    fixture = routed_fixture()
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import op_histogram

    histogram = op_histogram(fixture.schedule)
    assert all(histogram[name] > 0 for name in
               ("none", "append", "correct", "pop", "merge")), histogram
    weights, decay, schedule = fixture.result.weights, fixture.decay, fixture.schedule
    chunk = fixture.chunk_tokens
    floor, oracle_state, oracle_reads = sequential_noise_floor(weights, decay, schedule)
    assert floor > 0.0

    eager_reads, eager_state, eager_live = chunked_scan(weights, decay, schedule,
                                                        chunk_tokens=chunk, fused=False)
    fused_reads, fused_state, fused_live = chunked_scan(weights, decay, schedule,
                                                        chunk_tokens=chunk, fused=True)
    assert torch.equal(fused_live, eager_live)

    vs_eager = per_token_deviation(fused_reads, eager_reads)
    vs_oracle = per_token_deviation(fused_reads, oracle_reads)
    print(f"floor {floor:.3e} allowance {NOISE_FACTOR * floor:.3e}; "
          f"vs eager max {vs_eager[0]:.3e} median {vs_eager[1]:.3e}; "
          f"vs oracle max {vs_oracle[0]:.3e} median {vs_oracle[1]:.3e}")
    assert vs_oracle[0] <= NOISE_FACTOR * floor, (vs_oracle, floor)
    assert vs_eager[0] <= NOISE_FACTOR * floor, (vs_eager, floor)
    # The state is K1's, and K1 is bit-exact against the *oracle*; the eager chunked
    # path reaches the same place through a scatter-add, so it is held to the floor.
    assert float((fused_state - oracle_state).abs().max()) <= NOISE_FACTOR * floor
    assert float((fused_state - eager_state).abs().max()) <= NOISE_FACTOR * floor


@needs_cuda
def test_the_fused_read_is_the_sum_of_the_two_reference_terms():
    """``k23_read`` is exactly ``inter + intra`` in the reference's own expression.

    A wrong split can hide inside a correct sum, so each term is isolated by a gate-B
    fixture instead: (i) has no injection at all (pure inter) and (ii) starts from an
    empty store (pure intra).  This test pins the *composition*: one fp32 add, no
    renormalisation, no second softmax.
    """
    schedule, query, _, decay = random_schedule(2, 24, 2, 9, 5, lanes=2, seed=11)
    state, live = entry_state(2, 2, 9, 5, seed=7)
    weights = query.permute(0, 2, 1, 3).contiguous()
    chunk = k1_state_scan(state, live, decay, schedule)
    inter, intra = terms(weights, state, chunk, schedule)
    fused = k23_read(weights, state, chunk, schedule)
    eager = _reads_from_injections(weights, state, chunk, schedule)
    scale = max(1.0, float(inter.abs().max()), float(intra.abs().max()))
    print(f"inter scale {float(inter.abs().max()):.3e} "
          f"intra scale {float(intra.abs().max()):.3e}")
    assert float(intra.abs().max()) > 0.0, "the intra term must be exercised here"
    assert float((fused - eager).abs().max()) <= 1e-6 * scale
    assert float((fused - (inter + intra)).abs().max()) <= 1e-6 * scale


# --------------------------------------------------------------------------- gate B

@needs_cuda
def test_zero_writes_is_pure_inter():
    """(i) No write in the chunk: the read is ``exp(gc_read) w @ H`` and nothing else."""
    tokens, rank, value_dim = 8, 5, 4
    shape = (1, tokens, 1, 1)
    schedule = MatrixSchedule(
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
    chunk = k1_state_scan(state, live, decay, schedule)
    assert float(chunk.delta.abs().max()) == 0.0, "a no-write chunk injects nothing"
    fused = k23_read(weights, state, chunk, schedule)
    inter, intra = terms(weights, state, chunk, schedule)
    assert float(intra.abs().max()) == 0.0
    assert float((fused - inter).abs().max()) <= 1e-7 * max(1.0, float(inter.abs().max()))
    eager = _reads_from_injections(weights, state, chunk, schedule)
    assert float((fused - eager).abs().max()) <= 1e-7 * max(1.0, float(eager.abs().max()))


@needs_cuda
def test_a_write_on_every_token_is_heavy_intra():
    """(ii) A write on every lane: the intra block must be large and still match."""
    tokens, rank, value_dim = 9, 6, 4
    shape = (1, tokens, 2, 1)
    generator = torch.Generator().manual_seed(19)
    slot = torch.randint(0, rank, shape, generator=generator)
    payload = torch.randn(1, tokens, 2, 1, value_dim, generator=generator)
    schedule, decay, _ = scripted(slot, torch.full(shape, APPEND), payload,
                                  torch.full((1, tokens, 2), 0.9),
                                  torch.zeros(1, tokens, 2, rank))
    # An empty entry store: the inter term is exactly zero, so this fixture is the
    # intra block and nothing else.
    state = torch.zeros(1, 2, rank, value_dim, device=DEVICE)
    live = torch.zeros(1, 2, rank, dtype=torch.bool, device=DEVICE)
    weights = torch.rand(1, 2, tokens, rank, generator=generator).to(DEVICE)
    chunk = k1_state_scan(state, live, decay, schedule)
    assert float(chunk.delta.abs().max()) > 0.0
    fused = k23_read(weights, state, chunk, schedule)
    eager = _reads_from_injections(weights, state, chunk, schedule)
    inter, intra = terms(weights, state, chunk, schedule)
    scale = max(1.0, float(eager.abs().max()))
    deviation = float((fused - eager).abs().max())
    assert deviation <= 1e-6 * scale, (deviation, scale)
    # The fixture must be intra-dominated, or it tests nothing new.
    assert float(inter.abs().max()) == 0.0
    assert float(intra.abs().max()) > 0.0
    assert float(fused.abs().max()) > 0.0


@needs_cuda
def test_a_slot_that_dies_mid_chunk_leaves_the_read():
    """(iii) A POP mid-chunk: the entry term and the injection cancel exactly.

    Slot 2 is the read's only support.  Up to the POP the read is ``exp(gc) w[2] S[2]``;
    after it the decayed entry term and the POP's injection are the same magnitude with
    opposite signs, so the contribution must be gone -- a wrong decay weight or a
    double-counted slot leaves a visible residue here and nowhere else.
    """
    tokens, rank, value_dim = 8, 4, 3
    pop_at = 4
    shape = (1, tokens, 1, 1)
    slot = torch.full(shape, 2, dtype=torch.int64)
    op = torch.where(torch.arange(tokens).view(1, tokens, 1, 1) == pop_at,
                     torch.full(shape, POP), torch.full(shape, NONE))
    schedule, decay, _ = scripted(slot, op, torch.zeros(1, tokens, 1, 1, value_dim),
                                  torch.full((1, tokens, 1), 0.8),
                                  torch.zeros(1, tokens, 1, rank))
    content = torch.zeros(1, 1, rank, value_dim, device=DEVICE)
    content[0, 0, 2] = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    state = content
    live = torch.tensor([[[False, False, True, False]]], device=DEVICE)
    weights = torch.zeros(1, 1, tokens, rank, device=DEVICE)
    weights[..., 2] = 0.5
    chunk = k1_state_scan(state, live, decay, schedule)
    fused = k23_read(weights, state, chunk, schedule)
    eager = _reads_from_injections(weights, state, chunk, schedule)
    scale = float(fused[:pop_at].abs().max())
    assert scale > 0.0
    assert float((fused - eager).abs().max()) <= 1e-6 * max(1.0, scale)
    # Before the POP the read is the decayed row, exactly: token t sees t decays.
    # The POP's own token is still a pre-write read, so the death shows from t+1.
    for t in range(pop_at + 1):
        expected = 0.5 * 0.8 ** t * content[0, 0, 2]
        deviation = float((fused[0, 0, t] - expected).abs().max())
        assert deviation <= 1e-6 * scale, (t, deviation)
    residue = float(fused[0, 0, pop_at + 1:].abs().max())
    assert residue <= 1e-5 * scale, (residue, scale)
    assert torch.equal(chunk.live,
                       torch.tensor([[[False, False, False, False]]], device=DEVICE))


@needs_cuda
def test_a_token_supported_only_by_an_injection():
    """(iv) Live entry rows with zero content: the read is ``w[j] * payload`` alone.

    Nothing in the entry state can support the read, so the coefficient the intra kernel
    applies is the entire answer: any missing ``exp(gc_read)``/``exp(-gc_u)`` factor or a
    second normalisation is a scale error visible here.
    """
    tokens, rank, value_dim = 4, 4, 3
    shape = (1, tokens, 1, 1)
    slot = torch.zeros(shape, dtype=torch.int64)
    op = torch.tensor([APPEND, NONE, NONE, NONE]).view(shape)
    payload = torch.zeros(1, tokens, 1, 1, value_dim, device=DEVICE)
    payload[0, 0, 0, 0] = torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    schedule, decay, _ = scripted(slot, op, payload,
                                  torch.full((1, tokens, 1), 0.9),
                                  torch.zeros(1, tokens, 1, rank))
    state = torch.zeros(1, 1, rank, value_dim, device=DEVICE)
    live = torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE)
    weights = torch.zeros(1, 1, tokens, rank, device=DEVICE)
    weights[..., 0] = 0.75
    chunk = k1_state_scan(state, live, decay, schedule)
    fused = k23_read(weights, state, chunk, schedule)
    eager = _reads_from_injections(weights, state, chunk, schedule)
    # Token 0's read is pre-write and has no content behind it; token t >= 1 is the
    # injection alone, decayed by the t-1 steps since the write (lanes = 1 here).
    assert float(fused[0, 0, 0].abs().max()) == 0.0
    for t in (1, 2, 3):
        deviation = float((fused[0, 0, t] - eager[0, 0, t]).abs().max())
        assert deviation <= 1e-6 * max(1.0, float(eager.abs().max())), (t, deviation)
    expected = 0.75 * torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    assert float((fused[0, 0, 1] - expected).abs().max()) <= 1e-6, float(fused[0, 0, 1])
    decayed = 0.75 * 0.9 * torch.tensor([1.0, 2.0, 3.0], device=DEVICE)
    assert float((fused[0, 0, 2] - decayed).abs().max()) <= 1e-6, float(fused[0, 0, 2])
    # The entry term is exactly zero, so the whole read is the intra block.
    inter, _ = terms(weights, state, chunk, schedule)
    assert float(inter.abs().max()) == 0.0


# --------------------------------------------------------------------------- gate E

@needs_cuda
@pytest.mark.parametrize("chunk", [16, 128, 256])
def test_chunk_lengths_the_arm_can_use(chunk):
    """Chunk 16 / 128 / 256 at the arm's value dim, against the eager reduction."""
    tokens, rank, value_dim = chunk, 32, 128
    schedule, query, _, decay = random_schedule(1, tokens, 1, rank, value_dim, lanes=2,
                                                seed=chunk)
    state = torch.randn(1, 1, rank, value_dim, device=DEVICE) * 0.5
    live = torch.rand(1, 1, rank, device=DEVICE) < 0.6
    weights = torch.rand(1, 1, tokens, rank, device=DEVICE)
    chunked = k1_state_scan(state, live, decay, schedule)
    fused = k23_read(weights, state, chunked, schedule)
    eager = _reads_from_injections(weights, state, chunked, schedule)
    deviation = float((fused - eager).abs().max())
    scale = max(1.0, float(eager.abs().max()))
    assert deviation <= 1e-6 * scale, (chunk, deviation, scale)


@needs_cuda
@pytest.mark.parametrize("rank", [64, 2048])
def test_rank_at_both_ends_of_the_precompiled_set(rank):
    """The smallest and largest precompiled rank, through both kernels."""
    tokens, value_dim = 12, 8
    schedule, query, _, decay = random_schedule(1, tokens, 1, rank, value_dim, lanes=2,
                                                seed=rank)
    state = torch.randn(1, 1, rank, value_dim, device=DEVICE) * 0.3
    live = torch.rand(1, 1, rank, device=DEVICE) < 0.5
    weights = torch.rand(1, 1, tokens, rank, device=DEVICE)
    chunked = k1_state_scan(state, live, decay, schedule)
    fused = k23_read(weights, state, chunked, schedule)
    eager = _reads_from_injections(weights, state, chunked, schedule)
    deviation = float((fused - eager).abs().max())
    scale = max(1.0, float(eager.abs().max()))
    # Rank 2048 is 2048 fp32 terms in a different blocking order than cuBLAS's; K1
    # holds its rank-2048 reads to 1e-5 relative for the same reason.
    assert deviation <= 1e-5 * scale, (rank, deviation, scale)


@needs_cuda
def test_a_chunk_length_that_does_not_divide_the_token_count():
    tokens, rank, value_dim = 13, 6, 4
    schedule, query, _, decay = random_schedule(1, tokens, 2, rank, value_dim, lanes=2,
                                                seed=5)
    floor, oracle_state, oracle_reads = sequential_noise_floor(query, decay, schedule)
    for chunk in (2, 3, 4, 5, 13):
        reads, state, live = chunked_scan(query, decay, schedule, chunk_tokens=chunk,
                                          fused=True)
        eager_reads, eager_state, eager_live = chunked_scan(
            query, decay, schedule, chunk_tokens=chunk, fused=False)
        assert torch.equal(live, eager_live), chunk
        assert per_token_deviation(reads, eager_reads)[0] <= NOISE_FACTOR * floor, chunk
        assert per_token_deviation(reads, oracle_reads)[0] <= NOISE_FACTOR * floor, chunk


@needs_cuda
def test_the_fused_read_is_bit_deterministic():
    """Two runs are bit-identical: no atomics, one writer per output element."""
    schedule, query, _, decay = random_schedule(2, 32, 2, 64, 16, lanes=2, seed=5)
    state, live = entry_state(2, 2, 64, 16, seed=8)
    weights = torch.rand(2, 2, 8, 64, generator=torch.Generator().manual_seed(3)).to(DEVICE)
    chunked = k1_state_scan(state, live, decay[:, :8], schedule.chunk(0, 8))
    first = k23_read(weights, state, chunked, schedule.chunk(0, 8))
    second = k23_read(weights.clone(), state.clone(), chunked, schedule.chunk(0, 8))
    assert torch.equal(first, second)
    assert float(first.abs().max()) > 0.0


# ------------------------------------------------------- the flag, and the diagnostic

@needs_cuda
def test_the_fused_path_is_the_default_and_the_reference_is_still_reachable():
    """The default is the fused read; ``fused=False`` is the oracle, byte for byte."""
    assert FUSED_SCAN_DEFAULT is True
    schedule, query, _, decay = random_schedule(1, 12, 1, 5, 3, lanes=1, seed=19)
    default = chunked_scan(query, decay, schedule, chunk_tokens=4)
    fused = chunked_scan(query, decay, schedule, chunk_tokens=4, fused=True)
    eager = chunked_scan(query, decay, schedule, chunk_tokens=4, fused=False)
    for left, right in zip(default, fused):
        assert torch.equal(left, right)
    state = torch.zeros(1, 1, 5, 3, device=DEVICE)
    live = torch.zeros(1, 1, 5, dtype=torch.bool, device=DEVICE)
    weights = query.permute(0, 2, 1, 3)
    chunk_decay = decay.permute(0, 2, 1)
    reference = chunk_scan(state, live, weights, chunk_decay, schedule, fused=False)
    again = chunk_scan(state, live, weights, chunk_decay, schedule, fused=False)
    for left, right in zip(reference, again):
        assert torch.equal(left, right)
    assert not torch.equal(fused[0], eager[0]), "the fused path is not silently the reference"


@needs_cuda
def test_the_fused_read_refuses_to_build_a_graph_it_cannot_backward():
    schedule, query, _, decay = random_schedule(1, 8, 1, 5, 3, lanes=1, seed=27)
    state, live = entry_state(1, 1, 5, 3, seed=9)
    weights = query.permute(0, 2, 1, 3)
    chunked = k1_state_scan(state, live, decay, schedule)
    with pytest.raises(RuntimeError, match="forward only"):
        k23_read(weights.clone().requires_grad_(True), state, chunked, schedule)
    with pytest.raises(RuntimeError, match="forward only"):
        chunk_scan(state, live, weights.clone().requires_grad_(True),
                   decay.permute(0, 2, 1), schedule, fused=True)
    with torch.no_grad():
        reads = k23_read(weights.clone().requires_grad_(True), state, chunked, schedule)
    assert torch.isfinite(reads).all() and not reads.requires_grad


@needs_cuda
def test_the_kernels_reject_a_shape_they_are_not_compiled_for():
    schedule, query, _, decay = random_schedule(1, 4, 1, 5, 3, lanes=1, seed=21)
    state = torch.zeros(1, 1, 5, 3, device=DEVICE)
    live = torch.zeros(1, 1, 5, dtype=torch.bool, device=DEVICE)
    chunked = k1_state_scan(state, live, decay, schedule)
    weights = torch.zeros(1, 1, 4, 5, device=DEVICE)
    wide = K1Chunk(state=state, live=live, strength=chunked.strength,
                   delta=torch.zeros(1, 1, 4, MAX_VALUE_DIM * 2, device=DEVICE),
                   log_gate=chunked.log_gate, steps=chunked.steps)
    with pytest.raises(ValueError, match="value dims up to"):
        k23_read(weights, torch.zeros(1, 1, 5, MAX_VALUE_DIM * 2, device=DEVICE),
                 wide, schedule)
    with pytest.raises(ValueError, match="does not match the weights channel"):
        k23_read(torch.zeros(1, 1, 4, 4, device=DEVICE), state, chunked, schedule)


@needs_cuda
def test_the_union_normalised_diagnostic_is_pinned_where_it_can_be():
    """The diagnostic must be a *known* read where its two effects are separable.

    With no in-chunk write the live set and the cosines never move, so the per-token
    read differs from the registered one by exactly the strength channel it re-reads
    from the decayed rows: ``union_t = exp(gc_read_t) * registered_t``.  That is the
    second of the diagnostic's two effects (the first is the softmax's live set, which
    only a write can move), and pinning it here is what makes the number in
    ``K23_READ.md`` §7 interpretable instead of mysterious.
    """
    tokens, rank, value_dim = 6, 5, 3
    shape = (1, tokens, 1, 1)
    schedule = MatrixSchedule(
        slot=torch.zeros(shape, dtype=torch.int64, device=DEVICE),
        erase=torch.zeros(shape, dtype=torch.bool, device=DEVICE),
        payload=torch.zeros(1, tokens, 1, 1, value_dim, device=DEVICE),
        op=torch.full(shape, NONE, dtype=torch.int64, device=DEVICE),
        live_after=torch.zeros(shape, dtype=torch.bool, device=DEVICE))
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(1, tokens, 1, value_dim, generator=generator).to(DEVICE)
    state = torch.randn(1, 1, rank, value_dim, generator=generator).to(DEVICE)
    live = torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE)
    decay = torch.full((1, tokens, 1), 0.9, device=DEVICE)
    weights = read_weights(query, state, live, 2.0)
    reads, _, _ = chunked_scan(weights, decay, schedule, chunk_tokens=tokens,
                               initial_state=state, initial_live=live, fused=True)
    union = union_normalised_reads(query, decay, schedule, state, live, 2.0)
    # lanes = 1 here, so token t's read step carries t-1 decays: gc_read = cumsum shifted.
    log_gate = torch.log(decay).cumsum(dim=1)
    gc_read = torch.cat([torch.zeros_like(log_gate[:, :1]), log_gate[:, :-1]], dim=1)
    decayed = torch.exp(gc_read.unsqueeze(-1)) * reads        # (b, tokens, heads, v)
    deviation = float((union - decayed).abs().max())
    assert deviation <= 1e-6 * max(1.0, float(union.abs().max())), deviation
    # With a write inside the chunk the two reads are different objects.
    writing, decay_w, _ = scripted(torch.zeros(shape, dtype=torch.int64),
                                   torch.tensor([APPEND] + [NONE] * (tokens - 1)).view(shape),
                                   torch.ones(1, tokens, 1, 1, value_dim) * 2.0,
                                   decay.cpu(),
                                   torch.zeros(1, tokens, 1, rank))
    weights_w = read_weights(query, state, live, 2.0)
    registered, _, _ = chunked_scan(weights_w, decay_w, schedule=writing,
                                    chunk_tokens=tokens, initial_state=state,
                                    initial_live=live, fused=True)
    moved = union_normalised_reads(query, decay_w, writing, state, live, 2.0)
    assert float((registered - moved).abs().max()) > 0.0
