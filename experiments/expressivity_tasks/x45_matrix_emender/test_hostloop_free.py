"""Gates for the no-host-loop paths: parity, launch count, determinism, pathologies.

Gate discipline is the lane's (``test_k23_read.py``, ``test_k1_schedule_scan.py``): the
oracle is ``algebra.sequential_matrix_scan`` (bit-exact state and liveness, read within
``NOISE_FACTOR`` times a **measured** fp32 floor taken on the same fixture), and the
comparison path is the eager chunked reference ``chunk_scan(fused=False)``.

The four things this file has to establish, one per section:

* **parity** -- ``persistent_scan`` (Option A) reproduces the oracle: state and liveness
  *bit-exact* (it walks the oracle's order token by token), read within the floor;
  ``chunked_block_scan`` (Options B/C) is the same kernels as the receipt's baseline, so
  it must match ``scan.chunked_scan(fused=True)`` **bit for bit**;
* **the graph is the loop** -- ``BlockGraph`` must return exactly what the un-captured
  loop returns, because it captures the same launches; anything else is a stale buffer
  or a buffer aliased between the caller and the graph;
* **the size gate** -- rank 2048 does not fit one program's registers and must be
  refused by name, with ``block_scan(path='auto')`` falling back to the chunked path;
* **pathologies** -- rank 64/512, chunk 16/128/256, a chunk that does not divide the
  token count, no writes, a POP-only sequence, and determinism across runs.

Every test needs a CUDA device.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, NONE, POP, MatrixSchedule, sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.hostloop_free import (
    MAX_PERSISTENT_TILE, BlockGraph, block_scan, check_block_shapes,
    chunked_block_scan, persistent_available, persistent_scan, value_block, warp_count)
from experiments.expressivity_tasks.x45_matrix_emender.scan import chunked_scan
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    routed_fixture)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, random_schedule, sequential_noise_floor)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------- helpers

def head_major(decay_tokens_major: torch.Tensor) -> torch.Tensor:
    """``(batch, tokens, heads)`` -> ``(batch, heads, tokens)``, the block paths' layout."""
    return decay_tokens_major.permute(0, 2, 1).contiguous()


def oracle_reads(weights, decay, schedule, state, live):
    """The per-token oracle from a given entry store: ``(reads, state, live)``.

    ``weights``/``decay`` in the oracle's layout (``(b, tokens, h, rank)`` /
    ``(b, tokens, h)``), returned reads in the block paths' layout
    ``(b, heads, tokens, d_v)`` so the comparison is one subtraction.
    """
    reads, exit_state, exit_live = sequential_matrix_scan(
        weights, decay, schedule, initial_state=state.clone(), initial_live=live)
    return reads.permute(0, 2, 1, 3), exit_state, exit_live


def per_token_deviation(observed, expected):
    """``(max, median)`` of the per-token max deviation over ``(batch, heads, value)``."""
    flat = (observed - expected).abs().amax(dim=(0, 3))
    return float(flat.max()), float(flat.median())


def empty_store(batch, heads, rank, value_dim):
    return (torch.zeros(batch, heads, rank, value_dim, device=DEVICE),
            torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE))


def writing_schedule(tokens, heads, rank, value_dim, ops, lanes=1, seed=3, decay=0.9):
    """A scripted schedule cycling through ``ops``, with the layout the block paths take."""
    generator = torch.Generator().manual_seed(seed)
    shape = (1, tokens, heads, lanes)
    slot = torch.arange(tokens).view(1, tokens, 1, 1).expand(shape) % rank
    op = torch.tensor(ops, dtype=torch.int64).view(1, -1, 1, 1).expand(shape).contiguous()
    erase = (op == CORRECT) | (op == POP)
    live_after = (op == APPEND) | (op == CORRECT)
    payload = torch.randn(*shape, value_dim, generator=generator)
    carried = ((op == APPEND) | (op == CORRECT)).unsqueeze(-1)
    payload = torch.where(carried, payload, torch.zeros_like(payload))
    schedule = MatrixSchedule(slot=slot.to(DEVICE), erase=erase.to(DEVICE),
                              payload=payload.to(DEVICE),
                              op=op.to(DEVICE), live_after=live_after.to(DEVICE))
    weights = torch.rand(1, heads, tokens, rank, generator=generator).to(DEVICE)
    decay = torch.full((1, heads, tokens), decay, device=DEVICE)
    return schedule, weights, decay


# ------------------------------------------------------------------- gate A: parity

@needs_cuda
def test_persistent_scan_reproduces_the_oracle_on_a_real_routed_schedule():
    """Option A against the per-token oracle: state/liveness exact, read at the floor.

    The routed fixture is the one every gate in this lane uses (all five ops present,
    measured histogram in the docstring of ``routed_fixture``), and the floor is measured
    on it first.  ``persistent_scan`` applies the recorded schedule token by token in the
    oracle's own order, so the state and the liveness channel are expected to be *equal*,
    not close; only the read's reduction order can differ.
    """
    fixture = routed_fixture()
    weights = fixture.weights                      # (b, heads, tokens, rank)
    decay = head_major(fixture.decay)
    schedule = fixture.schedule
    floor, oracle_state, oracle_reads_tokens_major = sequential_noise_floor(
        fixture.result.weights, fixture.decay, schedule)
    state, live = empty_store(weights.shape[0], weights.shape[1], weights.shape[3],
                              schedule.payload.shape[-1])

    reads, exit_state, exit_live = persistent_scan(state, live, weights, decay, schedule)
    # ``sequential_noise_floor`` does not report the exit liveness channel, so the oracle
    # is run once more for it -- a tautological liveness assertion would be worse than none.
    _, _, oracle_live, _ = sequential_matrix_scan(
        fixture.result.weights, fixture.decay, schedule, initial_state=state.clone(),
        initial_live=live, return_liveness=True)
    expected_reads = oracle_reads_tokens_major.permute(0, 2, 1, 3)
    peak, median = per_token_deviation(reads, expected_reads)
    print(f"floor {floor:.3e} allowance {NOISE_FACTOR * floor:.3e}; "
          f"read max {peak:.3e} ({peak / floor:.2f}x floor) median {median:.3e}; "
          f"state {float((exit_state - oracle_state).abs().max()):.3e}; "
          f"live {int((exit_live != oracle_live).sum())} differing slots")
    assert torch.equal(exit_state, oracle_state)
    assert torch.equal(exit_live, oracle_live)
    assert bool(exit_live.any()) and not bool(exit_live.all()), "the fixture must not be trivial"
    assert peak <= NOISE_FACTOR * floor, (peak, floor)
    assert median <= NOISE_FACTOR * floor, (median, floor)
    # The read is the oracle's read: a per-token reduction over the rank axis, in the
    # kernel's own order -- so it sits at the floor rather than at the chunked path's
    # `exp(gc) * entry + injections` grouping.
    assert peak <= 4 * floor, (peak, floor)


@needs_cuda
def test_persistent_scan_matches_the_chunked_pipeline_at_every_token():
    """Option A and the chunked baseline are the same computation, token by token."""
    fixture = routed_fixture()
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    state, live = empty_store(weights.shape[0], weights.shape[1], weights.shape[3],
                              schedule.payload.shape[-1])
    persistent_reads, persistent_state, persistent_live = persistent_scan(
        state, live, weights, decay, schedule)
    chunked_reads, chunked_state, chunked_live = chunked_block_scan(
        state, live, weights, decay, schedule, fixture.chunk_tokens)
    assert torch.equal(persistent_live, chunked_live)
    assert torch.equal(persistent_state, chunked_state)
    scale = float(chunked_reads.abs().max())
    deviation = float((persistent_reads - chunked_reads).abs().max())
    print(f"read deviation {deviation:.3e} on a scale of {scale:.3e}")
    assert deviation <= 1e-5 * scale, (deviation, scale)


@needs_cuda
def test_chunked_block_scan_is_the_receipt_baseline_bit_for_bit():
    """The baseline this file wraps must be ``chunked_scan(fused=True)``, exactly.

    Otherwise every number in ``NO_HOST_LOOP.md`` would be measured against a different
    pipeline than ``K23_READ.md`` measured.  ``chunked_scan`` takes the token-major
    weights/decay layout and returns token-major reads, so both conversions are views.
    """
    fixture = routed_fixture()
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    state, live = empty_store(weights.shape[0], weights.shape[1], weights.shape[3],
                              schedule.payload.shape[-1])
    baseline = chunked_scan(fixture.result.weights, fixture.decay, schedule,
                            initial_state=state, initial_live=live,
                            chunk_tokens=fixture.chunk_tokens, fused=True)
    ours = chunked_block_scan(state, live, weights, decay, schedule, fixture.chunk_tokens)
    # ``chunked_scan`` reads and writes the token-major layout; the block paths use the
    # head-major one the kernels index, so the reads are compared through a view.
    for left, right in zip(ours, (baseline[0].permute(0, 2, 1, 3), baseline[1],
                                  baseline[2])):
        assert torch.equal(left, right)


# ------------------------------------------------------- gate B: the graph is the loop

@needs_cuda
def test_the_captured_block_is_bit_identical_to_the_uncaptured_loop():
    """A replay returns exactly what the loop returns, for every chunk size.

    Bit-identity is the right instrument: the graph captures the *same* launches, so a
    difference can only be a buffer the caller and the graph share, or a copy that does
    not cover an input.  The second replay is checked too, because the static outputs are
    overwritten in place.
    """
    schedule, query, _, decay = random_schedule(2, 64, 2, 16, 12, lanes=2, seed=31)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = head_major(decay)
    state, live = empty_store(2, 2, 16, 12)
    for chunk in (16, 32, 64):
        graph = BlockGraph(2, 2, 64, 16, 12, 2, chunk_tokens=chunk).capture()
        expected = chunked_block_scan(state, live, weights, decay, schedule, chunk)
        first = graph.run(state, live, weights, decay, schedule, clone=True)
        second = graph.run(state, live, weights, decay, schedule, clone=True)
        for left, right in zip(first, expected):
            assert torch.equal(left, right), chunk
        for left, right in zip(second, expected):
            assert torch.equal(left, right), chunk


@needs_cuda
def test_the_graph_copies_its_inputs_in_and_does_not_keep_the_callers_tensors():
    """The static buffers are filled from the caller, so a changed input changes the answer.

    This is the failure mode the lane hit at step granularity: a graph that silently keeps
    reading the buffers it was captured with.  Two different entry states must give two
    different results, and the caller's tensors must be free to be overwritten afterwards.
    """
    schedule, query, _, decay = random_schedule(1, 32, 1, 8, 6, lanes=2, seed=13)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = head_major(decay)
    graph = BlockGraph(1, 1, 32, 8, 6, 2, chunk_tokens=16).capture()
    first_state = torch.full((1, 1, 8, 6), 0.25, device=DEVICE)
    live = torch.zeros(1, 1, 8, dtype=torch.bool, device=DEVICE)
    first = graph.run(first_state, live, weights, decay, schedule, clone=True)
    second_state = torch.full((1, 1, 8, 6), -0.5, device=DEVICE)
    second = graph.run(second_state, live, weights, decay, schedule, clone=True)
    assert not torch.equal(first[0], second[0])
    expected = chunked_block_scan(second_state, live, weights, decay, schedule, 16)
    for left, right in zip(second, expected):
        assert torch.equal(left, right)
    # The caller's tensors can be freed and rewritten without touching the graph's.
    first_state.zero_()
    again = graph.run(second_state, live, weights, decay, schedule, clone=True)
    assert torch.equal(again[0], expected[0])


# -------------------------------------------------------------------- gate C: the gate

@needs_cuda
def test_rank_2048_does_not_fit_one_program_and_is_refused_by_name():
    """The size gate: rank 2048 x 16 value columns is 32 768 floats, four times the budget."""
    assert persistent_available(128, 128)
    assert persistent_available(512, 128)
    assert not persistent_available(2048, 128)
    assert MAX_PERSISTENT_TILE == 8192
    with pytest.raises(ValueError, match="does not fit, so it must take the chunked path"):
        value_block(2048, 128)
    schedule, query, _, decay = random_schedule(1, 8, 1, 2048, 8, lanes=2, seed=7)
    weights = query.permute(0, 2, 1, 3).contiguous()
    state, live = empty_store(1, 1, 2048, 8)
    with pytest.raises(ValueError, match="does not fit"):
        persistent_scan(state, live, weights, head_major(decay), schedule)
    with pytest.raises(ValueError, match="does not fit"):
        block_scan(state, live, weights, head_major(decay), schedule, path="persistent")
    # `auto` does not raise: it takes the chunked path, which is the fallback that the
    # gate exists to make automatic.
    reads, _, _ = block_scan(state, live, weights, head_major(decay), schedule,
                             path="auto", chunk_tokens=4)
    assert reads.shape == (1, 1, 8, 8)


@needs_cuda
def test_auto_picks_the_persistent_kernel_where_it_fits():
    schedule, query, _, decay = random_schedule(1, 16, 1, 32, 8, lanes=1, seed=17)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = head_major(decay)
    state, live = empty_store(1, 1, 32, 8)
    auto = block_scan(state, live, weights, decay, schedule, path="auto")
    explicit = persistent_scan(state, live, weights, decay, schedule)
    for left, right in zip(auto, explicit):
        assert torch.equal(left, right)
    chunked = block_scan(state, live, weights, decay, schedule, path="chunked",
                         chunk_tokens=8)
    assert not torch.equal(auto[0], chunked[0]), "auto must not silently be the loop"
    with pytest.raises(ValueError, match="unknown block path"):
        block_scan(state, live, weights, decay, schedule, path="graph")


# ------------------------------------------------------------------ gate E: pathology

@needs_cuda
@pytest.mark.parametrize("rank", [64, 512])
def test_rank_at_both_ends_of_the_persistent_kernels_range(rank):
    """Rank 64 (a 1 024-float tile) and rank 512 (8 192) against the per-token oracle.

    The two ends of the range that still fit one program: rank 64 is where the tile is
    smallest and rank 512 where it is at the budget, so these are the shapes that would
    catch a wrong ``RP`` masking or an off-by-one in the value-block offset.
    """
    tokens, value_dim = 32, 16
    schedule, query, _, decay = random_schedule(1, tokens, 2, rank, value_dim, lanes=2,
                                                seed=rank)
    weights = query.permute(0, 2, 1, 3).contiguous()
    floor, oracle_state, oracle_reads = sequential_noise_floor(query, decay, schedule)
    oracle_reads = oracle_reads.permute(0, 2, 1, 3)
    head_first_decay = head_major(decay)
    state, live = empty_store(1, 2, rank, value_dim)
    reads, exit_state, exit_live = persistent_scan(state, live, weights, head_first_decay,
                                                  schedule)
    peak, median = per_token_deviation(reads, oracle_reads)
    print(f"rank {rank}: floor {floor:.3e}, read max {peak:.3e} ({peak / floor:.2f}x), "
          f"median {median:.3e}, state {float((exit_state - oracle_state).abs().max()):.3e}")
    assert torch.equal(exit_state, oracle_state), rank
    assert peak <= NOISE_FACTOR * floor, (rank, peak, floor)
    assert median <= NOISE_FACTOR * floor, (rank, median, floor)
    # The chunked path reaches the same state (K1's walk is the same walk) and the same
    # liveness; only the read's grouping differs.
    chunked = chunked_block_scan(state, live, weights, head_first_decay, schedule, 16)
    assert torch.equal(chunked[1], exit_state), rank
    assert torch.equal(chunked[2], exit_live), rank


@needs_cuda
@pytest.mark.parametrize("chunk", [16, 128, 256])
def test_every_chunk_length_the_arm_can_use(chunk):
    """Option C: the chunk length is a cost knob, never a semantic one."""
    tokens, rank, value_dim = chunk, 24, 12
    schedule, query, _, decay = random_schedule(1, tokens, 1, rank, value_dim, lanes=2,
                                                seed=chunk)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay = head_major(decay)
    state, live = empty_store(1, 1, rank, value_dim)
    persistent = persistent_scan(state, live, weights, decay, schedule)
    chunked = chunked_block_scan(state, live, weights, decay, schedule, chunk)
    assert torch.equal(persistent[2], chunked[2])
    assert torch.equal(persistent[1], chunked[1])
    scale = max(1.0, float(chunked[0].abs().max()))
    assert float((persistent[0] - chunked[0]).abs().max()) <= 1e-5 * scale, chunk


@needs_cuda
def test_a_chunk_that_does_not_divide_the_token_count():
    """13 tokens at chunk 2/3/5/13, against the oracle: the last chunk is a short one."""
    tokens, rank, value_dim = 13, 6, 4
    schedule, query, _, decay = random_schedule(1, tokens, 2, rank, value_dim, lanes=2,
                                                seed=5)
    weights = query.permute(0, 2, 1, 3).contiguous()
    decay_head_major = head_major(decay)
    state, live = empty_store(1, 2, rank, value_dim)
    floor, expected_state, expected_reads = sequential_noise_floor(query, decay, schedule)
    persistent = persistent_scan(state, live, weights, decay_head_major, schedule)
    assert torch.equal(persistent[1], expected_state)
    assert per_token_deviation(persistent[0], expected_reads.permute(0, 2, 1, 3))[0] \
        <= NOISE_FACTOR * floor
    for chunk in (2, 3, 5, 13):
        chunked = chunked_block_scan(state, live, weights, decay_head_major, schedule,
                                    chunk)
        assert torch.equal(chunked[2], persistent[2]), chunk
        assert per_token_deviation(chunked[0],
                                  expected_reads.permute(0, 2, 1, 3))[0] \
            <= NOISE_FACTOR * floor, chunk


@needs_cuda
def test_a_no_write_block_and_a_pop_only_block():
    """The two degenerate walks: nothing happens, and everything dies.

    No writes: the read at token ``t`` is the entry store decayed through token ``t-1``,
    which the persistent kernel computes per token instead of through the injection list
    -- the fixtures where a wrong decay weight cannot hide behind an injection.
    POP-only: every lane frees its slot, so the exit liveness is all-False and every read
    after the first touch is zero.
    """
    tokens, rank, value_dim, heads = 9, 6, 4, 2
    schedule, weights, decay = writing_schedule(tokens, heads, rank, value_dim,
                                                [NONE] * tokens)
    state = torch.randn(1, heads, rank, value_dim,
                        generator=torch.Generator().manual_seed(4)).to(DEVICE) * 0.5
    live = torch.ones(1, heads, rank, dtype=torch.bool, device=DEVICE)
    reads, exit_state, _ = persistent_scan(state, live, weights, decay, schedule)
    for t in range(tokens):
        # The read at token t is the entry store decayed through the t tokens before it,
        # contracted with the recorded weights: no injection exists to contribute.
        expected = torch.einsum("bhr,bhrv->bhv", weights[:, :, t], state) * 0.9 ** t
        deviation = float((reads[:, :, t] - expected).abs().max())
        assert deviation <= 1e-6 * max(1.0, float(expected.abs().max())), (t, deviation)
    decayed = state
    for _ in range(tokens):
        decayed = decayed * 0.9
    assert float((exit_state - decayed).abs().max()) <= 1e-6 * float(state.abs().max())

    pops, weights_p, decay_p = writing_schedule(tokens, heads, rank, value_dim,
                                                [POP] * tokens)
    reads_p, _, live_p = persistent_scan(state, live, weights_p, decay_p, pops)
    assert not bool(live_p.any()), "a POP-only block leaves nothing live"
    # Slot ``t % rank`` dies at token ``t``, so the store is empty from token ``rank`` on
    # and every read after it (which sees the state left by the previous token) is zero.
    assert float(reads_p[:, :, 1].abs().max()) > 0.0
    assert float(reads_p[:, :, rank + 1:].abs().max()) == 0.0


@needs_cuda
def test_every_path_is_bit_deterministic_across_runs():
    """Two runs are bit-identical: no atomics, one writer per output element."""
    fixture = routed_fixture()
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    state, live = empty_store(weights.shape[0], weights.shape[1], weights.shape[3],
                              schedule.payload.shape[-1])
    first = persistent_scan(state, live, weights, decay, schedule)
    second = persistent_scan(state.clone(), live.clone(), weights.clone(), decay.clone(),
                             schedule)
    for left, right in zip(first, second):
        assert torch.equal(left, right)
    graph = BlockGraph(weights.shape[0], weights.shape[1], schedule.tokens,
                       weights.shape[3], schedule.payload.shape[-1], schedule.lanes,
                       chunk_tokens=fixture.chunk_tokens).capture()
    assert torch.equal(graph.run(state, live, weights, decay, schedule, clone=True)[0],
                       graph.run(state, live, weights, decay, schedule, clone=True)[0])
    assert float(first[0].abs().max()) > 0.0


@needs_cuda
def test_the_shape_audit_and_the_graph_refusal():
    """Mismatched shapes are refused before a launch, and none of the paths builds a graph."""
    fixture = routed_fixture()
    weights, decay, schedule = fixture.weights, head_major(fixture.decay), fixture.schedule
    state, live = empty_store(weights.shape[0], weights.shape[1], weights.shape[3],
                              schedule.payload.shape[-1])
    with pytest.raises(ValueError, match="decay must be"):
        check_block_shapes(state, live, weights, decay.permute(0, 2, 1), schedule)
    with pytest.raises(ValueError, match="weights .* do not match"):
        check_block_shapes(state, live, weights[:, :, :8], decay, schedule)
    with pytest.raises(ValueError, match="value dim|does not belong"):
        check_block_shapes(state[..., :4], live, weights, decay, schedule)
    requiring = weights.clone().requires_grad_(True)
    with pytest.raises(RuntimeError, match="forward only"):
        persistent_scan(state, live, requiring, decay, schedule)
    with pytest.raises(RuntimeError, match="forward only"):
        chunked_block_scan(state, live, requiring, decay, schedule, 16)
    with torch.no_grad():
        reads, _, _ = persistent_scan(state, live, requiring, decay, schedule)
    assert not reads.requires_grad and torch.isfinite(reads).all()


def test_the_tuning_rules_are_the_documented_ones():
    """``value_block``/``warp_count``: narrow blocks, 4 warps at and above a 2 048 tile."""
    assert value_block(64, 128) == 16
    assert value_block(128, 128) == 16
    assert value_block(512, 128) == 16
    assert value_block(128, 8) == 16
    assert warp_count(1024) == 2
    assert warp_count(2048) == 4
    assert warp_count(8192) == 4
    assert warp_count(256) == 1
