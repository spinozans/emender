"""Parity and pathological-case tests for the chunked parallel scan.

Three independent implementations of the same semantics are compared:

* ``naive_reference.tape_scan``  -- the certified X30 reference (oracle);
* ``sequential_replay``          -- a sync-free per-token loop over the same
                                    decision schedule (the speed baseline);
* ``chunked_tape_scan``          -- the chunked prefix scan under test.

The tolerance is not fitted: ``test_sequential_noise_floor`` measures the
deviation between the two *sequential* implementations, which differ only in
reduction order, and the chunked path is required to stay within
``NOISE_FACTOR`` times that measured floor.
"""
from __future__ import annotations

import types

import pytest
import torch

from experiments.expressivity_tasks.x30_tape_cell import naive_reference as naive
from experiments.expressivity_tasks.x30_tape_cell.routing_trace import FREED_DECAY, FREED_POP
from experiments.expressivity_tasks.x44_chunk_scan import chunked_cell as chunked
from experiments.expressivity_tasks.x44_chunk_scan import sequential_replay
from experiments.expressivity_tasks.x44_chunk_scan.affine import (AffineSummary, chunked_prefix_scan,
                                                                 compose, scan_with_entry)
from experiments.expressivity_tasks.x44_chunk_scan.schedule import (DecisionSchedule, NO_SLOT,
                                                                   from_naive_trace)

NOISE_FACTOR = 8.0   # chunked deviation allowed per unit of sequential noise


def config(budget=6, heads=2, decay=0.9, threshold=0.15, key_dim=4, value_dim=5,
           pop_threshold=0.5, write_threshold=0.5):
    return naive.TapeHeadConfig(key_dim=key_dim, value_dim=value_dim, n_heads=heads,
                                budget=budget, decay_rate=decay,
                                free_threshold=threshold,
                                pop_threshold=pop_threshold,
                                write_enable_threshold=write_threshold)


def inputs(cfg, seed, batch=2, tokens=19):
    torch.manual_seed(seed)
    shape = (batch, tokens, cfg.n_heads)
    k = torch.randn(*shape, cfg.key_dim)
    q = torch.randn(*shape, cfg.key_dim)
    v = torch.randn(*shape, cfg.value_dim)
    beta = (torch.rand(*shape) > .25).float()
    pop = torch.rand(*shape)
    logit = torch.randn(*shape)
    return k, q, v, beta, pop, logit


def flat(x, cfg, batch, tokens):
    shape = (batch, tokens, cfg.n_heads) + tuple(x.shape[3:])
    return x.reshape(*shape).transpose(1, 2).reshape(batch * cfg.n_heads, tokens, *x.shape[3:])


def run_naive(cfg, args, seed):
    k, q, v, beta, pop, logit = args
    state = naive.init_state(k.shape[0], cfg, k.device)
    out, final, trace = naive.tape_scan(k, q, v, beta, pop, logit, state, cfg)
    return out, final, trace


def schedule_of(cfg, args, trace):
    k, q, v, beta, pop, logit = args
    return from_naive_trace(trace, pop, beta, logit, cfg)


def chunked_run(cfg, args, sched, chunk_size, **kw):
    k, q, v, _, _, _ = args
    batch, tokens = k.shape[0], k.shape[1]
    return chunked.chunked_tape_scan(flat(k, cfg, batch, tokens), flat(q, cfg, batch, tokens),
                                     flat(v, cfg, batch, tokens), sched, cfg,
                                     chunk_size=chunk_size, **kw)


def sequential_run(cfg, args, sched, **kw):
    k, q, v, _, _, _ = args
    batch, tokens = k.shape[0], k.shape[1]
    return sequential_replay.sequential_tape_scan(
        flat(k, cfg, batch, tokens), flat(q, cfg, batch, tokens),
        flat(v, cfg, batch, tokens), sched, cfg, **kw)


def trace_layout(mask, cfg, batch, tokens):
    """``[rows, tokens, slots]`` -> the reference trace's ``[batch, tokens, heads, slots]``."""
    return (mask.reshape(batch, cfg.n_heads, tokens, cfg.slots)
            .permute(0, 2, 1, 3))


def deviation(cfg, args, read, final, out_ref, final_ref):
    """Max abs deviation of the read outputs and of every state channel."""
    batch, tokens = args[0].shape[0], args[0].shape[1]
    read = read.reshape(batch, cfg.n_heads, tokens, cfg.value_dim).permute(0, 2, 1, 3)
    rows = batch * cfg.n_heads
    worst = (read - out_ref).abs().max().item()
    detail = {'read': worst}
    for name in ('keys', 'values', 'strength', 'age', 'usage'):
        if not hasattr(final_ref, name):
            continue   # x30 materialize() omits the usage instrument
        ref = getattr(final_ref, name).reshape(rows, cfg.slots, *getattr(final_ref, name).shape[3:])
        got = getattr(final, name)
        value = (ref.float() - got.float()).abs().max().item()
        detail[name] = value
        worst = max(worst, value)
    if not bool((final_ref.live.reshape(rows, cfg.slots) == final.live).all()):
        detail['live_mismatch'] = float(
            (final_ref.live.reshape(rows, cfg.slots) != final.live).sum())
        worst = float('inf')
    return worst, detail


# ---------------------------------------------------------------------------
# The affine engine on its own: zero diagonal entries, chunk composition.
# ---------------------------------------------------------------------------
def test_affine_zero_diagonal_matches_fold():
    """POP death / overwrite erasure give d = 0; the composed scan must not
    divide by an accumulated decay and must reproduce the direct fold."""
    torch.manual_seed(0)
    chunks, steps, lanes = 5, 7, 11
    multiplier = torch.rand(chunks, steps, lanes) * 0.9 + 0.1
    injection = torch.randn(chunks, steps, lanes)
    # zero diagonals: mid-chunk erasure, at chunk boundaries, and a whole chunk
    multiplier[0, 3, :] = 0.0
    multiplier[1, 0, :] = 0.0
    multiplier[2, :, :] = 0.0
    multiplier[4, -1, ::3] = 0.0
    states, _ = chunked_prefix_scan(multiplier, injection)
    reference = torch.zeros(chunks, steps, lanes, dtype=multiplier.dtype)
    running = torch.zeros(lanes)
    for c in range(chunks):
        for i in range(steps):
            running = multiplier[c, i] * running + injection[c, i]
            reference[c, i] = running
    torch.testing.assert_close(states, reference, atol=1e-6, rtol=1e-6)
    assert torch.isfinite(states).all()


def test_affine_compose_is_associative_and_matches_scan():
    torch.manual_seed(1)
    first = AffineSummary(torch.rand(4) * .5, torch.randn(4))
    second = AffineSummary(torch.rand(4) * .5, torch.randn(4))
    third = AffineSummary(torch.rand(4) * .5, torch.randn(4))
    state = torch.randn(4)
    left = compose(compose(third, second), first)
    right = compose(third, compose(second, first))
    torch.testing.assert_close(left.apply(state), right.apply(state), atol=1e-6, rtol=0)
    torch.testing.assert_close(left.apply(state), third.apply(second.apply(first.apply(state))),
                               atol=1e-6, rtol=0)


def test_affine_prefix_compose_matches_sequential_entries():
    """Composed chunk entries must equal the exact carried fp32 state up to
    the fp32-vs-fp64 rounding of the fold itself (the noise floor)."""
    torch.manual_seed(2)
    chunks, steps, lanes = 9, 5, 7
    multiplier = torch.rand(chunks, steps, lanes) * .8 + .2
    injection = torch.randn(chunks, steps, lanes)
    multiplier[6, 2] = 0.0        # zero diagonal inside a chunk
    states, sequence = chunked_prefix_scan(multiplier, injection)
    entry = torch.zeros(lanes)
    running = entry.clone()
    exact = torch.zeros(chunks, steps, lanes, dtype=torch.float64)
    running64 = entry.double()
    for c in range(chunks):
        local, _ = scan_with_entry(multiplier[c], injection[c], running)
        if not torch.equal(multiplier[c].flatten(), multiplier[c].flatten()):  # pragma: no cover
            raise AssertionError('multiplier mutated')
        for i in range(steps):
            running64 = multiplier[c, i].double() * running64 + injection[c, i].double()
            exact[c, i] = running64
        running = local[-1]
    exact32 = exact.float()
    floor = (exact32 - exact).abs().max().item()   # casting noise of the fold
    torch.testing.assert_close(states, exact32, atol=32 * floor, rtol=0)
    torch.testing.assert_close(sequence.apply(entry).double(), running64, atol=32 * floor,
                               rtol=0)


def test_affine_single_step_and_empty_lane_shapes():
    multiplier = torch.ones(1, 1, 1)
    injection = torch.zeros(1, 1, 1)
    states, final = chunked_prefix_scan(multiplier, injection)
    torch.testing.assert_close(states, torch.zeros(1, 1, 1))
    torch.testing.assert_close(final.apply(torch.tensor([2.0])), torch.tensor([2.0]))


# ---------------------------------------------------------------------------
# Noise floor, then tolerance parity against the certified reference.
# ---------------------------------------------------------------------------
def _fixture(seed=0, **kw):
    cfg = config(**kw)
    args = inputs(cfg, seed)
    out_ref, final_ref, trace = run_naive(cfg, args, seed)
    sched = schedule_of(cfg, args, trace)
    return cfg, args, out_ref, final_ref, trace, sched


def test_sequential_noise_floor_define_tolerance():
    """The two sequential implementations differ only in reduction order.

    That deviation is the floor the chunked path is measured against; it is
    not a tolerance chosen to make the chunked path pass.
    """
    cfg, args, out_ref, final_ref, trace, sched = _fixture(seed=3)
    read, final = sequential_run(cfg, args, sched)
    worst, detail = deviation(cfg, args, read, final, out_ref, final_ref)
    print('sequential-vs-sequential noise floor', detail)
    assert worst < 1e-5
    assert 'live_mismatch' not in detail
    # The same decisions must reproduce the certified reference exactly enough
    # that routing-derived death codes agree slot for slot.
    _, (_, deaths) = sequential_run(cfg, args, sched, collect_deaths=True)
    batch, tokens = args[0].shape[:2]
    got_pop = torch.stack([d[0] for d in deaths], dim=1)
    got_decay = torch.stack([d[1] for d in deaths], dim=1)
    assert torch.equal(trace_layout(got_pop, cfg, batch, tokens),
                       trace.freed == FREED_POP)
    assert torch.equal(trace_layout(got_decay, cfg, batch, tokens),
                       trace.freed == FREED_DECAY)


@pytest.mark.parametrize('chunk', [1, 2, 3, 5, 19, 64])
@pytest.mark.parametrize('decay,threshold', [(0.9, 0.15), (0.5, 0.1), (0.999995, 1e-3)])
def test_parity_with_sequential_reference(chunk, decay, threshold):
    cfg, args, out_ref, final_ref, trace, sched = _fixture(
        seed=7, decay=decay, threshold=threshold, budget=8, heads=2)
    reference_read, reference_final = sequential_run(cfg, args, sched)
    floor, floor_detail = deviation(cfg, args, reference_read, reference_final,
                                   out_ref, final_ref)
    deaths = []
    read, final = chunked_run(cfg, args, sched, chunk, deaths=deaths)
    worst, detail = deviation(cfg, args, read, final, out_ref, final_ref)
    print(f'chunk={chunk} decay={decay} floor={floor:.3e} chunked={detail}')
    assert worst <= max(NOISE_FACTOR * floor, 1e-6)
    # Death schedule derived from the strength channel == the traced frees.
    batch, tokens = args[0].shape[:2]
    got_pop = torch.cat([d[0] for d in deaths], dim=1)
    got_decay = torch.cat([d[1] for d in deaths], dim=1)
    assert torch.equal(trace_layout(got_pop, cfg, batch, tokens),
                       trace.freed == FREED_POP)
    assert torch.equal(trace_layout(got_decay, cfg, batch, tokens),
                       trace.freed == FREED_DECAY)


def test_parity_through_the_x30_cell_replay_engine():
    """The same schedule through x30 ``cell.py``'s checkpointed replay path."""
    from experiments.expressivity_tasks.x30_tape_cell import cell as x30cell

    cfg, args, out_ref, final_ref, trace, sched = _fixture(seed=11, budget=8)
    k, q, v, beta, pop, logit = args
    read, final = chunked_run(cfg, args, sched, 4)
    state = x30cell.from_native(naive.init_state(k.shape[0], cfg, k.device), cfg)
    out_e, final_e, trace_e, _ = x30cell.tape_scan(*args, state, cfg, chunk_size=4)
    materialized = types.SimpleNamespace(**final_e.materialize())
    worst, detail = deviation(cfg, args, read, final, out_e, materialized)
    print('chunked vs x30 cell replay', detail)
    assert worst < 1e-5


# ---------------------------------------------------------------------------
# Pathological cases.
# ---------------------------------------------------------------------------
def _scripted_schedule(rows, tokens, slots):
    return dict(alloc=torch.full((rows, tokens), NO_SLOT, dtype=torch.int64),
                gate=torch.ones(rows, tokens),
                correct=torch.full((rows, tokens), NO_SLOT, dtype=torch.int64),
                beta=torch.ones(rows, tokens),
                pop_slot=torch.full((rows, tokens), NO_SLOT, dtype=torch.int64),
                pop_factor=torch.ones(rows, tokens))


def _compare_scripted(script, cfg, tokens, chunk, seeds=(0,), deaths=None):
    """Drive the three implementations with an explicitly written schedule."""
    rows = script['alloc'].shape[0]
    torch.manual_seed(seeds[0])
    k = torch.randn(rows, tokens, cfg.key_dim)
    q = torch.randn(rows, tokens, cfg.key_dim)
    v = torch.randn(rows, tokens, cfg.value_dim)
    sched = DecisionSchedule(**script)
    read_c, final_c = chunked.chunked_tape_scan(k, q, v, sched, cfg, chunk_size=chunk,
                                                deaths=deaths)
    read_s, final_s = sequential_replay.sequential_tape_scan(k, q, v, sched, cfg)
    torch.testing.assert_close(read_c, read_s, atol=2e-6, rtol=2e-6)
    for name in ('keys', 'values', 'strength', 'age', 'usage'):
        torch.testing.assert_close(getattr(final_c, name), getattr(final_s, name),
                                   atol=2e-6, rtol=2e-6)
    assert torch.equal(final_c.live, final_s.live)
    return read_c, final_c


def test_pop_death_then_reallocation_of_the_same_slot():
    """A slot dies at the POP free check and is re-allocated later: the append
    erases the lane (d = 0) after a zero diagonal, so the pre-death strength
    must not leak into the new life."""
    cfg = config(budget=2, heads=1, decay=0.9, threshold=0.3)
    tokens = 10
    script = _scripted_schedule(1, tokens, cfg.slots)
    script['alloc'][0, 0] = 0
    script['pop_slot'][0, 2] = 0
    script['pop_factor'][0, 2] = 0.3        # 0.9*0.9*0.3 = 0.243 <= 0.3 -> POP death
    script['alloc'][0, 5] = 0               # re-allocate the dead slot
    deaths = []
    read, final = _compare_scripted(script, cfg, tokens, 3, deaths=deaths)
    assert bool(final.live[0, 0])
    assert int(final.age[0, 0]) == tokens - 5
    torch.testing.assert_close(final.strength[0, 0],
                               torch.tensor(0.9 ** (tokens - 5)))
    pop_deaths = torch.cat([d[0] for d in deaths], dim=1)
    decay_deaths = torch.cat([d[1] for d in deaths], dim=1)
    assert bool(pop_deaths[0, 2, 0]) and not bool(decay_deaths[0, 2, 0])


def test_decay_death_freezes_strength_and_age():
    cfg = config(budget=2, heads=1, decay=0.5, threshold=0.2)
    tokens = 8
    script = _scripted_schedule(1, tokens, cfg.slots)
    script['alloc'][0, 1] = 1
    read, final = _compare_scripted(script, cfg, tokens, 4)
    assert not bool(final.live[0, 1])
    # 0.5**3 = 0.125 <= 0.2 dies at token 3; strength and age freeze there
    # (the reference increments age before the decay free check).
    torch.testing.assert_close(final.strength[0, 1], torch.tensor(0.5 ** 3))
    assert int(final.age[0, 1]) == 3


def test_overwrite_erasure_and_correction_chain():
    """Repeated writes to one slot inside a single chunk: the read of later
    tokens must see exactly the last payload, and a merge chains off the
    previous written value."""
    cfg = config(budget=3, heads=1, decay=0.9, threshold=1e-3)
    tokens = 9
    script = _scripted_schedule(1, tokens, cfg.slots)
    script['alloc'][0, 0] = 0
    script['alloc'][0, 1] = 1
    script['correct'][0, 2] = 1
    script['beta'][0, 2] = 0.5
    script['correct'][0, 3] = 1
    script['beta'][0, 3] = 0.25
    script['correct'][0, 4] = 1
    script['beta'][0, 4] = 0.75
    script['gate'][0, 5] = 0.3
    script['alloc'][0, 5] = 2
    read, final = _compare_scripted(script, cfg, tokens, 4)
    assert bool(final.live[0, 2])


def test_chunk_length_not_dividing_token_count_and_empty_writes():
    cfg = config(budget=4, heads=2, decay=0.8, threshold=0.2)
    tokens = 13
    script = _scripted_schedule(3, tokens, cfg.slots)
    script['alloc'][0, 0] = 0
    script['alloc'][2, 11] = 3
    for chunk in (1, 2, 3, 4, 5, 13, 64):
        _compare_scripted(script, cfg, tokens, chunk)


def test_no_writes_at_all_is_a_pure_decay():
    cfg = config(budget=3, heads=1, decay=0.5, threshold=1e-9)
    tokens = 7
    script = _scripted_schedule(2, tokens, cfg.slots)
    read, final = _compare_scripted(script, cfg, tokens, 3)
    assert not bool(final.live.any())
    assert float(final.strength.abs().max()) == 0.0
    assert float(read.abs().max()) == 0.0


def test_single_token_sequence_and_every_chunk_size():
    cfg = config(budget=3, heads=2, decay=0.9, threshold=0.15)
    for tokens in (1, 2, 3):
        _fixture_kwargs = dict(budget=3, heads=2, decay=0.9, threshold=0.15)
        fixture = _fixture(seed=5, **_fixture_kwargs)
        cfg2, args, out_ref, final_ref, trace, sched = fixture
        if args[0].shape[1] != tokens:
            continue
        for chunk in (1, 2, 7):
            read, final = chunked_run(cfg2, args, sched, chunk)
            worst, detail = deviation(cfg2, args, read, final, out_ref, final_ref)
            assert worst < 1e-5, detail


def test_schedule_rejects_append_and_correct_collision():
    cfg = config(budget=2, heads=1)
    script = _scripted_schedule(1, 3, cfg.slots)
    script['alloc'][0, 1] = 0
    script['correct'][0, 1] = 0
    k = torch.zeros(1, 3, cfg.key_dim)
    with pytest.raises(ValueError, match='appends and corrects'):
        chunked.chunked_tape_scan(k, k, torch.zeros(1, 3, cfg.value_dim),
                                  DecisionSchedule(**script), cfg, chunk_size=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
@pytest.mark.parametrize('steps,lanes', [(128, 7488), (4, 1000), (128, 140544)])
def test_triton_affine_scan_is_bit_exact(steps, lanes):
    """The scan kernel rounds once per operation (no FMA contraction), so the
    strength channel -- and therefore every derived death boundary -- is
    identical to the PyTorch pass.  Without that, a 1-ulp strength change flips
    a death boundary and moves the read output by ~5e-4."""
    from experiments.expressivity_tasks.x44_chunk_scan import triton_scan

    torch.manual_seed(0)
    multiplier = torch.rand(steps, lanes, device='cuda') * .9 + .05
    injection = torch.randn(steps, lanes, device='cuda')
    multiplier[3::7] = 0.0                      # zero diagonals in every chunk
    entry = torch.randn(lanes, device='cuda')
    exact, deviation = triton_scan.same_as_torch(multiplier, injection, entry)
    assert exact and deviation == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_cuda_graph_replay_is_bit_identical_to_eager():
    """Graph capture must not change a single bit: it only removes launches."""
    from experiments.expressivity_tasks.x44_chunk_scan import graph_scan

    cfg = config(budget=32, heads=1, decay=0.999, threshold=1e-3)
    args = tuple(x.cuda() for x in inputs(cfg, 17, batch=4, tokens=96))
    out_ref, final_ref, trace = run_naive(cfg, args, 17)
    sched = schedule_of(cfg, args, trace)
    eager_read, eager_final = chunked_run(cfg, args, sched, 32)
    graph = graph_scan.ChunkedScanGraph(cfg, 4, 96, chunk_size=32)
    graph_read, graph_final = graph(flat(args[0], cfg, 4, 96), flat(args[1], cfg, 4, 96),
                                   flat(args[2], cfg, 4, 96), sched)
    assert torch.equal(graph_read, eager_read)
    for name in ('keys', 'values', 'strength', 'age', 'usage'):
        assert torch.equal(getattr(graph_final, name), getattr(eager_final, name))
    assert torch.equal(graph_final.live, eager_final.live)
    worst, detail = deviation(cfg, args, graph_read, graph_final, out_ref, final_ref)
    assert worst < 1e-5, detail


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_parity_on_gpu_at_production_slot_counts():
    """Two rows, 64-slot store, 300 tokens, chunk sizes that stress the
    in-chunk write override; device and dtype identical to training."""
    cfg = config(budget=64, heads=1, decay=0.999, threshold=1e-3)
    args = tuple(x.cuda() for x in inputs(cfg, 13, batch=2, tokens=300))
    out_ref, final_ref, trace = run_naive(cfg, args, 13)
    sched = schedule_of(cfg, args, trace)
    reference_read, reference_final = sequential_run(cfg, args, sched)
    floor, _ = deviation(cfg, args, reference_read, reference_final, out_ref, final_ref)
    for chunk in (8, 32, 64, 128):
        read, final = chunked_run(cfg, args, sched, chunk)
        worst, detail = deviation(cfg, args, read, final, out_ref, final_ref)
        print('gpu chunk', chunk, 'floor', floor, detail)
        assert worst <= max(NOISE_FACTOR * floor, 1e-5), detail
