"""K7 gates: the high-rank store's parity, the split-softmax law, the walk's decisions.

Gate discipline is the lane's (``test_k23_read.py``, ``test_hostloop_free.py``): the
per-token oracle ``algebra.sequential_matrix_scan`` and the chunked twin are the
references, the state and the liveness channel are held to ``torch.equal`` (both K7
paths share K1's bit-exact walk / K6's exact decision arithmetic), and every read is
held to ``NOISE_FACTOR`` times a **measured** fp32 floor on the same fixture -- never
a fitted constant.  The Part-B walk additionally holds its DECISIONS, read WEIGHTS,
payload lanes and pre-decision liveness to ``torch.equal`` against the K7 eager oracle
(the K6 precedent: the oracle is written to the kernel's reduction order), and the
oracle itself is pinned to the K6 registration's oracle at rank 128.

The Lean-law gate (``test_the_split_softmax_normalises_by_the_full_population_mass``)
is the one that would catch a per-block renormalisation: with the softmax mass split
across slot blocks, renormalising inside a block doubles a survivor's coefficient
(``Obstructions.truncation_sharpens_survivor``), a deviation of order 1, while the
lawful full-population denominator lands at the measured floor.

Every test needs a CUDA device.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, MERGE, NONE, POP, sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.hostloop_free import (
    chunked_block_scan, persistent_available)
from experiments.expressivity_tasks.x45_matrix_emender.k23_read import k23_chunk
from experiments.expressivity_tasks.x45_matrix_emender.k7_high_rank_store import (
    MAX_VALUE_DIM, MAX_WALK_VALUE_DIM, k7_block_scan, k7_block_scan_routed,
    k7_chunk, k7_chunk_routed, k7_read_weights, k7_sequential_walk,
    sequential_oracle)
from experiments.expressivity_tasks.x45_matrix_emender.read import read_weights
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.test_k1_schedule_scan import (
    routed_fixture)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR, sequential_noise_floor)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
POLICY = RoutingPolicy(ops=True)


# --------------------------------------------------------------------------- fixtures


def routed_chunk_fixture(rank: int, tokens: int = 64, batch: int = 2, heads: int = 2,
                         value_dim: int = 128, seed: int | None = None,
                         live_fraction: float = 0.7, chunk: int = 0):
    """A real routed chunk at any rank: the router's schedule and weights, the
    entry store it routed against, and the decay channel in the block paths'
    head-major layout."""
    generator = torch.Generator().manual_seed(seed if seed is not None else rank)
    shape = (batch, heads, tokens, value_dim)
    query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(*shape, generator=generator).to(DEVICE)
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=generator).to(DEVICE) * 0.3
    live = (torch.rand(batch, heads, rank, generator=generator).to(DEVICE)
            < live_fraction)
    state = state * live.unsqueeze(-1)
    decay = torch.full((batch, heads, tokens), 0.95, device=DEVICE)
    schedule, weights = route_chunk(chunk, 0, state.clone(), live.clone(), query,
                                    payload, POLICY, rank)
    return dict(query=query, state=state, live=live, decay=decay,
                schedule=schedule, weights=weights)


def oracle_chunk(f: dict):
    """The per-token oracle over one chunk fixture: ``(reads, state)`` in the
    block paths' layouts."""
    batch, heads, tokens, rank = f["weights"].shape
    weights_tm = f["weights"].permute(0, 2, 1, 3).contiguous()
    reads, state = sequential_matrix_scan(
        weights_tm, f["decay"].permute(0, 2, 1), f["schedule"],
        initial_state=f["state"].clone(), initial_live=f["live"].clone())
    return reads.permute(0, 2, 1, 3), state


def chunk_floor(f: dict) -> float:
    """The measured fp32 sequential floor on the chunk fixture itself."""
    weights_tm = f["weights"].permute(0, 2, 1, 3).contiguous()
    floor, _, _ = sequential_noise_floor(weights_tm, f["decay"].permute(0, 2, 1),
                                         f["schedule"])
    return floor


def populated_walk_fixture(rank: int, tokens: int = 16, batch: int = 2, heads: int = 2,
                           value_dim: int = 128, seed: int = 11,
                           live_fraction: float = 0.8):
    """A populated store and a random stream: mostly APPEND/CORRECT, and every
    channel the walk records is compared against the oracle, so any arithmetic
    difference anywhere is a flip, not a tolerance."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, tokens, heads, value_dim)
    query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(*shape, generator=generator).to(DEVICE)
    state = torch.randn(batch, heads, rank, value_dim,
                        generator=generator).to(DEVICE) * 0.3
    live = (torch.rand(batch, heads, rank, generator=generator).to(DEVICE)
            < live_fraction)
    state = state * live.unsqueeze(-1)
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    return dict(query=query, payload=payload, state=state, live=live, decay=decay)


def run_walk(f: dict, rank: int, collect: bool = True):
    return k7_sequential_walk(f["query"], f["payload"], f["decay"], policy=POLICY,
                              rank=rank, initial_state=f["state"],
                              initial_live=f["live"], collect=collect)


def run_oracle(f: dict, rank: int):
    return sequential_oracle(f["query"], f["payload"], f["decay"], policy=POLICY,
                             rank=rank, initial_state=f["state"],
                             initial_live=f["live"])


def decision_flips(kernel: dict, oracle: dict) -> int:
    """Every bit the recorded channels can differ by: decisions, payload lanes,
    read weights, pre-decision liveness."""
    total = 0
    for field in ("slot", "op", "erase", "live_after"):
        total += int((kernel["schedule"].__dict__[field]
                      != oracle["schedule"].__dict__[field]).sum())
    total += int((kernel["schedule"].payload != oracle["schedule"].payload).sum())
    total += int((kernel["weights"] != oracle["weights"]).sum())
    total += int((kernel["live_pre"] != oracle["live_pre"]).sum())
    return total


# ------------------------------------------------------- Part A: the slot-blocked read


@needs_cuda
def test_k7_read_matches_k23_given_identical_weights():
    """Kernel against kernel, the same torch weights channel into both: the
    reads may differ only by reduction order (state and liveness are bit-equal
    by construction -- both paths run K1)."""
    for rank in (128, 512, 2048):
        f = routed_chunk_fixture(rank)
        r_k23, s_k23, l_k23 = k23_chunk(f["state"].clone(), f["live"].clone(),
                                        f["weights"], f["decay"], f["schedule"])
        r_k7, s_k7, l_k7 = k7_chunk(f["state"].clone(), f["live"].clone(),
                                    f["weights"], f["decay"], f["schedule"])
        assert torch.equal(s_k23, s_k7)
        assert torch.equal(l_k23, l_k7)
        floor = chunk_floor(f)
        deviation = float((r_k23 - r_k7).abs().max())
        assert deviation <= NOISE_FACTOR * floor, (rank, deviation, floor)


@needs_cuda
def test_k7_split_softmax_weights_match_the_registered_definition():
    """The in-kernel softmax against ``read.read_weights`` (the single-sourced
    definition), at a floor measured as the torch channel against its own fp64
    twin -- torch's own softmax order is not reproducible from a kernel (the
    K5A receipt), so this gate is floors, never bits."""
    for rank in (128, 2048):
        f = routed_chunk_fixture(rank)
        state, live = f["state"], f["live"]
        query = f["query"]
        got = k7_read_weights(query, state, live, POLICY.read_scale)
        want32 = read_weights(query, state, live, POLICY.read_scale)
        want64 = read_weights(query.to(torch.float64), state.to(torch.float64),
                              live, POLICY.read_scale)
        floor = float((want32 - want64).abs().max())
        deviation = float((got - want32).abs().max())
        assert deviation <= NOISE_FACTOR * max(floor, torch.finfo(torch.float32).eps), \
            (rank, deviation, floor)
        # the certified all-dead zero attention, exactly
        dead = torch.zeros_like(live)
        assert bool((k7_read_weights(query, state, dead, POLICY.read_scale) == 0).all())


@needs_cuda
def test_the_split_softmax_normalises_by_the_full_population_mass():
    """THE LEAN LAW, as a number: with the mass deliberately split across slot
    blocks (two clusters of equally-attractive rows, one per end of the rank
    axis), a per-block renormalisation doubles a survivor's coefficient
    (``truncation_sharpens_survivor``) -- an O(1) error.  The lawful
    full-population denominator lands at the fp32 floor."""
    rank, value_dim = 2048, 128
    batch, heads, tokens = 1, 1, 4
    # every row is +-e_0: two rows (one per end of the rank axis, hence in
    # DIFFERENT slot blocks) aligned with the query, the rest anti-aligned.
    # The full-population probability of each aligned row is then
    # ~1/(2 + 2046*exp(-4)) ~ 0.025; the per-block renormalisation the Lean
    # obstruction names would give ~0.46 -- an O(1) difference.
    state = torch.zeros(batch, heads, rank, value_dim, device=DEVICE)
    state[:, :, :, 0] = -1.0
    state[:, :, 0, 0] = 1.0
    state[:, :, rank - 1, 0] = 1.0
    live = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    query = torch.zeros(batch, heads, tokens, value_dim, device=DEVICE)
    query[:, :, :, 0] = 1.0
    got = k7_read_weights(query, state, live, POLICY.read_scale)
    want = read_weights(query, state, live, POLICY.read_scale)
    value = float(got[0, 0, 0, 0])
    assert abs(value - float(want[0, 0, 0, 0])) < 1e-6
    assert value < 0.1, (value, float(want[0, 0, 0, 0]))
    assert float((got - want).abs().max()) <= 8 * 2.4e-7


@needs_cuda
def test_k7_full_chunk_matches_k23_with_real_routing():
    """The end-to-end any-rank chunk (in-kernel weights) against the k23 drop-in
    fed the router's own torch weights: state and liveness bit-equal, reads at
    the measured floor."""
    for rank in (128, 512):
        f = routed_chunk_fixture(rank)
        r_k23, s_k23, l_k23 = k23_chunk(f["state"].clone(), f["live"].clone(),
                                        f["weights"], f["decay"], f["schedule"])
        r_k7, s_k7, l_k7 = k7_chunk_routed(f["query"], f["state"].clone(),
                                          f["live"].clone(), f["decay"],
                                          f["schedule"], POLICY.read_scale)
        assert torch.equal(s_k23, s_k7)
        assert torch.equal(l_k23, l_k7)
        floor = chunk_floor(f)
        deviation = float((r_k23 - r_k7).abs().max())
        assert deviation <= NOISE_FACTOR * floor, (rank, deviation, floor)


@needs_cuda
def test_the_canonical_routed_fixture_matches_the_oracle():
    """The lane's canonical fixture (all five ops, rank 5): state bit-exact
    against the per-token oracle, read within the measured floor, liveness
    equal -- the same numbers K23_READ.md section 2 reports for the chunked
    twin at this chunk length."""
    fixture = routed_fixture()
    schedule = fixture.schedule
    weights = fixture.weights
    decay = fixture.decay.permute(0, 2, 1).contiguous()
    floor, oracle_state, oracle_reads_tm = sequential_noise_floor(
        fixture.result.weights, fixture.decay, schedule)
    oracle_reads = oracle_reads_tm.permute(0, 2, 1, 3)
    state = torch.zeros(2, 2, 5, 8, device=DEVICE)
    live = torch.zeros(2, 2, 5, dtype=torch.bool, device=DEVICE)
    reads, exit_state, exit_live = k7_chunk(state, live, weights, decay, schedule)
    assert torch.equal(exit_state, oracle_state)
    deviation = float((reads - oracle_reads).abs().max())
    assert deviation <= NOISE_FACTOR * floor, (deviation, floor)


@needs_cuda
def test_k7_executes_at_rank_1024_and_2048_the_refusal_removed():
    """The growth snap set runs: k7 chunks and whole block calls execute at
    rank 1024 and 2048, with the state bit-exact against the oracle (and
    against the k23 twin, which was never refused), and the persistent
    kernel's by-name refusal receipted alongside -- the ceiling it names is
    the one this module removes.

    The block fixture's schedule and weights come from ``routed_chunk_scan``
    itself, so the weights channel is the chunk-granular one the registered
    router records (each chunk's against its own entry state) -- the honest
    twin for both the given-weights and the in-kernel forms.
    """
    from experiments.expressivity_tasks.x45_matrix_emender.scan import routed_chunk_scan
    for rank in (1024, 2048):
        f = routed_chunk_fixture(rank, tokens=64)
        o_reads, o_state = oracle_chunk(f)
        reads, state, live = k7_chunk_routed(f["query"], f["state"].clone(),
                                             f["live"].clone(), f["decay"],
                                             f["schedule"], POLICY.read_scale)
        assert torch.equal(state, o_state)
        assert float((reads - o_reads).abs().max()) <= NOISE_FACTOR * chunk_floor(f)
        # the whole block call, both forms, against the same twin
        generator = torch.Generator().manual_seed(rank + 1)
        batch, heads, tokens, value_dim = 2, 2, 256, 128
        shape = (batch, tokens, heads, value_dim)
        query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
        payload = torch.randn(*shape, generator=generator).to(DEVICE)
        decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
        recorded = routed_chunk_scan(query, payload, decay, router=route_chunk,
                                      policy=POLICY, rank=rank, rank_max=rank,
                                      chunk_tokens=128, collect=True)
        schedule = recorded.schedule
        weights = recorded.weights.permute(0, 2, 1, 3).contiguous()
        decay_hm = decay.permute(0, 2, 1).contiguous()
        state0 = torch.zeros(batch, heads, rank, value_dim, device=DEVICE)
        live0 = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
        r_k7, s_k7, l_k7 = k7_block_scan(state0.clone(), live0.clone(), weights,
                                        decay_hm, schedule, chunk_tokens=128)
        r_routed, s_r, l_r = k7_block_scan_routed(
            query.permute(0, 2, 1, 3).contiguous(), state0.clone(), live0.clone(),
            decay_hm, schedule, POLICY.read_scale, chunk_tokens=128)
        r_k23, s_k23, l_k23 = chunked_block_scan(state0.clone(), live0.clone(),
                                                 weights, decay_hm, schedule,
                                                 chunk_tokens=128)
        assert torch.equal(s_k7, s_k23) and torch.equal(l_k7, l_k23)
        assert torch.equal(s_r, s_k23) and torch.equal(l_r, l_k23)
        # the in-kernel weights land at the weights floor of the recorded channel
        weights_dev = float((r_routed - r_k23).abs().max())
        read_scale = float(r_k23.abs().max())
        assert weights_dev <= NOISE_FACTOR * max(chunk_floor(f), 1e-6 * read_scale), \
            (rank, weights_dev)
        assert float((r_k7 - r_k23).abs().max()) <= NOISE_FACTOR * chunk_floor(f)
        # the by-name refusal this module exists to remove, receipted
        assert not persistent_available(rank, value_dim)


@needs_cuda
def test_k7_block_scan_is_bit_deterministic():
    f = routed_chunk_fixture(2048, tokens=256, seed=99)
    a = k7_block_scan_routed(f["query"], f["state"].clone(), f["live"].clone(),
                             f["decay"], f["schedule"], POLICY.read_scale, 128)
    b = k7_block_scan_routed(f["query"], f["state"].clone(), f["live"].clone(),
                             f["decay"], f["schedule"], POLICY.read_scale, 128)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) \
        and torch.equal(a[2], b[2])
    w = k7_block_scan(f["state"].clone(), f["live"].clone(), f["weights"],
                      f["decay"], f["schedule"], 128)
    w2 = k7_block_scan(f["state"].clone(), f["live"].clone(), f["weights"],
                       f["decay"], f["schedule"], 128)
    assert torch.equal(w[0], w2[0])


@needs_cuda
def test_k7_refuses_what_it_cannot_express():
    f = routed_chunk_fixture(128)
    wide_query = torch.randn(1, 1, 4, MAX_VALUE_DIM + 1, device=DEVICE)
    wide_state = torch.randn(1, 1, 8, MAX_VALUE_DIM + 1, device=DEVICE)
    wide_live = torch.ones(1, 1, 8, dtype=torch.bool, device=DEVICE)
    with pytest.raises(ValueError, match="value dims"):
        k7_read_weights(wide_query, wide_state, wide_live, POLICY.read_scale)
    walk = populated_walk_fixture(64, tokens=4)
    generator = torch.Generator().manual_seed(9)
    wide = dict(
        query=torch.randn(2, 4, 2, MAX_WALK_VALUE_DIM + 1,
                          generator=generator).to(DEVICE),
        payload=torch.randn(2, 4, 2, MAX_WALK_VALUE_DIM + 1,
                            generator=generator).to(DEVICE),
        state=torch.randn(2, 2, 64, MAX_WALK_VALUE_DIM + 1,
                          generator=generator).to(DEVICE),
        live=torch.ones(2, 2, 64, dtype=torch.bool, device=DEVICE),
        decay=torch.full((2, 4, 2), 0.95, device=DEVICE))
    with pytest.raises(ValueError, match="probe ceiling"):
        k7_sequential_walk(wide["query"], wide["payload"], wide["decay"],
                           policy=POLICY, rank=64, initial_state=wide["state"],
                           initial_live=wide["live"])
    with pytest.raises(ValueError, match="growth is refused"):
        k7_sequential_walk(walk["query"], walk["payload"], walk["decay"],
                           policy=RoutingPolicy(ops=True, grow=True), rank=64)
    with pytest.raises(ValueError, match="append-only control"):
        k7_sequential_walk(walk["query"], walk["payload"], walk["decay"],
                           policy=RoutingPolicy(ops=False), rank=64)
    with pytest.raises(RuntimeError):   # FORWARD_ONLY, the lane's standing rule
        query = f["query"].requires_grad_(True)
        k7_read_weights(query, f["state"], f["live"], POLICY.read_scale)


# ------------------------------------------- Part B: the sequential decision walk


@needs_cuda
def test_walk_decisions_bit_exact_at_rank_1024_and_2048():
    """THE Part-B gate: every recorded channel bit-equal against the K7 oracle
    (the oracle written to the kernel's reduction order -- the K6 precedent),
    the exit state and liveness bit-equal, the read at a measured floor."""
    for rank in (128, 1024, 2048):
        f = populated_walk_fixture(rank)
        kernel = run_walk(f, rank)
        oracle = run_oracle(f, rank)
        assert decision_flips(kernel, oracle) == 0, rank
        assert torch.equal(kernel["state"], oracle["state"])
        assert torch.equal(kernel["live"], oracle["live"])
        # the read floor: the fp32 oracle against its own fp64 twin
        f64 = dict(f)
        f64.update(query=f["query"].to(torch.float64), payload=f["payload"].to(torch.float64),
                   decay=f["decay"].to(torch.float64), state=f["state"].to(torch.float64))
        oracle64 = sequential_oracle(f64["query"], f64["payload"], f64["decay"],
                                     policy=POLICY, rank=rank,
                                     initial_state=f64["state"],
                                     initial_live=f64["live"])
        floor = float((oracle["reads"] - oracle64["reads"].to(torch.float32)).abs().max())
        deviation = float((kernel["reads"] - oracle["reads"]).abs().max())
        assert deviation <= NOISE_FACTOR * max(floor, torch.finfo(torch.float32).eps), \
            (rank, deviation, floor)


@needs_cuda
def test_walk_covers_every_op_branch_bit_exactly():
    """Crafted stores forcing each branch of the five-op chain, in the K6
    suite's tradition: a repeated-direction store forces MERGE (both lanes), a
    full store with an orthogonal stream forces POP, and a full store whose
    every row sits at cosine in (tau_pop, tau_match) to the payload forces
    NONE.  Every one is bit-exact against the oracle."""
    value_dim, rank = 128, 64
    generator = torch.Generator().manual_seed(5)
    dirs = torch.randn(4, value_dim, generator=generator)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    # (i) MERGE + CORRECT: rows repeat three directions, payloads match them.
    state = dirs[torch.arange(rank) % 3].view(1, 1, rank, value_dim).to(DEVICE) * 0.7
    live = torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE)
    payload = dirs[torch.arange(16) % 3].view(1, 16, 1, value_dim).to(DEVICE)
    query = torch.randn(1, 16, 1, value_dim, generator=generator).to(DEVICE) * 0.5
    f = dict(query=query, payload=payload, state=state, live=live,
             decay=torch.full((1, 16, 1), 0.95, device=DEVICE))
    kernel, oracle = run_walk(f, rank), run_oracle(f, rank)
    assert decision_flips(kernel, oracle) == 0
    ops = kernel["schedule"].op[..., 0].reshape(-1)
    assert int((ops == MERGE).sum()) > 0
    assert int((kernel["schedule"].op[..., 1] == MERGE).sum()) > 0   # both lanes

    # (ii) POP: a full store of distinct directions, an orthogonal stream.
    rows = torch.randn(rank, value_dim, generator=generator)
    state = (rows / rows.norm(dim=-1, keepdim=True)).view(1, 1, rank, value_dim).to(DEVICE)
    orth = torch.randn(8, value_dim, generator=generator)
    orth = (orth / orth.norm(dim=-1, keepdim=True)).view(1, 8, 1, value_dim).to(DEVICE)
    f = dict(query=query[:, :8], payload=orth, state=state,
             live=torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE),
             decay=torch.full((1, 8, 1), 0.95, device=DEVICE))
    kernel, oracle = run_walk(f, rank), run_oracle(f, rank)
    assert decision_flips(kernel, oracle) == 0
    ops = kernel["schedule"].op[..., 0].reshape(-1)
    assert int((ops == POP).sum()) > 0 and int((ops == APPEND).sum()) > 0

    # (iii) NONE: every row at cosine 0.3 to the payload, store full -- nothing
    # matches (cos < tau_match), nothing is bad enough to POP (cos > tau_pop),
    # nothing is free.  The first four basis vectors carry the shared part.
    state = torch.zeros(1, 1, rank, value_dim, device=DEVICE)
    for j in range(rank):
        state[0, 0, j, 0] = 0.3
        state[0, 0, j, 1 + (j % 8)] = 0.95
    pl = torch.zeros(1, 4, 1, value_dim, device=DEVICE)
    pl[..., 0] = 1.0
    f = dict(query=query[:, :4], payload=pl, state=state,
             live=torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE),
             decay=torch.full((1, 4, 1), 0.95, device=DEVICE))
    kernel, oracle = run_walk(f, rank), run_oracle(f, rank)
    assert decision_flips(kernel, oracle) == 0
    ops = kernel["schedule"].op[..., 0].reshape(-1)
    assert int((ops == NONE).sum()) == 4


@needs_cuda
def test_the_k7_oracle_agrees_with_the_k6_registration():
    """The K7 oracle is the parity fixed point; pin it to the K6 draft
    registration's own oracle at the registered shape: decisions, weights,
    pre-decision liveness and the exit state bit-equal."""
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import \
        sequential_oracle_scan
    rank = 128
    f = populated_walk_fixture(rank, tokens=8, seed=2, live_fraction=0.6)
    # the walk fixture's layout is (b, T, h, .); both oracles take it verbatim
    o7 = sequential_oracle(f["query"], f["payload"], f["decay"], policy=POLICY,
                           rank=rank, initial_state=f["state"],
                           initial_live=f["live"])
    o6 = sequential_oracle_scan(f["query"], f["payload"], f["decay"], policy=POLICY,
                                rank=rank, rank_max=rank,
                                initial_state=f["state"], initial_live=f["live"],
                                collect=True)
    for field in ("slot", "op", "erase", "live_after"):
        assert torch.equal(o7["schedule"].__dict__[field],
                           o6["schedule"].__dict__[field]), field
    assert torch.equal(o7["weights"], o6["weights"])
    assert torch.equal(o7["live_pre"], o6["live_pre"])
    assert torch.equal(o7["state"], o6["state"])


@needs_cuda
def test_the_walk_is_bit_deterministic_and_collect_free():
    f = populated_walk_fixture(1024)
    a = run_walk(f, 1024)
    b = run_walk(f, 1024)
    for key in ("reads", "state", "live"):
        assert torch.equal(a[key], b[key]), key
    assert torch.equal(a["schedule"].slot, b["schedule"].slot)
    # collect=False is the same walk without the diagnostic channels
    c = run_walk(f, 1024, collect=False)
    assert torch.equal(a["reads"], c["reads"])
    assert torch.equal(a["state"], c["state"])
    assert torch.equal(a["schedule"].op, c["schedule"].op)


@needs_cuda
def test_the_walk_all_dead_read_is_exactly_zero():
    """The certified all-dead zero attention, at a rank no K6 kernel can run:
    token 0 reads an empty dead store, and its read and weights are exactly
    zero (later tokens see the appends, as the semantics says)."""
    rank = 2048
    f = populated_walk_fixture(rank, tokens=4, live_fraction=0.0)
    kernel = run_walk(f, rank)
    assert bool((kernel["reads"][:, 0] == 0).all())
    assert bool((kernel["weights"][:, 0] == 0).all())
