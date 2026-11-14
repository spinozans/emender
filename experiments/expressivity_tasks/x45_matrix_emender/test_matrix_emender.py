"""X45 gates A, B and E: orthonormality/exactness, scan parity, determinism.

The tolerance discipline is X44's: the sequential-vs-sequential noise floor is
measured on the same fixtures first (two reduction orderings, and float32 against
a float64 exact fold) and the chunked path is required to stay within
``NOISE_FACTOR`` times that measured floor -- never a fitted constant.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule, apply_lane, key_form_step,
    sequential_matrix_scan)
from experiments.expressivity_tasks.x45_matrix_emender.keys import (
    MAX_RANK, ORTHONORMAL_RANKS, KeyFamily, growth_target, snap_rank)
from experiments.expressivity_tasks.x45_matrix_emender import read as read_module
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    chunk_scan, chunked_scan, repeat_write_rate, routed_chunk_scan)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)

NOISE_FACTOR = 8.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- fixtures

def random_schedule(batch, tokens, heads, rank, value_dim, lanes=2, seed=0,
                    decay=0.9, device=DEVICE):
    """A synthetic fixed schedule exercising all five ops, with its live_after channel."""
    generator = torch.Generator().manual_seed(seed)
    shape = (batch, tokens, heads, lanes)
    slot = torch.randint(0, rank, shape, generator=generator)
    op = torch.randint(0, 5, shape, generator=generator)
    erase = (op == CORRECT) | (op == POP) | (op == MERGE)
    live_after = (op == APPEND) | (op == CORRECT)
    op = torch.where(live_after | (op == POP), op, torch.full_like(op, NONE))
    erase = (op == CORRECT) | (op == POP) | (op == MERGE)
    live_after = (op == APPEND) | (op == CORRECT)
    payload = torch.randn(*shape, value_dim, generator=generator)
    # Inactive and erasing lanes carry no payload: APPEND/CORRECT write their value,
    # POP writes exactly zero, an inactive lane is the identity.
    carried = ((op == APPEND) | (op == CORRECT)).unsqueeze(-1)
    payload = torch.where(carried, payload, torch.zeros_like(payload))
    schedule = MatrixSchedule(slot=slot.to(device), erase=erase.to(device),
                              payload=payload.to(device), op=op.to(device),
                              live_after=live_after.to(device))
    q = torch.randn(batch, tokens, heads, rank, generator=generator).to(device) * 0.3
    values = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(device)
    decay_t = (torch.full((batch, tokens, heads), decay) *
               (1 + 0.1 * torch.randn(batch, tokens, heads, generator=generator))).clamp(0.5, 1.0)
    return schedule, q, values, decay_t.to(device)


def sequential_noise_floor(q, decay, schedule, device=DEVICE):
    """Max deviation between two orderings of the same sequential math, and fp64."""
    reads_a, state_a = sequential_matrix_scan(q, decay, schedule)
    # Second ordering: the read as an explicit per-head matvec over a transposed state.
    state = torch.zeros_like(state_a)
    reads_b = torch.empty_like(reads_a)
    for t in range(q.shape[1]):
        reads_b[:, t] = (state.transpose(-1, -2) @ q[:, t].unsqueeze(-1)).squeeze(-1)
        state = state * decay[:, t, :, None, None]
        for lane in range(schedule.lanes):
            op = schedule.op[:, t, :, lane]
            if not bool(op.ne(NONE).any()):
                continue
            apply_lane(state, schedule.slot[:, t, :, lane], schedule.erase[:, t, :, lane],
                       schedule.payload[:, t, :, lane])
    reads_c, state_c = sequential_matrix_scan(q.double(), decay.double(), schedule)
    floor_order = float((reads_a - reads_b).abs().max())
    floor_exact = float((reads_a.double() - reads_c).abs().max())
    state_exact = float((state_a.double() - state_c).abs().max())
    return max(floor_order, floor_exact, state_exact), state_a, reads_a


# --------------------------------------------------------------------------- gate A

@pytest.mark.parametrize("kind", ["standard", "hadamard", "qr"])
def test_orthonormal_family_prefixes_are_identity(kind):
    """K[0:r] @ K[0:r].T == I to fp32 noise, for every precompiled rank."""
    family = KeyFamily(kind=kind, dim=MAX_RANK, seed=3401)
    for rank in ORTHONORMAL_RANKS:
        table = family.basis_rows(rank)
        assert table.shape == (rank, MAX_RANK)
        gram = table @ table.T
        deviation = float((gram - torch.eye(rank, dtype=gram.dtype)).abs().max())
        # f32 dot of length `rank` of unit entries: rounding grows like rank*eps.
        assert deviation <= 4 * rank * torch.finfo(torch.float32).eps, (kind, rank, deviation)


def test_prefix_ranks_are_nested_and_never_re_orthonormalised():
    """Growing the rank only reveals further rows; existing rows are byte-identical."""
    for kind in ("standard", "hadamard", "qr"):
        family = KeyFamily(kind=kind, dim=MAX_RANK)
        previous = family.basis_rows(ORTHONORMAL_RANKS[0])
        for rank in ORTHONORMAL_RANKS[1:]:
            grown = family.basis_rows(rank)
            assert torch.equal(grown[:previous.shape[0]], previous), (kind, rank)
            previous = grown


def test_pop_and_correct_are_exact_on_orthonormal_keys():
    """POP zeroes its own direction and leaves every other row BIT-identical."""
    rank, value_dim, heads = 8, 5, 2
    key = KeyFamily(kind="standard", dim=MAX_RANK)
    baseline = torch.randn(1, heads, rank, value_dim)
    schedule = MatrixSchedule(
        slot=torch.full((1, 1, heads, 1), 3, dtype=torch.int64),
        erase=torch.ones(1, 1, heads, 1, dtype=torch.bool),
        payload=torch.zeros(1, 1, heads, 1, value_dim),
        op=torch.full((1, 1, heads, 1), POP, dtype=torch.int64),
        live_after=torch.zeros(1, 1, heads, 1, dtype=torch.bool))
    q = torch.zeros(1, 1, heads, rank)
    values = torch.zeros(1, 1, heads, value_dim)
    decay = torch.ones(1, 1, heads)
    _, popped = sequential_matrix_scan(q, decay, schedule, initial_state=baseline.clone())
    assert torch.equal(popped[:, :, 3], torch.zeros(1, heads, value_dim))
    others = [j for j in range(rank) if j != 3]
    assert torch.equal(popped[:, :, others], baseline[:, :, others]), "POP touched another row"

    replacement = torch.randn(1, heads, value_dim)
    corrected_schedule = MatrixSchedule(
        slot=schedule.slot, erase=schedule.erase,
        payload=replacement.unsqueeze(1).unsqueeze(3),
        op=torch.full((1, 1, heads, 1), CORRECT, dtype=torch.int64),
        live_after=torch.ones(1, 1, heads, 1, dtype=torch.bool))
    _, corrected = sequential_matrix_scan(q, decay, corrected_schedule,
                                          initial_state=baseline.clone())
    assert torch.equal(corrected[:, :, 3], replacement)
    assert torch.equal(corrected[:, :, others], baseline[:, :, others])


@pytest.mark.parametrize("kind", ["standard", "hadamard", "qr"])
def test_key_form_ops_match_the_coefficient_form(kind):
    """The registration's key form and the coefficient form agree for a real family."""
    family = KeyFamily(kind=kind, dim=64, seed=11)
    rank, value_dim = 6, 4
    keys = family.basis_rows(rank)
    generator = torch.Generator().manual_seed(3)
    coefficients = torch.randn(rank, value_dim, generator=generator)

    for slot in range(rank):
        vector = torch.randn(value_dim, generator=generator)
        ambient = keys.T @ coefficients
        popped_key = key_form_step(ambient, keys, slot, True, torch.zeros(value_dim))

        state = coefficients.clone().unsqueeze(0).unsqueeze(0)
        apply_lane(state, torch.tensor([[slot]]), torch.tensor([[True]]),
                   torch.zeros(1, 1, value_dim))
        coefficient_form = keys.T @ state[0, 0]

        scale = float(ambient.abs().max())
        assert float((popped_key - coefficient_form).abs().max()) <= 1e-5 * max(1.0, scale)

        appended_key = key_form_step(ambient, keys, slot, False, vector)
        state = coefficients.clone().unsqueeze(0).unsqueeze(0)
        apply_lane(state, torch.tensor([[slot]]), torch.tensor([[False]]),
                   vector.unsqueeze(0).unsqueeze(0))
        assert float((appended_key - keys.T @ state[0, 0]).abs().max()) <= 1e-5 * max(1.0, scale)


def test_append_is_a_rank_one_update():
    rank, value_dim = 5, 3
    state = torch.zeros(1, 1, rank, value_dim)
    vector = torch.randn(value_dim)
    apply_lane(state, torch.tensor([[2]]), torch.tensor([[False]]), vector.unsqueeze(0).unsqueeze(0))
    expected = torch.zeros(1, 1, rank, value_dim)
    expected[0, 0, 2] = vector
    assert torch.equal(state, expected)


# --------------------------------------------------------------------------- gate B

@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 16, 64])
def test_chunked_matches_sequential(chunk):
    """Chunked scan against the per-token oracle, at a measured noise floor."""
    schedule, q, values, decay = random_schedule(2, 48, 2, 9, 5, lanes=2, seed=7, decay=0.9)
    floor, reference_state, reference_reads = sequential_noise_floor(q, decay, schedule)
    assert floor > 0.0
    reads, state, live = chunked_scan(q, decay, schedule, chunk_tokens=chunk,
                                      fused=False)
    read_deviation = float((reads - reference_reads).abs().max())
    state_deviation = float((state - reference_state).abs().max())
    assert read_deviation <= NOISE_FACTOR * floor, (chunk, read_deviation, floor)
    assert state_deviation <= NOISE_FACTOR * floor, (chunk, state_deviation, floor)
    _, _, reference_live, _ = sequential_matrix_scan(
        q, decay, schedule, return_liveness=True)
    assert torch.equal(live, reference_live), "final liveness differs"


@pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 16])
def test_liveness_is_bit_exact_at_every_chunk_boundary(chunk):
    """Death boundaries are integer bookkeeping: they must be identical, bitwise."""
    schedule, q, values, decay = random_schedule(2, 30, 2, 7, 4, lanes=2, seed=13)
    _, _, reference_live, trajectory = sequential_matrix_scan(
        q, decay, schedule, return_liveness=True)
    state = torch.zeros(q.shape[0], q.shape[2], q.shape[3], values.shape[-1],
                        device=q.device)
    live = torch.zeros(q.shape[0], q.shape[2], q.shape[3], dtype=torch.bool, device=q.device)
    for start in range(0, q.shape[1], chunk):
        stop = min(start + chunk, q.shape[1])
        _, state, live = chunk_scan(state, live, q[:, start:stop].permute(0, 2, 1, 3),
                                    decay[:, start:stop].permute(0, 2, 1),
                                    schedule.chunk(start, stop), fused=False)
        assert torch.equal(live, trajectory[stop - 1]), (chunk, stop)


def test_single_token_and_empty_writes():
    schedule, q, values, decay = random_schedule(1, 1, 1, 4, 3, lanes=1, seed=2)
    floor, reference_state, reference_reads = sequential_noise_floor(q, decay, schedule)
    for chunk in (1, 2, 8):
        reads, state, _ = chunked_scan(q, decay, schedule, chunk_tokens=chunk,
                                      fused=False)
        assert float((reads - reference_reads).abs().max()) <= NOISE_FACTOR * floor

    silent = MatrixSchedule(
        slot=torch.zeros(1, 5, 1, 1, dtype=torch.int64),
        erase=torch.zeros(1, 5, 1, 1, dtype=torch.bool),
        payload=torch.zeros(1, 5, 1, 1, 3),
        op=torch.zeros(1, 5, 1, 1, dtype=torch.int64),
        live_after=torch.zeros(1, 5, 1, 1, dtype=torch.bool))
    q_silent = torch.zeros(1, 5, 1, 4)
    values_silent = torch.zeros(1, 5, 1, 3)
    decay_silent = torch.full((1, 5, 1), 0.5)
    reads, state, _ = chunked_scan(q_silent, decay_silent, silent, chunk_tokens=2,
                                  fused=False)
    assert float(state.abs().max()) == 0.0 and float(reads.abs().max()) == 0.0


def test_chunk_length_not_dividing_token_count():
    schedule, q, values, decay = random_schedule(1, 13, 2, 6, 4, lanes=2, seed=5)
    floor, reference_state, reference_reads = sequential_noise_floor(q, decay, schedule)
    for chunk in (2, 3, 4, 5, 13):
        reads, state, _ = chunked_scan(q, decay, schedule, chunk_tokens=chunk,
                                      fused=False)
        assert float((reads - reference_reads).abs().max()) <= NOISE_FACTOR * floor


def test_pop_then_reallocation_of_the_same_slot():
    """POP, then APPEND back into the freed slot inside the same chunk."""
    rank, tokens = 3, 6
    shape = (1, tokens, 1, 1)
    slot = torch.tensor([0, 0, 0, 0, 0, 0], dtype=torch.int64).view(*shape)
    op = torch.tensor([APPEND, APPEND, POP, APPEND, CORRECT, NONE],
                      dtype=torch.int64).view(*shape)
    erase = (op == POP) | (op == CORRECT)
    payload = torch.zeros(1, tokens, 1, 1, 4)
    payload[:, 0, 0, 0] = torch.tensor([1.0, 0, 0, 0])
    payload[:, 1, 0, 0] = torch.tensor([0, 2.0, 0, 0])
    payload[:, 3, 0, 0] = torch.tensor([0, 0, 3.0, 0])
    payload[:, 4, 0, 0] = torch.tensor([0, 0, 0, 4.0])
    schedule = MatrixSchedule(slot=slot, erase=erase, op=op, payload=payload,
                              live_after=(op == APPEND) | (op == CORRECT))
    q = torch.ones(1, tokens, 1, rank) / 4
    values = payload[:, :, :, 0]
    decay = torch.ones(1, tokens, 1)
    floor, reference_state, reference_reads = sequential_noise_floor(q, decay, schedule)
    for chunk in (1, 2, 3, 6):
        reads, state, _ = chunked_scan(q, decay, schedule, chunk_tokens=chunk,
                                      fused=False)
        assert float((reads - reference_reads).abs().max()) <= NOISE_FACTOR * floor
        assert torch.equal(state[:, :, 2:], torch.zeros(1, 1, rank - 2, 4))


def test_backward_through_the_chunked_scan_matches_the_gradient_oracle():
    """Ordinary autograd through the chunked matmuls, checked against the oracle."""
    schedule, q, _, decay = random_schedule(1, 16, 1, 5, 3, lanes=1, seed=17, decay=0.9)

    def gradients(chunked):
        weights = q.clone().requires_grad_(True)
        payload = schedule.payload.clone().requires_grad_(True)
        gate = decay.clone().requires_grad_(True)
        graded = MatrixSchedule(slot=schedule.slot, erase=schedule.erase, payload=payload,
                                op=schedule.op, live_after=schedule.live_after)
        reads = (chunked_scan(weights, gate, graded, chunk_tokens=4, fused=False)[0]
                 if chunked
                 else sequential_matrix_scan(weights, gate, graded)[0])
        return torch.autograd.grad((reads ** 2).sum(), (weights, payload, gate))

    chunked_grads = gradients(True)
    oracle_grads = gradients(False)
    for name, a, b in zip(("weights", "payload", "decay"), chunked_grads, oracle_grads):
        assert torch.isfinite(a).all() and a.abs().max() > 0, name
        relative = float((a - b).abs().max() / b.abs().max().clamp_min(1e-6))
        assert relative <= 5e-3, (name, relative)


# --------------------------------------------------------------------------- gate E

def test_routing_is_deterministic_and_reproducible_from_its_record():
    generator = torch.Generator().manual_seed(23)
    batch, tokens, heads, rank, value_dim = 2, 40, 2, 8, 4
    q = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE) * 0.4
    payload = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE)
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    policy = RoutingPolicy(ops=True, tau_match=0.3, tau_merge=0.8, tau_pop=0.0)
    outputs = []
    for _ in range(2):
        outputs.append(routed_chunk_scan(
            q, payload, decay, router=route_chunk, policy=policy, rank=rank,
            rank_max=rank, chunk_tokens=8, collect=True, fused=False))
    first, second = outputs
    assert torch.equal(first.reads, second.reads)
    assert torch.equal(first.state, second.state)
    assert torch.equal(first.schedule.slot, second.schedule.slot)
    assert torch.equal(first.weights, second.weights)

    # The recorded schedule *and* read-weight channel reproduce the trajectory
    # without routing: that is the determinism claim (gate E).
    reads, state, live = chunked_scan(first.weights, decay, first.schedule,
                                     chunk_tokens=8, fused=False)
    assert torch.equal(reads, first.reads)
    assert torch.equal(state, first.state)
    assert torch.equal(live, first.live)
    assert first.stats.summary()["chunks"], "per-chunk telemetry missing"

    # The decision channel is detached: no gradient ever reaches a decision, so a
    # dtype-promoted but value-identical run reproduces it exactly.
    again = routed_chunk_scan(q.detach() * 1.0, payload.detach(), decay, router=route_chunk,
                              policy=policy, rank=rank, rank_max=rank, chunk_tokens=8,
                              collect=True, fused=False)
    assert torch.equal(again.schedule.slot, first.schedule.slot)
    assert torch.equal(again.weights, first.weights)


def test_append_only_policy_never_pops_and_uses_every_slot():
    generator = torch.Generator().manual_seed(29)
    batch, tokens, heads, rank, value_dim = 1, 32, 1, 6, 3
    q = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE)
    payload = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE)
    decay = torch.ones(batch, tokens, heads, device=DEVICE)
    policy = RoutingPolicy(ops=False)
    result = routed_chunk_scan(q, payload, decay, router=route_chunk, policy=policy,
                               rank=rank, rank_max=rank, chunk_tokens=8, collect=True,
                               fused=False)
    assert result.schedule.op.unique().tolist() == [APPEND]
    assert bool(result.live.all()), "append-only control must leave every slot live"
    expected = (torch.arange(tokens, device=DEVICE) % rank).view(1, tokens, 1)
    assert torch.equal(result.schedule.slot[..., 0], expected)


def test_growth_targets_snap_to_the_precompiled_set_and_are_monotone():
    assert snap_rank(1) == 64 and snap_rank(65) == 128 and snap_rank(2048) == 2048
    with pytest.raises(ValueError):
        snap_rank(4096)
    rank = 64
    seen = [rank]
    while rank < MAX_RANK:
        rank = growth_target(rank, 2.0, 64)
        seen.append(rank)
        assert rank in ORTHONORMAL_RANKS
    assert seen == [64, 128, 256, 512, 1024, 2048]
    assert growth_target(2048, 2.0, 64) == 2048
    assert growth_target(128, 1.5, 64) == 256  # max(192, 192) -> 256
    assert growth_target(128, 2.0, 16) == 256


def test_growth_fires_at_chunk_boundaries_and_stays_in_the_precompiled_set():
    generator = torch.Generator().manual_seed(31)
    batch, tokens, heads, rank, rank_max, value_dim = 1, 128, 1, 64, 256, 4
    q = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE)
    payload = torch.randn(batch, tokens, heads, value_dim, generator=generator).to(DEVICE)
    decay = torch.ones(batch, tokens, heads, device=DEVICE)
    # Round-robin appends over 64 slots with a 32-token window must read as saturated.
    policy = RoutingPolicy(ops=False, grow=True, grow_trigger=0.3, grow_window=2,
                           grow_window_tokens=32)
    result = routed_chunk_scan(q, payload, decay, router=route_chunk, policy=policy,
                               rank=rank, rank_max=rank_max, chunk_tokens=16,
                               collect=False, fused=False)
    state, rank_final, stats = result.state, result.rank, result.stats
    events = stats.growth_events
    assert events, "the growth trigger never fired under saturation"
    assert [event["rank_after"] for event in events] == sorted(
        {event["rank_after"] for event in events})
    for event in events:
        assert event["rank_after"] in ORTHONORMAL_RANKS
        assert event["rank_after"] > event["rank_before"]
    assert rank_final in ORTHONORMAL_RANKS and rank_final > rank
    # Growth is a zero-injection: rows above the grown rank were never written.
    assert float(state[:, :, rank_final:].abs().max()) == 0.0


def test_repeat_write_rate_counts_only_writes_inside_the_window():
    tokens, rank = 8, 2
    shape = (1, tokens, 1, 1)
    slot = torch.full(shape, 0, dtype=torch.int64)
    op = torch.full(shape, APPEND, dtype=torch.int64)
    schedule = MatrixSchedule(slot=slot, erase=torch.zeros(shape, dtype=torch.bool),
                              payload=torch.zeros(1, tokens, 1, 1, 2), op=op,
                              live_after=torch.ones(shape, dtype=torch.bool))
    last_write = torch.full((1, 1, rank), -1, dtype=torch.int64)
    first = repeat_write_rate(schedule, last_write, 0, 100)
    assert first == 0.0, "nothing was written before the first chunk"
    second = repeat_write_rate(schedule, last_write, tokens, 100)
    assert second == 1.0, "every write reuses the slot written in the same window"
    far = torch.full((1, 1, rank), -1, dtype=torch.int64)
    assert repeat_write_rate(schedule, far, 10_000, 100) == 0.0


# ------------------------------------------------------- read (single-sourced)

def test_read_matches_certified_reference():
    """Our read and the executed ``naive_reference`` read on identical state.

    The tape's keys are unit directions with the stored value as content, its
    strength channel is the row norm, and the query is unit-norm: then the certified
    ``keys . q`` score and our cosine score are the same number, so the two reads
    must agree to the fp32 matmul noise floor -- with no renormalisation of either.
    """
    from experiments.expressivity_tasks.x30_tape_cell import naive_reference as naive

    # key_dim == value_dim: the tape's key axis and the matrix store's row axis are
    # the same axis (that is what makes the tape a special case of the matrix).
    config = naive.TapeHeadConfig(key_dim=3, value_dim=3, n_heads=1, budget=5,
                                  decay_rate=1.0, free_threshold=0.0,
                                  pop_threshold=1.0, write_enable_threshold=0.5)
    generator = torch.Generator().manual_seed(5)
    slots, key_dim, value_dim = config.slots, config.key_dim, config.value_dim
    state = naive.init_state(1, config, "cpu")
    state.values = torch.randn(1, 1, slots, value_dim, generator=generator)
    state.keys = read_module.unit(state.values)
    state.strength = state.values.norm(dim=-1)
    state.live = torch.tensor([[[True, True, True, False, True]]])
    state.age = torch.ones(1, 1, slots, dtype=torch.int64)
    tokens = 2
    query = read_module.unit(torch.randn(1, tokens, 1, key_dim, generator=generator))
    reference, _, _ = naive.tape_scan(
        torch.zeros(1, tokens, 1, key_dim), query,
        torch.randn(1, tokens, 1, value_dim, generator=generator),
        torch.zeros(1, tokens, 1), torch.zeros(1, tokens, 1),
        torch.full((1, tokens, 1), -50.0), state, config)

    scale = 2.0 / key_dim ** 0.5       # the certified read's softmax scale
    rows = state.values[0, 0].unsqueeze(0).unsqueeze(0)
    live = state.live
    weights = read_module.read_weights(query[0, 0].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                       rows, live, scale)
    ours = read_module.read(weights, rows)
    deviation = float((ours[0, 0, 0] - reference[0, 0]).abs().max())
    assert deviation <= 1e-6, deviation
    # No strength renormalisation: the read magnitude carries the stored strengths.
    doubled = read_module.read(weights * 2.0, rows)
    assert torch.allclose(doubled, 2.0 * ours, atol=1e-6)


def test_a_pop_removes_a_row_from_the_read():
    """``s_j -> 0`` needs no special case: the row leaves the mixture entirely."""
    generator = torch.Generator().manual_seed(7)
    rows = torch.randn(1, 1, 4, 3, generator=generator)
    query = torch.randn(1, 1, 1, 3, generator=generator)
    live = torch.tensor([[[True, True, True, True]]])
    weights = read_module.read_weights(query, rows, live, 2.0)
    assert weights[0, 0, 0, 2] > 0

    popped = live.clone()
    popped[0, 0, 2] = False
    dead_rows = rows.clone()
    dead_rows[0, 0, 2] = 0.0
    after = read_module.read_weights(query, dead_rows, popped, 2.0)
    assert after[0, 0, 0, 2].item() == 0.0
    assert weights[0, 0, 0, 2].item() > 0.0
    # A POP is exactly equivalent to deleting the row from the store: the softmax
    # renormalises over the surviving live rows and nothing else moves.
    survivors = torch.tensor([0, 1, 3])
    deleted_rows = rows[:, :, survivors]
    deleted = read_module.read_weights(query, deleted_rows,
                                       torch.ones(1, 1, len(survivors), dtype=torch.bool),
                                       2.0)
    assert torch.allclose(read_module.read(after, dead_rows)[0, 0, 0],
                          read_module.read(deleted, deleted_rows)[0, 0, 0], atol=1e-6)
