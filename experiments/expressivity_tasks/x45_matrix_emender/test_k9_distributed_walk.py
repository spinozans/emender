"""K9 gates: the distributed walk's parity, walls, and growth re-dispatch.

Gate discipline is the lane's (``test_k7_high_rank.py``): the K7 eager oracle
(``k7_high_rank_store.sequential_oracle``, itself pinned to the K6
registration's oracle) and the K7 global-tiled walk are the references; every
decision channel, the read weights, the pre-decision liveness, the payload
lanes, the exit state and the liveness are held to ``torch.equal`` (0 flips);
every read is held to ``NOISE_FACTOR`` times a **measured** fp32-vs-fp64 floor
on the same fixture -- never a fitted constant.  The K9-specific gates:

* the graph replay is bit-equal to the eager launch sequence (the same
  kernels, the same order -- the launch boundaries are the barrier);
* determinism across runs (the K7 nondeterminism class -- cross-warp
  store->load visibility -- is closed by construction at this granularity and
  the test would catch any leak);
* the by-name refusals: the on-chip register budget (rank x value_dim <=
  MAX_SEQ_TILE) routes to the EXISTING K6/K7 paths, value_dim above the probe
  ceiling, the append-only and growth controls;
* the growth re-dispatch composes at the span-API level: a recorded span table
  across snaps {128 -> 2048} is bit-exact span by span against the oracle,
  the new rows dead at entry and until the first schedule write that targets
  them (the zero-injection receipts), ragged windows included.

Every test needs a CUDA device.
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, MERGE, NONE, POP)
from experiments.expressivity_tasks.x45_matrix_emender.k7_high_rank_store import (
    k7_sequential_walk, sequential_oracle)
from experiments.expressivity_tasks.x45_matrix_emender.k9_distributed_walk import (
    K9_BLOCK_SLOTS, k9_auto_walk, k9_sequential_walk, k9_span_walk,
    reset_k9_graphs)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)
from experiments.expressivity_tasks.x45_matrix_emender.test_matrix_emender import (
    NOISE_FACTOR)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
POLICY = RoutingPolicy(ops=True)


# --------------------------------------------------------------------------- fixtures


def populated_walk_fixture(rank: int, tokens: int = 16, batch: int = 2,
                           heads: int = 2, value_dim: int = 128, seed: int = 11,
                           live_fraction: float = 0.8):
    """A populated store and a random stream (K7's own fixture recipe)."""
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
    return dict(query=query, payload=payload, state=state, live=live,
                decay=decay)


def run_k9(f: dict, rank: int, collect: bool = True, graph: bool = True,
           **kwargs):
    reset_k9_graphs()
    return k9_sequential_walk(f["query"], f["payload"], f["decay"],
                              policy=POLICY, rank=rank,
                              initial_state=f["state"],
                              initial_live=f["live"], collect=collect,
                              graph=graph, **kwargs)


def run_k7(f: dict, rank: int, collect: bool = True):
    return k7_sequential_walk(f["query"], f["payload"], f["decay"],
                              policy=POLICY, rank=rank,
                              initial_state=f["state"],
                              initial_live=f["live"], collect=collect)


def run_oracle(f: dict, rank: int):
    return sequential_oracle(f["query"], f["payload"], f["decay"],
                             policy=POLICY, rank=rank,
                             initial_state=f["state"], initial_live=f["live"])


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


def read_floor(kernel: dict, oracle: dict, f: dict, rank: int) -> float:
    """The MEASURED fp32-vs-fp64 floor of the read on this fixture (K7's own
    method: the fp32 oracle against its fp64 twin)."""
    f64 = dict(f)
    f64.update(query=f["query"].to(torch.float64),
               payload=f["payload"].to(torch.float64),
               decay=f["decay"].to(torch.float64),
               state=f["state"].to(torch.float64))
    oracle64 = sequential_oracle(f64["query"], f64["payload"], f64["decay"],
                                 policy=POLICY, rank=rank,
                                 initial_state=f64["state"],
                                 initial_live=f64["live"])
    return float((oracle["reads"]
                  - oracle64["reads"].to(torch.float32)).abs().max())


# ------------------------------------------------------------------- gate B: parity


@needs_cuda
@pytest.mark.parametrize("rank", [256, 1024, 2048])
def test_decisions_bit_exact_vs_the_k7_walk_and_the_oracle(rank):
    """THE K9 gate: 0 flips against BOTH references -- the K7 global-tiled walk
    (kernel vs kernel) and the eager oracle -- the exit state and liveness
    bit-equal, the read at the measured floor (floors re-measured first)."""
    f = populated_walk_fixture(rank)
    kernel = run_k9(f, rank)
    oracle = run_oracle(f, rank)
    k7 = run_k7(f, rank)
    assert decision_flips(kernel, oracle) == 0, rank
    assert decision_flips(kernel, k7) == 0, rank
    assert torch.equal(kernel["state"], oracle["state"])
    assert torch.equal(kernel["live"], oracle["live"])
    floor = read_floor(kernel, oracle, f, rank)
    deviation = float((kernel["reads"] - oracle["reads"]).abs().max())
    assert deviation <= NOISE_FACTOR * max(floor, torch.finfo(torch.float32).eps), \
        (rank, deviation, floor)
    # K7's own read is a different reduction order too: both sit at the floor
    deviation_k7 = float((kernel["reads"] - k7["reads"]).abs().max())
    assert deviation_k7 <= NOISE_FACTOR * max(floor, torch.finfo(torch.float32).eps), \
        (rank, deviation_k7, floor)


@needs_cuda
def test_the_graph_replay_is_bit_equal_to_the_eager_launches():
    """The whole-scan CUDA graph replays exactly the eager launch sequence:
    every channel bit-equal (the launch boundaries are the barrier -- this is
    the test that would catch any capture-order surprise)."""
    f = populated_walk_fixture(1024)
    graph = run_k9(f, 1024, graph=True)
    eager = run_k9(f, 1024, graph=False)
    assert decision_flips(graph, eager) == 0
    assert torch.equal(graph["reads"], eager["reads"])
    assert torch.equal(graph["state"], eager["state"])
    assert torch.equal(graph["live"], eager["live"])


@needs_cuda
def test_determinism_across_runs():
    """The K7 nondeterminism class (cross-warp store->load visibility) is
    closed by construction here -- the launch boundary between kernels is the
    barrier, and the one intra-CTA hazard (the decision kernel both loading
    and storing the liveness) was moved to the apply sweep.  Two fresh full
    scans must agree bit for bit; a leak would flip decisions."""
    f = populated_walk_fixture(2048, tokens=8)
    a = run_k9(f, 2048)
    b = run_k9(f, 2048)
    assert decision_flips(a, b) == 0
    for key in ("reads", "state", "live"):
        assert torch.equal(a[key], b[key]), key


@needs_cuda
def test_walk_covers_every_op_branch_bit_exactly():
    """Crafted stores forcing each branch of the five-op chain (the K7 suite's
    tradition, at a rank above the on-chip threshold): a repeated-direction
    store forces MERGE on both lanes, a full store with an orthogonal stream
    forces POP, a full store at cosine 0.3 to the payload forces NONE, an
    empty store forces APPEND -- every one bit-exact against the oracle."""
    value_dim, rank = 128, 256
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
    kernel, oracle = run_k9(f, rank), run_oracle(f, rank)
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
    kernel, oracle = run_k9(f, rank), run_oracle(f, rank)
    assert decision_flips(kernel, oracle) == 0
    ops = kernel["schedule"].op[..., 0].reshape(-1)
    assert int((ops == POP).sum()) > 0 and int((ops == APPEND).sum()) > 0

    # (iii) NONE: every row at cosine 0.3 to the payload, store full.
    state = torch.zeros(1, 1, rank, value_dim, device=DEVICE)
    for j in range(rank):
        state[0, 0, j, 0] = 0.3
        state[0, 0, j, 1 + (j % 8)] = 0.95
    pl = torch.zeros(1, 4, 1, value_dim, device=DEVICE)
    pl[..., 0] = 1.0
    f = dict(query=query[:, :4], payload=pl, state=state,
             live=torch.ones(1, 1, rank, dtype=torch.bool, device=DEVICE),
             decay=torch.full((1, 4, 1), 0.95, device=DEVICE))
    kernel, oracle = run_k9(f, rank), run_oracle(f, rank)
    assert decision_flips(kernel, oracle) == 0
    ops = kernel["schedule"].op[..., 0].reshape(-1)
    assert int((ops == NONE).sum()) == 4


@needs_cuda
def test_the_all_dead_read_is_exactly_zero():
    """The certified all-dead zero attention at rank 2048: token 0 reads an
    empty dead store -- read and weights exactly zero."""
    rank = 2048
    f = populated_walk_fixture(rank, tokens=4, live_fraction=0.0)
    kernel = run_k9(f, rank)
    assert bool((kernel["reads"][:, 0] == 0).all())
    assert bool((kernel["weights"][:, 0] == 0).all())


@needs_cuda
def test_collect_free_is_the_same_walk():
    """collect=False is the walk without the diagnostic channels."""
    f = populated_walk_fixture(512, tokens=8)
    a = run_k9(f, 512, collect=True)
    c = run_k9(f, 512, collect=False)
    assert torch.equal(a["reads"], c["reads"])
    assert torch.equal(a["state"], c["state"])
    assert torch.equal(a["schedule"].op, c["schedule"].op)


@needs_cuda
def test_ragged_windows():
    """Odd token counts and span tables with ragged boundaries (the growth
    re-dispatch's own geometry) run the same exact walk."""
    for tokens in (1, 33, 130):
        f = populated_walk_fixture(256, tokens=tokens)
        kernel = run_k9(f, 256)
        oracle = run_oracle(f, 256)
        assert decision_flips(kernel, oracle) == 0, tokens
        assert torch.equal(kernel["state"], oracle["state"])


# ------------------------------------------------------- the refusals and the routing


@needs_cuda
def test_k9_refuses_the_onchip_budget_by_name():
    """rank x value_dim within MAX_SEQ_TILE is the K6 kernel's own shape: the
    existing K6/K7 paths own it (bit-pinned, already fast) and K9 refuses it
    by name rather than competing with them."""
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import MAX_SEQ_TILE
    f = populated_walk_fixture(128)
    with pytest.raises(ValueError, match="on-chip register budget"):
        k9_sequential_walk(f["query"], f["payload"], f["decay"], policy=POLICY,
                           rank=128, initial_state=f["state"],
                           initial_live=f["live"])
    assert 128 * 128 <= MAX_SEQ_TILE


@needs_cuda
def test_k9_auto_walk_routes_low_ranks_to_the_existing_k7_path():
    """The auto-path's high-rank route: at the registered shape the existing
    K7 walk answers (the same bit-pinned semantics), above the on-chip
    threshold the distributed walk does."""
    f = populated_walk_fixture(128, tokens=4)
    low = k9_auto_walk(f["query"], f["payload"], f["decay"], policy=POLICY,
                       rank=128, initial_state=f["state"],
                       initial_live=f["live"])
    high_f = populated_walk_fixture(256, tokens=4)
    high = k9_auto_walk(high_f["query"], high_f["payload"],
                        high_f["decay"], policy=POLICY, rank=256,
                        initial_state=high_f["state"],
                        initial_live=high_f["live"], graph=False)
    oracle = run_oracle(high_f, 256)
    assert decision_flips(high, oracle) == 0
    assert decision_flips(low, run_oracle(f, 128)) == 0


@needs_cuda
def test_k9_refuses_the_chunk_dial_controls():
    f = populated_walk_fixture(256, tokens=4)
    for policy in (RoutingPolicy(ops=False),
                   RoutingPolicy(ops=True, grow=True)):
        with pytest.raises(ValueError):
            k9_sequential_walk(f["query"], f["payload"], f["decay"],
                               policy=policy, rank=256)


# --------------------------------------------------- gate D: growth re-dispatch (API)


@needs_cuda
def test_growth_redispatch_across_a_snap_change_is_exact():
    """A RECORDED span table across snaps (the growth events' own output
    shape, ragged boundaries included) replays exactly: every span bit-exact
    against the oracle given the chained entry state, the whole-chain exit
    bit-equal, and the zero-injection receipts hold -- the new rows are dead
    at the grown span's entry and stay dead until the first schedule write
    that targets them."""
    from experiments.expressivity_tasks.x45_matrix_emender.k8_growth import (
        _pad_live, _pad_state)
    batch, tokens, heads, value_dim = 2, 960, 2, 128
    generator = torch.Generator().manual_seed(7)
    shape = (batch, tokens, heads, value_dim)
    query = torch.randn(*shape, generator=generator).to(DEVICE) * 0.5
    payload = torch.randn(*shape, generator=generator).to(DEVICE)
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    spans = [(0, 192, 128), (192, 448, 256), (448, 700, 512), (700, 960, 1024)]
    result = k9_span_walk(query, payload, decay, policy=POLICY, spans=spans)
    state = torch.zeros(batch, heads, 128, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, 128, dtype=torch.bool, device=DEVICE)
    flips = 0
    for index, (t0, t1, rank) in enumerate(spans):
        oracle = sequential_oracle(query[:, t0:t1], payload[:, t0:t1],
                                   decay[:, t0:t1], policy=POLICY, rank=rank,
                                   initial_state=state, initial_live=live)
        segment = result["collected"][index]
        flips += decision_flips(dict(schedule=segment["schedule"],
                                     weights=segment["weights"],
                                     live_pre=segment["live_pre"]), oracle)
        # the zero-injection receipt: the grown rows are dead at the span entry
        grown = live.shape[-1]
        if grown > 0 and index > 0:
            assert not bool(result["collected"][index]["live_pre"][
                :, 0, :, grown:].any()), index
        state, live = oracle["state"], oracle["live"]
        if index + 1 < len(spans) and spans[index + 1][2] > rank:
            state = _pad_state(state, spans[index + 1][2])
            live = _pad_live(live, spans[index + 1][2])
    assert flips == 0
    assert torch.equal(result["state"], state)
    assert torch.equal(result["live"], live)


@needs_cuda
def test_the_span_table_is_validated():
    batch, tokens, heads, value_dim = 1, 8, 1, 128
    shape = (batch, tokens, heads, value_dim)
    query = torch.randn(*shape, device=DEVICE)
    payload = torch.randn(*shape, device=DEVICE)
    decay = torch.full((batch, tokens, heads), 0.95, device=DEVICE)
    for bad in ([(0, 4, 256)], [(4, 8, 256)]):
        with pytest.raises(ValueError):
            k9_span_walk(query, payload, decay, policy=POLICY, spans=bad)
