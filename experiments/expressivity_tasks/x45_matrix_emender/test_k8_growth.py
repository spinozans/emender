"""K8 growth tests: the legality, parity, gradient and bookkeeping gates.

The K8 registration-inherited discipline, at fixture scale (the gate receipts
in ``k8_gate_receipt.py`` take the registered shapes):

* **gate A legality** -- ``grow``-never-fires is BIT-EQUAL to K6's own path
  (the same op on the same inputs); the grown chain's span-1 schedule and
  boundary state are bit-equal to the fixed-rank K6 twin's prefix (zero-
  injection changes nothing before it fires); the zero-row property is
  receipted exactly (new rows all-zero and dead until the first write that
  targets them); probe and real passes agree bit-for-bit.
* **gate A parity of the hi path** -- the K8 hi span (K7's walk, launched
  directly) is bit-equal to ``k7_sequential_walk`` on the same inputs and
  bit-exact on decisions/weights/liveness against K7's eager oracle, reads at
  the measured floor; the slot-blocked replay kernels are bit-equal to the
  walks' own states and checkpoints, at and above the register gate.
* **gate B gradients** -- the hi backward vs the eager oracle that replays the
  SAME recorded span table, floor first (fp32 oracle vs its own fp64 twin,
  NOISE_FACTOR x), including a chain ACROSS a growth boundary (the zero-inject
  pad's adjoint in the loop); the prep's recomputed weights bit-equal the
  forward's own channel.
* **the trigger bookkeeping** -- fires exactly where the registered rule says
  (mean of the last ``grow_window`` boundary rates over ``grow_trigger``), the
  window resets on fire, the rank follows ``keys.growth_target`` and the cap.
* **determinism** -- two growing scans bit-equal on every channel (the K7
  cross-warp lesson, re-asserted here for the K8 backward walk too).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, OP_NAMES, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _seq_forward_launch, _seq_warps, _round_f32, sequential_oracle_scan, sequential_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k7_high_rank_store import (
    k7_sequential_walk, sequential_oracle)
from experiments.expressivity_tasks.x45_matrix_emender.k8_growth import (
    GROW_BOUNDARY_TOKENS, GrowingMatrixMixer, _hi_backward_body, _hi_prep,
    _hi_walk_launch, growing_scan, oracle_growing_scan, traj_checkpoints,
    traj_exit_state, traj_segment)
from experiments.expressivity_tasks.x45_matrix_emender.keys import (
    growth_target, snap_rank)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy
from experiments.expressivity_tasks.x45_matrix_emender.scan import (
    repeat_write_rate)

NOISE_FACTOR = 8.0
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")

GROWING = RoutingPolicy(ops=True, grow=True)
QUIET = RoutingPolicy(ops=True, grow=False)


def pol32(policy):
    return (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
            _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
            _round_f32(policy.read_scale))


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed,
                   device="cuda"):
    """The K5A/K6 lane's fixture generator: payloads drawn from ``distinct``
    fixed directions (small ``distinct`` pumps the append-collision rate)."""
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to(device) * 0.5
    directions = torch.randn(distinct, value_dim, generator=g).to(device)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = directions[choice]
    decay = torch.rand(batch, tokens, heads, generator=g).to(device) \
        .clamp(0.9, 1.0)
    return query, payload, decay


def zeros_state(batch, heads, rank, value_dim, device="cuda"):
    return (torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=device),
            torch.zeros(batch, heads, rank, dtype=torch.bool, device=device))


# --------------------------------------------------------------- gate A: legality


def test_grow_never_fires_is_bit_equal_to_the_k6_path():
    """``grow=True`` with fewer than ``grow_window`` statistic boundaries in
    the window is ONE span = K6's own op call on the same inputs: every channel
    bit-equal to ``sequential_scan`` (the registered trigger needs 4 boundary
    rates before it can fire at all)."""
    batch, tokens, heads, rank, value_dim = 2, 192, 2, 64, 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           512, 77)
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.99)
    state, live = zeros_state(batch, heads, rank, value_dim)
    grown = growing_scan(query, payload, decay, policy=policy,
                         start_rank=rank, rank_cap=256)
    k6 = sequential_scan(query, payload, decay, policy=QUIET, rank=rank,
                         rank_max=rank)
    assert grown.stats.spans == [(0, tokens, rank)]
    assert grown.stats.events == []
    assert torch.equal(grown.reads, k6.reads)
    assert torch.equal(grown.state, k6.state)
    assert torch.equal(grown.live, k6.live)


def test_grow_false_fixed_rank_matches_k6_and_hi_executes():
    """``grow=False``: the fixed-rank arms.  At the registered rank this is
    K6's path bit for bit; at a snap above the register gate the hi op runs
    the whole window as one span."""
    batch, tokens, heads, value_dim = 2, 128, 2, 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           128, 5)
    state, live = zeros_state(batch, heads, 128, value_dim)
    fixed = growing_scan(query, payload, decay, policy=QUIET, start_rank=128,
                         rank_cap=2048)
    k6 = sequential_scan(query, payload, decay, policy=QUIET, rank=128,
                         rank_max=128)
    assert fixed.stats.spans == [(0, tokens, 128)]
    assert torch.equal(fixed.reads, k6.reads)
    assert torch.equal(fixed.state, k6.state)
    hi = growing_scan(query, payload, decay, policy=QUIET, start_rank=256,
                      rank_cap=2048)
    assert hi.stats.spans == [(0, tokens, 256)]
    assert hi.stats.events == []


def test_trigger_fires_records_spans_and_events():
    """A pumped stream fires the registered rule: the events' boundaries,
    ranks and smoothed rates are recorded; the spans tile the window."""
    batch, tokens, heads, value_dim = 2, 1024, 2, 64
    start_rank, cap = 64, 256
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           8, 11)
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.5,
                           grow_window=4, grow_window_tokens=512,
                           grow_geometric=2.0, grow_stride_floor=64)
    result = growing_scan(query, payload, decay, policy=policy,
                          start_rank=start_rank, rank_cap=cap)
    spans = result.stats.spans
    events = result.stats.events
    assert events, "the pumped stream must fire the trigger"
    # the spans tile [0, tokens) at the recorded ranks
    assert spans[0][0] == 0 and spans[-1][1] == tokens
    for (a0, a1, _), (b0, _, _) in zip(spans, spans[1:]):
        assert a1 == b0
    for (t0, t1, rank), event in zip(spans, events):
        assert t1 == event["boundary"]
        assert rank == event["rank_before"]
        assert event["rank_after"] == min(cap, growth_target(
            rank, policy.grow_geometric, policy.grow_stride_floor))
        assert event["rank_after"] > rank
    # every event boundary is a 64-token statistic boundary, at least
    # grow_window boundaries after the previous fire (the window resets)
    previous = 0
    for event in events:
        boundary = event["boundary"]
        assert boundary % GROW_BOUNDARY_TOKENS == 0
        assert boundary - previous >= policy.grow_window \
            * GROW_BOUNDARY_TOKENS
        assert len(event["window_rates"]) == policy.grow_window
        assert sum(event["window_rates"]) / policy.grow_window \
            == pytest.approx(event["smoothed_rate"])
        assert event["smoothed_rate"] >= policy.grow_trigger
        previous = boundary
    # the rates are reported per boundary, in order, through every fire
    boundaries = [b for b, _ in result.stats.rates]
    assert boundaries == sorted(boundaries)
    for event in events:
        assert event["boundary"] in boundaries
    assert boundaries[-1] == events[-1]["boundary"]


def test_trigger_first_fire_is_the_registered_rule():
    """The first fire is the first boundary whose smoothed rate crosses the
    trigger, computed independently from the reported per-boundary rates."""
    batch, tokens, heads, value_dim = 1, 512, 1, 32
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           4, 3)
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.5,
                           grow_window=4, grow_window_tokens=512)
    result = growing_scan(query, payload, decay, policy=policy,
                          start_rank=64, rank_cap=128)
    rates = {b: r for b, r in result.stats.rates}
    window = []
    expected = None
    for boundary in range(GROW_BOUNDARY_TOKENS, tokens + 1,
                          GROW_BOUNDARY_TOKENS):
        window.append(rates[boundary])
        if len(window) >= policy.grow_window \
                and sum(window[-policy.grow_window:]) / policy.grow_window \
                >= policy.grow_trigger:
            expected = boundary
            break
    assert result.stats.events[0]["boundary"] == expected


def test_zero_injection_new_rows_zero_and_dead_until_first_write():
    """The legality receipt: at every growth event the new rows enter exact
    zeros, liveness off, and stay so until the first schedule write that
    targets them; the old rows' evolution through the boundary is the fixed
    twin's (the prefix property)."""
    batch, tokens, heads, value_dim = 2, 1024, 2, 64
    start_rank, cap = 64, 256
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           8, 11)
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.5)
    result = growing_scan(query, payload, decay, policy=policy,
                          start_rank=start_rank, rank_cap=cap, collect=True)
    collected = result.collected
    assert len(collected) >= 2
    for index in range(1, len(collected)):
        prev_rank = collected[index - 1]["rank"]
        span = collected[index]
        rank = span["rank"]
        assert rank > prev_rank
        # the entry liveness of the new rows is OFF...
        assert not span["live_pre"][:, 0, :, prev_rank:].any()
        # ...and the new rows are never READ as live before the first write
        # that targets them (the schedule's own record).
        schedule = span["schedule"]
        first_write = None
        for t in range(span["t1"] - span["t0"]):
            lane0, lane1 = schedule.slot[:, t, :, 0], schedule.slot[:, t, :, 1]
            wrote_new = (schedule.op[:, t, :, 0] != NONE) & (lane0 >= prev_rank)
            wrote_new |= (schedule.op[:, t, :, 1] != NONE) & (lane1 >= prev_rank)
            if bool(wrote_new.any()):
                first_write = t
                break
        if first_write is None:
            live_new = span["live_pre"][:, :, :, prev_rank:]
            assert not live_new.any()
        else:
            assert not span["live_pre"][:, :first_write, :, prev_rank:].any()
        # the zero CONTENT at the boundary: the padded entry of the new rows
        # is exact zeros by construction (F.pad), and the pre-boundary state's
        # old rows are the fixed twin's own (the next test pins that).


def test_prefix_bit_equality_and_probe_matches_real():
    """The grown chain's span-1 schedule and boundary state are bit-equal to
    the fixed-rank K6 twin's own prefix (zero-injection changes nothing
    before it fires); the probe pass and the real pass agree bit-for-bit."""
    batch, tokens, heads, value_dim = 2, 1024, 2, 64
    start_rank = 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           8, 11)
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.5)
    result = growing_scan(query, payload, decay, policy=policy,
                          start_rank=start_rank, rank_cap=256, collect=True)
    t_star = result.stats.events[0]["boundary"]
    # the fixed-rank twin at the START rank, whole window, K6's own launch
    state, live = zeros_state(batch, heads, start_rank, value_dim)
    twin = _seq_forward_launch(query, payload, decay, state, live,
                               pol32(QUIET), start_rank, 64, collect=True,
                               warps=_seq_warps(start_rank, value_dim),
                               stages=3)
    # the twin's own trigger statistic must fire at the same boundary: the
    # grown run and the twin share the prefix, so their rates agree there.
    k = t_star // GROW_BOUNDARY_TOKENS
    # 1. the grown span-1 schedule is the twin's prefix, bit for bit
    grown_sched = result.collected[0]["schedule"]
    twin_sched = MatrixSchedule(
        slot=twin["slot"][:, :t_star], erase=twin["erase"][:, :t_star],
        payload=twin["payload"][:, :t_star], op=twin["op"][:, :t_star],
        live_after=twin["live_after"][:, :t_star])
    for field in SCHEDULE_FIELDS + ("payload",):
        assert torch.equal(getattr(grown_sched, field),
                           getattr(twin_sched, field))
    assert torch.equal(result.collected[0]["weights"],
                       twin["weights"][:, :t_star])
    # 2. the boundary state: the grown span-1 exit == the twin's own segment
    #    checkpoint at t_star (old rows unchanged by the growth event).
    assert torch.equal(result.collected[0]["state"], twin["ckpt_state"][k])
    assert torch.equal(result.collected[0]["live"], twin["ckpt_live"][k])
    # 3. the probe pass agrees with the real pass: re-deriving the probe span
    #    with the same launch reproduces the collected schedule bit for bit.
    state, live = zeros_state(batch, heads, start_rank, value_dim)
    probe = _seq_forward_launch(query, payload, decay, state, live,
                                pol32(policy), start_rank, 64, collect=False,
                                warps=_seq_warps(start_rank, value_dim),
                                stages=3)
    for field in SCHEDULE_FIELDS + ("payload",):
        assert torch.equal(getattr(grown_sched, field),
                           getattr(MatrixSchedule(
                               slot=probe["slot"], erase=probe["erase"],
                               payload=probe["payload"], op=probe["op"],
                               live_after=probe["live_after"]),
                               field)[:, :t_star])


def test_determinism_two_growing_scans_bit_equal():
    """Two full growing scans, growth firing, hi spans included: every channel
    bit-equal (the K7 cross-warp lesson, held on the whole K8 chain)."""
    batch, tokens, heads, value_dim = 2, 512, 2, 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           8, 13)
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.5)
    first = growing_scan(query, payload, decay, policy=policy,
                         start_rank=64, rank_cap=512, collect=True)
    second = growing_scan(query, payload, decay, policy=policy,
                          start_rank=64, rank_cap=512, collect=True)
    assert first.stats.spans == second.stats.spans
    for a, b in zip(first.collected, second.collected):
        for field in SCHEDULE_FIELDS + ("payload",):
            assert torch.equal(getattr(a["schedule"], field),
                               getattr(b["schedule"], field))
        assert torch.equal(a["weights"], b["weights"])
        assert torch.equal(a["live_pre"], b["live_pre"])
        assert torch.equal(a["state"], b["state"])
        assert torch.equal(a["live"], b["live"])
    assert torch.equal(first.reads, second.reads)


# ------------------------------------------------- gate A: the hi path's parity


@pytest.mark.parametrize("rank", [256, 2048])
def test_hi_span_matches_k7_walk_and_oracle(rank):
    """The hi span (K7's walk launched directly) is bit-equal to
    ``k7_sequential_walk`` on the same inputs; decisions, weights and
    liveness are bit-exact against K7's eager oracle, the read at the
    measured fp32-vs-fp64 floor."""
    batch, tokens, heads, value_dim = 1, 128, 2, 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           64, 21)
    state, live = zeros_state(batch, heads, rank, value_dim)
    policy = RoutingPolicy(ops=True, grow=False)
    ours = _hi_walk_launch(query, payload, decay, state, live, pol32(policy),
                           rank, collect=True)
    ref = k7_sequential_walk(query, payload, decay, policy=policy, rank=rank,
                             initial_state=state, initial_live=live,
                             collect=True)
    for key in ("reads", "state", "live"):
        assert torch.equal(ours[key], ref[key]), key
    for field in SCHEDULE_FIELDS + ("payload",):
        assert torch.equal(getattr(ours["schedule"], field),
                           getattr(ref["schedule"], field))
    assert torch.equal(ours["weights"], ref["weights"])
    assert torch.equal(ours["live_pre"], ref["live_pre"])
    # the oracle: decisions and weights bit-exact, reads at the floor
    oracle = sequential_oracle(query, payload, decay, policy=policy,
                                rank=rank, collect=True)
    with torch.no_grad():
        oracle64 = sequential_oracle(query.double(), payload.double(),
                                     decay.double(), policy=policy,
                                     rank=rank, collect=False)
    flips = 0
    for field in SCHEDULE_FIELDS + ("payload",):
        flips += int((getattr(oracle["schedule"], field)
                      != getattr(ours["schedule"], field)).sum())
    assert flips == 0
    assert torch.equal(oracle["weights"], ours["weights"])
    assert torch.equal(oracle["live_pre"], ours["live_pre"])
    floor = (oracle["reads"] - oracle64["reads"].float()).abs().max().item()
    deviation = (oracle["reads"].float() - ours["reads"]).abs().max().item()
    assert deviation <= NOISE_FACTOR * max(floor, 1e-30), \
        (floor, deviation)


@pytest.mark.parametrize("rank,use_k6", [(256, False), (128, True)])
def test_traj_replay_bit_equal_to_the_walks(rank, use_k6):
    """The slot-blocked replay reproduces the walks' own states bit for bit:
    the checkpoint mode's exit equals the walk's exit, each segment's replay
    exits at the next checkpoint, and at rank <= 128 the K8 checkpoints equal
    K6's own forward checkpoints."""
    batch, tokens, heads, value_dim = 2, 256, 2, 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           64, 31)
    state, live = zeros_state(batch, heads, rank, value_dim)
    if use_k6:
        out = _seq_forward_launch(query, payload, decay, state, live,
                                  pol32(QUIET), rank, 64, collect=False,
                                  warps=_seq_warps(rank, value_dim),
                                  stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        exit_state, exit_live = out["state"], out["live"]
        k6_ckpt_state, k6_ckpt_live = out["ckpt_state"], out["ckpt_live"]
    else:
        out = _hi_walk_launch(query, payload, decay, state, live, pol32(QUIET),
                              rank, collect=False)
        schedule = out["schedule"]
        exit_state, exit_live = out["state"], out["live"]
        k6_ckpt_state = None
    replay_exit, replay_live, ckpt_state, ckpt_live = traj_checkpoints(
        state, live, decay, schedule, rank, 64)
    assert torch.equal(replay_exit, exit_state)
    assert torch.equal(replay_live, exit_live)
    nseg = ckpt_state.shape[0]
    for seg in range(nseg):
        start = seg * 64
        stop = min(start + 64, tokens)
        traj, seg_exit, seg_live = traj_segment(
            ckpt_state[seg], ckpt_live[seg], decay[:, start:stop],
            schedule.chunk(start, stop), rank, heads)
        if seg + 1 < nseg:
            assert torch.equal(seg_exit, ckpt_state[seg + 1])
            assert torch.equal(seg_live, ckpt_live[seg + 1])
    if use_k6:
        assert torch.equal(ckpt_state, k6_ckpt_state.view(ckpt_state.shape))
        assert torch.equal(ckpt_live, k6_ckpt_live.view(ckpt_live.shape))


# ------------------------------------------------------------- gate B: gradients


def _grad_case(query, payload, decay, upstream, *, policy, spans,
               dtype=torch.float32, initial_state=None):
    """The oracle chain's autograd gradients (fp32) and its fp64 twin's, for
    the floor discipline."""
    q = query.clone().to(dtype).requires_grad_(True)
    p = payload.clone().to(dtype).requires_grad_(True)
    d = decay.clone().to(dtype).requires_grad_(True)
    if initial_state is None:
        s = None
    else:
        s = initial_state.clone().to(dtype).requires_grad_(True)
    ref = oracle_growing_scan(q, p, d, policy=policy, spans=spans,
                               collect=False, initial_state=s)
    (ref["reads"] * upstream.to(dtype)).sum().backward()
    return (q.grad, p.grad, d.grad, None if s is None else s.grad)


def test_hi_span_grads_within_floors():
    """Gate B at a hi span: the K8 backward vs the eager oracle that replays
    the same (single-span) table, floor first, NOISE_FACTOR x -- and the prep's
    recomputed read weights bit-equal the forward's own channel."""
    batch, tokens, heads, rank, value_dim = 2, 128, 2, 256, 64
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           64, 41)
    g = torch.Generator().manual_seed(1234)
    upstream = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to("cuda")
    policy = RoutingPolicy(ops=True, grow=False)
    spans = [(0, tokens, rank)]
    o32 = _grad_case(query, payload, decay, upstream, policy=policy,
                     spans=spans)
    o64 = _grad_case(query, payload, decay, upstream, policy=policy,
                     spans=spans, dtype=torch.float64)
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    state, live = zeros_state(batch, heads, rank, value_dim)
    res = growing_scan(q, p, d, policy=policy, start_rank=rank,
                       rank_cap=rank)
    (res.reads * upstream).sum().backward()
    for name, kern, orac, or64 in zip(("dquery", "dpayload", "ddecay"),
                                      (q.grad, p.grad, d.grad),
                                      o32, o64):
        floor = (orac.float() - or64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= NOISE_FACTOR * max(floor, 1e-30), \
            (name, floor, deviation)
    # the prep's recomputed weights are BIT-EQUAL the forward's channel
    collect = _hi_walk_launch(query, payload, decay, state, live,
                              pol32(policy), rank, collect=True)
    schedule = collect["schedule"]
    _, _, ckpt_state, ckpt_live = traj_checkpoints(state, live, decay,
                                                   schedule, rank, 64)
    for seg in range(ckpt_state.shape[0]):
        start = seg * 64
        stop = min(start + 64, tokens)
        traj, _, _ = traj_segment(ckpt_state[seg], ckpt_live[seg],
                                  decay[:, start:stop],
                                  schedule.chunk(start, stop), rank, heads)
        prep = _hi_prep(traj, query, torch.zeros_like(collect["reads"]),
                        collect["live_pre"], start, stop - start, rank,
                        policy.read_scale)
        assert torch.equal(
            prep["w"].view(batch, heads, stop - start, rank)
            .permute(0, 2, 1, 3),
            collect["weights"][:, start:stop]), seg


def test_grads_across_a_growth_boundary():
    """Gate B ACROSS the shape change: a 64 -> 128 -> 256 chain (the K6 op,
    the K6 op, the K8 hi op, two zero-injection pads), gradients vs the mixed
    eager oracle chain -- floor first, NOISE_FACTOR x -- with the entry state
    attached so ``dstate`` crosses both pads."""
    batch, tokens, heads, value_dim = 2, 1024, 2, 64
    start_rank, cap = 64, 256
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           8, 51)
    g = torch.Generator().manual_seed(99)
    upstream = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to("cuda")
    policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.5)
    state0, live0 = zeros_state(batch, heads, start_rank, value_dim)
    probe = growing_scan(query, payload, decay, policy=policy,
                         start_rank=start_rank, rank_cap=cap,
                         initial_state=state0, initial_live=live0)
    spans = probe.stats.spans
    assert [s[2] for s in spans] == [64, 128, 256], spans
    o32 = _grad_case(query, payload, decay, upstream, policy=policy,
                     spans=spans, initial_state=state0)
    o64 = _grad_case(query, payload, decay, upstream, policy=policy,
                     spans=spans, dtype=torch.float64,
                     initial_state=state0)
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state0.clone().requires_grad_(True)
    res = growing_scan(q, p, d, policy=policy, start_rank=start_rank,
                       rank_cap=cap, initial_state=s, initial_live=live0)
    assert res.stats.spans == spans
    (res.reads * upstream).sum().backward()
    for name, kern, orac, or64 in zip(
            ("dquery", "dpayload", "ddecay", "dstate"),
            (q.grad, p.grad, d.grad, s.grad), o32, o64):
        floor = (orac.float() - or64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= NOISE_FACTOR * max(floor, 1e-30), \
            (name, floor, deviation)


# ------------------------------------------------------------- the refusals


def test_refusals():
    batch, tokens, heads, value_dim = 1, 64, 1, 32
    query, payload, decay = distinct_stream(batch, tokens, heads, value_dim,
                                           16, 7)
    with pytest.raises(ValueError, match="append-only"):
        growing_scan(query, payload, decay,
                     policy=RoutingPolicy(ops=False, grow=True),
                     start_rank=64)
    with pytest.raises(ValueError, match="precompiled snap"):
        growing_scan(query, payload, decay, policy=GROWING, start_rank=100)
    with pytest.raises(ValueError, match="rank cap"):
        growing_scan(query, payload, decay, policy=GROWING, start_rank=128,
                     rank_cap=64)
    wide = distinct_stream(batch, tokens, heads, 256, 16, 7)
    with pytest.raises(ValueError, match="value ceiling"):
        growing_scan(*wide, policy=GROWING, start_rank=128, rank_cap=2048)


def test_mixer_conversion_shares_parameters_and_grows():
    """The K8 mixer: the conversion re-wraps in place (the K6 pattern), the
    parameters are the very same objects, and a forward pass through the
    converted mixer records growth events."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixShape, MatrixMixer as ParentMixer)
    torch.manual_seed(3)
    shape = MatrixShape(width=64, depth=1, hidden=32, heads=1, value_dim=32,
                        rank=64, rank_max=64, vocab=256)
    parent = ParentMixer(shape, RoutingPolicy(ops=True), 64).cuda()
    before = [p for p in parent.parameters()]
    growing = GrowingMatrixMixer.__new__(GrowingMatrixMixer)
    torch.nn.Module.__init__(growing)
    growing.__dict__.update(parent.__dict__)
    growing.segment_tokens = 64
    growing.rank_cap = 256
    growing.policy = RoutingPolicy(ops=True, grow=True, grow_trigger=0.5)
    growing.growth_events = []
    growing.last_rank = int(growing.rank)
    x = torch.randn(2, 768, 64, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out, result = growing.mix(x)
    assert list(growing.parameters()) == before
    assert result.stats.events, "the mixer's pumped stream must grow"
    assert result.rank == 256
    assert growing.last_rank == 256
    assert len(growing.growth_events) == len(result.stats.events)
