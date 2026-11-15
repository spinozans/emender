"""K6 -- the sequential matrix arm, gated.

The parity discipline at the new granularity (``PREREGISTRATION_SEQUENTIAL_MATRIX.md``):

* **Bit-equality by construction is the headline**: the spec DEFINES the
  reduction order (the balanced tree folds, fp32, correctly-rounded IEEE ops,
  libdevice exp), the eager oracle implements it in torch and the kernels in
  Triton, so every DECISION channel -- slot, op, erase, live_after, the payload
  lane (the MERGE blend included), the read WEIGHTS, the state and the
  liveness -- must be ``torch.equal`` against the oracle on every fixture,
  including the bf16 autocast regime.  A difference is a bug, not a floor.
* **The read is the one channel at the measured floor** (its rank-axis
  reduction is a native sum, the K1/K23 precedent): the fp32 floor is measured
  on the same fixture first (the oracle's fp32 read against its own fp64
  twin), and the kernel's read must sit within ``NOISE_FACTOR`` of it.
* **The gradient gates are floor-discipline too** (the softmax adjoint's
  cross-axis reductions are autograd-ordered on the oracle side): the fp64
  oracle validates the oracle itself (a gradcheck over every adjoint branch),
  the fp32-vs-fp64 floor is measured per output, and the kernel path's four
  gradients sit within ``NOISE_FACTOR`` of it.  The segment-recompute
  trajectory is asserted bit-equal to the forward's own checkpoints.
* **The refusals are by name** and take the EAGER ORACLE (the same semantics)
  or raise outright for the chunk-dial concepts this registration does not
  carry (ops=False, grow, a rank prefix).
* **Determinism**: no atomics anywhere; two runs bit-identical.

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, OP_NAMES, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    MAX_SEQ_TILE, SequentialMatrixMixer, _seq_backward_body, _seq_forward_launch,
    convert_model_to_sequential, sequential_available, sequential_oracle_scan,
    sequential_scan, tree_fold, _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    EPS, RoutingPolicy)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NOISE_FACTOR = 8.0
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))


def distinct_stream(batch, tokens, heads, value_dim, distinct, seed,
                    decay=0.95):
    """The lane's canonical fixture: payload values drawn from ``distinct``
    fixed directions (what makes MERGE reachable at all), query random."""
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(batch, tokens, heads, value_dim, generator=g) \
        .to(DEVICE) * 0.5
    directions = torch.randn(distinct, value_dim, generator=g).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = directions[choice]
    if isinstance(decay, float):
        decay = torch.full((batch, tokens, heads), decay, device=DEVICE)
    else:
        decay = decay.to(DEVICE)
    return query, payload, decay


def kernel_forward(query, payload, decay, rank, segment_tokens=64,
                   collect=True, initial_state=None, initial_live=None):
    state = (torch.zeros(query.shape[0], query.shape[2], rank,
                         query.shape[-1], dtype=torch.float32, device=DEVICE)
             if initial_state is None else initial_state.contiguous())
    live = (torch.zeros(query.shape[0], query.shape[2], rank, dtype=torch.bool,
                        device=DEVICE) if initial_live is None
            else initial_live.contiguous())
    return _seq_forward_launch(query, payload, decay, state, live, POL32,
                               rank, segment_tokens, collect=collect,
                               warps=4 if rank * query.shape[-1] < 8192 else 8,
                               stages=3), state


def crafted_store(rows, batch=1, heads=1, V=8):
    """An initial state + liveness with the given (live) rows per head."""
    rank = len(rows)
    state = torch.zeros(batch, heads, rank, V, device=DEVICE)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    for r, row in enumerate(rows):
        state[:, :, r] = torch.as_tensor(row, device=DEVICE)
        live[:, :, r] = True
    return state, live


def assert_decisions_bit_equal(out, ref, query, payload, decay, rank, label):
    """Gate A: every decision channel, the payload lane, the weights, the state
    and the liveness bit-equal to the oracle; the read within the floor."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(ref["schedule"], name), out[name]), \
            (label, name)
    assert torch.equal(ref["schedule"].payload, out["payload"]), (label, "payload")
    assert torch.equal(ref["weights"], out["weights"]), (label, "weights")
    assert torch.equal(ref["state"], out["state"]), (label, "state")
    assert torch.equal(ref["live"], out["live"]), (label, "live")
    assert torch.equal(ref["live_pre"], out["live_pre"]), (label, "live_pre")
    # the read: the fp32 floor measured on the same fixture (fp32 vs fp64
    # oracle), the kernel within NOISE_FACTOR of it.
    with torch.no_grad():
        ref64 = sequential_oracle_scan(query.double(), payload.double(),
                                       decay.double(), policy=POLICY,
                                       rank=rank, rank_max=rank)
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    assert deviation <= max(NOISE_FACTOR * max(floor, 1e-30), 1e-6), \
        (label, "read", deviation, floor)


# --------------------------------------------------------------------- gate A


@needs_cuda
def test_gate_a_all_five_ops_bit_equal():
    """The K1 gate fixture's regime: all five ops present, bit-equal."""
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    out, _ = kernel_forward(query, payload, decay, rank=5, segment_tokens=16)
    ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                 rank=5, rank_max=5)
    assert_decisions_bit_equal(out, ref, query, payload, decay, 5, "all-five-ops")
    hist = op_histogram(ref["schedule"])
    assert hist["correct"] > 0 and hist["append"] > 0, hist


@needs_cuda
def test_gate_a_merge_and_pop_reachable():
    """Crafted stores force MERGE (a 0.9-cosine second row) and POP (a full
    store, an orthogonal payload) -- the fresh-state regime rarely reaches
    either, which is itself the sweep's finding."""
    g = torch.Generator().manual_seed(31)
    V = 8
    e = torch.eye(V, device=DEVICE)
    # MERGE: rows 0 and 1 are 0.9-cosine to each other; the payload is row 0's
    # direction, so best=row 0 (cos 1.0) and second=row 1 (cos 0.9 >= tau_merge)
    row1 = 0.9 * e[0] + (1 - 0.81) ** 0.5 * e[1]    # unit, cos(e0, row1) = 0.9
    state, live = crafted_store([e[0], row1, e[2], e[3]])
    payload = e[0].view(1, 1, 1, V).repeat(1, 16, 1, 1)
    query = torch.randn(1, 16, 1, V, generator=g).to(DEVICE)
    decay = torch.full((1, 16, 1), 0.999, device=DEVICE)
    out, _ = kernel_forward(query, payload, decay, rank=4, segment_tokens=8,
                            initial_state=state, initial_live=live)
    ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                 rank=4, rank_max=4, initial_state=state,
                                 initial_live=live)
    assert_decisions_bit_equal(out, ref, query, payload, decay, 4, "merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0, f"the fixture must reach MERGE: {hist}"
    # POP: a full live store, a payload orthogonal to every row
    state2, live2 = crafted_store([e[i] for i in range(4)])
    orth = e[4]
    payload2 = orth.view(1, 1, 1, V).repeat(1, 16, 1, 1)
    out2, _ = kernel_forward(query, payload2, decay, rank=4, segment_tokens=8,
                            initial_state=state2, initial_live=live2)
    ref2 = sequential_oracle_scan(query, payload2, decay, policy=POLICY,
                                  rank=4, rank_max=4, initial_state=state2,
                                  initial_live=live2)
    assert_decisions_bit_equal(out2, ref2, query, payload2, decay, 4, "pop")
    hist2 = op_histogram(ref2["schedule"])
    assert hist2["pop"] > 0, f"the fixture must reach POP: {hist2}"


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    """T not a multiple of the segment length, and of the block width."""
    query, payload, decay = distinct_stream(2, 96, 2, 8, 6, 23, decay=0.9)
    out, _ = kernel_forward(query, payload, decay, rank=5, segment_tokens=64)
    ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                 rank=5, rank_max=5)
    assert_decisions_bit_equal(out, ref, query, payload, decay, 5, "ragged-segments")
    out2, _ = kernel_forward(query, payload, decay, rank=5, segment_tokens=7)
    assert_decisions_bit_equal(out2, ref, query, payload, decay, 5, "ragged-segments-7")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    """The registered regime: bf16 inputs, fp32 defined-order math."""
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    query, payload, decay = (query.bfloat16(), payload.bfloat16(),
                             decay.bfloat16())
    out, _ = kernel_forward(query, payload, decay, rank=5, segment_tokens=16)
    ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                 rank=5, rank_max=5)
    assert_decisions_bit_equal(out, ref, query, payload, decay, 5, "bf16")


@needs_cuda
def test_gate_a_edge_shapes():
    """One token, one head, one rank: the degenerate corners."""
    for batch, tokens, heads, rank, V in [(1, 1, 1, 1, 8), (1, 3, 1, 2, 8),
                                          (2, 65, 1, 16, 16)]:
        query, payload, decay = distinct_stream(batch, tokens, heads, V, 3, 7,
                                                decay=0.9)
        out, _ = kernel_forward(query, payload, decay, rank=rank,
                                segment_tokens=5)
        ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                     rank=rank, rank_max=rank)
        assert_decisions_bit_equal(out, ref, query, payload, decay, rank, f"edge-{batch}-{tokens}-{rank}")


@needs_cuda
def test_gate_a_all_dead_read_is_exactly_zero():
    """The certified all-dead rule: the first chunk of an empty store reads 0."""
    query, payload, decay = distinct_stream(2, 8, 2, 8, 4, 3)
    out, _ = kernel_forward(query, payload, decay, rank=5, segment_tokens=4)
    assert torch.equal(out["reads"][:, :1], torch.zeros_like(out["reads"][:, :1]))


@needs_cuda
def test_gate_a_tie_break_first_index():
    """Equal scores: the lowest index wins, torch's argmax rule."""
    # two identical rows: every similarity ties; the tie-break penalty and the
    # first-index rule must pick row 0 in both implementations.
    g = torch.Generator().manual_seed(5)
    directions = torch.randn(2, 8, generator=g).to(DEVICE)
    payload = directions[torch.randint(0, 2, (1, 32, 1), generator=g).to(DEVICE)]
    query = torch.randn(1, 32, 1, 8, generator=g).to(DEVICE)
    decay = torch.full((1, 32, 1), 0.999, device=DEVICE)
    out, _ = kernel_forward(query, payload, decay, rank=4, segment_tokens=16)
    ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                 rank=4, rank_max=4)
    assert_decisions_bit_equal(out, ref, query, payload, decay, 4, "ties")
    correct = (ref["schedule"].op[..., 0] != NONE)
    assert correct.any()
    assert (ref["schedule"].slot[..., 0][correct] == 0).all(), \
        "a tie must resolve to the lowest index in both implementations"


@needs_cuda
def test_gate_a_registered_shape_flip_test():
    """The decision flip test at a registered-class shape (bf16): 0 flips.

    The full 8.65M-lane bar at the whole registered shape is the receipt's
    (``receipts/k6_gates.json``); this is the suite's fast slice of it.
    """
    query, payload, decay = distinct_stream(2, 512, 2, 128, 512, 77)
    query, payload, decay = (query.bfloat16(), payload.bfloat16(),
                             decay.bfloat16())
    out, _ = kernel_forward(query, payload, decay, rank=128, segment_tokens=64)
    with torch.no_grad():
        ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                     rank=128, rank_max=128)
    flips = 0
    lanes = 0
    for name in SCHEDULE_FIELDS:
        a, b = getattr(ref["schedule"], name), out[name]
        flips += int((a != b).sum())
        lanes += a.numel()
    flips += int((ref["schedule"].payload != out["payload"]).sum())
    lanes += ref["schedule"].payload.numel()
    assert flips == 0, f"{flips} flips in {lanes} lanes"


# --------------------------------------------------------------------- gate B


def _oracle_grads(query, payload, decay, rank, dtype=torch.float32,
                  state_grad=True):
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = torch.zeros(query.shape[0], query.shape[2], rank, query.shape[-1],
                    dtype=dtype, device=DEVICE).requires_grad_(state_grad)
    ref = sequential_oracle_scan(q.to(dtype), p.to(dtype), d.to(dtype),
                                  policy=POLICY, rank=rank, rank_max=rank,
                                  initial_state=s)
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(dtype).to(DEVICE)
    (ref["reads"] * upstream).sum().backward()
    return ref, (q.grad, p.grad, d.grad, s.grad if state_grad else None)


@needs_cuda
def test_gate_b_oracle_gradcheck_fp64():
    """The oracle is validated before it is trusted: an fp64 gradcheck over a
    fixture with every adjoint branch (NONE, APPEND, CORRECT, POP, MERGE)."""
    query, payload, decay = distinct_stream(2, 32, 2, 8, 5, 11)
    query, payload, decay = (query.double(), payload.double(), decay.double())
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = torch.zeros(2, 2, 5, 8, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    hist_holder = {}

    def f(q_, p_, d_, s_):
        ref = sequential_oracle_scan(q_, p_, d_, policy=POLICY, rank=5,
                                     rank_max=5, initial_state=s_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"]

    assert torch.autograd.gradcheck(f, (q, p, d, s), eps=1e-6, atol=1e-4,
                                    rtol=1e-4)
    hist = hist_holder["hist"]
    assert all(hist[n] > 0 for n in ("append", "correct")), hist


@needs_cuda
def test_gate_b_grads_within_measured_floors():
    """The kernel path's four gradients vs the oracle's autograd, inside
    NOISE_FACTOR of the fp32-vs-fp64 floor measured on the same fixture."""
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    rank, seg = 5, 16
    ref, oracle_grads = _oracle_grads(query, payload, decay, rank)
    _, oracle_grads64 = _oracle_grads(query, payload, decay, rank,
                                      dtype=torch.float64)
    out, _ = kernel_forward(query, payload, decay, rank=rank,
                            segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _seq_backward_body(query, payload, decay, upstream, schedule,
                               out["live_pre"], out["ckpt_state"],
                               out["ckpt_live"], rank, seg, POLICY.read_scale)
    for name, kern, orac, orac64 in zip(("dquery", "dpayload", "ddecay",
                                         "dstate"),
                                        grads, oracle_grads, oracle_grads64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= NOISE_FACTOR * max(floor, 1e-6) or \
            deviation <= 1e-5, (name, deviation, floor)
        assert torch.isfinite(kern).all(), name


@needs_cuda
def test_gate_b_trajectory_replay_is_bit_exact():
    """The segment recompute (K4's replay kernel) reproduces the forward's own
    checkpoints bit for bit -- the recompute-over-store contract."""
    from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward \
        import persistent_scan_trajectory
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    rank, seg = 5, 16
    out, _ = kernel_forward(query, payload, decay, rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    decay32 = decay.float().permute(0, 2, 1).contiguous()
    tokens = query.shape[1]
    for index, start in enumerate(range(0, tokens, seg)):
        stop = min(start + seg, tokens)
        zeros_w = torch.zeros(2, 2, stop - start, rank, dtype=torch.float32,
                              device=DEVICE)
        _, exit_state, exit_live, _traj = persistent_scan_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index], zeros_w,
            decay32[:, :, start:stop], schedule.chunk(start, stop))
        if stop == tokens:
            assert torch.equal(exit_state, out["state"]), "the last segment"
            assert torch.equal(exit_live, out["live"])
        else:
            assert torch.equal(exit_state, out["ckpt_state"][index + 1]), \
                f"segment {index} replay vs the forward's own checkpoint"


@needs_cuda
def test_gate_b_pop_terminates_the_row_gradient():
    """A POPped row's gradient terminates at its death token: an upstream
    gradient strictly after the death reaches neither the earlier payload nor
    the entry row (the K4 case-iii boundary, at the new granularity)."""
    V = 8
    e = torch.eye(V, device=DEVICE)
    state, live = crafted_store([e[i] for i in range(4)])
    payload = e[4].view(1, 1, 1, V).repeat(1, 16, 1, 1)   # orthogonal: POPs
    g = torch.Generator().manual_seed(17)
    query = torch.randn(1, 16, 1, V, generator=g).to(DEVICE)
    decay = torch.full((1, 16, 1), 0.999, device=DEVICE)
    rank, seg = 4, 8
    out, _ = kernel_forward(query, payload, decay, rank=rank,
                            segment_tokens=seg, initial_state=state,
                            initial_live=live)
    death = None
    op0 = out["op"][0, :, 0, 0]
    for t in range(16):
        if int(op0[t]) == 3:                  # POP
            death = t
            break
    assert death is not None and death < 15, "the fixture must POP early"
    slot = int(out["slot"][0, death, 0, 0])
    # gradient ONLY after the death token: nothing may reach the row's writes
    upstream = torch.zeros(1, 16, 1, V, device=DEVICE)
    upstream[:, death + 1:] = torch.randn(1, 16 - death - 1, 1, V,
                                          generator=torch.Generator()
                                          .manual_seed(9)).to(DEVICE)
    sched = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                           payload=out["payload"], op=out["op"],
                           live_after=out["live_after"])
    dq, dp, dd, dst = _seq_backward_body(query, payload, decay, upstream,
                                         sched, out["live_pre"],
                                         out["ckpt_state"], out["ckpt_live"],
                                         rank, seg, POLICY.read_scale)
    # the oracle under the same masked upstream, from the same entry store
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    r2 = sequential_oracle_scan(q, p, d, policy=POLICY, rank=rank,
                                rank_max=rank, initial_state=s,
                                initial_live=live)
    (r2["reads"] * upstream).sum().backward()
    before = slice(0, death)
    dp_before = dp[:, before].abs().max().item() if death > 0 else 0.0
    op_before = p.grad[:, before].abs().max().item() if death > 0 else 0.0
    assert dp_before == 0.0, \
        "a POPped row's payload gradient must terminate at the death token"
    assert op_before == 0.0, \
        "the oracle agrees: nothing before the death token carries gradient"
    # and the two adjoints agree within floor on every output
    for name, kern, orac in (("dquery", dq, q.grad), ("dpayload", dp, p.grad),
                             ("ddecay", dd, d.grad), ("dstate", dst, s.grad)):
        deviation = (kern.float() - orac.float()).abs().max().item()
        scale = orac.float().abs().max().item()
        assert deviation <= 1e-4 * max(scale, 1.0) or deviation < 1e-5, \
            (name, deviation, scale)


@needs_cuda
def test_gate_b_merge_gradient_reaches_both_source_rows():
    """The MERGE blend's content is attached: its gradient reaches BOTH source
    rows through the payload channel's adjoint (the K4 case-iv discipline)."""
    V = 8
    e = torch.eye(V, device=DEVICE)
    row1 = 0.9 * e[0] + (1 - 0.81) ** 0.5 * e[1]   # cos(e0, row1) = 0.9
    state, live = crafted_store([e[0], row1, e[2], e[3]])
    payload = e[0].view(1, 1, 1, V).repeat(1, 16, 1, 1)
    g = torch.Generator().manual_seed(31)
    query = torch.randn(1, 16, 1, V, generator=g).to(DEVICE)
    decay = torch.full((1, 16, 1), 0.999, device=DEVICE)
    rank, seg = 4, 8
    out, _ = kernel_forward(query, payload, decay, rank=rank,
                            segment_tokens=seg, initial_state=state,
                            initial_live=live)
    merges = (out["op"][0, :, 0, 0] == 4).nonzero()
    assert len(merges) > 0, "the fixture must MERGE"
    t = int(merges[0])
    upstream = torch.zeros(1, 16, 1, V, device=DEVICE)
    upstream[:, t + 1:] = 1.0
    sched = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                           payload=out["payload"], op=out["op"],
                           live_after=out["live_after"])
    dq, dp, dd, dst = _seq_backward_body(query, payload, decay, upstream,
                                         sched, out["live_pre"],
                                         out["ckpt_state"], out["ckpt_live"],
                                         rank, seg, POLICY.read_scale)
    # the oracle under the same upstream, from the same entry store
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    r2 = sequential_oracle_scan(q, p, d, policy=POLICY, rank=rank,
                                rank_max=rank, initial_state=s,
                                initial_live=live)
    (r2["reads"] * upstream).sum().backward()
    for name, kern, orac in (("dquery", dq, q.grad), ("dpayload", dp, p.grad),
                             ("ddecay", dd, d.grad), ("dstate", dst, s.grad)):
        deviation = (kern.float() - orac.float()).abs().max().item()
        scale = orac.float().abs().max().item()
        assert deviation <= 1e-4 * max(scale, 1.0) or deviation < 1e-5, \
            (name, deviation, scale)


@needs_cuda
def test_refusals_are_by_name():
    query, payload, decay = distinct_stream(2, 8, 2, 8, 4, 3)
    with pytest.raises(ValueError, match="append-only"):
        sequential_scan(query, payload, decay, policy=RoutingPolicy(ops=False),
                        rank=5, rank_max=5)
    with pytest.raises(ValueError, match="growth"):
        sequential_scan(query, payload, decay,
                        policy=RoutingPolicy(ops=True, grow=True), rank=5,
                        rank_max=5)
    with pytest.raises(ValueError, match="rank == rank_max"):
        sequential_scan(query, payload, decay, policy=POLICY, rank=5,
                        rank_max=8)


@needs_cuda
def test_size_gate_falls_back_to_the_oracle():
    """A store too big for one program takes the eager oracle verbatim (the
    same semantics, not a granularity change)."""
    rank, V = 256, 128
    assert not sequential_available(rank, V)
    assert sequential_available(128, 128)
    query = torch.randn(1, 16, 1, V, device=DEVICE)
    payload = torch.randn(1, 16, 1, V, device=DEVICE)
    decay = torch.full((1, 16, 1), 0.95, device=DEVICE)
    res = sequential_scan(query, payload, decay, policy=POLICY, rank=rank,
                         rank_max=rank)
    ref = sequential_oracle_scan(query, payload, decay, policy=POLICY,
                                 rank=rank, rank_max=rank)
    assert torch.equal(res.reads, ref["reads"])
    assert torch.equal(res.state, ref["state"])


@needs_cuda
def test_bit_determinism_two_runs():
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    with torch.no_grad():
        r1 = sequential_scan(query, payload, decay, policy=POLICY, rank=5,
                             rank_max=5)
        r2 = sequential_scan(query, payload, decay, policy=POLICY, rank=5,
                             rank_max=5)
    assert torch.equal(r1.reads, r2.reads)
    assert torch.equal(r1.state, r2.state)
    assert torch.equal(r1.live, r2.live)


@needs_cuda
def test_op_path_matches_the_direct_body():
    """The custom-op pair is the raw launch's wrapper: forward outputs and
    gradients agree with calling the pieces directly."""
    query, payload, decay = distinct_stream(2, 64, 2, 8, 6, 23)
    q = query.clone().requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    res = sequential_scan(q, p, d, policy=POLICY, rank=5, rank_max=5)
    upstream = torch.randn(res.reads.shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    (res.reads * upstream).sum().backward()
    out, _ = kernel_forward(query, payload, decay, rank=5, segment_tokens=64)
    assert torch.equal(res.reads, out["reads"])
    assert torch.equal(res.state, out["state"])
    ref, (rq, rp, rd, _) = _oracle_grads(query, payload, decay, 5)
    for name, kern, orac in (("dquery", q.grad, rq), ("dpayload", p.grad, rp),
                             ("ddecay", d.grad, rd)):
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation < 1e-4, (name, deviation)


@needs_cuda
def test_tree_fold_is_the_defined_order():
    """The spec's fold: padding with exact zeros changes nothing, and the fold
    is bit-deterministic (the same input twice, the same bits)."""
    x = torch.randn(4, 5, 8, device=DEVICE)
    assert torch.equal(tree_fold(x, -1, 8), tree_fold(x, -1, 8))
    padded = torch.randn(4, 5, 3, device=DEVICE)
    zero_padded = torch.zeros(4, 5, 5, device=DEVICE)
    zero_padded[:, :, :3] = padded
    assert torch.equal(tree_fold(padded, -1, 8),
                       tree_fold(zero_padded, -1, 8)), \
        "exact-zero padding must contribute exactly 0 at every fold level"
    # the middle-axis fold is the same defined tree
    y = torch.randn(3, 6, 4, device=DEVICE)
    assert torch.equal(tree_fold(y, -2, 8), tree_fold(y, 1, 8))
