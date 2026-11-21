"""K17 -- THE WRITE FORM UNDER THE ADDITIVE READ (the dense gated write on
the composed machine), gated.

The composed machine's parity disciplines, extended by the dense lane
(``PREREGISTRATION_K17_WRITE_FORM.md``, committed before any probe
measurement):

* **Bit-equality by construction**: the eager oracle and the kernels
  implement the identical defined-order expressions -- the K15 composed walk
  (K13's additive read block under K11B's per-channel DECAY vector and
  gated lane-0 arithmetic) plus the dense lane 2 (the best-match-live-row
  deposit, K11 G2's semantics at the per-channel granularity) -- so every
  channel -- the 3-lane schedule, the payload lanes, the read WEIGHTS, the
  state, the liveness, the pre-decision liveness, the key table -- must be
  ``torch.equal`` against the oracle on every fixture, in both GATED
  combinations, including the bf16 autocast regime.  A difference is a bug,
  not a floor.
* **THE BIRTH RECEIPT (the K15 discipline, carried)**: at the zero birth
  (``d_proj == 0``, ``m_d == 0`` -> ``dense == 0`` exactly) the K17 forward
  is BIT-IDENTICAL to the K15 COMPOSED kernel's own committed launch on
  every shared channel (the machine is BORN AS THE COMPOSED MACHINE bit for
  bit), and the discrete op sequence is bit-invariant to EVERY dense deposit
  value (P2's 0-flip receipt).
* **The gradient gates are floor-discipline**: the fp64 ORACLE validates
  itself first -- a gradcheck THROUGH THE DENSE-WRITE PATH (the dense lane's
  own adjoint, the additive read's raw-score weights ATTACHED under the
  per-channel DECAY vector and write gate, over every adjoint branch) --
  then the kernel path's EIGHT gradients sit within ``NOISE_FACTOR`` of the
  measured fp32-vs-fp64 floor.
* **The replay receipt**: the lane-generic per-channel trajectory replay
  rebuilds the forward's own segment checkpoints bit-exactly from the
  3-LANE recorded schedule, dense lane included.
* **The conversion receipts**: the parameter delta is the registration's
  (p3's own + the dense channel's b_proj-form count), the zero-born dense
  channel (the mixer born as the composed machine bit for bit), and the
  refusals are by name.

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, NONE, POP, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    convert_model_to_separated_keys)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates import (
    channel_persistent_trajectory)
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    ComposedReadMixer, _k15_forward_launch, composed_oracle_scan,
    composed_parameter_delta, composed_scan, convert_model_to_composition)
from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write import (
    LANES, DenseWriteComposedMixer, _k17_backward_body, _k17_forward_launch,
    convert_model_to_dense_write, dense_oracle_scan, dense_parameter_delta,
    dense_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NOISE_FACTOR = 8.0
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))


def keyed_stream(batch, tokens, heads, value_dim, key_dim, distinct, seed,
                 decay=0.95):
    """The lane's canonical fixture (K10/K11/K11B/K15's): payload AND
    write-key values drawn from ``distinct`` fixed directions, query keys
    random.  The decay channel is the PER-CHANNEL vector (uniform around
    the scalar default)."""
    g = torch.Generator().manual_seed(seed)
    payload_dirs = torch.randn(distinct, value_dim, generator=g).to(DEVICE)
    key_dirs = torch.randn(distinct, key_dim, generator=g).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = payload_dirs[choice.cpu()].to(DEVICE)
    write_keys = key_dirs[choice.cpu()].to(DEVICE)
    query_keys = (torch.randn(batch, tokens, heads, key_dim, generator=g)
                  .to(DEVICE) * 0.5)
    if isinstance(decay, float):
        base = torch.full((batch, tokens, heads, 1), decay, device=DEVICE)
        spread = (torch.rand(batch, tokens, heads, value_dim, generator=g)
                  * 0.2 - 0.1).to(DEVICE)
        decay = (base + spread).clamp(min=1e-4, max=1.0 - 1e-6)
    else:
        decay = decay.to(DEVICE)
    return payload, write_keys, query_keys, decay


def gate_stream(batch, tokens, heads, value_dim, seed):
    """Random PER-CHANNEL gates in (0.05, 0.95) -- the off-identity fixture."""
    g = torch.Generator().manual_seed(seed)
    gates = (torch.rand(batch, tokens, heads, value_dim, generator=g)
             * 0.9 + 0.05).to(DEVICE)
    return gates


def dense_stream(batch, tokens, heads, value_dim, seed, scale=0.05):
    """Random dense deposits at a small scale (the G2 fixture convention)."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, tokens, heads, value_dim, generator=g)
            .to(DEVICE) * scale)


def kernel_forward(payload, dense, write_keys, query_keys, decay, gates,
                   rank, segment_tokens=64, gated=True, use_dense=True,
                   collect=True, initial_state=None, initial_live=None,
                   initial_keys=None, warps=None):
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                         device=DEVICE) if initial_state is None
             else initial_state.contiguous())
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
            if initial_live is None else initial_live.contiguous())
    keys = (torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                        device=DEVICE) if initial_keys is None
            else initial_keys.contiguous())
    if gates is None:
        gates = torch.ones(batch, tokens, heads, value_dim, device=DEVICE)
    if not use_dense:
        dense = payload.new_zeros((1, 1, 1, 1))
    return _k17_forward_launch(payload, dense, write_keys, query_keys,
                               decay, gates, state, live, keys, POL32, rank,
                               segment_tokens, gated, use_dense,
                               collect=collect,
                               warps=warps if warps is not None else
                               (4 if rank * value_dim < 8192 else 8),
                               stages=3), state


def composed_kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank, segment_tokens=64, gated=True, collect=True,
                            initial_state=None, initial_live=None,
                            initial_keys=None, warps=None):
    """The K15 composed launch, the birth-parity reference."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                         device=DEVICE) if initial_state is None
             else initial_state.contiguous())
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
            if initial_live is None else initial_live.contiguous())
    keys = (torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                        device=DEVICE) if initial_keys is None
            else initial_keys.contiguous())
    if gates is None:
        gates = torch.ones(batch, tokens, heads, value_dim, device=DEVICE)
    return _k15_forward_launch(payload, write_keys, query_keys, decay,
                               gates, state, live, keys, POL32, rank,
                               segment_tokens, gated,
                               collect=collect,
                               warps=warps if warps is not None else
                               (4 if rank * value_dim < 8192 else 8),
                               stages=3)


def crafted_store(rows, key_rows=None, batch=1, heads=1, V=8):
    """An initial state + key table + liveness with the given (live) rows."""
    rank = len(rows)
    key_rows = rows if key_rows is None else key_rows
    K = len(key_rows[0])
    state = torch.zeros(batch, heads, rank, V, device=DEVICE)
    keys = torch.zeros(batch, heads, rank, K, device=DEVICE)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    for r, row in enumerate(rows):
        state[:, :, r] = torch.as_tensor(row, device=DEVICE)
        keys[:, :, r] = torch.as_tensor(key_rows[r], device=DEVICE)
        live[:, :, r] = True
    return state, keys, live


def oracle_reference(payload, dense, write_keys, query_keys, decay, gates,
                     rank, gated=True, use_dense=True, collect=True,
                     initial_state=None, initial_live=None, initial_keys=None):
    return dense_oracle_scan(payload, dense, write_keys, query_keys, decay,
                             gates, policy=POLICY, gated=gated,
                             use_dense=use_dense, rank=rank, rank_max=rank,
                             initial_state=initial_state,
                             initial_live=initial_live,
                             initial_keys=initial_keys,
                             collect=collect)


def assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, rank, label,
                               use_dense=True):
    """Gate A: every decision channel, the payload lanes (3), the read
    WEIGHTS (the raw scores), the state, the liveness, the pre-decision
    liveness AND the key table bit-equal to the oracle; the read within the
    measured fp32-vs-fp64 floor (the native-sum floor, the K13/K15
    convention)."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(ref["schedule"], name), out[name]), \
            (label, name)
    assert torch.equal(ref["schedule"].payload, out["payload"]), (label, "payload")
    assert torch.equal(ref["weights"], out["weights"]), (label, "weights")
    assert torch.equal(ref["state"], out["state"]), (label, "state")
    assert torch.equal(ref["live"], out["live"]), (label, "live")
    assert torch.equal(ref["live_pre"], out["live_pre"]), (label, "live_pre")
    assert torch.equal(ref["keys"], out["keys"]), (label, "keys")
    with torch.no_grad():
        ref64 = dense_oracle_scan(
            payload.double(), dense.double(), write_keys.double(),
            query_keys.double(), decay.double(), gates.double(),
            policy=POLICY, rank=rank, rank_max=rank, use_dense=use_dense)
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    assert deviation <= max(NOISE_FACTOR * max(floor, 1e-30), 1e-6), \
        (label, "read", deviation, floor)
    assert out["slot"].shape[-1] == LANES


# --------------------------------------------------------------------- gate A


@needs_cuda
@pytest.mark.parametrize("gated", [True, False])
def test_gate_a_identity_regime_bit_equal(gated):
    """The canonical fixture, both GATED combinations (the per-channel DECAY
    vector and the dense lane are active in both): every channel bit-equal,
    and the read weights carry the RAW scores (the additive signature,
    unchanged by the gates and the deposits)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    dense = dense_stream(2, 64, 2, 8, 7)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5, segment_tokens=16, gated=gated)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5, gated=gated)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 5,
                               f"identity-{gated}")
    hist = op_histogram(ref["schedule"])
    assert hist["correct"] > 0 and hist["append"] > 0, hist
    # the dense lane fired somewhere on this fixture
    assert int((out["op"][..., 2] != NONE).sum()) > 0, \
        "the fixture must fire the dense lane"
    # the additive signature under the gates: raw signed scores, bounded by
    # the gain on live rows, exactly zero on dead rows, NOT a simplex.
    w = out["weights"]
    live_pre = out["live_pre"]
    assert (w[live_pre].abs() <= POLICY.read_scale + 1e-6).all()
    assert (w[~live_pre] == 0).all()
    assert (w[live_pre] < 0).any(), \
        "the composed read's weights must carry negative similarities"
    assert not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1))), \
        "the composed weights are raw scores, not a probability simplex"


@needs_cuda
def test_gate_a_merge_and_pop_reachable():
    """Crafted KEY tables force MERGE and POP; the per-channel channels and
    the dense lane ride along bit-exactly."""
    g = torch.Generator().manual_seed(31)
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    key1 = 0.9 * ek[0] + (1 - 0.81) ** 0.5 * ek[1]
    state, keys, live = crafted_store([e[0], e[2], e[3], e[4]],
                                      key_rows=[ek[0], key1, ek[2], ek[3]])
    payload = e[0].view(1, 1, 1, V).repeat(1, 16, 1, 1)
    write_keys = ek[0].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    query_keys = torch.randn(1, 16, 1, K, generator=g).to(DEVICE)
    decay = torch.full((1, 16, 1, V), 0.999, device=DEVICE)
    gates = gate_stream(1, 16, 1, V, 5)
    dense = dense_stream(1, 16, 1, V, 6)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, segment_tokens=8,
                           initial_state=state, initial_live=live,
                           initial_keys=keys)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 4, "merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0, f"the fixture must reach MERGE: {hist}"
    assert (out["keys"][..., 1, :] == 0).all()
    # POP: a full live store, a write key orthogonal to every row key
    state2, keys2, live2 = crafted_store([e[i] for i in range(4)],
                                         key_rows=[ek[i] for i in range(4)])
    write_keys2 = ek[4].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    out2, _ = kernel_forward(payload, dense, write_keys2, query_keys, decay,
                             gates, rank=4, segment_tokens=8,
                             initial_state=state2, initial_live=live2,
                             initial_keys=keys2)
    ref2 = oracle_reference(payload, dense, write_keys2, query_keys, decay,
                            gates, rank=4, initial_state=state2,
                            initial_live=live2, initial_keys=keys2)
    assert_decisions_bit_equal(out2, ref2, payload, dense, write_keys2,
                               query_keys, decay, gates, 4, "pop")
    hist2 = op_histogram(ref2["schedule"])
    assert hist2["pop"] > 0, f"the fixture must reach POP: {hist2}"


@needs_cuda
def test_gate_a_dense_lane_targets_the_best_match_live_row():
    """P3, receipted: the deposit lands on the argmax-live row (recorded in
    lane 2's slot), never on a dead row, and never when no live row existed
    at decision time -- INCLUDING at NONE tokens (the deposit fires below
    tau_match: every token writes its best-match live row)."""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    # a FULL live store whose keys all share one direction, with a write key
    # at a moderate cosine to it (below tau_match, above tau_pop): every
    # token is a NONE -- the deposit fires below tau_match (P3)
    state, keys, live = crafted_store([e[0], e[1], e[2], e[3]],
                                      key_rows=[ek[0], ek[0], ek[0], ek[0]])
    payload = e[4].view(1, 1, 1, V).repeat(1, 16, 1, 1) * 0.5
    moderate = 0.3 * ek[0] + (1 - 0.09) ** 0.5 * ek[1]   # cos = 0.3 to ek[0]
    write_keys = moderate.view(1, 1, 1, K).repeat(1, 16, 1, 1)
    query_keys = torch.zeros(1, 16, 1, K, device=DEVICE)
    decay = torch.full((1, 16, 1, V), 0.999, device=DEVICE)
    gates = gate_stream(1, 16, 1, V, 5)
    dense = dense_stream(1, 16, 1, V, 6, scale=0.1)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=4, segment_tokens=16,
                            initial_state=state, initial_live=live,
                            initial_keys=keys)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 4, "dense-target")
    # every fired deposit's target is live at decision time
    fired = out["op"][..., 2] != NONE
    assert int(fired.sum()) > 0, "the fixture must fire the dense lane"
    live_pre = out["live_pre"]
    targets = out["slot"][..., 2]
    for idx in fired.nonzero()[:64]:
        b, t, h = idx.tolist()
        assert bool(live_pre[b, t, h, targets[b, t, h]]), \
            "the dense lane wrote a row that was not live at decision time"
    # the deposit fires at NONE tokens too (below tau_match, no POP)
    none_tokens = out["op"][..., 0] == NONE
    assert bool((none_tokens).all()), \
        "the fixture must make every op a NONE (cos 0.3 in the dead zone)"
    assert int((fired & none_tokens).sum()) > 0, \
        "the dense lane must fire below tau_match (P3: argmax over live)"


@needs_cuda
def test_gate_a_pop_kills_the_best_row_no_deposit():
    """P3's in-token guards, receipted on a rank-1 store (best == worst;
    the write keys ROTATE at cosine 0.6 -- CORRECT every token from t=2,
    MERGE never: the K12-flagged best==second blend corner stays out of
    the fixture, exactly as the committed engines' own suites keep it):
    (i) a POP that kills the best-match row in-token suppresses that
    token's deposit (the target must still be live after lanes 0/1); (ii)
    the token AFTER (an APPEND into the freed slot) has NO live row at
    decision time, so its deposit goes nowhere too; (iii) the next token
    CORRECTs the re-filed row and the deposit fires again."""
    V, K = 8, 16
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    # rank 1: one slot, live, keyed away from the write key -> POP first
    state, keys, live = crafted_store([e[0]], key_rows=[ek[0]])
    payload = e[4].view(1, 1, 1, V).repeat(1, 8, 1, 1) * 0.5
    # rotating write keys: t=0 orthogonal (POP), then consecutive cos 0.6
    # (CORRECT from t=2 on; MERGE never -- second_value == best_value at
    # rank 1, kept below tau_merge by construction)
    write_keys = ek[4].view(1, 1, 1, K).repeat(1, 8, 1, 1).clone()
    for t in range(1, 8):
        write_keys[0, t, 0] = 0.6 * write_keys[0, t - 1, 0] \
            + 0.8 * ek[4 + t]
    query_keys = torch.zeros(1, 8, 1, K, device=DEVICE)
    decay = torch.full((1, 8, 1, V), 0.999, device=DEVICE)
    gates = torch.ones(1, 8, 1, V, device=DEVICE)
    dense = dense_stream(1, 8, 1, V, 6, scale=0.1)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=1, segment_tokens=8,
                            initial_state=state, initial_live=live,
                            initial_keys=keys)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=1, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 1, "pop-guard")
    assert int(out["op"][0, 0, 0, 0]) == int(POP), \
        "the fixture's first token must POP"
    assert int(out["op"][0, 0, 0, 2]) == int(NONE), \
        "a POP that kills the best row must suppress the deposit"
    assert int(out["op"][0, 1, 0, 0]) == int(APPEND), \
        "the second token must APPEND into the freed slot"
    assert int(out["op"][0, 1, 0, 2]) == int(NONE), \
        "no live row at decision time: the deposit must go nowhere"
    assert int(out["op"][0, 2, 0, 0]) == int(CORRECT), \
        "the third token must CORRECT the re-filed row"
    assert int(out["op"][0, 2, 0, 2]) != int(NONE), \
        "the third token's deposit must fire on the live best row"


@needs_cuda
def test_gate_a_heterogeneous_merge_corner():
    """The K12-flagged multi-lane MERGE corner (the K11 gate-E fixture
    verbatim): the dense machine's oracle guards the MERGE-source close PER
    LANE and agrees with the kernel BIT FOR BIT on a fixture where head 0
    MERGEs and head 1 CORRECTs with a live second row."""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    key1 = 0.9 * ek[0] + (1 - 0.81) ** 0.5 * ek[1]
    state = torch.zeros(1, 2, 4, V, dtype=torch.float32, device=DEVICE)
    keys = torch.zeros(1, 2, 4, K, dtype=torch.float32, device=DEVICE)
    live = torch.zeros(1, 2, 4, dtype=torch.bool, device=DEVICE)
    for h, rows in enumerate(([e[0], e[2], e[3], e[4]],
                              [e[1], e[2], e[3], e[4]])):
        key_rows = ([ek[0], key1, ek[2], ek[3]] if h == 0
                    else [ek[1], ek[2], ek[3], ek[4]])
        for r, (row, krow) in enumerate(zip(rows, key_rows)):
            state[0, h, r] = row
            keys[0, h, r] = krow
            live[0, h, r] = True
    payload = torch.cat([e[0].view(1, 1, 1, V), e[1].view(1, 1, 1, V)],
                        dim=2).repeat(1, 8, 1, 1)
    write_keys = torch.cat([ek[0].view(1, 1, 1, K),
                            ek[1].view(1, 1, 1, K)],
                           dim=2).repeat(1, 8, 1, 1)
    query_keys = torch.zeros(1, 8, 2, K, device=DEVICE)
    decay = torch.full((1, 8, 2, V), 0.999, device=DEVICE)
    gates = torch.ones(1, 8, 2, V, device=DEVICE)
    dense = dense_stream(1, 8, 2, V, 6)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, segment_tokens=8,
                           initial_state=state, initial_live=live,
                           initial_keys=keys)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 4,
                               "heterogeneous-merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0 and hist["correct"] > 0, hist


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    """T not a multiple of the segment length, and of the block width."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23,
                                                          decay=0.9)
    gates = gate_stream(2, 96, 2, 8, 13)
    dense = dense_stream(2, 96, 2, 8, 3)
    for seg in (64, 7):
        out, _ = kernel_forward(payload, dense, write_keys, query_keys,
                                decay, gates, rank=5, segment_tokens=seg)
        ref = oracle_reference(payload, dense, write_keys, query_keys,
                               decay, gates, rank=5)
        assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                                   query_keys, decay, gates, 5,
                                   f"ragged-{seg}")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    """The registered regime: bf16 inputs, fp32 defined-order math (the K11B
    lesson: parity fixtures through the production expression)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    gates = gate_stream(2, 64, 2, 8, 91).bfloat16()
    dense = dense_stream(2, 64, 2, 8, 7).bfloat16()
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5, segment_tokens=16)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 5, "bf16")


@needs_cuda
def test_gate_a_edge_shapes():
    """One token, one head, one rank: the degenerate corners."""
    for batch, tokens, heads, rank, V, K in [(1, 1, 1, 1, 8, 4),
                                             (1, 3, 1, 2, 8, 4),
                                             (2, 65, 1, 16, 16, 16)]:
        payload, write_keys, query_keys, decay = keyed_stream(
            batch, tokens, heads, V, K, 3, 7, decay=0.9)
        gates = gate_stream(batch, tokens, heads, V, 9)
        dense = dense_stream(batch, tokens, heads, V, 4)
        out, _ = kernel_forward(payload, dense, write_keys, query_keys,
                                decay, gates, rank=rank, segment_tokens=5)
        ref = oracle_reference(payload, dense, write_keys, query_keys,
                               decay, gates, rank=rank)
        assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                                   query_keys, decay, gates, rank,
                                   f"edge-{batch}-{tokens}-{rank}-{K}")


@needs_cuda
def test_gate_a_all_dead_read_zero_and_deposit_nowhere():
    """The all-dead rule, unchanged: an empty store reads 0, and the first
    token's deposit goes nowhere (no live row at decision time)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 4, 3)
    gates = gate_stream(2, 8, 2, 8, 5)
    dense = dense_stream(2, 8, 2, 8, 5)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5, segment_tokens=4)
    assert torch.equal(out["reads"][:, :1], torch.zeros_like(out["reads"][:, :1]))
    assert bool((out["op"][:, 0, :, 2] == NONE).all()), \
        "no live row at decision time: the deposit must go nowhere"


@needs_cuda
def test_gate_a_tie_break_first_index():
    """Equal KEY similarities: the lowest index wins in both implementations
    (the decision chain is the banked read's own)."""
    g = torch.Generator().manual_seed(5)
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    state, keys, live = crafted_store([e[0], e[1], e[2], e[3]],
                                      key_rows=[ek[0], ek[0], ek[2], ek[3]])
    payload = torch.randn(1, 32, 1, V, generator=g).to(DEVICE)
    write_keys = ek[0].view(1, 1, 1, K).repeat(1, 32, 1, 1)
    query_keys = torch.randn(1, 32, 1, K, generator=g).to(DEVICE)
    decay = torch.full((1, 32, 1, V), 0.999, device=DEVICE)
    gates = gate_stream(1, 32, 1, V, 3)
    dense = dense_stream(1, 32, 1, V, 3)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, segment_tokens=16,
                           initial_state=state, initial_live=live,
                           initial_keys=keys)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 4, "ties")
    correct = (ref["schedule"].op[..., 0] != NONE)
    assert correct.any()
    assert (ref["schedule"].slot[..., 0][correct] == 0).all(), \
        "a tie must resolve to the lowest index in both implementations"


@needs_cuda
def test_gate_a_keys_not_decayed_and_refiled():
    """The K10 semantic carried verbatim: keys do not age, and neither the
    gates nor the deposits ever touch the key lane."""
    payload, write_keys, query_keys, decay = keyed_stream(1, 32, 1, 8, 4, 3, 11,
                                                          decay=0.5)
    gates = gate_stream(1, 32, 1, 8, 7)
    dense = dense_stream(1, 32, 1, 8, 7)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=4, segment_tokens=8)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4)
    assert_decisions_bit_equal(out, ref, payload, dense, write_keys,
                               query_keys, decay, gates, 4, "undecayed-keys")
    assert (out["keys"] != 0).any()


@needs_cuda
def test_gate_a_registered_shape_flip_test():
    """The decision flip test at a registered-class shape (bf16): 0 flips,
    kernel vs oracle, the per-channel gates, the decay and the dense lane
    all active."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 512, 2, 128, 16,
                                                          512, 77)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    gates = gate_stream(2, 512, 2, 128, 55).bfloat16()
    dense = dense_stream(2, 512, 2, 128, 9).bfloat16()
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=128, segment_tokens=64)
    with torch.no_grad():
        ref = dense_oracle_scan(payload, dense, write_keys, query_keys,
                                decay, gates, policy=POLICY,
                                rank=128, rank_max=128, gated=True,
                                use_dense=True)
    flips = 0
    lanes = 0
    for name in SCHEDULE_FIELDS:
        a, b = getattr(ref["schedule"], name), out[name]
        flips += int((a != b).sum())
        lanes += a.numel()
    flips += int((ref["schedule"].payload != out["payload"]).sum())
    lanes += ref["schedule"].payload.numel()
    flips += int((ref["keys"] != out["keys"]).sum())
    lanes += ref["keys"].numel()
    flips += int((ref["weights"] != out["weights"]).sum())
    lanes += ref["weights"].numel()
    assert flips == 0, f"{flips} flips in {lanes} lanes"


@needs_cuda
def test_gate_a_zero_flip_decision_invariance():
    """P2's receipt, carried to the dense machine: the discrete op sequence
    (lanes 0/1) is bit-identical under ANY per-channel gate values, ANY
    per-channel decay vectors and ANY dense deposits at fixed inputs (no
    gate value and no deposit feeds op selection)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 16, 8, 6,
                                                          41)
    ones = torch.ones(2, 128, 2, 16, device=DEVICE)
    zero_dense = torch.zeros_like(payload)
    base_decay = torch.full_like(decay, 0.95)
    base, _ = kernel_forward(payload, zero_dense, write_keys, query_keys,
                             base_decay, ones, rank=8, segment_tokens=32)
    for seed in (1, 2, 3):
        g = torch.Generator().manual_seed(seed)
        gates = (torch.rand(2, 128, 2, 16, generator=g) * 0.95
                 + 0.025).to(DEVICE)
        vary_decay = (torch.rand(2, 128, 2, 16, generator=g) * 0.9 + 0.05
                     ).to(DEVICE)
        dense = torch.randn(2, 128, 2, 16, generator=g).to(DEVICE)
        vary, _ = kernel_forward(payload, dense, write_keys, query_keys,
                                 vary_decay, gates, rank=8,
                                 segment_tokens=32)
        for name in SCHEDULE_FIELDS:
            assert torch.equal(base[name][..., :2], vary[name][..., :2]), \
                (seed, name)
        assert torch.equal(base["live_pre"], vary["live_pre"]), (seed, "live_pre")
        assert torch.equal(base["keys"], vary["keys"]), (seed, "keys")
        # the lane-2 RECORD targets the same best row under any deposit
        assert torch.equal(base["slot"][..., 2], vary["slot"][..., 2]), \
            (seed, "dense-target")


@needs_cuda
def test_gate_a_determinism():
    """No atomics: two kernel runs bit-identical."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    dense = dense_stream(2, 64, 2, 8, 7)
    out1, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                             gates, rank=5, segment_tokens=16)
    out2, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                             gates, rank=5, segment_tokens=16)
    for name in ("reads", "state", "keys", "slot", "op", "payload", "weights",
                 "live_pre"):
        assert torch.equal(out1[name], out2[name]), name


# --------------------------------------------- the write-form machine's birth receipts


@needs_cuda
def test_birth_bit_identical_to_the_composed_machine():
    """THE WRITE-FORM MACHINE'S BIRTH RECEIPT (the K15 discipline): at the
    zero birth (``d_proj == 0``, ``m_d == 0`` -> ``dense == 0`` exactly) the
    K17 forward launch is BIT-IDENTICAL to the K15 COMPOSED machine's own
    committed kernel launch on EVERY shared channel -- the machine is BORN
    AS THE COMPOSED MACHINE (and hence as the K13 additive machine: the
    gates born at the registered scalar equivalents).  The -0.0 IEEE corner
    of the zero deposit is the only legal escape; its count is asserted
    zero on this fixture."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 16, 8, 6,
                                                          43, decay=0.95)
    # the mixer's OWN birth deposit: 0*sigmoid(0)*p -- per-element +-0.0 with
    # the payload's sign (the IEEE identity either way)
    zero_dense = 0.0 * payload
    for gated in (False, True):
        k15 = composed_kernel_forward(payload, write_keys, query_keys, decay,
                                      None, rank=8, segment_tokens=16,
                                      gated=gated)
        k17, _ = kernel_forward(payload, zero_dense, write_keys, query_keys,
                                decay, gates=None, rank=8, segment_tokens=16,
                                gated=gated)
        for name in SCHEDULE_FIELDS:
            assert torch.equal(k15[name], k17[name][..., :2]), (gated, name)
        assert torch.equal(k15["payload"], k17["payload"][..., :2, :]), \
            (gated, "payload")
        for name in ("reads", "state", "live", "live_pre", "weights", "keys"):
            assert torch.equal(k15[name], k17[name]), (gated, name)
        # the dense lane FIRES (on the live guards alone) with an exact-zero
        # payload wherever a live best row exists -- the birth gradient path
        fired = k17["op"][..., 2] != NONE
        assert int(fired.sum()) > 0, "the dense lane must fire at birth"
        assert (k17["payload"][..., 2, :][fired] == 0).all(), \
            "the birth deposit must be exact zero where it fires"


@needs_cuda
def test_birth_minus_zero_corner_receipt():
    """The -0.0 IEEE corner, receipted (the K11 precedent): at the zero birth
    the deposit is ``0*sigmoid(0)*p`` whose SIGN is the payload's -- a
    NEGATIVE payload element deposits -0.0, and the birth receipt (the K17
    kernel vs the K15 composed kernel, the committed launch) holds BIT FOR
    BIT on an adversarial fixture whose live rows hold exact -0.0 elements
    and whose every op is a NONE (the deposit is the ONLY write): both
    engines share the committed walk's row-extraction arithmetic, so the
    zero deposit moves nothing relative to the composed machine.  (A
    positive payload element deposits +0.0, which would flip an
    exactly--0.0 element -- the one corner the birth receipt counts; this
    fixture keeps every deposit at -0.0 and the count at zero.)"""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    # live rows holding exact -0.0 elements; keys at moderate cosines to
    # the shared write key (all in the NONE dead zone: below tau_match,
    # above tau_pop -- the deposit is the only write)
    row0 = e[0].clone()
    row0[3] = -0.0
    row1 = e[1].clone()
    row1[5] = -0.0
    key0 = 0.3 * ek[0] + (1 - 0.09) ** 0.5 * ek[1]
    key1 = 0.25 * ek[0] + (1 - 0.0625) ** 0.5 * ek[2]
    key2 = 0.2 * ek[0] + (1 - 0.04) ** 0.5 * ek[3]
    state, keys, live = crafted_store([row0, row1, e[2]],
                                      key_rows=[key0, key1, key2])
    # every payload element NEGATIVE -> every birth deposit is -0.0 (the
    # mixer's own expression 0*sigmoid(0)*p)
    payload = -torch.rand(1, 16, 1, V, device=DEVICE) - 0.1
    write_keys = ek[0].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    query_keys = torch.zeros(1, 16, 1, K, device=DEVICE)
    decay = torch.full((1, 16, 1, V), 0.999, device=DEVICE)
    zero_dense = 0.0 * payload
    k17, _ = kernel_forward(payload, zero_dense, write_keys, query_keys,
                            decay, gates=None, rank=3, segment_tokens=16,
                            initial_state=state, initial_live=live,
                            initial_keys=keys)
    k15 = composed_kernel_forward(payload, write_keys, query_keys, decay,
                                  None, rank=3, segment_tokens=16,
                                  initial_state=state, initial_live=live,
                                  initial_keys=keys)
    # every op is a NONE and the dense lane fired (the deposit is the only
    # write on this fixture)
    assert bool((k17["op"][..., 0] == NONE).all()), \
        "the fixture must make every op a NONE"
    fired = k17["op"][..., 2] != NONE
    assert int(fired.sum()) > 0, "the dense lane must fire"
    assert torch.equal(k15["state"], k17["state"]), \
        "the zero deposit must be the IEEE identity vs the composed machine"
    assert torch.equal(k15["live"], k17["live"])
    assert torch.equal(k15["keys"], k17["keys"])


@needs_cuda
def test_decisions_bit_identical_to_the_composed_kernel():
    """The decisions-bit-identical receipt, carried to the write-form
    machine: at fixed inputs the dense scan's recorded schedule (lanes 0/1)
    is BIT-IDENTICAL to the K15 composed kernel's own launch under ANY
    dense deposits (no deposit feeds op selection), while the value state
    differs as designed off the zero birth."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23)
    k15 = composed_kernel_forward(payload, write_keys, query_keys, decay,
                                  gates=None, rank=5, segment_tokens=16)
    dense = dense_stream(2, 96, 2, 8, 7)
    k17, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates=None, rank=5, segment_tokens=16)
    for name in SCHEDULE_FIELDS:
        assert torch.equal(k15[name], k17[name][..., :2]), name
    assert torch.equal(k15["live_pre"], k17["live_pre"]), "live_pre"
    assert torch.equal(k15["keys"], k17["keys"]), "keys"
    # off the zero birth the deposit DOES change the value channel (the
    # design)
    assert not torch.equal(k15["state"], k17["state"]), \
        "a nonzero deposit must move the value state"
    assert torch.equal(k15["weights"], k17["weights"]), \
        "the read weights are the decision channel's own (P2)"


@needs_cuda
def test_mixer_birth_bit_identical_to_the_composed_mixer():
    """The conversion's plumbing receipt: the dense-write mixer at birth is
    BIT-IDENTICAL to the K15 composed mixer's own output (the dense channel
    at the zero birth, the gates at the registered scalar equivalents, the
    additive read K13's own) -- the machine-level birth receipt."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    def build():
        torch.manual_seed(1234)
        stub = torch.nn.Module()
        stub.blocks = [torch.nn.Module()]
        stub.blocks[0].mixer = MatrixMixer(MatrixShape(
            width=64, depth=1, hidden=128, heads=2, value_dim=32,
            rank=16, rank_max=16), POLICY, 64)
        return convert_model_to_separated_keys(stub)

    stub15 = build()
    mixer15 = convert_model_to_composition(stub15).blocks[0].mixer.to(DEVICE)
    stub17 = build()
    mixer17 = convert_model_to_dense_write(
        convert_model_to_composition(stub17)).blocks[0].mixer.to(DEVICE)
    # the two builds share the seed: the carried projections must agree bit
    # for bit (the conversions carry them over untouched)
    for name in ("value", "key", "query_key", "out", "decay", "b_proj",
                 "f_proj", "g_proj"):
        for p15, p17 in zip(getattr(mixer15, name).parameters(),
                            getattr(mixer17, name).parameters()):
            assert torch.equal(p15, p17), name
    x = torch.randn(2, 64, 64, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out15, _ = mixer15.mix(x)
        out17, _ = mixer17.mix(x)
    assert torch.equal(out15, out17), \
        "the born dense-write mixer must reproduce the composed mixer's " \
        "output bit for bit"


@needs_cuda
def test_zero_born_dense_channel_gradient_receipt():
    """P5's no-dead-channel receipt and the two-step wake-up, carried to the
    dense channel: at birth the emitted deposit is EXACTLY zero, the
    zero-born scalar receives gradient at step 0 (through sigmoid(0) = 0.5),
    and the zero-born projection's step-0 gradient is EXACTLY zero (the K11
    two-step wake-up pattern -- receipted so the step-0 zero is not read as
    a dead channel)."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    torch.manual_seed(99)
    stub = torch.nn.Module()
    stub.blocks = [torch.nn.Module()]
    stub.blocks[0].mixer = MatrixMixer(MatrixShape(
        width=64, depth=1, hidden=128, heads=2, value_dim=32,
        rank=16, rank_max=16), POLICY, 64)
    stub = convert_model_to_separated_keys(stub)
    mixer = convert_model_to_dense_write(
        convert_model_to_composition(stub)).blocks[0].mixer.to(DEVICE)
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = mixer.mix(x)
    out.float().pow(2).sum().backward()
    assert float(mixer.dense_scalar.grad.abs().sum()) > 0.0, \
        "the zero-born dense scalar must receive gradient at birth"
    assert float(mixer.d_proj.weight.grad.abs().sum()) == 0.0, \
        "the zero-born d_proj's step-0 gradient must be exactly zero " \
        "(the two-step wake-up)"
    # the p3 gates' own wake-up receipt, unchanged by the dense channel
    for name in ("decay_scalar", "write_scalar", "read_scalar"):
        assert float(getattr(mixer, name).grad.abs().sum()) > 0.0, name
    for name in ("f_proj", "b_proj", "g_proj"):
        for param in getattr(mixer, name).parameters():
            assert float(param.grad.abs().sum()) == 0.0, name
    # and at birth the emitted dense payload is exactly zero (P5)
    with torch.no_grad():
        payload = mixer.value(x).view(2, 32, mixer.heads, mixer.value_dim)
        dense = mixer.dense_scalar.view(1, 1, -1, 1) \
            * torch.sigmoid(mixer.d_proj(x)).view(
                2, 32, mixer.heads, mixer.value_dim) * payload
    assert float(dense.abs().max()) == 0.0


# --------------------------------------------------------------------- gate B


def _oracle_grads(payload, dense, write_keys, query_keys, decay, gates, rank,
                  dtype=torch.float32, state_grad=True, keys_grad=True,
                  gated=True, initial_state=None, initial_live=None,
                  initial_keys=None):
    p = payload.clone().requires_grad_(True)
    dn = dense.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    g_ = gates.clone().requires_grad_(True)
    s = (torch.zeros(payload.shape[0], payload.shape[2], rank,
                     payload.shape[-1], dtype=dtype,
                     device=DEVICE) if initial_state is None
         else initial_state.clone()).requires_grad_(state_grad)
    k = (torch.zeros(payload.shape[0], payload.shape[2], rank,
                     write_keys.shape[-1], dtype=dtype,
                     device=DEVICE) if initial_keys is None
         else initial_keys.clone()).requires_grad_(keys_grad)
    ref = dense_oracle_scan(p.to(dtype), dn.to(dtype), wk.to(dtype),
                            qk.to(dtype), d.to(dtype), g_.to(dtype),
                            policy=POLICY, gated=gated, rank=rank,
                            rank_max=rank, initial_state=s,
                            initial_live=initial_live,
                            initial_keys=k)
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(dtype).to(DEVICE)
    (ref["reads"] * upstream).sum().backward()
    return ref, (p.grad, dn.grad, wk.grad, qk.grad, d.grad, g_.grad,
                 s.grad if state_grad else None, k.grad if keys_grad else None)


GRAD_NAMES = ("dpayload", "ddense", "dwrite_keys", "dquery_keys", "ddecay",
              "dgates", "dstate", "dkeys")


@needs_cuda
@pytest.mark.parametrize("gated", [True, False])
def test_gate_b_oracle_gradcheck_fp64_through_the_dense_path(gated):
    """MANDATORY (the registration's 5): the fp64 gradcheck THROUGH THE
    DENSE-WRITE PATH, over a fixture that fires every content-carrying
    adjoint branch -- the dense lane's own deposit (fired on live guards),
    the additive read's raw-score weights ATTACHED (no softmax jacobian
    anywhere), the payload lanes under the per-channel write gate, and the
    PER-CHANNEL DECAY vector -- both GATED combinations (the eager fp64 oracle's
    cost scales with tokens; 12 tokens fire every branch at a sixth of the
    wall)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 12, 2, 8, 4, 5, 11)
    gates = gate_stream(2, 12, 2, 8, 17)
    dense = dense_stream(2, 12, 2, 8, 19, scale=0.1)
    p = payload.double().clone().requires_grad_(True)
    dn = dense.double().clone().requires_grad_(True)
    wk = write_keys.double().clone().requires_grad_(True)
    qk = query_keys.double().clone().requires_grad_(True)
    d = decay.double().clone().requires_grad_(True)
    g_ = gates.double().clone().requires_grad_(True)
    s = torch.zeros(2, 2, 5, 8, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    k = torch.zeros(2, 2, 5, 4, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    hist_holder = {}
    def f(p_, dn_, wk_, qk_, d_, g__, s_, k_):
        ref = dense_oracle_scan(p_, dn_, wk_, qk_, d_, g__, policy=POLICY,
                                gated=gated, rank=5, rank_max=5,
                                initial_state=s_, initial_keys=k_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    assert torch.autograd.gradcheck(f, (p, dn, wk, qk, d, g_, s, k),
                                    eps=1e-6, atol=1e-4, rtol=1e-4)
    hist = hist_holder["hist"]
    assert all(hist[n] > 0 for n in ("append", "correct")), hist
    # the dense lane must have fired for the gradcheck to cover its branch
    ref_probe = dense_oracle_scan(payload, dense, write_keys, query_keys,
                                  decay, gates, policy=POLICY, gated=gated,
                                  rank=5, rank_max=5)
    fired = ref_probe["schedule"].op[..., 2] != NONE
    assert int(fired.sum()) > 0, "the gradcheck fixture must fire the dense lane"


@needs_cuda
def test_gate_b_grads_within_measured_floors():
    """The kernel path's EIGHT gradients vs the oracle's autograd, inside
    NOISE_FACTOR of the fp32-vs-fp64 floor measured on the same fixture
    (the dense-composed adjoint: the additive prep + the K17 value walk
    with the dense lane's ``ddense = dS[best]``)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    dense = dense_stream(2, 64, 2, 8, 7)
    rank, seg = 5, 16
    ref, oracle_grads = _oracle_grads(payload, dense, write_keys, query_keys,
                                      decay, gates, rank)
    _, oracle_grads64 = _oracle_grads(payload, dense, write_keys, query_keys,
                                      decay, gates, rank, dtype=torch.float64)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k17_backward_body(payload, dense, write_keys, query_keys, decay,
                               gates, upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank, seg, POLICY.read_scale,
                               True)
    for name, kern, orac, orac64 in zip(GRAD_NAMES, grads, oracle_grads,
                                        oracle_grads64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= NOISE_FACTOR * max(floor, 1e-6) or \
            deviation <= 1e-5, (name, deviation, floor)
        assert torch.isfinite(kern).all(), name


@needs_cuda
def test_gate_b_dense_lane_adjoint_is_the_best_row():
    """The dense lane's own adjoint receipt: ``ddense = dS[best]`` (additive,
    ``ds`` passes through) -- verified against the oracle's autograd on a
    fixture whose rows hold WILDLY different norms and whose deposits are
    large (the adjoint must pick the best row's incoming adjoint EXACTLY,
    with no decay, gate or blend leakage into it)."""
    g = torch.Generator().manual_seed(17)
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    state, keys, live = crafted_store(
        [e[0], 5.0 * e[1], 0.2 * e[2], 3.0 * e[3]],
        key_rows=[ek[0], ek[1], ek[2], ek[3]], batch=2, heads=2)
    payload = torch.randn(2, 48, 2, V, generator=g).to(DEVICE)
    write_keys = torch.randn(2, 48, 2, K, generator=g).to(DEVICE)
    query_keys = torch.randn(2, 48, 2, K, generator=g).to(DEVICE)
    decay = (torch.rand(2, 48, 2, V, generator=g) * 0.1 + 0.9).to(DEVICE)
    gates = (torch.rand(2, 48, 2, V, generator=g) * 0.9 + 0.05).to(DEVICE)
    dense = (torch.rand(2, 48, 2, V, generator=g).to(DEVICE) * 0.2 - 0.1)
    rank, seg = 4, 16
    ref, oracle_grads = _oracle_grads(payload, dense, write_keys, query_keys,
                                      decay, gates, rank, initial_state=state,
                                      initial_live=live, initial_keys=keys)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=rank, segment_tokens=seg,
                            initial_state=state, initial_live=live,
                            initial_keys=keys)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k17_backward_body(payload, dense, write_keys, query_keys, decay,
                               gates, upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank, seg, POLICY.read_scale,
                               True)
    for name, kern, orac in zip(GRAD_NAMES, grads, oracle_grads):
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= 1e-4, (name, deviation)
        assert torch.isfinite(kern).all(), name
    # the dense adjoint is EXACTLY zero where the lane did not fire
    ddense = grads[1]
    fired = schedule.op[..., 2] != NONE
    if bool((~fired).any()):
        assert float(ddense[~fired].abs().max()) == 0.0
    assert float(ddense[fired].abs().max()) > 0.0


@needs_cuda
def test_gate_b_trajectory_replay_bit_exact():
    """The replay receipt: K11B's lane-generic per-channel value replay
    rebuilds the forward's own segment checkpoints bit-exactly from the
    3-LANE recorded schedule, dense lane included (the recompute-over-store
    contract, unchanged by the write form)."""
    import triton
    from experiments.expressivity_tasks.x45_matrix_emender.k11_learned_gates \
        import _k11_key_traj_kernel
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    dense = dense_stream(2, 64, 2, 8, 7)
    rank, seg = 5, 16
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    decay32 = decay.float().permute(0, 2, 1, 3).contiguous()
    tokens = query_keys.shape[1]
    batch, heads = 2, 2
    rows = batch * heads
    key_dim = write_keys.shape[-1]
    KP = triton.next_power_of_2(key_dim)
    RP = triton.next_power_of_2(rank)
    for index, start in enumerate(range(0, tokens, seg)):
        stop = min(start + seg, tokens)
        segt = stop - start
        _r, exit_state, exit_live, _traj = channel_persistent_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index],
            decay32[:, :, start:stop, :], schedule.chunk(start, stop))
        ktraj = torch.empty((rows, segt, rank, key_dim), dtype=torch.float32,
                            device=DEVICE)
        _k11_key_traj_kernel[(rows,)](
            out["ckpt_keys"][index], write_keys, schedule.slot, schedule.op,
            ktraj, segt, tokens, start, rank, key_dim,
            *schedule.slot.stride(), *write_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        if stop == tokens:
            assert torch.equal(exit_state, out["state"]), "the last segment"
            assert torch.equal(exit_live, out["live"])
        else:
            assert torch.equal(exit_state, out["ckpt_state"][index + 1]), \
                f"segment {index} value replay vs the forward's checkpoint"
        if stop < tokens:
            # apply the last token's key lane to the replayed table
            # (ktraj's layout is (rows, segt, rank, key_dim), rows = b*h)
            last = segt - 1
            t = start + last
            final = ktraj[:, last].clone()
            for b in range(batch):
                for h in range(heads):
                    row = b * heads + h
                    o = int(schedule.op[b, t, h, 0])
                    if o != NONE:
                        final[row, int(schedule.slot[b, t, h, 0])] = \
                            write_keys[b, t, h].float() * \
                            (0.0 if o == 3 else 1.0)
            assert torch.equal(final, out["ckpt_keys"][index + 1].reshape(
                rows, rank, key_dim)), f"segment {index} key replay"


# ------------------------------------------------------- the conversion receipts


@needs_cuda
def test_conversion_delta_and_dense_tensors():
    """The conversion re-wraps a K15-composed mixer sharing every parameter
    object, adds the dense channel (the b_proj form) at the zero birth, and
    the parameter delta is the registration's (p3's own + the dense
    channel's count, computed not hardcoded)."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)

    stub = Stub()
    stub = convert_model_to_composition(convert_model_to_separated_keys(stub))
    before = {n: p for n, p in stub.blocks[0].mixer.named_parameters()}
    stub = convert_model_to_dense_write(stub)
    mixer = stub.blocks[0].mixer
    mixer.to(DEVICE)
    assert isinstance(mixer, DenseWriteComposedMixer)
    assert isinstance(mixer, ComposedReadMixer)
    after = dict(mixer.named_parameters())
    for name, param in before.items():
        assert after[name] is param, \
            f"the conversion must SHARE the composed parameters: {name}"
    delta = sum(p.numel() for p in mixer.parameters()) \
        - sum(p.numel() for p in before.values())
    assert delta == dense_parameter_delta(1, 64, 2, 32) \
        - composed_parameter_delta(1, 64, 2, 32)
    assert delta == 64 * 2 * 32 + 2           # the b_proj form + the scalar
    assert dense_parameter_delta(24, 576, 2, 128) \
        == composed_parameter_delta(24, 576, 2, 128) + 3_538_992
    assert dense_parameter_delta(24, 576, 2, 128) == 12_196_032
    # the dense channel is born at exact zero
    assert float(mixer.d_proj.weight.abs().max()) == 0.0
    assert float(mixer.dense_scalar.abs().max()) == 0.0
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.no_grad():
        out, result = mixer.mix(x)
    assert out.shape == (2, 32, 64)
    assert result.reads.shape == (2, 32, 2, 32)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="expects a K15-composed model"):
        convert_model_to_dense_write(Stub())


@needs_cuda
def test_refusals_are_by_name():
    """ops=False, grow and a rank prefix are refused outright; the wrapper
    takes the EAGER ORACLE verbatim otherwise (same semantics), and the
    oracle itself refuses a mis-shaped dense tensor."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 3, 5)
    gates = gate_stream(2, 8, 2, 8, 9)
    dense = dense_stream(2, 8, 2, 8, 9)
    with pytest.raises(ValueError, match="append-only"):
        dense_scan(payload, dense, write_keys, query_keys, decay, gates,
                   policy=RoutingPolicy(ops=False), rank=5)
    with pytest.raises(ValueError, match="growth"):
        dense_scan(payload, dense, write_keys, query_keys, decay, gates,
                   policy=RoutingPolicy(grow=True), rank=5)
    with pytest.raises(ValueError, match="rank == rank_max"):
        dense_scan(payload, dense, write_keys, query_keys, decay, gates,
                   policy=POLICY, rank=4, rank_max=8)
    with pytest.raises(ValueError, match="dense deposit must share"):
        dense_scan(payload, torch.zeros_like(dense[..., :4]), write_keys,
                  query_keys, decay, gates, policy=POLICY, rank=5)
    result = dense_scan(payload, dense, write_keys, query_keys, decay, gates,
                       policy=POLICY, rank=5, rank_max=5)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5, collect=False)
    assert torch.equal(result.reads, ref["reads"])
    assert torch.equal(result.state, ref["state"])
