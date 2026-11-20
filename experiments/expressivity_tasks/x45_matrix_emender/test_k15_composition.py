"""K15 -- THE COMPOSITION (the additive read + the p3 gate stack), gated.

The two committed engines' parity disciplines, composed
(``PREREGISTRATION_COMPOSITION.md``, committed before any probe
measurement):

* **Bit-equality by construction**: the eager oracle and the kernels
  implement the identical defined-order expressions -- K13's additive read
  block (the raw banked score, live-gated; NO softmax, NO ``||S_j||``) under
  K11B's per-channel DECAY vector and gated lane-0 arithmetic (the
  per-channel CORRECT blend's pre-decay form ``g*p + (1-g)*(lam_vec*v_pre)``,
  the per-channel DECAY ``mul.rn(acc, lam_vec)``) -- so every channel --
  the 2-lane schedule, the payload lanes, the read WEIGHTS (the raw scores),
  the state, the liveness, the pre-decision liveness, the key table -- must
  be ``torch.equal`` against the oracle on every fixture, in both GATED
  combinations, including the bf16 autocast regime (the K11B lesson:
  bit-parity fixtures go through the production expression under autocast).
  A difference is a bug, not a floor.
* **The read is the one channel at the measured floor** (its rank-axis
  reduction is a native sum, the K1/K23/K13 precedent).
* **THE BIRTH RECEIPT (the composition's own)**: at ``lam_vec ==`` the
  registered scalar on every channel and ``g_vec == 1`` per channel the
  K15 forward is BIT-IDENTICAL to the K13 ADDITIVE machine's own committed
  kernel launch on every channel (the machine-variant's nearest kin -- the
  composition is born as the additive machine), and the discrete op
  sequence is bit-invariant to EVERY per-channel gate/decay value (P2) and
  to the read form (the K13 decisions-bit-identical receipt carried over).
* **The gradient gates are floor-discipline**: the fp64 ORACLE validates
  itself first -- a gradcheck THROUGH THE COMPOSED PATH (the additive read's
  raw-score weights ATTACHED under the per-channel DECAY vector and write
  gate, over every adjoint branch) -- then the kernel path's SEVEN
  gradients sit within ``NOISE_FACTOR`` of the measured fp32-vs-fp64 floor.
* **The strength path is GONE with the softmax** (R13-2's engine-side
  receipt, carried over): the value walk's strength term dies by the
  prep's EXACT-ZERO ``dstrength`` channel -- IEEE add of 0.0 is exact --
  verified against the oracle's own autograd on a fixture with wildly
  different row norms.
* **The replay receipt**: the per-channel trajectory replay (K11B's) and
  the key-lane replay (K10's) rebuild the forward's own segment
  checkpoints bit-exactly from the 2-lane recorded schedule.
* **The conversion receipts**: the parameter delta is p3's own (the read
  channel instantiated ONCE, never double-added), the zero-born gates (the
  mixer born as the K13 additive machine bit for bit, the clamp-bound
  carrier corner included -- the falsifier's standing remedy), and the
  refusals are by name.

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    _k10_forward_launch, convert_model_to_separated_keys)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates import (
    LANES, channel_persistent_trajectory)
from experiments.expressivity_tasks.x45_matrix_emender.k13_additive_read import (
    _k13_backward_body, _k13_forward_launch, convert_model_to_additive_read)
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition import (
    ComposedReadMixer, _k15_backward_body, _k15_forward_launch,
    composed_oracle_scan, composed_parameter_delta, composed_scan,
    convert_model_to_composition)
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
    """The lane's canonical fixture (K10/K11/K11B's): payload AND write-key
    values drawn from ``distinct`` fixed directions, query keys random.  The
    decay channel is the PER-CHANNEL vector (uniform around the scalar
    default)."""
    g = torch.Generator().manual_seed(seed)
    payload_dirs = torch.randn(distinct, value_dim, generator=g).to(DEVICE)
    key_dirs = torch.randn(distinct, key_dim, generator=g).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = payload_dirs[choice.cpu()].to(DEVICE)
    write_keys = key_dirs[choice.cpu()].to(DEVICE)
    query_keys = (torch.randn(batch, tokens, heads, key_dim, generator=g)
                  .to(DEVICE) * 0.5)
    if isinstance(decay, float):
        # a per-channel vector around the scalar: distinct channels, in range
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


def kernel_forward(payload, write_keys, query_keys, decay, gates, rank,
                   segment_tokens=64, gated=True, collect=True,
                   initial_state=None, initial_live=None, initial_keys=None,
                   warps=None):
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
                               stages=3), state


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


def oracle_reference(payload, write_keys, query_keys, decay, gates,
                     rank, gated=True, collect=True,
                     initial_state=None, initial_live=None, initial_keys=None):
    return composed_oracle_scan(payload, write_keys, query_keys, decay,
                                gates, policy=POLICY, gated=gated,
                                rank=rank, rank_max=rank,
                                initial_state=initial_state,
                                initial_live=initial_live,
                                initial_keys=initial_keys,
                                collect=collect)


def assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, gates, rank, label):
    """Gate A: every decision channel, the payload lanes (2), the read
    WEIGHTS (the raw scores), the state, the liveness, the pre-decision
    liveness AND the key table bit-equal to the oracle; the read within the
    measured fp32-vs-fp64 floor (the native-sum floor, the K13 convention)."""
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
        ref64 = composed_oracle_scan(
            payload.double(), write_keys.double(), query_keys.double(),
            decay.double(), gates.double(), policy=POLICY, rank=rank,
            rank_max=rank)
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
    vector is active in both): every channel bit-equal, and the read weights
    carry the RAW scores (the signed-weight signature of the additive form,
    unchanged by the gates)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=5, segment_tokens=16, gated=gated)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=5, gated=gated)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, gates, 5, f"identity-{gated}")
    hist = op_histogram(ref["schedule"])
    assert hist["correct"] > 0 and hist["append"] > 0, hist
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
    """Crafted KEY tables force MERGE and POP; the per-channel channels ride
    along bit-exactly."""
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
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=4, segment_tokens=8, initial_state=state,
                            initial_live=live, initial_keys=keys)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=4, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, gates, 4, "merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0, f"the fixture must reach MERGE: {hist}"
    assert (out["keys"][..., 1, :] == 0).all()
    # POP: a full live store, a write key orthogonal to every row key
    state2, keys2, live2 = crafted_store([e[i] for i in range(4)],
                                         key_rows=[ek[i] for i in range(4)])
    write_keys2 = ek[4].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    out2, _ = kernel_forward(payload, write_keys2, query_keys, decay, gates,
                             rank=4, segment_tokens=8,
                             initial_state=state2, initial_live=live2,
                             initial_keys=keys2)
    ref2 = oracle_reference(payload, write_keys2, query_keys, decay, gates,
                            rank=4, initial_state=state2,
                            initial_live=live2, initial_keys=keys2)
    assert_decisions_bit_equal(out2, ref2, payload, write_keys2, query_keys,
                               decay, gates, 4, "pop")
    hist2 = op_histogram(ref2["schedule"])
    assert hist2["pop"] > 0, f"the fixture must reach POP: {hist2}"


@needs_cuda
def test_gate_a_heterogeneous_merge_corner():
    """The K12-flagged multi-lane MERGE corner (the K11 gate-E fixture
    verbatim): the composed oracle guards the MERGE-source close PER LANE
    (K11B's committed fix -- the K10/K11/K13 oracles' ``if any:`` block is
    NOT copied), so the oracle agrees with the kernel BIT FOR BIT on a
    fixture where head 0 MERGEs and head 1 CORRECTs with a live second row."""
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
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=4, segment_tokens=8, initial_state=state,
                            initial_live=live, initial_keys=keys)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=4, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, gates, 4, "heterogeneous-merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0 and hist["correct"] > 0, hist


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    """T not a multiple of the segment length, and of the block width."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23,
                                                          decay=0.9)
    gates = gate_stream(2, 96, 2, 8, 13)
    for seg in (64, 7):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                                rank=5, segment_tokens=seg)
        ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                               rank=5)
        assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                                   decay, gates, 5, f"ragged-{seg}")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    """The registered regime: bf16 inputs, fp32 defined-order math (the K11B
    lesson: parity fixtures through the production expression)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    gates = gate_stream(2, 64, 2, 8, 91).bfloat16()
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=5, segment_tokens=16)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=5)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, gates, 5, "bf16")


@needs_cuda
def test_gate_a_edge_shapes():
    """One token, one head, one rank: the degenerate corners."""
    for batch, tokens, heads, rank, V, K in [(1, 1, 1, 1, 8, 4),
                                             (1, 3, 1, 2, 8, 4),
                                             (2, 65, 1, 16, 16, 16)]:
        payload, write_keys, query_keys, decay = keyed_stream(
            batch, tokens, heads, V, K, 3, 7, decay=0.9)
        gates = gate_stream(batch, tokens, heads, V, 9)
        out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                                rank=rank, segment_tokens=5)
        ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                               rank=rank)
        assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                                   decay, gates, rank,
                                   f"edge-{batch}-{tokens}-{rank}-{K}")


@needs_cuda
def test_gate_a_all_dead_read_is_exactly_zero():
    """The all-dead rule, unchanged by the form swap: an empty store reads 0
    (the raw-score weights are zero on every row)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 4, 3)
    gates = gate_stream(2, 8, 2, 8, 5)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=5, segment_tokens=4)
    assert torch.equal(out["reads"][:, :1], torch.zeros_like(out["reads"][:, :1]))


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
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=4, segment_tokens=16, initial_state=state,
                            initial_live=live, initial_keys=keys)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=4, initial_state=state, initial_live=live,
                           initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, gates, 4, "ties")
    correct = (ref["schedule"].op[..., 0] != NONE)
    assert correct.any()
    assert (ref["schedule"].slot[..., 0][correct] == 0).all(), \
        "a tie must resolve to the lowest index in both implementations"


@needs_cuda
def test_gate_a_keys_not_decayed_and_refiled():
    """The K10 semantic carried verbatim: keys do not age, and the gates
    never touch the key lane."""
    payload, write_keys, query_keys, decay = keyed_stream(1, 32, 1, 8, 4, 3, 11,
                                                          decay=0.5)
    gates = gate_stream(1, 32, 1, 8, 7)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=4, segment_tokens=8)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=4)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, gates, 4, "undecayed-keys")
    assert (out["keys"] != 0).any()


@needs_cuda
def test_gate_a_registered_shape_flip_test():
    """The decision flip test at a registered-class shape (bf16): 0 flips,
    kernel vs oracle, the per-channel gates and decay active."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 512, 2, 128, 16,
                                                          512, 77)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    gates = gate_stream(2, 512, 2, 128, 55).bfloat16()
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=128, segment_tokens=64)
    with torch.no_grad():
        ref = composed_oracle_scan(payload, write_keys, query_keys,
                                   decay, gates, policy=POLICY,
                                   rank=128, rank_max=128, gated=True)
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
    """P2's receipt, composed: the discrete op sequence (lanes 0/1) is
    bit-identical under ANY per-channel gate values and ANY per-channel
    decay vectors at fixed inputs (no gate value, scalar or vector, feeds op
    selection; the read form cannot either)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 16, 8, 6,
                                                          41)
    ones = torch.ones(2, 128, 2, 16, device=DEVICE)
    base_decay = torch.full_like(decay, 0.95)
    base, _ = kernel_forward(payload, write_keys, query_keys, base_decay,
                             ones, rank=8, segment_tokens=32)
    for seed in (1, 2, 3):
        g = torch.Generator().manual_seed(seed)
        gates = (torch.rand(2, 128, 2, 16, generator=g) * 0.95
                 + 0.025).to(DEVICE)
        vary_decay = (torch.rand(2, 128, 2, 16, generator=g) * 0.9 + 0.05
                     ).to(DEVICE)
        vary, _ = kernel_forward(payload, write_keys, query_keys, vary_decay,
                                 gates, rank=8, segment_tokens=32)
        for name in SCHEDULE_FIELDS:
            assert torch.equal(base[name], vary[name]), (seed, name)
        assert torch.equal(base["live_pre"], vary["live_pre"]), (seed, "live_pre")
        assert torch.equal(base["keys"], vary["keys"]), (seed, "keys")


@needs_cuda
def test_gate_a_determinism():
    """No atomics: two kernel runs bit-identical."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    out1, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                             rank=5, segment_tokens=16)
    out2, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                             rank=5, segment_tokens=16)
    for name in ("reads", "state", "keys", "slot", "op", "payload", "weights",
                 "live_pre"):
        assert torch.equal(out1[name], out2[name]), name


# ------------------------------------------- the composition's own birth receipts


@needs_cuda
def test_birth_bit_identical_to_the_additive_machine():
    """THE COMPOSITION'S BIRTH RECEIPT: at ``lam_vec ==`` the registered
    scalar on every channel and ``g_vec == 1`` per channel, the K15 forward
    launch is BIT-IDENTICAL to the K13 ADDITIVE machine's own committed
    kernel launch on EVERY channel -- the composition is born as the
    additive machine (the K11B gates' mimetic-neutral birth through the
    channel-decay wrapper; the additive read is K13's own form).  The -0.0
    IEEE corner of the unity-gate CORRECT blend is the only legal escape
    (the K11-receipted corner); its count is asserted zero on this fixture."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 16, 8, 6,
                                                          43, decay=0.95)
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    rank = 8
    # the registered scalar, broadcast to the per-channel form EXACTLY as
    # the mixer's birth constructs it (the channel-decay wrapper's factor is
    # the exact 1.0)
    scalar = decay[..., 0].contiguous()                    # (b, T, h)
    lam_vec = scalar.unsqueeze(-1).expand(batch, tokens, heads, value_dim
                                          ).contiguous()
    ones = torch.ones(batch, tokens, heads, value_dim, device=DEVICE)
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device=DEVICE)
    k13 = _k13_forward_launch(payload, write_keys, query_keys, scalar,
                              state, live, keys, POL32, rank, 16,
                              collect=True, warps=4, stages=3)
    for gated in (False, True):
        k15, _ = kernel_forward(payload, write_keys, query_keys, lam_vec,
                                ones if gated else None, rank=rank,
                                segment_tokens=16, gated=gated)
        for name in SCHEDULE_FIELDS:
            assert torch.equal(k13[name], k15[name]), (gated, name)
        assert torch.equal(k13["payload"], k15["payload"]), (gated, "payload")
        for name in ("reads", "state", "live", "live_pre", "weights", "keys"):
            assert torch.equal(k13[name], k15[name]), (gated, name)


@needs_cuda
def test_decisions_bit_identical_to_the_additive_kernel():
    """The K13 decisions-bit-identical receipt, carried over to the
    composition: at fixed inputs the composed scan's recorded schedule is
    BIT-IDENTICAL to the K13 additive kernel's own launch under ANY
    per-channel gate/decay values (no gate value feeds op selection), while
    the payload lanes and the state differ as designed off identity."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23)
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    rank = 5
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device=DEVICE)
    k13 = _k13_forward_launch(payload, write_keys, query_keys,
                              decay[..., 0].contiguous(), state, live, keys,
                              POL32, rank, 16, collect=True, warps=4,
                              stages=3)
    gates = gate_stream(batch, tokens, heads, value_dim, 91)
    k15, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=rank, segment_tokens=16, gated=True)
    for name in SCHEDULE_FIELDS:
        assert torch.equal(k13[name], k15[name]), name
    assert torch.equal(k13["live_pre"], k15["live_pre"]), "live_pre"
    assert torch.equal(k13["keys"], k15["keys"]), "keys"
    # off identity the gates DO change the value channel (the design):
    assert not torch.equal(k13["payload"], k15["payload"]), \
        "the per-channel write gate must modulate the payload lane"
    assert not torch.equal(k13["state"], k15["state"]), \
        "the per-channel decay must modulate the state"


@needs_cuda
def test_mixer_birth_bit_identical_to_the_additive_mixer():
    """The conversion's plumbing receipt: the composed mixer at birth is
    BIT-IDENTICAL to the K13 additive mixer's own output (the gates at the
    registered scalar equivalents, the read gate at exact identity, the
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

    stub13 = build()
    mixer13 = convert_model_to_additive_read(stub13).blocks[0].mixer.to(DEVICE)
    stub15 = build()
    mixer15 = convert_model_to_composition(stub15).blocks[0].mixer.to(DEVICE)
    # the two builds share the seed: the banked projections must agree bit
    # for bit (the conversions carry them over untouched)
    for name in ("value", "key", "query_key", "out", "decay"):
        assert torch.equal(getattr(mixer13, name).weight,
                           getattr(mixer15, name).weight), name
    x = torch.randn(2, 64, 64, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out13, _ = mixer13.mix(x)
        out15, _ = mixer15.mix(x)
    assert torch.equal(out13, out15), \
        "the born composed mixer must reproduce the K13 additive mixer's " \
        "output bit for bit"


@needs_cuda
def test_mixer_birth_survives_the_clamp_bound_carrier_corner():
    """The falsifier's corner, PINNED PERMANENTLY on the composition
    (receipt 57e1f942, the K11B standing remedy): a bf16-SATURATING sigmoid
    position through the PRODUCTION autocast expression.  The bounds act in
    the incumbent's carrier dtype, so the born composed mixer reproduces
    the K13 additive mixer bit for bit INCLUDING this corner."""
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

    stub13 = build()
    mixer13 = convert_model_to_additive_read(stub13).blocks[0].mixer.to(DEVICE)
    stub15 = build()
    mixer15 = convert_model_to_composition(stub15).blocks[0].mixer.to(DEVICE)
    # craft the corner on BOTH mixers identically: head 0's decay bias
    # saturates the bf16 sigmoid to EXACTLY 1.0; head 1 stays mid-range
    with torch.no_grad():
        for mixer in (mixer13, mixer15):
            mixer.decay.weight.zero_()
            mixer.decay.bias[0].fill_(20.0)
            mixer.decay.bias[1].fill_(0.0)
    x = torch.randn(2, 64, 64, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lam_vec, lam_inc = mixer15.channel_decay(x)
        assert bool((lam_inc[:, :, 0] == 1.0).all()), \
            "the fixture must saturate head 0's bf16 sigmoid to exactly 1.0"
        assert float(lam_vec[:, :, 0].max()) == 1.0, \
            ("the saturating positions must be EXACTLY 1.0 (the incumbent's "
             "own bf16-rounded ceiling), never the naive fp32 clamp's "
             "0.9999989867210388")
        out13, _ = mixer13.mix(x)
        out15, _ = mixer15.mix(x)
    assert torch.equal(out13, out15), \
        "the born composed mixer must reproduce the K13 additive mixer's " \
        "output bit for bit at the saturating corner"


@needs_cuda
def test_zero_born_scalars_receive_gradient_projections_wait():
    """P5's no-dead-channel receipt and the two-step wake-up, carried to the
    composition: at birth the zero-born scalars receive gradient at step 0
    (through sigmoid(0) = 0.5) while the projections' step-0 gradient is
    EXACTLY zero (the K11 two-step wake-up pattern -- receipted so the
    step-0 zero is not read as a dead channel)."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    torch.manual_seed(99)
    stub = torch.nn.Module()
    stub.blocks = [torch.nn.Module()]
    stub.blocks[0].mixer = MatrixMixer(MatrixShape(
        width=64, depth=1, hidden=128, heads=2, value_dim=32,
        rank=16, rank_max=16), POLICY, 64)
    stub = convert_model_to_separated_keys(stub)
    mixer = convert_model_to_composition(stub).blocks[0].mixer.to(DEVICE)
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = mixer.mix(x)
    out.float().pow(2).sum().backward()
    for name in ("decay_scalar", "write_scalar", "read_scalar"):
        assert float(getattr(mixer, name).grad.abs().sum()) > 0.0, \
            f"the zero-born {name} must receive gradient at birth"
    for name in ("f_proj", "b_proj", "g_proj"):
        for param in getattr(mixer, name).parameters():
            assert float(param.grad.abs().sum()) == 0.0, \
                (f"the zero-born {name} projections' step-0 gradient must "
                 f"be exactly zero (the two-step wake-up)")


# --------------------------------------------------------------------- gate B


def _oracle_grads(payload, write_keys, query_keys, decay, gates, rank,
                  dtype=torch.float32, state_grad=True, keys_grad=True,
                  gated=True, initial_state=None, initial_live=None,
                  initial_keys=None):
    p = payload.clone().requires_grad_(True)
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
    ref = composed_oracle_scan(p.to(dtype), wk.to(dtype), qk.to(dtype),
                               d.to(dtype), g_.to(dtype), policy=POLICY,
                               gated=gated, rank=rank, rank_max=rank,
                               initial_state=s,
                               initial_live=initial_live,
                               initial_keys=k)
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(dtype).to(DEVICE)
    (ref["reads"] * upstream).sum().backward()
    return ref, (p.grad, wk.grad, qk.grad, d.grad, g_.grad,
                 s.grad if state_grad else None, k.grad if keys_grad else None)


GRAD_NAMES = ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dgates",
              "dstate", "dkeys")


@needs_cuda
@pytest.mark.parametrize("gated", [True, False])
def test_gate_b_oracle_gradcheck_fp64_through_the_composed_path(gated):
    """MANDATORY (the registration's 2.4): the fp64 gradcheck THROUGH THE
    COMPOSED PATH, over a fixture that fires every content-carrying adjoint
    branch -- the additive read's raw-score weights ATTACHED (no softmax
    jacobian anywhere: the read's direct term into the value state, the
    score path into the query keys and the key table), the payload lanes
    under the per-channel write gate, and the PER-CHANNEL DECAY vector --
    both GATED combinations."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 32, 2, 8, 4, 5, 11)
    gates = gate_stream(2, 32, 2, 8, 17)
    p = payload.double().clone().requires_grad_(True)
    wk = write_keys.double().clone().requires_grad_(True)
    qk = query_keys.double().clone().requires_grad_(True)
    d = decay.double().clone().requires_grad_(True)
    g_ = gates.double().clone().requires_grad_(True)
    s = torch.zeros(2, 2, 5, 8, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    k = torch.zeros(2, 2, 5, 4, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    hist_holder = {}

    def f(p_, wk_, qk_, d_, g__, s_, k_):
        ref = composed_oracle_scan(p_, wk_, qk_, d_, g__, policy=POLICY,
                                   gated=gated, rank=5, rank_max=5,
                                   initial_state=s_, initial_keys=k_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    assert torch.autograd.gradcheck(f, (p, wk, qk, d, g_, s, k),
                                    eps=1e-6, atol=1e-4, rtol=1e-4)
    hist = hist_holder["hist"]
    assert all(hist[n] > 0 for n in ("append", "correct")), hist


@needs_cuda
def test_gate_b_grads_within_measured_floors():
    """The kernel path's SEVEN gradients vs the oracle's autograd, inside
    NOISE_FACTOR of the fp32-vs-fp64 floor measured on the same fixture
    (the composed adjoint: the additive prep + the per-channel value walk)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    rank, seg = 5, 16
    ref, oracle_grads = _oracle_grads(payload, write_keys, query_keys,
                                      decay, gates, rank)
    _, oracle_grads64 = _oracle_grads(payload, write_keys, query_keys,
                                      decay, gates, rank, dtype=torch.float64)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k15_backward_body(payload, write_keys, query_keys, decay, gates,
                               upstream, schedule, out["live_pre"],
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
def test_gate_b_the_strength_path_is_gone_with_the_softmax():
    """R13-2's engine-side receipt, carried over to the composed walk: the
    value walk's strength term dies by the prep's EXACT-ZERO ``dstrength``
    channel -- the read adjoint's value contribution is the DIRECT term
    alone (``dS[j] += score_j * dy``), verified against the oracle's own
    autograd on a fixture with live rows of WILDLY different norms (where
    the banked softmax read's strength path would be O(1))."""
    g = torch.Generator().manual_seed(17)
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    # rows of WILDLY different norms: the strength channel is the thing the
    # composed read must NOT consume.
    state, keys, live = crafted_store(
        [e[0], 5.0 * e[1], 0.2 * e[2], 3.0 * e[3]],
        key_rows=[ek[0], ek[1], ek[2], ek[3]], batch=2, heads=2)
    payload = torch.randn(2, 48, 2, V, generator=g).to(DEVICE)
    write_keys = torch.randn(2, 48, 2, K, generator=g).to(DEVICE)
    query_keys = torch.randn(2, 48, 2, K, generator=g).to(DEVICE)
    decay = (torch.rand(2, 48, 2, V, generator=g) * 0.1 + 0.9).to(DEVICE)
    gates = (torch.rand(2, 48, 2, V, generator=g) * 0.9 + 0.05).to(DEVICE)
    rank, seg = 4, 16
    ref, oracle_grads = _oracle_grads(payload, write_keys, query_keys,
                                      decay, gates, rank, initial_state=state,
                                      initial_live=live, initial_keys=keys)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=rank, segment_tokens=seg,
                            initial_state=state, initial_live=live,
                            initial_keys=keys)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k15_backward_body(payload, write_keys, query_keys, decay, gates,
                               upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank, seg, POLICY.read_scale,
                               True)
    for name, kern, orac in zip(GRAD_NAMES, grads, oracle_grads):
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= 1e-4, (name, deviation)
        assert torch.isfinite(kern).all(), name


@needs_cuda
def test_gate_b_trajectory_replay_bit_exact():
    """The replay receipt: K11B's per-channel value replay and K10's key-lane
    replay rebuild the forward's own segment checkpoints bit-exactly from the
    2-lane recorded schedule (the recompute-over-store contract, both
    channels, unchanged by the composition)."""
    import triton
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import _k10_key_traj_kernel
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    rank, seg = 5, 16
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=rank, segment_tokens=seg)
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
        _k10_key_traj_kernel[(rows,)](
            out["ckpt_keys"][index], write_keys, schedule.slot, schedule.op,
            ktraj, segt, tokens, start, rank, key_dim,
            *write_keys.stride(), H=heads, KP=KP, RP=RP, num_warps=4)
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
def test_mixer_conversion_swaps_the_scan_and_adds_the_p3_stack():
    """The conversion re-wraps a K10-converted mixer sharing every parameter
    object, adds the p3 gate channels at the zero-born identity, and the
    parameter delta is the registration's (p3's own count -- the read
    channel is K13's own construction, instantiated ONCE)."""
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
    stub = convert_model_to_separated_keys(stub)
    before = {n: p for n, p in stub.blocks[0].mixer.named_parameters()}
    stub = convert_model_to_composition(stub)
    mixer = stub.blocks[0].mixer
    mixer.to(DEVICE)
    assert isinstance(mixer, ComposedReadMixer)
    after = dict(mixer.named_parameters())
    for name, param in before.items():
        assert after[name] is param, \
            f"the conversion must SHARE the banked parameters: {name}"
    delta = sum(p.numel() for p in mixer.parameters()) \
        - sum(p.numel() for p in before.values())
    assert delta == composed_parameter_delta(1, 64, 2, 32)
    # p3's own per-layer accounting, computed not hardcoded
    assert delta == (64 * 32 + 32 * 2 * 32 + 2) \
        + (64 * 2 * 32 + 2) \
        + (64 * 32 + 32 * 2 * 32 + 2 * 32 + 2)
    assert composed_parameter_delta(24, 576, 2, 128) == 8_657_040
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.no_grad():
        out, result = mixer.mix(x)
    assert out.shape == (2, 32, 64)
    assert result.reads.shape == (2, 32, 2, 32)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="expects a K10-converted model"):
        convert_model_to_composition(Stub())


@needs_cuda
def test_refusals_are_by_name():
    """ops=False, grow and a rank prefix are refused outright; the wrapper
    takes the EAGER ORACLE verbatim otherwise (same semantics)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 3, 5)
    gates = gate_stream(2, 8, 2, 8, 9)
    with pytest.raises(ValueError, match="append-only"):
        composed_scan(payload, write_keys, query_keys, decay, gates,
                      policy=RoutingPolicy(ops=False), rank=5)
    with pytest.raises(ValueError, match="growth"):
        composed_scan(payload, write_keys, query_keys, decay, gates,
                      policy=RoutingPolicy(grow=True), rank=5)
    with pytest.raises(ValueError, match="rank == rank_max"):
        composed_scan(payload, write_keys, query_keys, decay, gates,
                      policy=POLICY, rank=4, rank_max=8)
    result = composed_scan(payload, write_keys, query_keys, decay, gates,
                           policy=POLICY, rank=5, rank_max=5)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=5, collect=False)
    assert torch.equal(result.reads, ref["reads"])
    assert torch.equal(result.state, ref["state"])
