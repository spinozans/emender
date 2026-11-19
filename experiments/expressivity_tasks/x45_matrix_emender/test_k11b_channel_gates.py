"""K11B -- the per-channel-gate arm, gated.

The K11 parity discipline, extended to the per-channel DECAY vector and the
per-channel write gate (``PREREGISTRATION_CHANNEL_GATES.md`` 3):

* **Bit-equality by construction**: the eager oracle and the kernels
  implement the identical defined-order expressions (the per-channel CORRECT
  blend's pre-decay form ``g*p + (1-g)*(lam_vec*v_pre)`` and the per-channel
  DECAY ``mul.rn(acc, lam_vec)`` included), so every channel -- the 2-lane
  schedule, the payload lanes, the read weights, the state, the liveness,
  the pre-decision liveness, the key table -- must be ``torch.equal``
  against the oracle on every fixture, in both GATED combinations, including
  the bf16 autocast regime.  The READ is floor-discipline (the oracle's
  tree-fold reduction vs the kernel's ``tl.sum`` -- K10/K11's own
  convention), asserted inside the measured fp32-vs-fp64 floor.  A
  difference is a bug, not a floor.
* **The 0-flip decision-invariance receipt (P2)**: the discrete op sequence
  (lanes 0/1) is bit-identical under ANY per-channel gate values and ANY
  per-channel decay vectors at fixed inputs.
* **The birth-parity / plumbing receipt (P5)**: at ``lam_vec ==`` the
  registered scalar on every channel and ``g_vec == 1`` per channel the
  k11b scan is bit-identical to K10's OWN launch on every channel (the
  differing-element count is receipted -- the only legal difference is the
  -0.0 IEEE corner), and the MIXER at birth (the conversion's output) is
  bit-identical to the K10 mixer's own output including the per-channel
  read gate at exact identity.
* **The gradient gates are floor-discipline**: the fp64 oracle gradcheck
  first (every adjoint branch, the per-channel DECAY and gate channels
  included), the fp32-vs-fp64 floor per output, the kernel path's SEVEN
  gradients within ``NOISE_FACTOR`` of it; the zero-born scalars must
  receive gradient at step 0 (P5's no-dead-channel receipt) while the
  projections' step-0 gradient is exactly zero (the K11 two-step wake-up
  pattern, receipted so the step-0 zero is not read as a dead channel).
* **The replay receipt**: the lane-specific per-channel trajectory replay
  rebuilds the forward's own segment checkpoints bit-exactly from the 2-lane
  recorded schedule (K4's imported kernel cannot carry the per-channel
  decay -- the engine limitation the supervisor ruled on).
* **The conversion receipts**: the arm's parameter delta (GDN2's own
  projection shapes, computed not hardcoded), the zero-born gates (the arm
  is born as the banked winner, bit-equal per channel), and the optimizer
  routing (the gate projections land on the hybrid's AdamW side, the
  incumbent's own "a gate, not a hidden map" convention).

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, OP_NAMES, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    _k10_forward_launch)
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates import (
    LANES, ChannelGatedSeparatedMixer, _k11b_backward_body,
    _k11b_forward_launch, arm_channels, arm_parameter_delta,
    channel_gated_oracle_scan, channel_gated_scan,
    channel_persistent_trajectory, convert_model_to_channel_gates)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
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


def keyed_stream(batch, tokens, heads, value_dim, key_dim, distinct, seed,
                 decay=0.95):
    """The lane's canonical fixture (K10/K11's): payload AND write-key values
    drawn from ``distinct`` fixed directions, query keys random.  The decay
    channel is the PER-CHANNEL vector (uniform around the scalar default)."""
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
    return _k11b_forward_launch(payload, write_keys, query_keys, decay,
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


def assert_decisions_bit_equal(out, ref, label):
    """Gate A: every decision channel, the payload lanes (2), the weights,
    the state, the liveness, the pre-decision liveness AND the key table
    bit-equal to the oracle; the read at the floor-discipline bar."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(ref["schedule"], name), out[name]), \
            (label, name)
    assert torch.equal(ref["schedule"].payload, out["payload"]), (label, "payload")
    assert torch.equal(ref["weights"], out["weights"]), (label, "weights")
    assert torch.equal(ref["state"], out["state"]), (label, "state")
    assert torch.equal(ref["live"], out["live"]), (label, "live")
    assert torch.equal(ref["live_pre"], out["live_pre"]), (label, "live_pre")
    assert torch.equal(ref["keys"], out["keys"]), (label, "keys")
    read_dev = (out["reads"] - ref["reads"]).abs().max().item()
    read_scale = ref["reads"].abs().max().item()
    assert read_dev <= max(1e-5, 1e-5 * read_scale), (label, "reads", read_dev)
    assert out["slot"].shape[-1] == LANES


def oracle_reference(payload, write_keys, query_keys, decay, gates,
                     rank, gated=True, collect=True,
                     initial_state=None, initial_live=None, initial_keys=None):
    return channel_gated_oracle_scan(payload, write_keys, query_keys, decay,
                                     gates, policy=POLICY, gated=gated,
                                     rank=rank, rank_max=rank,
                                     initial_state=initial_state,
                                     initial_live=initial_live,
                                     initial_keys=initial_keys,
                                     collect=collect)


# --------------------------------------------------------------------- gate A


@needs_cuda
@pytest.mark.parametrize("gated", [True, False])
def test_gate_a_identity_regime_bit_equal(gated):
    """The canonical fixture, both GATED combinations (the per-channel DECAY
    vector is active in both)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=5, segment_tokens=16, gated=gated)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=5, gated=gated)
    assert_decisions_bit_equal(out, ref, f"identity-{gated}")
    hist = op_histogram(ref["schedule"])
    assert hist["correct"] > 0 and hist["append"] > 0, hist


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
    assert_decisions_bit_equal(out, ref, "merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0, f"the fixture must reach MERGE: {hist}"
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
    assert_decisions_bit_equal(out2, ref2, "pop")
    hist2 = op_histogram(ref2["schedule"])
    assert hist2["pop"] > 0, f"the fixture must reach POP: {hist2}"


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23,
                                                          decay=0.9)
    gates = gate_stream(2, 96, 2, 8, 13)
    for seg in (64, 7):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                                rank=5, segment_tokens=seg)
        ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                               rank=5)
        assert_decisions_bit_equal(out, ref, f"ragged-{seg}")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    """The registered regime: bf16 inputs, fp32 defined-order math."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    gates = gate_stream(2, 64, 2, 8, 91).bfloat16()
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=5, segment_tokens=16)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=5)
    assert_decisions_bit_equal(out, ref, "bf16")


@needs_cuda
def test_gate_a_all_dead_read_is_exactly_zero():
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 4, 3)
    gates = gate_stream(2, 8, 2, 8, 5)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=5, segment_tokens=4)
    assert torch.equal(out["reads"][:, :1], torch.zeros_like(out["reads"][:, :1]))


@needs_cuda
def test_gate_a_determinism():
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    out1, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                             rank=5, segment_tokens=16)
    out2, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                             rank=5, segment_tokens=16)
    for name in ("reads", "state", "keys", "slot", "op", "payload", "weights",
                 "live_pre"):
        assert torch.equal(out1[name], out2[name]), name


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
        ref = channel_gated_oracle_scan(payload, write_keys, query_keys,
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
    assert flips == 0, f"{flips} flips in {lanes} lanes"


@needs_cuda
def test_gate_a_zero_flip_decision_invariance():
    """P2's receipt: the discrete op sequence (lanes 0/1) is bit-identical
    under ANY per-channel gate values and ANY per-channel decay vectors at
    fixed inputs."""
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


@needs_cuda
def test_gate_a_birth_parity_receipt():
    """P5's plumbing receipt: at lam_vec == the registered scalar on every
    channel and g_vec == 1 per channel the k11b scan is bit-identical to
    K10's OWN launch on every channel (the -0.0 corner is the only legal
    difference; its count is asserted zero on this fixture)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 16, 8, 6,
                                                          43, decay=0.95)
    # the registered scalar, broadcast to the per-channel form EXACTLY as
    # the mixer's birth constructs it (mul by the exact 1.0 factor)
    scalar = torch.full((2, 128, 2), 0.95, device=DEVICE)
    lam_vec = scalar.unsqueeze(-1) * torch.ones_like(decay)
    ones = torch.ones(2, 128, 2, 16, device=DEVICE)
    k10 = _k10_forward_launch(payload, write_keys, query_keys, scalar,
                              torch.zeros(2, 2, 8, 16, dtype=torch.float32,
                                          device=DEVICE),
                              torch.zeros(2, 2, 8, dtype=torch.bool,
                                          device=DEVICE),
                              torch.zeros(2, 2, 8, 8, dtype=torch.float32,
                                          device=DEVICE),
                              POL32, 8, 16, collect=True, warps=4, stages=3)
    for gated in (False, True):
        k11b, _ = kernel_forward(payload, write_keys, query_keys, lam_vec,
                                 ones, rank=8, segment_tokens=16, gated=gated)
        for name in SCHEDULE_FIELDS:
            assert torch.equal(k10[name], k11b[name]), (gated, name)
        assert torch.equal(k10["payload"], k11b["payload"]), (gated, "payload")
        for name in ("reads", "state", "live", "live_pre", "weights", "keys"):
            assert torch.equal(k10[name], k11b[name]), (gated, name)


# --------------------------------------------------------------------- gate B


def _oracle_grads(payload, write_keys, query_keys, decay, gates, rank,
                   dtype=torch.float32, state_grad=True, keys_grad=True,
                   gated=True):
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    g_ = gates.clone().requires_grad_(True)
    s = torch.zeros(payload.shape[0], payload.shape[2], rank, payload.shape[-1],
                     dtype=dtype, device=DEVICE).requires_grad_(state_grad)
    k = torch.zeros(payload.shape[0], payload.shape[2], rank,
                    write_keys.shape[-1], dtype=dtype,
                    device=DEVICE).requires_grad_(keys_grad)
    ref = channel_gated_oracle_scan(p.to(dtype), wk.to(dtype), qk.to(dtype),
                                    d.to(dtype), g_.to(dtype),
                                    policy=POLICY, gated=gated, rank=rank,
                                    rank_max=rank, initial_state=s,
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
def test_gate_b_oracle_gradcheck_fp64(gated):
    """The oracle validates before it is trusted: an fp64 gradcheck through
    the payload, the PER-CHANNEL DECAY vector, the PER-CHANNEL GATES, the
    key channel, the value state and the key table, both GATED
    combinations."""
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
        ref = channel_gated_oracle_scan(p_, wk_, qk_, d_, g__, policy=POLICY,
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
    """The kernel path's seven gradients vs the oracle's autograd, inside
    NOISE_FACTOR of the fp32-vs-fp64 floor measured on the same fixture."""
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
    grads = _k11b_backward_body(payload, write_keys, query_keys, decay, gates,
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
def test_gate_b_scalar_gradient_reaches_birth_but_projections_wait():
    """P5's no-dead-channel receipt and the two-step wake-up, together: at
    the birth configuration the zero-born scalars receive gradient at step 0
    (through sigmoid(0) = 0.5) while the projections' step-0 gradient is
    EXACTLY zero (the K11 two-step wake-up pattern -- receipted so the
    step-0 zero is not read as a dead channel)."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)
            self.blocks[0].mixer = convert_model_to_separated_keys(
                self).blocks[0].mixer

    stub = Stub()
    stub = convert_model_to_channel_gates(stub, arm="p3")
    mixer = stub.blocks[0].mixer.to(DEVICE)
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = mixer.mix(x)
    out.float().pow(2).sum().backward()
    for name in ("decay_scalar", "write_scalar", "read_scalar"):
        scalar = getattr(mixer, name)
        assert float(scalar.grad.abs().sum()) > 0.0, \
            f"the zero-born {name} must receive gradient at birth"
    for name in ("f_proj", "b_proj", "g_proj"):
        projection = getattr(mixer, name)
        for param in projection.parameters():
            assert float(param.grad.abs().sum()) == 0.0, \
                (f"the zero-born {name} projections' step-0 gradient must "
                 f"be exactly zero (the two-step wake-up)")
    # and at birth the emitted channels are at the registered values (P5)
    with torch.no_grad():
        lam_vec, lam_inc = mixer.channel_decay(x)
        assert torch.equal(lam_vec, lam_inc.unsqueeze(-1).expand_as(lam_vec)
                           .to(lam_vec.dtype)), "the per-channel decay must " \
            "be bit-equal to the registered scalar on every channel at birth"
        g_vec = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
            * torch.sigmoid(mixer.b_proj(x)).view(2, 32, mixer.heads,
                                                   mixer.value_dim)
        assert float((g_vec - 1.0).abs().max()) == 0.0
        r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
            * torch.sigmoid(mixer.g_proj(x)).view(2, 32, mixer.heads,
                                                  mixer.value_dim)
        assert float((r_vec - 1.0).abs().max()) == 0.0


@needs_cuda
def test_gate_b_mixer_birth_output_bit_identical_to_k10():
    """The conversion's own plumbing receipt: the p3-converted mixer's output
    at birth is bit-identical to the un-converted K10 mixer's own output
    (the read gate multiplies by exact 1.0; the per-channel decay equals the
    registered scalar per channel)."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)
            self.blocks[0].mixer = convert_model_to_separated_keys(
                self).blocks[0].mixer

    stub = Stub()
    k10_mixer = stub.blocks[0].mixer
    stub = convert_model_to_channel_gates(stub, arm="p3")
    mixer = stub.blocks[0].mixer.to(DEVICE)
    k10_mixer = k10_mixer.to(DEVICE)
    x = torch.randn(2, 64, 64, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_k11b, _ = mixer.mix(x)
        out_k10, _ = k10_mixer.mix(x)
    assert torch.equal(out_k11b, out_k10), \
        "the born mixer must reproduce the K10 mixer's output bit for bit"


@needs_cuda
def test_gate_b_birth_survives_the_clamp_bound_carrier_corner():
    """The falsifier's corner, PINNED PERMANENTLY (receipt 57e1f942): a
    bf16-SATURATING sigmoid position through the PRODUCTION autocast
    expression. The incumbent's clamp bound 1-1e-6 rounds to exactly 1.0 in
    its bf16 carrier (its own effective ceiling); a naive fp32 clamp of the
    promoted per-channel product would preserve 0.999999 and break the
    mimetic-neutral birth -- one such position in 32,768 at the probe's
    layer 4 amplified to a 6.6e-5 step-1 loss delta and the falsifier
    correctly refused the lane. The bounds act in the incumbent's carrier
    dtype, so the born mixer reproduces the K10 mixer bit for bit INCLUDING
    this corner; bit-parity fixtures must go through the production
    expression under autocast, and this test is the standing remedy."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape, DECAY_FLOOR)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)
            self.blocks[0].mixer = convert_model_to_separated_keys(
                self).blocks[0].mixer

    stub = Stub()
    k10_mixer = stub.blocks[0].mixer
    stub = convert_model_to_channel_gates(stub, arm="p1")
    mixer = stub.blocks[0].mixer.to(DEVICE)
    k10_mixer = k10_mixer.to(DEVICE)
    # craft the corner: head 0's decay bias saturates the bf16 sigmoid to
    # EXACTLY 1.0 everywhere; head 1 stays mid-range (the heterogeneous
    # fixture -- the conversion shares the module objects, so one mutation
    # sets both mixers)
    with torch.no_grad():
        k10_mixer.decay.weight.zero_()
        k10_mixer.decay.bias[0].fill_(20.0)
        k10_mixer.decay.bias[1].fill_(0.0)
    x = torch.randn(2, 64, 64, device=DEVICE)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lam_vec, lam_inc = mixer.channel_decay(x)
        decay10 = torch.sigmoid(k10_mixer.decay(x)).clamp(
            min=DECAY_FLOOR, max=1.0 - 1e-6)
    assert lam_inc.dtype == torch.bfloat16, \
        "the production autocast regime must carry the incumbent's dtype"
    assert bool((lam_inc[:, :, 0] == 1.0).all()), \
        "the fixture must saturate head 0's bf16 sigmoid to exactly 1.0"
    assert bool((lam_inc[:, :, 1] == 0.5).all()), \
        "head 1 must stay mid-range (the heterogeneous fixture)"
    assert torch.equal(
        lam_vec,
        decay10.unsqueeze(-1).expand_as(lam_vec).to(lam_vec.dtype)), \
        "the born per-channel decay must equal the incumbent's expression " \
        "value-for-value INCLUDING the saturating corner"
    assert float(lam_vec[:, :, 0].max()) == 1.0, \
        ("the saturating positions must be EXACTLY 1.0 (the incumbent's own "
         "bf16-rounded ceiling), never the naive fp32 clamp's "
         "0.9999989867210388")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out_k11b, _ = mixer.mix(x)
        out_k10, _ = k10_mixer.mix(x)
    assert torch.equal(out_k11b, out_k10), \
        "the born mixer must reproduce the K10 mixer's output bit for bit " \
        "at the saturating corner"


@needs_cuda
def test_gate_b_trajectory_replay_bit_exact():
    """The replay receipt: the lane-specific per-channel trajectory kernel
    rebuilds the forward's own segment checkpoints bit-exactly from the
    2-lane recorded schedule."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 8, 91)
    rank, seg = 5, 16
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    decay32 = decay.float().permute(0, 2, 1, 3).contiguous()
    batch, tokens, heads = 2, 64, 2
    for index, start in enumerate(range(0, tokens, seg)):
        stop = min(start + seg, tokens)
        _, exit_state, exit_live, _traj = channel_persistent_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index],
            decay32[:, :, start:stop, :], schedule.chunk(start, stop))
        if stop == tokens:
            assert torch.equal(exit_state, out["state"]), "the last segment"
            assert torch.equal(exit_live, out["live"])
        else:
            assert torch.equal(exit_state, out["ckpt_state"][index + 1]), \
                f"segment {index} value replay vs the forward's checkpoint"


@needs_cuda
def test_gate_b_pop_terminates_the_row_gradient():
    """K10's boundary re-pinned on the per-channel oracle: a POPped row's
    value gradient terminates at its death token."""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    state, keys, live = crafted_store([e[i] for i in range(4)],
                                      key_rows=[ek[i] for i in range(4)])
    payload = e[4].view(1, 1, 1, V).repeat(1, 16, 1, 1)
    write_keys = ek[4].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    g = torch.Generator().manual_seed(17)
    query_keys = torch.randn(1, 16, 1, K, generator=g).to(DEVICE)
    decay = torch.full((1, 16, 1, V), 0.999, device=DEVICE)
    gates = gate_stream(1, 16, 1, V, 5)
    rank, seg = 4, 8
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, gates,
                            rank=rank, segment_tokens=seg,
                            initial_state=state, initial_live=live,
                            initial_keys=keys)
    death = None
    op0 = out["op"][0, :, 0, 0]
    for t in range(16):
        if int(op0[t]) == 3:                  # POP
            death = t
            break
    assert death is not None and death < 15, "the fixture must POP early"
    upstream = torch.zeros(1, 16, 1, V, device=DEVICE)
    upstream[:, death + 1:] = torch.randn(1, 16 - death - 1, 1, V,
                                          generator=torch.Generator()
                                          .manual_seed(9)).to(DEVICE)
    sched = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                           payload=out["payload"], op=out["op"],
                           live_after=out["live_after"])
    dp, dwk, dqk, dd, dg, dst, dki = _k11b_backward_body(
        payload, write_keys, query_keys, decay, gates, upstream, sched,
        out["live_pre"], out["ckpt_state"], out["ckpt_live"],
        out["ckpt_keys"], rank, seg, POLICY.read_scale, True)
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    g_ = gates.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    k = keys.clone().requires_grad_(True)
    r2 = channel_gated_oracle_scan(p, wk, qk, d, g_, policy=POLICY, rank=rank,
                                   rank_max=rank, initial_state=s,
                                   initial_live=live, initial_keys=k)
    (r2["reads"] * upstream).sum().backward()
    before = slice(0, death)
    assert (dp[:, before].abs().max().item() if death > 0 else 0.0) == 0.0, \
        "a POPped row's payload gradient must terminate at the death token"
    for name, kern, orac in zip(GRAD_NAMES,
                                (dp, dwk, dqk, dd, dg, dst, dki),
                                (p.grad, wk.grad, qk.grad, d.grad, g_.grad,
                                 s.grad, k.grad)):
        deviation = (kern.float() - orac.float()).abs().max().item()
        scale = orac.float().abs().max().item()
        assert deviation <= 1e-4 * max(scale, 1.0) or deviation < 1e-5, \
            (name, deviation, scale)


# ------------------------------------------------------------- the conversion


@needs_cuda
def test_conversion_arm_deltas_and_birth():
    """The registration's 1.5 accounting, computed not hardcoded (GDN2's own
    projection shapes); and P5's birth: every gate channel is born at exact
    identity, the per-channel decay bit-equal to the registered scalar."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys
    width, heads, value_dim = 64, 2, 32

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=width, depth=1, hidden=128, heads=heads,
                value_dim=value_dim, rank=16, rank_max=16), POLICY, 64)

    for arm in ("p1", "p2", "p3"):
        stub = Stub()
        stub = convert_model_to_separated_keys(stub)
        pre = stub.blocks[0].mixer
        pre_params = sum(p.numel() for p in pre.parameters())
        stub = convert_model_to_channel_gates(stub, arm=arm)
        mixer = stub.blocks[0].mixer
        delta = sum(p.numel() for p in mixer.parameters()) - pre_params
        assert delta == arm_parameter_delta(arm, 1, width, heads,
                                            value_dim), \
            (arm, delta, arm_parameter_delta(arm, 1, width, heads, value_dim))
        # the shared modules are the very same objects (the conversion
        # re-wraps THIS mixer's own modules)
        for name in ("value", "key", "query_key", "out", "decay"):
            assert getattr(mixer, name) is getattr(pre, name), name
        # P5's birth, per channel: the decay vector bit-equal to the
        # registered scalar on every channel; the gates at exact 1
        mixer = mixer.to(DEVICE)
        x = torch.randn(2, 8, width, device=DEVICE)
        with torch.no_grad():
            lam_vec, lam_inc = mixer.channel_decay(x)
            assert torch.equal(
                lam_vec, lam_inc.unsqueeze(-1).expand_as(lam_vec)
                .to(lam_vec.dtype)), (arm, "the per-channel decay birth")
            if mixer.channels["write"]:
                g_vec = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.b_proj(x)).view(
                        2, 8, heads, value_dim)
                assert float((g_vec - 1.0).abs().max()) == 0.0
            if mixer.channels["read"]:
                r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.g_proj(x)).view(
                        2, 8, heads, value_dim)
                assert float((r_vec - 1.0).abs().max()) == 0.0


@needs_cuda
def test_conversion_routes_gate_params_to_adamw():
    """The incumbent's own optimizer convention: the bake-off's
    ``split_parameters`` routes the per-channel gate projections to the
    hybrid's AdamW side (the DECAY gate's own side -- "a gate, not a hidden
    map")."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys
    from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
        import split_parameters

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)

    stub = Stub()
    stub = convert_model_to_separated_keys(stub)
    stub = convert_model_to_channel_gates(stub, arm="p3")
    mixer = stub.blocks[0].mixer

    class Holder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].mixer = m

    _muon, _adamw, muon_names, adamw_names = split_parameters(Holder(mixer))
    for name in ("f_proj.0.weight", "f_proj.1.weight", "decay_scalar",
                 "b_proj.weight", "write_scalar",
                 "g_proj.0.weight", "g_proj.1.weight", "g_proj.1.bias",
                 "read_scalar"):
        full = f"blocks.0.mixer.{name}"
        assert full in adamw_names, f"{full} must route to AdamW"
        assert full not in muon_names


# ------------------------------------------------------------- the refusals


@needs_cuda
def test_refusals_are_by_name():
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 3, 5)
    gates = torch.ones(2, 8, 2, 8, device=DEVICE)
    with pytest.raises(ValueError, match="append-only"):
        channel_gated_scan(payload, write_keys, query_keys, decay, gates,
                           policy=RoutingPolicy(ops=False), rank=5)
    with pytest.raises(ValueError, match="growth"):
        channel_gated_scan(payload, write_keys, query_keys, decay, gates,
                           policy=RoutingPolicy(grow=True), rank=5)
    with pytest.raises(ValueError, match="rank == rank_max"):
        channel_gated_scan(payload, write_keys, query_keys, decay, gates,
                           policy=POLICY, rank=4, rank_max=8)
    # the size gate: a store too big for one program takes the oracle verbatim
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import separated_available
    assert not separated_available(2048, 128, 16)
    result = channel_gated_scan(payload, write_keys, query_keys, decay, gates,
                                policy=POLICY, rank=5, rank_max=5)
    ref = oracle_reference(payload, write_keys, query_keys, decay, gates,
                           rank=5, collect=False)
    assert torch.equal(result.reads, ref["reads"])
    assert torch.equal(result.state, ref["state"])
