"""K11 -- the learned-gate arm, gated.

The K10 parity discipline, extended to the gate channel and the dense lane
(``PREREGISTRATION_LEARNED_GATES.md`` 3):

* **Bit-equality by construction**: the eager oracle and the kernels implement
  the identical defined-order expressions (the CORRECT blend's pre-decay form
  ``g*p + (1-g)*lam*v_pre`` included), so every channel -- the 3-lane schedule,
  the payload lanes, the read weights, the state, the liveness, the
  pre-decision liveness, the key table -- must be ``torch.equal`` against the
  oracle on every fixture, in all four GATED/DENSE combinations, including the
  bf16 autocast regime.  A difference is a bug, not a floor.
* **The 0-flip decision-invariance receipt (P2)**: the discrete op sequence
  (lanes 0/1) is bit-identical under ANY gate values and ANY dense deposits
  at fixed inputs.
* **The birth-parity / plumbing receipt (P5)**: at ``g_write == 1``,
  ``g_out == 1``, ``dense == 0`` the k11 scan is bit-identical to K10's OWN
  launch on every channel (the differing-element count is receipted -- the
  only legal difference is the -0.0 IEEE corner).
* **The gradient gates are floor-discipline**: the fp64 oracle gradcheck first
  (every adjoint branch), the fp32-vs-fp64 floor per output, the kernel path's
  EIGHT gradients within ``NOISE_FACTOR`` of it; the zero-born scalars must
  receive gradient at step 0 (P5's no-dead-channel receipt); K10's structural
  boundaries re-pinned.
* **The replay receipt**: K4's imported ``persistent_scan_trajectory`` rebuilds
  the forward's own segment checkpoints bit-exactly from the 3-lane recorded
  schedule, dense lane included.
* **The conversion receipts**: the arm's parameter delta, the zero-born gates
  (the arm is born as the banked winner), and the optimizer routing (the gate
  projections land on the hybrid's AdamW side, the incumbent's own
  "a gate, not a hidden map" convention).

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, OP_NAMES, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    _k10_forward_launch)
from experiments.expressivity_tasks.x45_matrix_emender.k11_learned_gates import (
    LANES, GatedSeparatedMixer, _k11_backward_body, _k11_forward_launch,
    arm_channels, arm_parameter_delta, convert_model_to_gated_writes,
    gated_oracle_scan, gated_scan)
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
    """The lane's canonical fixture (K10's): payload AND write-key values
    drawn from ``distinct`` fixed directions, query keys random."""
    g = torch.Generator().manual_seed(seed)
    payload_dirs = torch.randn(distinct, value_dim, generator=g).to(DEVICE)
    key_dirs = torch.randn(distinct, key_dim, generator=g).to(DEVICE)
    choice = torch.randint(0, distinct, (batch, tokens, heads), generator=g)
    payload = payload_dirs[choice.cpu()].to(DEVICE)
    write_keys = key_dirs[choice.cpu()].to(DEVICE)
    query_keys = (torch.randn(batch, tokens, heads, key_dim, generator=g)
                  .to(DEVICE) * 0.5)
    if isinstance(decay, float):
        decay = torch.full((batch, tokens, heads), decay, device=DEVICE)
    else:
        decay = decay.to(DEVICE)
    return payload, write_keys, query_keys, decay


def gate_stream(batch, tokens, heads, seed):
    """Random gates in (0.05, 0.95) and a random dense payload -- the
    off-identity gate/dense fixture."""
    g = torch.Generator().manual_seed(seed)
    gates = (torch.rand(batch, tokens, heads, generator=g) * 0.9 + 0.05
             ).to(DEVICE)
    return gates


def kernel_forward(payload, dense, write_keys, query_keys, decay, gates, rank,
                   segment_tokens=64, gated=True, use_dense=True,
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
    if dense is None:
        dense = torch.zeros_like(payload)
    if gates is None:
        gates = torch.ones(batch, tokens, heads, device=DEVICE)
    return _k11_forward_launch(payload, dense, write_keys, query_keys, decay,
                                gates, state, live, keys, POL32, rank,
                                segment_tokens, gated, use_dense,
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
    """Gate A: every decision channel, the payload lanes (3), the weights,
    the state, the liveness, the pre-decision liveness AND the key table
    bit-equal to the oracle."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(ref["schedule"], name), out[name]), \
            (label, name)
    assert torch.equal(ref["schedule"].payload, out["payload"]), (label, "payload")
    assert torch.equal(ref["weights"], out["weights"]), (label, "weights")
    assert torch.equal(ref["state"], out["state"]), (label, "state")
    assert torch.equal(ref["live"], out["live"]), (label, "live")
    assert torch.equal(ref["live_pre"], out["live_pre"]), (label, "live_pre")
    assert torch.equal(ref["keys"], out["keys"]), (label, "keys")
    assert out["slot"].shape[-1] == LANES


def oracle_reference(payload, dense, write_keys, query_keys, decay, gates,
                     rank, gated=True, use_dense=True, collect=True,
                     initial_state=None, initial_live=None, initial_keys=None):
    return gated_oracle_scan(payload, dense, write_keys, query_keys, decay,
                             gates, policy=POLICY, gated=gated,
                             use_dense=use_dense, rank=rank, rank_max=rank,
                             initial_state=initial_state,
                             initial_live=initial_live,
                             initial_keys=initial_keys, collect=collect)


# --------------------------------------------------------------------- gate A


@needs_cuda
@pytest.mark.parametrize("gated,use_dense", [(True, True), (True, False),
                                            (False, True), (False, False)])
def test_gate_a_identity_regime_bit_equal(gated, use_dense):
    """The canonical fixture, all four GATED/DENSE combinations."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 91)
    dense = torch.randn(2, 64, 2, 8, generator=torch.Generator()
                        .manual_seed(7)).to(DEVICE) * 0.05
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=5, segment_tokens=16, gated=gated,
                            use_dense=use_dense)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5, gated=gated, use_dense=use_dense)
    assert_decisions_bit_equal(out, ref, f"identity-{gated}-{use_dense}")
    hist = op_histogram(ref["schedule"])
    assert hist["correct"] > 0 and hist["append"] > 0, hist


@needs_cuda
def test_gate_a_merge_and_pop_reachable():
    """Crafted KEY tables force MERGE and POP; the gate channel and the dense
    lane ride along bit-exactly."""
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
    decay = torch.full((1, 16, 1), 0.999, device=DEVICE)
    gates = gate_stream(1, 16, 1, 5)
    dense = torch.randn(1, 16, 1, V, generator=g).to(DEVICE) * 0.05
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=4, segment_tokens=8, initial_state=state,
                            initial_live=live, initial_keys=keys)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, initial_state=state,
                           initial_live=live, initial_keys=keys)
    assert_decisions_bit_equal(out, ref, "merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0, f"the fixture must reach MERGE: {hist}"
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
    assert_decisions_bit_equal(out2, ref2, "pop")
    hist2 = op_histogram(ref2["schedule"])
    assert hist2["pop"] > 0, f"the fixture must reach POP: {hist2}"
    # the dense lane never wrote a dead row (P3): dead rows are exactly zero
    dead = ~out2["live"][0, 0]
    if bool(dead.any()):
        assert (out2["state"][0, 0][dead].abs().max().item()
                if dead.any() else 0.0) == 0.0


@needs_cuda
def test_gate_a_dense_lane_targets_the_best_match_live_row():
    """P3, receipted: the deposit lands on the argmax-live row (recorded in
    lane 2's slot), never on a dead row, and never when no live row existed at
    decision time."""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    state, keys, live = crafted_store([e[0], e[1], e[2], e[3]],
                                     key_rows=[ek[0], ek[1], ek[2], ek[3]])
    payload = e[4].view(1, 1, 1, V).repeat(1, 16, 1, 1) * 0.5
    write_keys = ek[4].view(1, 1, 1, K).repeat(1, 16, 1, 1)  # matches nothing
    query_keys = torch.zeros(1, 16, 1, K, device=DEVICE)
    decay = torch.full((1, 16, 1), 0.999, device=DEVICE)
    gates = torch.ones(1, 16, 1, device=DEVICE)
    dense = e[5].view(1, 1, 1, V).to(DEVICE).repeat(1, 16, 1, 1) * 0.1
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=4, segment_tokens=16,
                            gated=False, use_dense=True,
                            initial_state=state, initial_live=live,
                            initial_keys=keys)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=4, gated=False,
                           initial_state=state, initial_live=live,
                           initial_keys=keys)
    assert_decisions_bit_equal(out, ref, "dense-target")
    # every fired deposit's target is live at decision time
    fired = out["op"][..., 2] != NONE
    live_pre = out["live_pre"]
    targets = out["slot"][..., 2]
    fired_idx = fired.nonzero()
    for idx in fired_idx[:64]:
        b, t, h = idx.tolist()
        assert bool(live_pre[b, t, h, targets[b, t, h]]), \
            "the dense lane wrote a row that was not live at decision time"


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23,
                                                          decay=0.9)
    gates = gate_stream(2, 96, 2, 13)
    dense = torch.randn(2, 96, 2, 8, generator=torch.Generator()
                        .manual_seed(3)).to(DEVICE) * 0.05
    for seg in (64, 7):
        out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                                gates, rank=5, segment_tokens=seg)
        ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                               gates, rank=5)
        assert_decisions_bit_equal(out, ref, f"ragged-{seg}")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    """The registered regime: bf16 inputs, fp32 defined-order math."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    gates = gate_stream(2, 64, 2, 91).bfloat16()
    dense = (torch.randn(2, 64, 2, 8, generator=torch.Generator()
                         .manual_seed(7)).to(DEVICE) * 0.05).bfloat16()
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=5, segment_tokens=16)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5)
    assert_decisions_bit_equal(out, ref, "bf16")


@needs_cuda
def test_gate_a_all_dead_read_is_exactly_zero():
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 4, 3)
    gates = gate_stream(2, 8, 2, 5)
    dense = torch.zeros_like(payload)
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=5, segment_tokens=4)
    assert torch.equal(out["reads"][:, :1], torch.zeros_like(out["reads"][:, :1]))
    # and the first token's deposit went nowhere (no live row at decision time)
    assert bool((out["op"][:, 0, :, 2] == NONE).all())


@needs_cuda
def test_gate_a_determinism():
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 91)
    dense = torch.randn(2, 64, 2, 8, generator=torch.Generator()
                        .manual_seed(7)).to(DEVICE) * 0.05
    out1, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                             gates, rank=5, segment_tokens=16)
    out2, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                             gates, rank=5, segment_tokens=16)
    for name in ("reads", "state", "keys", "slot", "op", "payload", "weights",
                 "live_pre"):
        assert torch.equal(out1[name], out2[name]), name


@needs_cuda
def test_gate_a_registered_shape_flip_test():
    """The decision flip test at a registered-class shape (bf16): 0 flips,
    kernel vs oracle, gates and dense lane active."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 512, 2, 128, 16,
                                                          512, 77)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    gates = gate_stream(2, 512, 2, 55).bfloat16()
    dense = (torch.randn(2, 512, 2, 128, generator=torch.Generator()
                         .manual_seed(9)).to(DEVICE) * 0.05).bfloat16()
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=128, segment_tokens=64)
    with torch.no_grad():
        ref = gated_oracle_scan(payload, dense, write_keys, query_keys,
                                decay, gates, policy=POLICY, rank=128,
                                rank_max=128, gated=True, use_dense=True)
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
    under ANY gate values and ANY dense deposits at fixed inputs."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 16, 8, 6,
                                                          41)
    ones = torch.ones(2, 128, 2, device=DEVICE)
    zero_dense = torch.zeros_like(payload)
    base, _ = kernel_forward(payload, zero_dense, write_keys, query_keys,
                             decay, ones, rank=8, segment_tokens=32)
    for seed in (1, 2, 3):
        g = torch.Generator().manual_seed(seed)
        gates = (torch.rand(2, 128, 2, generator=g) * 0.95 + 0.025).to(DEVICE)
        dense = torch.randn(2, 128, 2, 16, generator=g).to(DEVICE)
        vary, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                                 gates, rank=8, segment_tokens=32)
        for name in SCHEDULE_FIELDS:
            assert torch.equal(base[name][..., :2], vary[name][..., :2]), \
                (seed, name)


@needs_cuda
def test_gate_a_birth_parity_receipt():
    """P5's plumbing receipt: at g_write == 1, dense == 0 the k11 scan is
    bit-identical to K10's OWN launch on every channel (the -0.0 corner is
    the only legal difference; its count is asserted zero on this fixture)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 16, 8, 6,
                                                          43)
    ones = torch.ones(2, 128, 2, device=DEVICE)
    zero_dense = torch.zeros_like(payload)
    k10 = _k10_forward_launch(payload, write_keys, query_keys, decay,
                              torch.zeros(2, 2, 8, 16, dtype=torch.float32,
                                          device=DEVICE),
                              torch.zeros(2, 2, 8, dtype=torch.bool,
                                          device=DEVICE),
                              torch.zeros(2, 2, 8, 8, dtype=torch.float32,
                                          device=DEVICE),
                              POL32, 8, 16, collect=True, warps=4, stages=3)
    k11, _ = kernel_forward(payload, zero_dense, write_keys, query_keys,
                            decay, ones, rank=8, segment_tokens=16)
    for name in SCHEDULE_FIELDS:
        assert torch.equal(k10[name], k11[name][..., :2]), name
    assert torch.equal(k10["payload"], k11["payload"][..., :2, :]), "payload"
    for name in ("reads", "state", "live", "live_pre", "weights", "keys"):
        assert torch.equal(k10[name], k11[name]), name
    # and the same receipt through the GATED=0/DENSE=0 configuration
    k11b, _ = kernel_forward(payload, zero_dense, write_keys, query_keys,
                            decay, ones, rank=8, segment_tokens=16,
                            gated=False, use_dense=False)
    assert torch.equal(k10["state"], k11b["state"])
    assert torch.equal(k10["reads"], k11b["reads"])


# --------------------------------------------------------------------- gate B


def _oracle_grads(payload, dense, write_keys, query_keys, decay, gates, rank,
                  dtype=torch.float32, state_grad=True, keys_grad=True,
                  gated=True, use_dense=True):
    p = payload.clone().requires_grad_(True)
    dn = dense.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    g_ = gates.clone().requires_grad_(True)
    s = torch.zeros(payload.shape[0], payload.shape[2], rank, payload.shape[-1],
                    dtype=dtype, device=DEVICE).requires_grad_(state_grad)
    k = torch.zeros(payload.shape[0], payload.shape[2], rank,
                    write_keys.shape[-1], dtype=dtype,
                    device=DEVICE).requires_grad_(keys_grad)
    ref = gated_oracle_scan(p.to(dtype), dn.to(dtype), wk.to(dtype),
                            qk.to(dtype), d.to(dtype), g_.to(dtype),
                            policy=POLICY, gated=gated, use_dense=use_dense,
                            rank=rank, rank_max=rank, initial_state=s,
                            initial_keys=k)
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(dtype).to(DEVICE)
    (ref["reads"] * upstream).sum().backward()
    return ref, (p.grad, dn.grad, wk.grad, qk.grad, d.grad, g_.grad,
                 s.grad if state_grad else None, k.grad if keys_grad else None)


GRAD_NAMES = ("dpayload", "ddense", "dwrite_keys", "dquery_keys", "ddecay",
              "dgates", "dstate", "dkeys")


@needs_cuda
def test_gate_b_oracle_gradcheck_fp64():
    """The oracle validates before it is trusted: an fp64 gradcheck through
    the payload, the dense lane, the GATES, the key channel, the value state
    and the key table, both GATED/DENSE combinations."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 32, 2, 8, 4, 5, 11)
    gates = gate_stream(2, 32, 2, 17)
    dense = torch.randn(2, 32, 2, 8, generator=torch.Generator()
                        .manual_seed(19)).to(DEVICE) * 0.1
    for gated, use_dense in [(True, True), (False, True)]:
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
            ref = gated_oracle_scan(p_, dn_, wk_, qk_, d_, g__, policy=POLICY,
                                    gated=gated, use_dense=use_dense, rank=5,
                                    rank_max=5, initial_state=s_,
                                    initial_keys=k_)
            hist_holder["hist"] = op_histogram(ref["schedule"])
            return ref["reads"], ref["state"], ref["keys"]

        assert torch.autograd.gradcheck(f, (p, dn, wk, qk, d, g_, s, k),
                                        eps=1e-6, atol=1e-4, rtol=1e-4)
        hist = hist_holder["hist"]
        assert all(hist[n] > 0 for n in ("append", "correct")), hist


@needs_cuda
def test_gate_b_grads_within_measured_floors():
    """The kernel path's eight gradients vs the oracle's autograd, inside
    NOISE_FACTOR of the fp32-vs-fp64 floor measured on the same fixture."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 91)
    dense = torch.randn(2, 64, 2, 8, generator=torch.Generator()
                        .manual_seed(7)).to(DEVICE) * 0.05
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
    grads = _k11_backward_body(payload, dense, write_keys, query_keys, decay,
                               gates, upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank, seg, POLICY.read_scale,
                               True, True)
    for name, kern, orac, orac64 in zip(GRAD_NAMES, grads, oracle_grads,
                                        oracle_grads64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= NOISE_FACTOR * max(floor, 1e-6) or \
            deviation <= 1e-5, (name, deviation, floor)
        assert torch.isfinite(kern).all(), name


@needs_cuda
def test_gate_b_dense_gradient_reaches_the_zero_born_scalar():
    """P5's no-dead-channel receipt: at the birth configuration (dense == 0
    exactly) the dense lane's adjoint is NONZERO -- the zero-born scalar
    receives gradient at step 0 through the opaque op pair."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
                import convert_model_to_separated_keys
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=64, depth=1, hidden=128, heads=2, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)
            self.blocks[0].mixer = convert_model_to_separated_keys(
                self).blocks[0].mixer

    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys
    stub = Stub()
    stub = convert_model_to_gated_writes(stub, arm="g2")
    mixer = stub.blocks[0].mixer.to(DEVICE)
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = mixer.mix(x)
    out.float().pow(2).sum().backward()
    assert float(mixer.dense_scalar.grad.abs().sum()) > 0.0, \
        "the zero-born dense scalar must receive gradient at birth"
    # and at birth the emitted dense payload is exactly zero (P5)
    with torch.no_grad():
        payload = mixer.value(x).view(2, 32, mixer.heads, mixer.value_dim)
        dense = mixer.dense_scalar.view(1, 1, -1, 1) \
            * torch.sigmoid(mixer.dense_gate(x)).unsqueeze(-1) * payload
    assert float(dense.abs().max()) == 0.0


@needs_cuda
def test_gate_b_trajectory_replay_bit_exact():
    """The replay receipt: K4's imported kernel rebuilds the forward's own
    segment checkpoints bit-exactly from the 3-lane recorded schedule, dense
    lane included."""
    from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward \
        import persistent_scan_trajectory
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    gates = gate_stream(2, 64, 2, 91)
    dense = torch.randn(2, 64, 2, 8, generator=torch.Generator()
                        .manual_seed(7)).to(DEVICE) * 0.05
    rank, seg = 5, 16
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    decay32 = decay.float().permute(0, 2, 1).contiguous()
    batch, tokens = 2, 64
    heads = 2
    for index, start in enumerate(range(0, tokens, seg)):
        stop = min(start + seg, tokens)
        zeros_w = torch.zeros(batch, heads, stop - start, rank,
                              dtype=torch.float32, device=DEVICE)
        _, exit_state, exit_live, _traj = persistent_scan_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index], zeros_w,
            decay32[:, :, start:stop], schedule.chunk(start, stop))
        if stop == tokens:
            assert torch.equal(exit_state, out["state"]), "the last segment"
            assert torch.equal(exit_live, out["live"])
        else:
            assert torch.equal(exit_state, out["ckpt_state"][index + 1]), \
                f"segment {index} value replay vs the forward's checkpoint"


@needs_cuda
def test_gate_b_pop_terminates_the_row_gradient():
    """K10's boundary re-pinned on the gated oracle: a POPped row's value
    gradient terminates at its death token."""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    state, keys, live = crafted_store([e[i] for i in range(4)],
                                     key_rows=[ek[i] for i in range(4)])
    payload = e[4].view(1, 1, 1, V).repeat(1, 16, 1, 1)
    write_keys = ek[4].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    g = torch.Generator().manual_seed(17)
    query_keys = torch.randn(1, 16, 1, K, generator=g).to(DEVICE)
    decay = torch.full((1, 16, 1), 0.999, device=DEVICE)
    gates = gate_stream(1, 16, 1, 5)
    dense = torch.randn(1, 16, 1, V, generator=g).to(DEVICE) * 0.05
    rank, seg = 4, 8
    out, _ = kernel_forward(payload, dense, write_keys, query_keys, decay,
                            gates, rank=rank, segment_tokens=seg,
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
    dp, ddn, dwk, dqk, dd, dg, dst, dki = _k11_backward_body(
        payload, dense, write_keys, query_keys, decay, gates, upstream, sched,
        out["live_pre"], out["ckpt_state"], out["ckpt_live"],
        out["ckpt_keys"], rank, seg, POLICY.read_scale, True, True)
    p = payload.clone().requires_grad_(True)
    dn = dense.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    g_ = gates.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    k = keys.clone().requires_grad_(True)
    r2 = gated_oracle_scan(p, dn, wk, qk, d, g_, policy=POLICY, rank=rank,
                           rank_max=rank, initial_state=s, initial_live=live,
                           initial_keys=k)
    (r2["reads"] * upstream).sum().backward()
    before = slice(0, death)
    assert (dp[:, before].abs().max().item() if death > 0 else 0.0) == 0.0, \
        "a POPped row's payload gradient must terminate at the death token"
    for name, kern, orac in zip(GRAD_NAMES,
                                (dp, ddn, dwk, dqk, dd, dg, dst, dki),
                                (p.grad, dn.grad, wk.grad, qk.grad, d.grad,
                                 g_.grad, s.grad, k.grad)):
        deviation = (kern.float() - orac.float()).abs().max().item()
        scale = orac.float().abs().max().item()
        assert deviation <= 1e-4 * max(scale, 1.0) or deviation < 1e-5, \
            (name, deviation, scale)


# ------------------------------------------------------------- the conversion


@needs_cuda
def test_conversion_arm_deltas_and_birth():
    """The registration's 1.5 accounting, computed not hardcoded; and P5's
    birth: every gate channel is born at exact identity/zero."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import convert_model_to_separated_keys
    width, heads = 64, 2

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(MatrixShape(
                width=width, depth=1, hidden=128, heads=heads, value_dim=32,
                rank=16, rank_max=16), POLICY, 64)

    for arm in ("g1", "g2", "g3"):
        stub = Stub()
        stub = convert_model_to_separated_keys(stub)
        pre = stub.blocks[0].mixer
        pre_params = sum(p.numel() for p in pre.parameters())
        stub = convert_model_to_gated_writes(stub, arm=arm)
        mixer = stub.blocks[0].mixer
        delta = sum(p.numel() for p in mixer.parameters()) - pre_params
        assert delta == arm_parameter_delta(arm, 1, width, heads), \
            (arm, delta, arm_parameter_delta(arm, 1, width, heads))
        # the shared modules are the very same objects (the conversion
        # re-wraps THIS mixer's own modules)
        for name in ("value", "key", "query_key", "out", "decay"):
            assert getattr(mixer, name) is getattr(pre, name), name
        # P5's birth: write gate exactly 1, readout exactly 1, dense exactly 0
        mixer = mixer.to(DEVICE)
        x = torch.randn(2, 8, width, device=DEVICE)
        with torch.no_grad():
            if mixer.channels["write"]:
                gates = 1.0 - mixer.write_gate_scalar * torch.sigmoid(
                    mixer.write_gate(x))
                assert float((gates - 1.0).abs().max()) == 0.0
            if mixer.channels["readout"]:
                g_out = 1.0 + mixer.readout_scalar * torch.sigmoid(
                    mixer.readout_gate(x))
                assert float((g_out - 1.0).abs().max()) == 0.0
            if mixer.channels["dense"]:
                payload = mixer.value(x).view(2, 8, heads, 32)
                dense = mixer.dense_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.dense_gate(x)).unsqueeze(-1) * payload
                assert float(dense.abs().max()) == 0.0


@needs_cuda
def test_conversion_routes_gate_params_to_adamw():
    """The incumbent's own optimizer convention: the bake-off's
    ``split_parameters`` routes the gate projections to the hybrid's AdamW
    side (the DECAY gate's own side -- "a gate, not a hidden map")."""
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
    stub = convert_model_to_gated_writes(stub, arm="g3")
    mixer = stub.blocks[0].mixer

    class Holder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Module()])
            self.blocks[0].mixer = m

    _muon, _adamw, muon_names, adamw_names = split_parameters(Holder(mixer))
    for name in ("write_gate.weight", "write_gate.bias",
                 "readout_gate.weight", "readout_gate.bias",
                 "dense_gate.weight", "dense_gate.bias",
                 "write_gate_scalar", "readout_scalar", "dense_scalar"):
        full = f"blocks.0.mixer.{name}"
        assert full in adamw_names, f"{full} must route to AdamW"
        assert full not in muon_names


# ------------------------------------------------------------- the refusals


@needs_cuda
def test_refusals_are_by_name():
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 3, 5)
    gates = torch.ones(2, 8, 2, device=DEVICE)
    dense = torch.zeros_like(payload)
    with pytest.raises(ValueError, match="append-only"):
        gated_scan(payload, dense, write_keys, query_keys, decay, gates,
                   policy=RoutingPolicy(ops=False), rank=5)
    with pytest.raises(ValueError, match="growth"):
        gated_scan(payload, dense, write_keys, query_keys, decay, gates,
                   policy=RoutingPolicy(grow=True), rank=5)
    with pytest.raises(ValueError, match="rank == rank_max"):
        gated_scan(payload, dense, write_keys, query_keys, decay, gates,
                   policy=POLICY, rank=4, rank_max=8)
    # the size gate: a store too big for one program takes the oracle verbatim
    from experiments.expressivity_tasks.x45_matrix_emender.k11_learned_gates \
        import separated_available
    assert not separated_available(2048, 128, 16)
    result = gated_scan(payload, dense, write_keys, query_keys, decay, gates,
                       policy=POLICY, rank=5, rank_max=5)
    ref = oracle_reference(payload, dense, write_keys, query_keys, decay,
                           gates, rank=5, collect=False)
    assert torch.equal(result.reads, ref["reads"])
    assert torch.equal(result.state, ref["state"])
