"""K13 -- the additive read, gated.

The K10 parity discipline, extended to the read-form swap
(``PREREGISTRATION_ADDITIVE_READ.md``, committed before any probe
measurement):

* **Bit-equality by construction**: the K10 defined-order spec is untouched
  (the score channel is the banked read's own VERBATIM); the eager oracle
  implements the additive read in torch and the kernel in Triton, so every
  decision channel -- slot, op, erase, live_after, the payload lane (the
  MERGE blend included), the read WEIGHTS (the raw scores, live-gated), the
  state, the liveness, the pre-decision liveness AND the key table -- must
  be ``torch.equal`` against the oracle on every fixture, including the bf16
  autocast regime (the K11B lesson: bit-parity fixtures go through the
  production expression under autocast).  A difference is a bug, not a floor.
* **The read is the one channel at the measured floor** (its rank-axis
  reduction is a native sum, the K1/K23 precedent).
* **THE DECISIONS-BIT-IDENTICAL receipt** (the machine-variant's own birth
  receipt, replacing the mimetic-neutral 0-flip bar a different function
  cannot carry): at fixed inputs the additive read's recorded schedule is
  bit-identical to the BANKED read's (K10's own launch) -- the read form
  cannot feed op selection, and this pins it engine-side.
* **The gradient gates are floor-discipline**: the fp64 ORACLE validates
  itself first -- a gradcheck THROUGH THE ADDITIVE READ PATH over every
  adjoint branch (the read's direct term into the state, the score path into
  the query keys and the key table, the lanes, DECAY) -- then the kernel
  path's SIX gradients sit within ``NOISE_FACTOR`` of the measured
  fp32-vs-fp64 floor.
* **The g_proj read gate is born at exact identity**: ``r_vec == 1``
  bit for bit at birth, so the arm is the PURE additive read (the
  registration's 1, ruling R13-3).
* **The refusals are by name** and take the EAGER ORACLE (the same
  semantics), or raise outright for the chunk-dial concepts this
  registration does not carry (ops=False, grow, a rank prefix).
* **Determinism**: no atomics anywhere; two runs bit-identical.

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    _k10_forward_launch, convert_model_to_separated_keys)
from experiments.expressivity_tasks.x45_matrix_emender.k13_additive_read import (
    AdditiveReadMixer, _k13_backward_body, _k13_forward_launch,
    additive_oracle_scan, additive_scan, convert_model_to_additive_read,
    read_gate_parameter_delta)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy)
from experiments.expressivity_tasks.x45_matrix_emender.test_k10_separated_keys import (
    DEVICE, crafted_store, keyed_stream)

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
NOISE_FACTOR = 8.0
SCHEDULE_FIELDS = ("slot", "erase", "op", "live_after")
POLICY = RoutingPolicy(ops=True)
POL32 = (_round_f32(POLICY.tau_match), _round_f32(POLICY.tau_merge),
         _round_f32(POLICY.tau_pop), _round_f32(POLICY.tie_break),
         _round_f32(POLICY.read_scale))


def kernel_forward(payload, write_keys, query_keys, decay, rank,
                   key_dim=None, segment_tokens=64, collect=True,
                   initial_state=None, initial_live=None, initial_keys=None):
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1] if key_dim is None else key_dim
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                         device=DEVICE) if initial_state is None
             else initial_state.contiguous())
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
            if initial_live is None else initial_live.contiguous())
    keys = (torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                        device=DEVICE) if initial_keys is None
            else initial_keys.contiguous())
    return _k13_forward_launch(payload, write_keys, query_keys, decay, state,
                               live, keys, POL32, rank, segment_tokens,
                               collect=collect,
                               warps=4 if rank * value_dim < 8192 else 8,
                               stages=3), state


def oracle_reference(payload, write_keys, query_keys, decay, rank,
                     collect=True, initial_state=None, initial_live=None,
                     initial_keys=None):
    return additive_oracle_scan(payload, write_keys, query_keys,
                                decay, policy=POLICY, rank=rank,
                                rank_max=rank, collect=collect,
                                initial_state=initial_state,
                                initial_live=initial_live,
                                initial_keys=initial_keys)


def assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, rank, label):
    """Gate A: every decision channel, the payload lane, the weights (the raw
    scores), the state, the liveness, the pre-decision liveness AND the key
    table bit-equal to the oracle; the read within the measured floor."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(ref["schedule"], name), out[name]), \
            (label, name)
    assert torch.equal(ref["schedule"].payload, out["payload"]), (label, "payload")
    assert torch.equal(ref["weights"], out["weights"]), (label, "weights")
    assert torch.equal(ref["state"], out["state"]), (label, "state")
    assert torch.equal(ref["live"], out["live"]), (label, "live")
    assert torch.equal(ref["live_pre"], out["live_pre"]), (label, "live_pre")
    assert torch.equal(ref["keys"], out["keys"]), (label, "keys")
    # the read: the fp32 floor measured on the same fixture (fp32 vs fp64
    # oracle), the kernel within NOISE_FACTOR of it.
    with torch.no_grad():
        ref64 = additive_oracle_scan(
            payload.double(), write_keys.double(), query_keys.double(),
            decay.double(), policy=POLICY, rank=rank, rank_max=rank)
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    assert deviation <= max(NOISE_FACTOR * max(floor, 1e-30), 1e-6), \
        (label, "read", deviation, floor)


# --------------------------------------------------------------------- gate A


@needs_cuda
def test_gate_a_all_five_ops_regime_bit_equal():
    """The lane's canonical fixture: repeated payload+key directions, the
    identity regime -- every channel bit-equal, and the read weights carry
    the RAW scores (the signed-weight signature of the additive form)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                            segment_tokens=16)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 5)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 5, "identity-regime")
    hist = op_histogram(ref["schedule"])
    assert hist["correct"] > 0 and hist["append"] > 0, hist
    # the additive signature: the weights are the raw signed scores, bounded
    # by the gain on live rows, exactly zero on dead rows, and NOT a simplex
    # (they do not sum to 1).
    w = out["weights"]
    live_pre = out["live_pre"]
    assert (w[live_pre].abs() <= POLICY.read_scale + 1e-6).all()
    assert (w[~live_pre] == 0).all()
    assert (w[live_pre] < 0).any(), \
        "the additive read's weights must carry negative similarities"
    assert not torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1))), \
        "the additive weights are raw scores, not a probability simplex"


@needs_cuda
def test_gate_a_merge_and_pop_reachable():
    """Crafted KEY tables force MERGE and POP -- the value channel's content
    is irrelevant to the decisions now (K10 verbatim)."""
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
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=4,
                            segment_tokens=8, initial_state=state,
                            initial_live=live, initial_keys=keys)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 4,
                           initial_state=state, initial_live=live,
                           initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 4, "merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0, f"the fixture must reach MERGE: {hist}"
    assert (out["keys"][..., 1, :] == 0).all()
    state2, keys2, live2 = crafted_store([e[i] for i in range(4)],
                                         key_rows=[ek[i] for i in range(4)])
    write_keys2 = ek[4].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    out2, _ = kernel_forward(payload, write_keys2, query_keys, decay, rank=4,
                             segment_tokens=8, initial_state=state2,
                             initial_live=live2, initial_keys=keys2)
    ref2 = oracle_reference(payload, write_keys2, query_keys, decay, 4,
                            initial_state=state2, initial_live=live2,
                            initial_keys=keys2)
    assert_decisions_bit_equal(out2, ref2, payload, write_keys2, query_keys,
                               decay, 4, "pop")
    hist2 = op_histogram(ref2["schedule"])
    assert hist2["pop"] > 0, f"the fixture must reach POP: {hist2}"


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    """T not a multiple of the segment length, and of the block width."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23,
                                                          decay=0.9)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                            segment_tokens=64)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 5)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 5, "ragged-segments")
    out2, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                             segment_tokens=7)
    assert_decisions_bit_equal(out2, ref, payload, write_keys, query_keys,
                               decay, 5, "ragged-segments-7")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    """The registered regime: bf16 inputs, fp32 defined-order math (the K11B
    lesson: parity fixtures through the production expression)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                            segment_tokens=16)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 5)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 5, "bf16")


@needs_cuda
def test_gate_a_edge_shapes():
    """One token, one head, one rank: the degenerate corners."""
    for batch, tokens, heads, rank, V, K in [(1, 1, 1, 1, 8, 4),
                                             (1, 3, 1, 2, 8, 4),
                                             (2, 65, 1, 16, 16, 16)]:
        payload, write_keys, query_keys, decay = keyed_stream(
            batch, tokens, heads, V, K, 3, 7, decay=0.9)
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                rank=rank, segment_tokens=5)
        ref = oracle_reference(payload, write_keys, query_keys, decay, rank)
        assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                                   decay, rank,
                                   f"edge-{batch}-{tokens}-{rank}-{K}")


@needs_cuda
def test_gate_a_all_dead_read_is_exactly_zero():
    """The all-dead rule, unchanged by the form swap: an empty store reads 0
    (the raw-score weights are zero on every row)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 4, 3)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                            segment_tokens=4)
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
    decay = torch.full((1, 32, 1), 0.999, device=DEVICE)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=4,
                            segment_tokens=16, initial_state=state,
                            initial_live=live, initial_keys=keys)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 4,
                           initial_state=state, initial_live=live,
                           initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 4, "ties")
    correct = (ref["schedule"].op[..., 0] != NONE)
    assert correct.any()
    assert (ref["schedule"].slot[..., 0][correct] == 0).all(), \
        "a tie must resolve to the lowest index in both implementations"


@needs_cuda
def test_gate_a_keys_not_decayed_and_refiled():
    """The K10 semantic carried verbatim: keys do not age."""
    payload, write_keys, query_keys, decay = keyed_stream(1, 32, 1, 8, 4, 3, 11,
                                                          decay=0.5)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=4,
                            segment_tokens=8)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 4)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 4, "undecayed-keys")
    assert (out["keys"] != 0).any()


@needs_cuda
def test_gate_a_registered_shape_flip_test():
    """The decision flip test at a registered-class shape (bf16): 0 flips."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 512, 2, 128, 16,
                                                          512, 77)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=128,
                            segment_tokens=64)
    with torch.no_grad():
        ref = additive_oracle_scan(payload, write_keys, query_keys,
                                   decay, policy=POLICY, rank=128, rank_max=128)
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
def test_gate_a_determinism():
    """No atomics: two kernel runs bit-identical."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    out1, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                             segment_tokens=16)
    out2, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                             segment_tokens=16)
    for name in ("reads", "state", "keys", "slot", "op", "payload", "weights",
                 "live_pre"):
        assert torch.equal(out1[name], out2[name]), name


# ------------------------------------------- the machine-variant's own receipts


@needs_cuda
def test_decisions_bit_identical_to_the_banked_read():
    """THE BIRTH RECEIPT (the machine-variant's 0-flip analog): at fixed
    inputs the additive read's recorded schedule -- every slot, op, erase,
    live_after, payload lane and the pre-decision liveness -- is BIT-IDENTICAL
    to the BANKED read's own kernel launch (K10's), while the weights and the
    reads differ as designed.  The read form cannot feed op selection; this
    pins it engine-side, against the committed banked kernel itself."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23)
    out_add, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                                segment_tokens=16)
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    state = torch.zeros(batch, heads, 5, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, 5, dtype=torch.bool, device=DEVICE)
    keys = torch.zeros(batch, heads, 5, key_dim, dtype=torch.float32,
                       device=DEVICE)
    out_bank, _ = _k10_forward_launch(payload, write_keys, query_keys, decay,
                                      state, live, keys, POL32, 5, 16,
                                      collect=True, warps=4, stages=3), state
    for name in SCHEDULE_FIELDS:
        assert torch.equal(out_bank[name], out_add[name]), name
    assert torch.equal(out_bank["payload"], out_add["payload"]), "payload"
    assert torch.equal(out_bank["live_pre"], out_add["live_pre"]), "live_pre"
    assert torch.equal(out_bank["state"], out_add["state"]), "state"
    assert torch.equal(out_bank["live"], out_add["live"]), "live"
    assert torch.equal(out_bank["keys"], out_add["keys"]), "keys"
    # the read DID change: different weights, different reads (a different
    # function -- no birth parity, by registration)
    assert not torch.equal(out_bank["weights"], out_add["weights"])
    assert not torch.equal(out_bank["reads"], out_add["reads"])


@needs_cuda
def test_additive_read_magnitude_carries_the_rows():
    """The |w|-mass instrument's engine-side fact (ruling R13-4): the
    effective-rows instrument is defined on the |w|-mass distribution because
    the additive weights are signed; and the additive read spreads mass over
    MANY rows where the banked softmax concentrates (the raw score has no
    simplex to dilute)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 96, 2, 8, 4, 6, 23)
    out_add, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                                segment_tokens=16)
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    state = torch.zeros(batch, heads, 5, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, 5, dtype=torch.bool, device=DEVICE)
    keys = torch.zeros(batch, heads, 5, key_dim, dtype=torch.float32,
                       device=DEVICE)
    out_bank = _k10_forward_launch(payload, write_keys, query_keys, decay,
                                     state, live, keys, POL32, 5, 16,
                                     collect=True, warps=4, stages=3)
    w_add = out_add["weights"].abs()
    w_bank = out_bank["weights"].abs()
    # the |w|-mass effective rows (R13-4), vs the banked entropy's own rows
    # on the same fixture: the additive read must not be MORE concentrated.
    def eff_rows(w, live_pre):
        mass = w.sum(-1).clamp_min(1e-30)
        probs = w / mass.unsqueeze(-1)
        contrib = torch.where(probs > 0, probs * probs.log(),
                              torch.zeros_like(probs))
        has = live_pre.any(-1) & (mass > 0)
        return torch.exp(-contrib.sum(-1))[has].mean().item()
    rows_add = eff_rows(w_add, out_add["live_pre"])
    rows_bank = eff_rows(w_bank, out_add["live_pre"])
    assert rows_add > rows_bank, \
        (f"the additive read must not be more concentrated than the banked "
         f"softmax on the same fixture: {rows_add:.3f} vs {rows_bank:.3f}")


@needs_cuda
def test_read_gate_born_at_exact_identity():
    """R13-3: the g_proj gate is born at exact 1 -- at birth the mixer's
    output IS the pure additive read's, bit for bit, and the gate's parameters
    receive gradient at step 0 (the K11 two-step wake-up pattern)."""
    mixer = AdditiveReadMixer(64, 2, 32, 16, 16, 16, POLICY).to(DEVICE)
    assert float(mixer.read_scalar.abs().max()) == 0.0
    for module in mixer.g_proj.modules():
        if isinstance(module, torch.nn.Linear):
            assert float(module.weight.abs().max()) == 0.0
            if module.bias is not None:
                assert float(module.bias.abs().max()) == 0.0
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 32, 64, generator=g).to(DEVICE)
    with torch.no_grad():
        out, result = mixer.mix(x)
        # the pure additive scan's own output through the out projection
        expected = mixer.out(result.reads.reshape(2, 32, 2 * 32))
    assert torch.equal(out, expected), \
        "at birth the gate must be an exact identity (r_vec == 1)"
    # the gate is live at step 0: gradient reaches the zero-born scalar
    out_grad, _result = mixer.mix(x)
    out_grad.sum().backward()
    assert mixer.read_scalar.grad is not None
    assert float(mixer.read_scalar.grad.abs().max()) > 0.0


@needs_cuda
def test_mixer_conversion_swaps_the_read_form():
    """The conversion re-wraps a K10-converted mixer sharing every parameter
    object, adds the zero-born gate, and the parameter delta is the
    registration's 1.5 (the k11b read channel's own count)."""
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
    stub = convert_model_to_additive_read(stub)
    mixer = stub.blocks[0].mixer
    mixer.to(DEVICE)
    assert isinstance(mixer, AdditiveReadMixer)
    after = dict(mixer.named_parameters())
    for name, param in before.items():
        assert after[name] is param, \
            f"the conversion must SHARE the banked parameters: {name}"
    delta = sum(p.numel() for p in mixer.parameters()) \
        - sum(p.numel() for p in before.values())
    assert delta == read_gate_parameter_delta(1, 64, 2, 32)
    assert delta == 64 * 32 + 32 * 2 * 32 + 2 * 32 + 2
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.no_grad():
        out, result = mixer.mix(x)
    assert out.shape == (2, 32, 64)
    assert result.reads.shape == (2, 32, 2, 32)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="expects a K10-converted model"):
        convert_model_to_additive_read(Stub())


@needs_cuda
def test_refusals_are_by_name():
    """ops=False, grow and a rank prefix are refused outright; the wrapper
    takes the EAGER ORACLE verbatim otherwise (same semantics)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 3, 5)
    with pytest.raises(ValueError, match="append-only"):
        additive_scan(payload, write_keys, query_keys, decay,
                      policy=RoutingPolicy(ops=False), rank=5)
    with pytest.raises(ValueError, match="growth"):
        additive_scan(payload, write_keys, query_keys, decay,
                      policy=RoutingPolicy(grow=True), rank=5)
    with pytest.raises(ValueError, match="rank == rank_max"):
        additive_scan(payload, write_keys, query_keys, decay, policy=POLICY,
                      rank=4, rank_max=8)
    result = additive_scan(payload, write_keys, query_keys, decay,
                           policy=POLICY, rank=5, rank_max=5)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 5,
                           collect=False)
    assert torch.equal(result.reads, ref["reads"])
    assert torch.equal(result.state, ref["state"])


# --------------------------------------------------------------------- gate B


def _oracle_grads(payload, write_keys, query_keys, decay, rank,
                  dtype=torch.float32, state_grad=True, keys_grad=True):
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = torch.zeros(payload.shape[0], payload.shape[2], rank, payload.shape[-1],
                    dtype=dtype, device=DEVICE).requires_grad_(state_grad)
    k = torch.zeros(payload.shape[0], payload.shape[2], rank,
                    write_keys.shape[-1], dtype=dtype,
                    device=DEVICE).requires_grad_(keys_grad)
    ref = additive_oracle_scan(p.to(dtype), wk.to(dtype), qk.to(dtype),
                               d.to(dtype), policy=POLICY, rank=rank,
                               rank_max=rank, initial_state=s,
                               initial_keys=k)
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                          .manual_seed(1234)).to(dtype).to(DEVICE)
    (ref["reads"] * upstream).sum().backward()
    return ref, (p.grad, wk.grad, qk.grad, d.grad,
                s.grad if state_grad else None, k.grad if keys_grad else None)


@needs_cuda
def test_gate_b_oracle_gradcheck_fp64_through_the_additive_read():
    """MANDATORY (the registration's 2.4): the fp64 gradcheck THROUGH THE
    ADDITIVE READ PATH, over a fixture that fires every adjoint branch -- the
    read's DIRECT term into the value state, the SCORE path into the query
    keys and the key table (the raw-score weights are attached, no softmax
    jacobian anywhere), the payload lanes, and DECAY."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 32, 2, 8, 4, 5, 11)
    payload, write_keys = payload.double(), write_keys.double()
    query_keys, decay = query_keys.double(), decay.double()
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = torch.zeros(2, 2, 5, 8, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    k = torch.zeros(2, 2, 5, 4, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    hist_holder = {}

    def f(p_, wk_, qk_, d_, s_, k_):
        ref = additive_oracle_scan(p_, wk_, qk_, d_, policy=POLICY, rank=5,
                                   rank_max=5, initial_state=s_,
                                   initial_keys=k_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    assert torch.autograd.gradcheck(f, (p, wk, qk, d, s, k), eps=1e-6,
                                    atol=1e-4, rtol=1e-4)
    hist = hist_holder["hist"]
    assert all(hist[n] > 0 for n in ("append", "correct")), hist


@needs_cuda
def test_gate_b_grads_within_measured_floors():
    """The kernel path's six gradients vs the oracle's autograd, inside
    NOISE_FACTOR of the fp32-vs-fp64 floor measured on the same fixture."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    rank, seg = 5, 16
    ref, oracle_grads = _oracle_grads(payload, write_keys, query_keys, decay,
                                      rank)
    _, oracle_grads64 = _oracle_grads(payload, write_keys, query_keys, decay,
                                      rank, dtype=torch.float64)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=rank,
                            segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k13_backward_body(payload, write_keys, query_keys, decay,
                               upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank, seg, POLICY.read_scale)
    for name, kern, orac, orac64 in zip(
            ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dstate",
             "dkeys"), grads, oracle_grads, oracle_grads64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= NOISE_FACTOR * max(floor, 1e-6) or \
            deviation <= 1e-5, (name, deviation, floor)
        assert torch.isfinite(kern).all(), name


@needs_cuda
def test_gate_b_the_strength_path_is_gone_with_the_softmax():
    """R13-2's engine-side receipt: the value walk's strength term dies by
    the EXACT-ZERO dstrength channel -- the read adjoint's value contribution
    is the DIRECT term alone (dS[j] += score_j * dy), verified against the
    oracle's own autograd on a fixture with live rows of differing norms
    (where the banked read's strength path would be O(1))."""
    g = torch.Generator().manual_seed(17)
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    # rows of WILDLY different norms: the strength channel is the thing the
    # additive read must NOT consume.
    state, keys, live = crafted_store(
        [e[0], 5.0 * e[1], 0.2 * e[2], 3.0 * e[3]],
        key_rows=[ek[0], ek[1], ek[2], ek[3]], batch=2, heads=2)
    payload = torch.randn(2, 48, 2, V, generator=g).to(DEVICE)
    write_keys = torch.randn(2, 48, 2, K, generator=g).to(DEVICE)
    query_keys = torch.randn(2, 48, 2, K, generator=g).to(DEVICE)
    decay = torch.full((2, 48, 2), 0.999, device=DEVICE)
    rank, seg = 4, 16
    ref, oracle_grads = _oracle_grads_with_init(
        payload, write_keys, query_keys, decay, rank, state, live, keys)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=rank,
                            segment_tokens=seg, initial_state=state,
                            initial_live=live, initial_keys=keys)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k13_backward_body(payload, write_keys, query_keys, decay,
                               upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], rank, seg, POLICY.read_scale)
    for name, kern, orac in zip(("dpayload", "dwrite_keys", "dquery_keys",
                                 "ddecay", "dstate", "dkeys"),
                                grads, oracle_grads):
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= 1e-4, (name, deviation)
        assert torch.isfinite(kern).all(), name


def _oracle_grads_with_init(payload, write_keys, query_keys, decay, rank,
                             state, live, keys):
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    k = keys.clone().requires_grad_(True)
    ref = additive_oracle_scan(p, wk, qk, d, policy=POLICY, rank=rank,
                              rank_max=rank, initial_state=s,
                              initial_live=live, initial_keys=k)
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    (ref["reads"] * upstream).sum().backward()
    return ref, (p.grad, wk.grad, qk.grad, d.grad, s.grad, k.grad)


@needs_cuda
def test_gate_b_trajectory_replay_bit_exact():
    """The segment recompute reproduces the forward's own checkpoints bit for
    bit -- the value side through K4's imported replay, the key side through
    K10's imported key-lane replay (the recompute-over-store contract, both
    channels, unchanged by the read form)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward \
        import persistent_scan_trajectory
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import _k10_key_traj_kernel
    import triton
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    rank, seg = 5, 16
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=rank,
                           segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    decay32 = decay.float().permute(0, 2, 1).contiguous()
    tokens = query_keys.shape[1]
    batch, heads = 2, 2
    rows = batch * heads
    key_dim = write_keys.shape[-1]
    KP = triton.next_power_of_2(key_dim)
    RP = triton.next_power_of_2(rank)
    for index, start in enumerate(range(0, tokens, seg)):
        stop = min(start + seg, tokens)
        zeros_w = torch.zeros(batch, heads, stop - start, rank,
                              dtype=torch.float32, device=DEVICE)
        _, exit_state, exit_live, _traj = persistent_scan_trajectory(
            out["ckpt_state"][index], out["ckpt_live"][index], zeros_w,
            decay32[:, :, start:stop], schedule.chunk(start, stop))
        segt = stop - start
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
                        if o == 3:             # POP clears the address
                            final[row, int(schedule.slot[b, t, h, 0])] = 0
                        else:
                            final[row, int(schedule.slot[b, t, h, 0])] = \
                                write_keys[b, t, h].float()
                    if int(schedule.op[b, t, h, 1]) == 4:   # MERGE source
                        second = int(schedule.slot[b, t, h, 1])
                        final[row, second] = 0
            expected = out["ckpt_keys"][index + 1].reshape(rows, rank, key_dim)
            assert torch.equal(final, expected), \
                f"segment {index} key replay vs the forward's checkpoint"
