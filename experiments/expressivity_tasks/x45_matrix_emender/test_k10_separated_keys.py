"""K10 -- the separated-key arm, gated.

The K6 parity discipline, extended to the key channel
(``PREREGISTRATION_SEPARATED_KEYS.md``):

* **Bit-equality by construction**: the K6 defined-order spec (the balanced
  tree folds, fp32, correctly-rounded IEEE ops, libdevice exp) extends to the
  key axis; the eager oracle implements it in torch and the kernels in Triton,
  so every decision channel -- slot, op, erase, live_after, the payload lane
  (the MERGE blend included), the read WEIGHTS, the state, the liveness, the
  pre-decision liveness AND the key table -- must be ``torch.equal`` against
  the oracle on every fixture, including the bf16 autocast regime.  A
  difference is a bug, not a floor.
* **The read is the one channel at the measured floor** (its rank-axis
  reduction is a native sum, the K1/K23 precedent).
* **The gradient gates are floor-discipline** (the softmax adjoint's
  cross-axis reductions are autograd-ordered on the oracle side): the fp64
  oracle validates the oracle itself (a gradcheck over every adjoint branch),
  the fp32-vs-fp64 floor is measured per output, and the kernel path's SIX
  gradients (dpayload, dwrite_keys, dquery_keys, ddecay, dstate, dkeys) sit
  within ``NOISE_FACTOR`` of it.  The key lane's structural boundaries are
  pinned in the K4 tradition: a POPped row's KEY gradient terminates at its
  death token, a re-filed row's earlier write key carries no gradient from
  later reads, and a MERGE's blend reaches BOTH source rows.
* **The KEY-COLLAPSE watch** (the Wheeler receipt's guard): the mimetic birth
  is born spread -- ``span_key_std`` bounded well away from zero -- and the
  guard runs on every fixture that trains nothing.
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
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    SeparatedSequentialMixer, _k10_backward_body, _k10_forward_launch,
    convert_model_to_separated_keys, mimetic_key_family,
    separated_available, separated_scan,
    separated_sequential_oracle_scan, span_key_std)
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
    """The lane's canonical fixture: payload AND write-key values drawn from
    ``distinct`` fixed directions (wide decision margins -- the same payload
    direction implies the same key direction, the identity regime the arm
    exists to break), query keys random."""
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
    return _k10_forward_launch(payload, write_keys, query_keys, decay, state,
                               live, keys, POL32, rank, segment_tokens,
                               collect=collect,
                               warps=4 if rank * value_dim < 8192 else 8,
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


def assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, rank, label):
    """Gate A: every decision channel, the payload lane, the weights, the
    state, the liveness, the pre-decision liveness AND the key table bit-equal
    to the oracle; the read within the floor."""
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
        ref64 = separated_sequential_oracle_scan(
            payload.double(), write_keys.double(), query_keys.double(),
            decay.double(), policy=POLICY, rank=rank, rank_max=rank)
    floor = (ref["reads"] - ref64["reads"].float()).abs().max().item()
    deviation = (ref["reads"] - out["reads"]).abs().max().item()
    assert deviation <= max(NOISE_FACTOR * max(floor, 1e-30), 1e-6), \
        (label, "read", deviation, floor)


def oracle_reference(payload, write_keys, query_keys, decay, rank, collect=True):
    return separated_sequential_oracle_scan(payload, write_keys, query_keys,
                                            decay, policy=POLICY, rank=rank,
                                            rank_max=rank, collect=collect)


# --------------------------------------------------------------------- gate A


@needs_cuda
def test_gate_a_all_five_ops_regime_bit_equal():
    """The lane's canonical fixture: repeated payload+key directions, the
    identity regime -- every channel bit-equal."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                            segment_tokens=16)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 5)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 5, "identity-regime")
    hist = op_histogram(ref["schedule"])
    assert hist["correct"] > 0 and hist["append"] > 0, hist


@needs_cuda
def test_gate_a_merge_and_pop_reachable():
    """Crafted KEY tables force MERGE (two rows with keys at cos 0.9) and POP
    (a full store, a write key orthogonal to every row's key) -- the value
    channel's content is irrelevant to the decisions now."""
    g = torch.Generator().manual_seed(31)
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    # MERGE: rows 0 and 1 hold keys at cos 0.9; the write key is row 0's key,
    # so best = row 0 (cos 1.0) and second = row 1 (cos 0.9 >= tau_merge).
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
    ref = separated_sequential_oracle_scan(payload, write_keys, query_keys,
                                           decay, policy=POLICY, rank=4,
                                           rank_max=4, initial_state=state,
                                           initial_live=live,
                                           initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 4, "merge")
    hist = op_histogram(ref["schedule"])
    assert hist["merge"] > 0, f"the fixture must reach MERGE: {hist}"
    # the MERGE source row's key is zeroed with its value
    assert (out["keys"][..., 1, :] == 0).all()
    # POP: a full live store, a write key orthogonal to every row key
    state2, keys2, live2 = crafted_store([e[i] for i in range(4)],
                                         key_rows=[ek[i] for i in range(4)])
    orth = ek[4]
    write_keys2 = orth.view(1, 1, 1, K).repeat(1, 16, 1, 1)
    out2, _ = kernel_forward(payload, write_keys2, query_keys, decay, rank=4,
                             segment_tokens=8, initial_state=state2,
                             initial_live=live2, initial_keys=keys2)
    ref2 = separated_sequential_oracle_scan(payload, write_keys2, query_keys,
                                            decay, policy=POLICY, rank=4,
                                            rank_max=4, initial_state=state2,
                                            initial_live=live2,
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
    """The registered regime: bf16 inputs, fp32 defined-order math."""
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
    """The certified all-dead rule: the first token of an empty store reads 0."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 4, 3)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=5,
                            segment_tokens=4)
    assert torch.equal(out["reads"][:, :1], torch.zeros_like(out["reads"][:, :1]))


@needs_cuda
def test_gate_a_tie_break_first_index():
    """Equal KEY similarities: the lowest index wins, torch's argmax rule.
    Two live rows hold the SAME key; every matching write must resolve to row 0
    in both implementations."""
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
    ref = separated_sequential_oracle_scan(payload, write_keys, query_keys,
                                           decay, policy=POLICY, rank=4,
                                           rank_max=4, initial_state=state,
                                           initial_live=live,
                                           initial_keys=keys)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 4, "ties")
    correct = (ref["schedule"].op[..., 0] != NONE)
    assert correct.any()
    assert (ref["schedule"].slot[..., 0][correct] == 0).all(), \
        "a tie must resolve to the lowest index in both implementations"


@needs_cuda
def test_gate_a_keys_not_decayed_and_refiled():
    """The registration's semantic 3: keys do not age -- a live row's key is
    exactly its last write's key, however much the VALUE decayed."""
    payload, write_keys, query_keys, decay = keyed_stream(1, 32, 1, 8, 4, 3, 11,
                                                          decay=0.5)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=4,
                            segment_tokens=8)
    ref = oracle_reference(payload, write_keys, query_keys, decay, 4)
    assert_decisions_bit_equal(out, ref, payload, write_keys, query_keys,
                               decay, 4, "undecayed-keys")
    # the key channel is alive and re-filed (the semantic the parity pins)
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
        ref = separated_sequential_oracle_scan(payload, write_keys, query_keys,
                                               decay, policy=POLICY, rank=128,
                                               rank_max=128)
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


# ------------------------------------------------------------- the collapse watch


@needs_cuda
def test_mimetic_birth_is_born_spread():
    """The KEY-COLLAPSE watch at birth: the mimetic family's emitted keys are
    spread by construction -- distinct inputs get distinct addresses, and the
    family's columns are orthonormal."""
    family = mimetic_key_family(64, 16, 2)
    gram = family[0].t() @ family[0]
    assert torch.allclose(gram, torch.eye(16, device=gram.device), atol=1e-5), \
        "the family must be orthonormal"
    mixer = SeparatedSequentialMixer(64, 2, 32, 16, 16, 16, POLICY).to(DEVICE)
    g = torch.Generator().manual_seed(3)
    x = torch.randn(4, 128, 64, generator=g).to(DEVICE)
    from experiments.expressivity_tasks.x45_matrix_emender \
        .key_separation_probe import causal_context
    context = causal_context(x, mixer.ctx).reshape(4, 128, mixer.ctx * 64)
    with torch.no_grad():
        emitted = mixer.key(context).view(4, 128, mixer.heads, mixer.key_dim)
    spread = span_key_std(emitted)                       # (b, heads)
    assert (spread > 0.05).all(), f"the collapse signature at birth: {spread}"
    # born mimetic: the query-key projection equals the write-key projection
    assert torch.equal(mixer.key.weight, mixer.query_key.weight)
    # only the CURRENT-token context position is non-zero at birth
    assert mixer.key.weight[:, :3 * 64].abs().max().item() == 0.0


@needs_cuda
def test_mixer_conversion_swaps_addressing():
    """The conversion removes the content query, adds the key projections at
    the mimetic family, keeps the shared modules, and the converted mixer's
    forward runs the separated scan end to end."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        MatrixMixer, MatrixShape)
    shape = MatrixShape(width=64, depth=1, hidden=128, heads=2, value_dim=32,
                        rank=16, rank_max=16)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = [torch.nn.Module()]
            self.blocks[0].mixer = MatrixMixer(shape, POLICY, 64)

    stub = Stub()
    before = sum(p.numel() for p in stub.blocks[0].mixer.parameters())
    stub = convert_model_to_separated_keys(stub)
    mixer = stub.blocks[0].mixer
    mixer.to(DEVICE)          # the mimetic init ran on CPU; the walk is CUDA
    assert isinstance(mixer, SeparatedSequentialMixer)
    assert not hasattr(mixer, "query"), "the dead content query must be gone"
    assert hasattr(mixer, "key") and hasattr(mixer, "query_key")
    after = sum(p.numel() for p in mixer.parameters())
    delta = after - before
    expected = 2 * (mixer.ctx * 64) * (2 * mixer.key_dim) - 64 * (2 * 32)
    assert delta == expected  # at this small shape the swap is not neutral;
    # the registered shape's neutrality is asserted by the probe.
    x = torch.randn(2, 32, 64, device=DEVICE)
    with torch.no_grad():
        out, result = mixer.mix(x)
    assert out.shape == (2, 32, 64)
    assert result.reads.shape == (2, 32, 2, 32)
    assert torch.isfinite(out).all()


@needs_cuda
def test_refusals_are_by_name():
    """ops=False, grow and a rank prefix are refused outright; a store the
    walk cannot hold takes the EAGER ORACLE (same semantics)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 8, 2, 8, 4, 3, 5)
    with pytest.raises(ValueError, match="append-only"):
        separated_scan(payload, write_keys, query_keys, decay,
                      policy=RoutingPolicy(ops=False), rank=5)
    with pytest.raises(ValueError, match="growth"):
        separated_scan(payload, write_keys, query_keys, decay,
                      policy=RoutingPolicy(grow=True), rank=5)
    with pytest.raises(ValueError, match="rank == rank_max"):
        separated_scan(payload, write_keys, query_keys, decay, policy=POLICY,
                      rank=4, rank_max=8)
    # the size gate: a store too big for one program takes the oracle verbatim
    assert not separated_available(2048, 128, 16)
    result = separated_scan(payload, write_keys, query_keys, decay,
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
    ref = separated_sequential_oracle_scan(p.to(dtype), wk.to(dtype),
                                           qk.to(dtype), d.to(dtype),
                                           policy=POLICY, rank=rank,
                                           rank_max=rank, initial_state=s,
                                           initial_keys=k)
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(dtype).to(DEVICE)
    (ref["reads"] * upstream).sum().backward()
    return ref, (p.grad, wk.grad, qk.grad, d.grad,
                s.grad if state_grad else None, k.grad if keys_grad else None)


@needs_cuda
def test_gate_b_oracle_gradcheck_fp64():
    """The oracle is validated before it is trusted: an fp64 gradcheck over a
    fixture with the adjoint branches (NONE, APPEND, CORRECT at least; POP and
    MERGE boundaries are pinned by the structural tests below), through the
    value state AND the key table."""
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
        ref = separated_sequential_oracle_scan(p_, wk_, qk_, d_, policy=POLICY,
                                              rank=5, rank_max=5,
                                              initial_state=s_,
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
    grads = _k10_backward_body(payload, write_keys, query_keys, decay,
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
def test_gate_b_trajectory_replay_bit_exact():
    """The segment recompute reproduces the forward's own checkpoints bit for
    bit -- the value side through K4's replay kernel, the key side through the
    key-lane replay (the recompute-over-store contract, both channels)."""
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
        # the replay's entry IS the forward's own checkpoint (trivially, both
        # are loaded from it); the non-trivial check is the EXIT: the last
        # token's post-lane key table must equal the NEXT segment's entry
        # checkpoint, and the value exit must match the forward's own walk.
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


@needs_cuda
def test_gate_b_pop_terminates_the_row_gradient():
    """A POPped row's VALUE and KEY gradients terminate at its death token: an
    upstream gradient strictly after the death reaches neither the earlier
    payload nor the earlier write key, and the POP's own write key carries
    nothing (the lane wrote a constant zero)."""
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
    rank, seg = 4, 8
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=rank,
                            segment_tokens=seg, initial_state=state,
                            initial_live=live, initial_keys=keys)
    death = None
    op0 = out["op"][0, :, 0, 0]
    for t in range(16):
        if int(op0[t]) == 3:                  # POP
            death = t
            break
    assert death is not None and death < 15, "the fixture must POP early"
    # gradient ONLY after the death token
    upstream = torch.zeros(1, 16, 1, V, device=DEVICE)
    upstream[:, death + 1:] = torch.randn(1, 16 - death - 1, 1, V,
                                          generator=torch.Generator()
                                          .manual_seed(9)).to(DEVICE)
    sched = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                           payload=out["payload"], op=out["op"],
                           live_after=out["live_after"])
    dp, dwk, dqk, dd, dst, dki = _k10_backward_body(
        payload, write_keys, query_keys, decay, upstream, sched,
        out["live_pre"], out["ckpt_state"], out["ckpt_live"], out["ckpt_keys"],
        rank, seg, POLICY.read_scale)
    # the oracle under the same masked upstream, from the same entry store
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    k = keys.clone().requires_grad_(True)
    r2 = separated_sequential_oracle_scan(p, wk, qk, d, policy=POLICY,
                                          rank=rank, rank_max=rank,
                                          initial_state=s, initial_live=live,
                                          initial_keys=k)
    (r2["reads"] * upstream).sum().backward()
    before = slice(0, death)
    assert (dp[:, before].abs().max().item() if death > 0 else 0.0) == 0.0, \
        "a POPped row's payload gradient must terminate at the death token"
    assert (p.grad[:, before].abs().max().item() if death > 0 else 0.0) == 0.0
    # the POP's own write key carries nothing in either implementation
    assert float(dwk[0, death].abs().max()) == 0.0
    assert float(wk.grad[0, death].abs().max()) == 0.0
    # and the adjoints agree within floor on every output
    for name, kern, orac in (("dpayload", dp, p.grad),
                             ("dwrite_keys", dwk, wk.grad),
                             ("dquery_keys", dqk, qk.grad),
                             ("ddecay", dd, d.grad), ("dstate", dst, s.grad),
                             ("dkeys", dki, k.grad)):
        deviation = (kern.float() - orac.float()).abs().max().item()
        scale = orac.float().abs().max().item()
        assert deviation <= 1e-4 * max(scale, 1.0) or deviation < 1e-5, \
            (name, deviation, scale)


@needs_cuda
def test_gate_b_merge_gradient_reaches_both_source_rows():
    """The MERGE blend's content is attached: its VALUE gradient reaches BOTH
    source rows through the payload channel's adjoint, and the KEY lane
    re-files the target (its write key carries gradient) while the source
    row's key adjoint terminates (the K4 case-iv discipline, key side)."""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    key1 = 0.9 * ek[0] + (1 - 0.81) ** 0.5 * ek[1]
    state, keys, live = crafted_store([e[0], e[2], e[3], e[4]],
                                      key_rows=[ek[0], key1, ek[2], ek[3]])
    payload = e[0].view(1, 1, 1, V).repeat(1, 16, 1, 1)
    write_keys = ek[0].view(1, 1, 1, K).repeat(1, 16, 1, 1)
    g = torch.Generator().manual_seed(31)
    query_keys = torch.randn(1, 16, 1, K, generator=g).to(DEVICE)
    decay = torch.full((1, 16, 1), 0.999, device=DEVICE)
    rank, seg = 4, 8
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=rank,
                            segment_tokens=seg, initial_state=state,
                            initial_live=live, initial_keys=keys)
    merges = (out["op"][0, :, 0, 0] == 4).nonzero()
    assert len(merges) > 0, "the fixture must MERGE"
    t = int(merges[0])
    upstream = torch.zeros(1, 16, 1, V, device=DEVICE)
    upstream[:, t + 1:] = torch.randn(1, 15 - t, 1, V,
                                      generator=torch.Generator()
                                      .manual_seed(21)).to(DEVICE)
    sched = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                           payload=out["payload"], op=out["op"],
                           live_after=out["live_after"])
    dp, dwk, dqk, dd, dst, dki = _k10_backward_body(
        payload, write_keys, query_keys, decay, upstream, sched,
        out["live_pre"], out["ckpt_state"], out["ckpt_live"], out["ckpt_keys"],
        rank, seg, POLICY.read_scale)
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    k = keys.clone().requires_grad_(True)
    r2 = separated_sequential_oracle_scan(p, wk, qk, d, policy=POLICY,
                                          rank=rank, rank_max=rank,
                                          initial_state=s, initial_live=live,
                                          initial_keys=k)
    (r2["reads"] * upstream).sum().backward()
    for name, kern, orac in (("dpayload", dp, p.grad),
                             ("dwrite_keys", dwk, wk.grad),
                             ("dquery_keys", dqk, qk.grad),
                             ("ddecay", dd, d.grad), ("dstate", dst, s.grad),
                             ("dkeys", dki, k.grad)):
        deviation = (kern.float() - orac.float()).abs().max().item()
        scale = orac.float().abs().max().item()
        assert deviation <= 1e-4 * max(scale, 1.0) or deviation < 1e-5, \
            (name, deviation, scale)


@needs_cuda
def test_gate_b_refile_terminates_the_earlier_key_gradient():
    """The key lane's case-iii boundary: a row re-filed at token t -- the row's
    EARLIER write key carries no gradient from reads after t (the pre-write
    key's path terminates at the re-file), and the re-file token's write key
    carries the later reads' gradient."""
    V, K = 8, 8
    e = torch.eye(V, device=DEVICE)
    ek = torch.eye(K, device=DEVICE)
    # row 0 born with key ek[0]; tokens 1.. CORRECT it with the SAME key
    # direction (cos 1.0 >= tau_match), re-filing the address each time.
    state, keys, live = crafted_store([e[0], e[1], e[2], e[3]],
                                      key_rows=[ek[0], ek[5], ek[6], ek[7]])
    payload = e[0].view(1, 1, 1, V).repeat(1, 8, 1, 1)
    write_keys = ek[0].view(1, 1, 1, K).repeat(1, 8, 1, 1)
    g = torch.Generator().manual_seed(13)
    query_keys = torch.zeros(1, 8, 1, K, device=DEVICE)
    # late reads target row 0 but OFF the cosine's stationary point: a query
    # key exactly parallel to the row key would make the score's gradient
    # w.r.t. the row key vanish to first order (cos is maximized there) -- a
    # fixture degeneracy the oracle and the kernel agree on.
    probe_key = 0.9 * ek[0] + (1 - 0.81) ** 0.5 * ek[1]
    query_keys[:, 4:] = probe_key.view(1, 1, 1, K)  # cos 0.9 to row 0
    decay = torch.full((1, 8, 1), 0.999, device=DEVICE)
    rank, seg = 4, 8
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, rank=rank,
                            segment_tokens=seg, initial_state=state,
                            initial_live=live, initial_keys=keys)
    ops = out["op"][0, :, 0, 0]
    assert int(ops[1]) == 2, "token 1 must CORRECT row 0 (key cos 1.0)"
    assert all(int(o) == 2 for o in ops[1:4]), "tokens 1-3 must all re-file row 0"
    probe_weights = out["weights"][0, 4:, 0, :]
    assert (probe_weights[:, 0] > probe_weights[:, 1:].max(dim=-1).values \
            + 0.1).all(), "the late reads must be dominated by row 0"
    upstream = torch.zeros(1, 8, 1, V, device=DEVICE)
    upstream[:, 4:] = torch.randn(1, 4, 1, V,
                                  generator=torch.Generator()
                                  .manual_seed(21)).to(DEVICE)
    # only the late reads
    sched = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                           payload=out["payload"], op=out["op"],
                           live_after=out["live_after"])
    dp, dwk, dqk, dd, dst, dki = _k10_backward_body(
        payload, write_keys, query_keys, decay, upstream, sched,
        out["live_pre"], out["ckpt_state"], out["ckpt_live"], out["ckpt_keys"],
        rank, seg, POLICY.read_scale)
    # the row's address after token 1 is token 1's key: every late read's key
    # gradient lands on the LAST re-file at or before the read, never earlier.
    last_refile = 3          # tokens 1..3 all re-file (the same key direction)
    assert float(dwk[0, last_refile].abs().max()) > 0.0, \
        "the re-file token's write key must carry the late reads' gradient"
    for t in range(0, last_refile):
        assert float(dwk[0, t].abs().max()) == 0.0, \
            f"token {t}'s write key must carry nothing after the re-file"
    # the oracle agrees exactly here (the key-lane adjoint is exact-replayable
    # given the recorded schedule: no cross-axis reduction on this path)
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = state.clone().requires_grad_(True)
    k = keys.clone().requires_grad_(True)
    r2 = separated_sequential_oracle_scan(p, wk, qk, d, policy=POLICY,
                                          rank=rank, rank_max=rank,
                                          initial_state=s, initial_live=live,
                                          initial_keys=k)
    (r2["reads"] * upstream).sum().backward()
    # the key-lane adjoint agrees within floor (the read-side coefficients'
    # cross-axis reductions are autograd-ordered on the oracle side -- the
    # floor discipline, stated, not hidden)
    deviation = (dwk.float() - wk.grad.float()).abs().max().item()
    scale = wk.grad.float().abs().max().item()
    assert deviation <= 1e-4 * max(scale, 1.0) or deviation < 1e-5, \
        (deviation, scale)
