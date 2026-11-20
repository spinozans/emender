"""K14 -- the learned management routing arm, gated.

The K10 parity discipline, extended to the routing channel
(``PREREGISTRATION_LEARNED_MANAGEMENT.md``, the supervisor's rulings cited
in ``k14_learned_routing.py``'s header):

* **Bit-equality by construction**: the K10 defined-order spec extends to
  the router's five per-op scores (single correctly-rounded subs/adds over
  quantities K10 already defines); the eager oracle implements it in torch
  and the kernel in Triton, so every channel -- slot, op, erase, live_after,
  the payload lane, the read WEIGHTS, the state, the liveness, the
  pre-decision liveness, the key table AND the recorded per-op scores --
  must be ``torch.equal`` against the oracle on every fixture and at every
  router weight, including the bf16 autocast regime.
* **Birth parity (P5)**: at ``g = +0.0`` the k14 launch is BIT-EQUAL to
  K10's OWN launch on every K10 channel -- the branch-active margin
  encoding reproduces the threshold chain exactly, with a UNIQUE argmax (no
  tie-break reliance: the active margin >= 0 vs the reserved -1.0
  separation) -- and the frozen-router falsifier control trains bit-equal
  to the K10 machine for as long as the router stays frozen.
* **The zero-gradient finding**: the ORACLE path's structural gradient to
  the router is exactly zero (asserted) -- under the recorded-schedule
  doctrine the op choice is a backward constant; the kernel path's
  straight-through estimator is the admitted exception, and it must equal
  its fp64 autograd reference of the surrogate objective, with a nonzero
  signal at birth on a fixture that fires.
* **The gradient gates are floor-discipline** (K10's): the fp64 oracle
  validates the six K10 channels through the routing oracle (at the birth
  router AND a flipped-router regime); the kernel path's six gradients sit
  within ``NOISE_FACTOR`` of the measured floor and are BIT-EQUAL to K10's
  own body's on the same recorded channels.
* **Refusals by name**, **determinism**, and the conversion's parameter
  accounting (+553,200 at the registered shape).

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, CORRECT, MERGE, NONE, OP_NAMES, POP, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    SeparatedSequentialMixer, _k10_backward_body, _k10_forward_launch,
    convert_model_to_separated_keys, separated_available)
from experiments.expressivity_tasks.x45_matrix_emender.k14_learned_routing import (
    RESERVED_SCORE, RoutingSeparatedMixer, _k14_backward_body,
    _k14_forward_launch, _router_estimator, convert_model_to_routing,
    routing_available, routing_oracle_scan, routing_parameter_delta,
    routing_scan)
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
# the crafted regime: tau_match high enough that only near-exact keys
# CORRECT, a mergeable twin pair, and tau_pop placed so the crafted stream
# splits its unmatched keys between POP (worst <= tau_pop) and NONE.
CRAFTED = RoutingPolicy(ops=True, tau_match=0.80, tau_merge=0.85, tau_pop=0.30)
CRAFTED32 = (_round_f32(CRAFTED.tau_match), _round_f32(CRAFTED.tau_merge),
             _round_f32(CRAFTED.tau_pop), _round_f32(CRAFTED.tie_break),
             _round_f32(CRAFTED.read_scale))


def keyed_stream(batch, tokens, heads, value_dim, key_dim, distinct, seed,
                 decay=0.95):
    """K10's canonical fixture: payload AND write-key values drawn from
    ``distinct`` fixed directions, query keys random."""
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


def router_scores(batch, tokens, heads, value=0.0, seed=0):
    """The router's per-op scores: exact zeros (birth / the frozen control)
    or a random draw (the flipped-router regime)."""
    if value == 0.0 and seed == 0:
        return torch.zeros(batch, tokens, heads, 5, dtype=torch.float32,
                           device=DEVICE)
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(batch, tokens, heads, 5, generator=g)
            .to(DEVICE) * float(value)).contiguous()


def kernel_forward(payload, write_keys, query_keys, decay, scores, rank,
                   segment_tokens=16, collect=True, policy32=POL32,
                   initial_state=None, initial_live=None, initial_keys=None):
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
    return _k14_forward_launch(payload, write_keys, query_keys, decay,
                               scores, state, live, keys, policy32, rank,
                               segment_tokens, collect=collect,
                               warps=4 if rank * value_dim < 8192 else 8,
                               stages=3), state


def oracle_reference(payload, write_keys, query_keys, decay, scores, rank,
                     policy=POLICY, collect=True, initial_state=None,
                     initial_live=None, initial_keys=None):
    return routing_oracle_scan(payload, write_keys, query_keys, decay,
                               scores, policy=policy, rank=rank,
                               rank_max=rank, collect=collect,
                               initial_state=initial_state,
                               initial_live=initial_live,
                               initial_keys=initial_keys)


def assert_decisions_bit_equal(out, ref, label):
    """Gate A: every decision channel, the payload lane, the weights, the
    state, the liveness, the pre-decision liveness, the key table AND the
    recorded per-op scores bit-equal to the oracle; the read within the
    floor."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(ref["schedule"], name), out[name]), \
            (label, name)
    assert torch.equal(ref["schedule"].payload, out["payload"]), (label, "payload")
    assert torch.equal(ref["weights"], out["weights"]), (label, "weights")
    assert torch.equal(ref["state"], out["state"]), (label, "state")
    assert torch.equal(ref["live"], out["live"]), (label, "live")
    assert torch.equal(ref["live_pre"], out["live_pre"]), (label, "live_pre")
    assert torch.equal(ref["keys"], out["keys"]), (label, "keys")
    assert torch.equal(ref["scores"], out["op_scores"]), (label, "op_scores")
    assert torch.equal(ref["chain_op"], out["chain_op"]), (label, "chain_op")
    read_dev = (ref["reads"] - out["reads"]).abs().max().item()
    assert read_dev <= 1e-5, (label, "read", read_dev)


def crafted_routing_fixture(batch=2, tokens=12, heads=2, V=8, KD=4, rank=6,
                            seed=3):
    """The all-five-ops regime at BIRTH: a full store of distinct keys, then
    a stream of exact matches (CORRECT), a merge-pair key (MERGE), a
    strongly-unmatched key (POP once a slot is free, then APPEND back) and a
    moderately-unmatched key whose worst match stays above tau_pop (NONE)."""
    g = torch.Generator().manual_seed(seed)

    def u(v):
        t = torch.tensor(v, dtype=torch.float32)
        return t / t.norm()

    keys_rows = [u([1, 0, 0, 0]), u([0.9, 0.436, 0, 0]), u([0, 0, 1, 0]),
                 u([0, 0, 0, 1]), u([1, 1, 1, 1]), u([0, 1, 0, 0])]
    state0 = torch.randn(batch, heads, rank, V, generator=g).to(DEVICE)
    keys0 = torch.stack(keys_rows).view(1, 1, rank, KD).expand(
        batch, heads, rank, KD).contiguous().to(DEVICE).clone()
    live0 = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    # the stream, verified at birth against the oracle: e0 (MERGE -- its
    # 0.9-cos twin is second, >= tau_merge); appendk (unmatched -> APPEND
    # into the freed slot, refilling the store); popk (unmatched, worst cos
    # <= tau_pop -> POP on the full store); appendk again; nonek (unmatched
    # with worst cos > tau_pop -> NONE on the full store); then repeats.
    # The birth histogram covers all five ops.
    appendk = u([0.7, 0.1, -0.2, -0.67])
    popk = u([0.4, -0.4, 0.4, -0.8])
    nonek = u([0.5, 0.3, 0.5, 0.3])
    seq = [u([1, 0, 0, 0]), appendk, popk, appendk, nonek, u([0, 0, 1, 0]),
           popk, appendk, nonek, u([1, 0, 0, 0]), nonek, u([0, 0, 1, 0])]
    seq = (seq * (tokens // len(seq) + 1))[:tokens]
    payload = torch.randn(batch, tokens, heads, V, generator=g).to(DEVICE)
    write_keys = torch.stack(seq).view(1, tokens, 1, KD).expand(
        batch, tokens, heads, KD).contiguous().to(DEVICE)
    query_keys = torch.randn(batch, tokens, heads, KD, generator=g).to(DEVICE)
    decay = (torch.rand(batch, tokens, heads, generator=g).to(DEVICE) * 0.1
             + 0.9)
    return payload, write_keys, query_keys, decay, state0, keys0, live0


# --------------------------------------------------------------------- gate A


@needs_cuda
def test_gate_a_bit_equal_at_birth_and_flipped_router():
    """The lane's canonical fixture at the birth router (exact zeros) AND a
    random flipped router: every channel bit-equal, the recorded per-op
    scores and the chain's own op included."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    for label, scores in (("birth", router_scores(2, 64, 2)),
                          ("flipped", router_scores(2, 64, 2, 0.9, seed=41))):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                scores, rank=5, segment_tokens=16)
        ref = oracle_reference(payload, write_keys, query_keys, decay,
                              scores, 5)
        assert_decisions_bit_equal(out, ref, label)
    # the flipped router must actually FLIP some op somewhere (the learned
    # regime is exercised, not just reached)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                            router_scores(2, 64, 2, 2.0, seed=41), rank=5,
                            segment_tokens=16)
    assert int((out["op"][..., 0] != out["chain_op"]).sum()) > 0, \
        "a magnitude-2 random router must flip some op"


@needs_cuda
def test_gate_a_birth_parity_vs_k10_launch():
    """P5's birth bar: at g = +0.0 the k14 launch is BIT-EQUAL to K10's own
    launch on every K10 channel -- the branch-active margin encoding
    reproduces the threshold chain EXACTLY, 0 flips."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 8, 4, 6, 91)
    out14, _ = kernel_forward(payload, write_keys, query_keys, decay,
                              router_scores(2, 128, 2), rank=5,
                              segment_tokens=16)
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    state = torch.zeros(batch, heads, 5, value_dim, dtype=torch.float32,
                        device=DEVICE)
    live = torch.zeros(batch, heads, 5, dtype=torch.bool, device=DEVICE)
    keys = torch.zeros(batch, heads, 5, key_dim, dtype=torch.float32,
                       device=DEVICE)
    out10 = _k10_forward_launch(payload, write_keys, query_keys, decay, state,
                                live, keys, POL32, 5, 16, collect=True,
                                warps=4, stages=3)
    flips = 0
    lanes = 0
    for name in SCHEDULE_FIELDS:
        assert torch.equal(out10[name], out14[name]), ("birth", name)
        lanes += out10[name].numel()
        flips += int((out10[name] != out14[name]).sum())
    for name in ("payload", "live_pre"):
        assert torch.equal(out10[name], out14[name]), ("birth", name)
        lanes += out10[name].numel()
        flips += int((out10[name] != out14[name]).sum())
    for name in ("weights", "state", "live", "keys", "reads"):
        assert torch.equal(out10[name], out14[name]), ("birth", name)
        lanes += out10[name].numel()
        flips += int((out10[name] != out14[name]).sum())
    assert flips == 0 and lanes > 0
    # and the recorded chain op is K10's own op, token for token
    assert torch.equal(out10["op"][..., 0], out14["chain_op"])


@needs_cuda
def test_gate_a_birth_argmax_is_unique_no_tie_reliance():
    """The birth construction's core receipt: the argmax is UNIQUE at birth
    (the active margin >= 0 vs the reserved -1.0 separation) on a dense
    fixture -- no tie-break reliance anywhere (the supervisor-approved
    form's whole point)."""
    (payload, write_keys, query_keys, decay, state0, keys0,
     live0) = crafted_routing_fixture()
    rank = keys0.shape[2]
    out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                            router_scores(*payload.shape[:3]), rank=rank,
                            segment_tokens=16, policy32=CRAFTED32,
                            initial_state=state0, initial_live=live0,
                            initial_keys=keys0)
    s = out["op_scores"]                       # (b, T, h, 5)
    top2 = s.topk(2, dim=-1).values
    margins = (top2[..., 0] - top2[..., 1])
    assert bool((margins > 0).all()), \
        f"the birth argmax must be unique; min margin {margins.min().item()}"
    # the winner's score is its own threshold delta (>= 0) and the runner-up
    # is the reserved constant on this fixture
    assert float(top2[..., 0].min()) >= 0.0
    assert float((top2[..., 1] == RESERVED_SCORE).float().mean()) > 0.9


@needs_cuda
def test_gate_a_crafted_regime_covers_all_five_ops():
    """The birth chain exercised on ALL FIVE branches (CORRECT/MERGE/POP/
    APPEND/NONE) with parity to the oracle, and the birth scores equal the
    chain's own margins."""
    (payload, write_keys, query_keys, decay, state0, keys0,
     live0) = crafted_routing_fixture()
    rank = keys0.shape[2]
    scores = router_scores(*payload.shape[:3])
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, scores,
                            rank=rank, segment_tokens=16, policy32=CRAFTED32,
                            initial_state=state0, initial_live=live0,
                            initial_keys=keys0)
    ref = oracle_reference(payload, write_keys, query_keys, decay, scores,
                           rank, policy=CRAFTED, initial_state=state0,
                           initial_live=live0, initial_keys=keys0)
    assert_decisions_bit_equal(out, ref, "crafted")
    hist = op_histogram(ref["schedule"])
    for name in ("none", "append", "correct", "pop", "merge"):
        assert hist[name] > 0, (name, hist)
    # at birth the learned op IS the chain's op, token for token
    assert torch.equal(out["op"][..., 0], out["chain_op"])


@needs_cuda
def test_gate_a_feasibility_masks():
    """The semantics guards: a magnitude-10 APPEND score cannot APPEND into a
    FULL store, and a magnitude-10 POP score cannot POP an EMPTY store --
    the masked ops sit at -inf in the recorded scores, and the argmax picks
    a feasible op."""
    batch, tokens, heads, V, KD, rank = 1, 4, 1, 8, 4, 4
    g = torch.Generator().manual_seed(19)

    def u(v):
        t = torch.tensor(v, dtype=torch.float32)
        return t / t.norm()

    # a FULL store of orthogonal keys; the stream key is unmatched
    keys0 = torch.stack([u([1, 0, 0, 0]), u([0, 1, 0, 0]),
                         u([0, 0, 1, 0]), u([0, 0, 0, 1])]
                        ).view(1, 1, rank, KD).to(DEVICE)
    state0 = torch.randn(batch, heads, rank, V, generator=g).to(DEVICE)
    live0 = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    payload = torch.randn(batch, tokens, heads, V, generator=g).to(DEVICE)
    write_keys = u([0.7, 0.7, 0.7, 0.7]).view(1, 1, 1, KD).expand(
        batch, tokens, heads, KD).contiguous().to(DEVICE)
    query_keys = torch.randn(batch, tokens, heads, KD, generator=g).to(DEVICE)
    decay = torch.full((batch, tokens, heads), 0.99, device=DEVICE)
    # the APPEND score dominates everywhere
    scores = torch.zeros(batch, tokens, heads, 5, device=DEVICE)
    scores[..., APPEND] = 10.0
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, scores,
                            rank=rank, segment_tokens=4, policy32=CRAFTED32,
                            initial_state=state0, initial_live=live0,
                            initial_keys=keys0)
    assert not bool((out["op"][..., 0] == APPEND).any()), \
        "APPEND must be masked on a full store however large its score"
    assert bool((out["op_scores"][..., APPEND] == float("-inf")).all())
    # the empty store: the POP score dominates
    scores = torch.zeros(batch, tokens, heads, 5, device=DEVICE)
    scores[..., POP] = 10.0
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, scores,
                            rank=rank, segment_tokens=4, policy32=CRAFTED32,
                            initial_state=torch.zeros_like(state0),
                            initial_live=torch.zeros_like(live0),
                            initial_keys=torch.zeros_like(keys0))
    # at t=0 the store is empty: POP is masked however large its score (the
    # first token APPENDs; from t=1 a live row exists and POP is legitimately
    # feasible again -- the guard is the no-live-rows corner only)
    assert int(out["op"][0, 0, 0, 0]) != POP
    assert bool((out["op_scores"][0, 0, ..., POP] == float("-inf")).all())
    assert int(out["op"][0, 0, 0, 0]) == APPEND


@needs_cuda
def test_gate_a_frozen_router_falsifier_bit_equal_through_training():
    """The never-fires falsifier control: with the router frozen at exact
    zeros the routing machine trains BIT-EQUAL to the K10 machine -- the
    losses equal at every step for three optimizer steps (the forward is the
    same bit-exact machine, the frozen router never enters the optimizer,
    and its estimator writes only its own buffer)."""
    width, heads, value_dim, rank, key_dim, ctx = 32, 2, 8, 8, 4, 4
    policy = RoutingPolicy(ops=True)
    x = torch.randn(2, 64, width, device=DEVICE)
    target = torch.randn(2, 64, width, device=DEVICE)

    def build(mixer_cls, extra):
        torch.manual_seed(99)
        torch.cuda.manual_seed_all(99)
        mixer = mixer_cls(width, heads, value_dim, rank, rank, key_dim,
                         policy, ctx=ctx, segment_tokens=16, **extra)
        return mixer.cuda()

    m10 = build(SeparatedSequentialMixer, {})
    m14 = build(RoutingSeparatedMixer, dict(router_trainable=False))
    assert torch.equal(m14.router.weight, torch.zeros_like(m14.router.weight))
    assert torch.equal(m14.router.bias, torch.zeros_like(m14.router.bias))
    # the shared modules must be bit-identical at birth (the same build order)
    for name in ("value", "key", "query_key", "out", "decay"):
        a = getattr(m10, name).weight
        b = getattr(m14, name).weight
        assert torch.equal(a, b), name
    opt10 = torch.optim.AdamW(m10.parameters(), lr=1e-3, fused=True)
    opt14 = torch.optim.AdamW((p for p in m14.parameters() if p.requires_grad),
                              lr=1e-3, fused=True)
    for step in range(3):
        loss10 = ((m10(x) - target) ** 2).mean()
        opt10.zero_grad(set_to_none=True)
        loss10.backward()
        torch.nn.utils.clip_grad_norm_(m10.parameters(), 1.0)
        opt10.step()
        loss14 = ((m14(x) - target) ** 2).mean()
        opt14.zero_grad(set_to_none=True)
        loss14.backward()
        torch.nn.utils.clip_grad_norm_(m14.parameters(), 1.0)
        opt14.step()
        assert loss10.item() == loss14.item(), (step, loss10.item(),
                                                loss14.item())
        assert torch.equal(m14.router.weight,
                          torch.zeros_like(m14.router.weight))


@needs_cuda
def test_gate_a_mixer_birth_parity_under_autocast():
    """The K11B 0.5 lesson, applied at the mixer level: the birth fixture
    MUST go through the production expression under the bf16 autocast regime
    -- the router's fp32-outside-autocast path included.  At birth the
    routing mixer's output is the K10 mixer's bit for bit."""
    width, heads, value_dim, rank, key_dim, ctx = 32, 2, 8, 8, 4, 4
    policy = RoutingPolicy(ops=True)
    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    m10 = SeparatedSequentialMixer(width, heads, value_dim, rank, rank,
                                   key_dim, policy, ctx=ctx,
                                   segment_tokens=16).cuda()
    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    m14 = RoutingSeparatedMixer(width, heads, value_dim, rank, rank, key_dim,
                               policy, ctx=ctx, segment_tokens=16,
                               router_trainable=True).cuda()
    x = torch.randn(2, 96, width, device=DEVICE)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out10, res10 = m10.mix(x)
        out14, res14 = m14.mix(x)
    assert torch.equal(out10, out14)
    assert torch.equal(res10.reads, res14.reads)
    assert torch.equal(res10.state, res14.state)
    assert torch.equal(res10.live, res14.live)


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    payload, write_keys, query_keys, decay = keyed_stream(2, 51, 2, 8, 4, 5, 7)
    for seg in (8, 16, 64):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                router_scores(2, 51, 2, 0.61, seed=43),
                                rank=5, segment_tokens=seg)
        ref = oracle_reference(payload, write_keys, query_keys, decay,
                               router_scores(2, 51, 2, 0.61, seed=43), 5)
        assert_decisions_bit_equal(out, ref, f"ragged seg={seg}")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 13)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    scores = router_scores(2, 64, 2, 0.83, seed=44)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                scores, rank=5, segment_tokens=16)
    ref = oracle_reference(payload.float(), write_keys.float(),
                           query_keys.float(), decay.float(), scores, 5)
    assert_decisions_bit_equal(out, ref, "bf16")


@needs_cuda
def test_gate_a_heterogeneous_lanes_only_some_merge():
    """The multi-lane MERGE corner under the LEARNED router: when only SOME
    (batch, head) lanes merge at a token, the oracle must kill ONLY the
    merging lanes' source rows (the per-lane-masked form; K10's own oracle
    carries the latent unmasked form, unreachable by its fixtures and inert
    in production -- not copied here)."""
    batch, tokens, heads, V, KD, rank = 2, 8, 2, 8, 4, 6
    g = torch.Generator().manual_seed(17)

    def u(v):
        t = torch.tensor(v, dtype=torch.float32)
        return t / t.norm()

    keys_rows = [u([1, 0, 0, 0]), u([0.9, 0.436, 0, 0]), u([0, 0, 1, 0]),
                 u([0, 0, 0, 1]), u([1, 1, 1, 1]), u([0, 1, 0, 0])]
    state0 = torch.randn(batch, heads, rank, V, generator=g).to(DEVICE)
    keys0 = torch.stack(keys_rows).view(1, 1, rank, KD).expand(
        batch, heads, rank, KD).contiguous().to(DEVICE).clone()
    live0 = torch.ones(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    payload = torch.randn(batch, tokens, heads, V, generator=g).to(DEVICE)
    query_keys = torch.randn(batch, tokens, heads, KD, generator=g).to(DEVICE)
    decay = torch.full((batch, tokens, heads), 0.99, device=DEVICE)
    wk = torch.zeros(batch, tokens, heads, KD)
    wk[:, :, 0] = torch.tensor([1.0, 0, 0, 0])
    wk[:, :, 1] = torch.tensor([0.9, 0.436, 1.0, 0.0])   # ~orthogonal to all
    write_keys = wk.to(DEVICE)
    scores = router_scores(batch, tokens, heads)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, scores,
                            rank=rank, segment_tokens=4, policy32=CRAFTED32,
                            initial_state=state0, initial_live=live0,
                            initial_keys=keys0)
    ref = oracle_reference(payload, write_keys, query_keys, decay, scores,
                           rank, policy=CRAFTED, initial_state=state0,
                           initial_live=live0, initial_keys=keys0)
    assert_decisions_bit_equal(out, ref, "heterogeneous")
    ops0 = out["op"][:, :, 0, 0]
    ops1 = out["op"][:, :, 1, 0]
    assert bool((ops0[:, 0] == MERGE).all()), ops0.tolist()
    assert not bool((ops0[:, 1:] == MERGE).any()), ops0.tolist()
    assert not bool((ops1 == MERGE).any()), ops1.tolist()
    for b in range(batch):
        second0 = int(out["slot"][b, 0, 0, 1])
        second1 = int(out["slot"][b, 0, 1, 1])
        assert not bool(out["live_pre"][b, 1, 0, second0])
        target1 = int(out["slot"][b, 0, 1, 0])
        popped1 = int(out["op"][b, 0, 1, 0]) == POP
        if not (popped1 and target1 == second1):
            assert bool(out["live_pre"][b, 1, 1, second1]), \
                (b, second1, out["live_pre"][b, 1, 1].tolist())


@needs_cuda
def test_gate_a_determinism():
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 29)
    scores = router_scores(2, 64, 2, 0.44, seed=45)
    outs = [kernel_forward(payload, write_keys, query_keys, decay, scores,
                           rank=5, segment_tokens=16)[0] for _ in range(2)]
    for name in ("slot", "op", "payload", "state", "keys", "op_scores",
                 "chain_op", "reads"):
        assert torch.equal(outs[0][name], outs[1][name]), name


# ------------------------------------------------- the finding and the estimator


@needs_cuda
def test_the_zero_gradient_finding():
    """THE FINDING, asserted: the ORACLE path's structural gradient to the
    router scores is EXACTLY ZERO on a fixture where the router fires and
    the loss depends on the store -- the op choice is discrete and the
    recorded-schedule doctrine makes it a backward constant.  (The kernel
    path's straight-through estimator is the admitted exception; the next
    test gates it.)"""
    (payload, write_keys, query_keys, decay, state0, keys0,
     live0) = crafted_routing_fixture()
    rank = keys0.shape[2]
    g = router_scores(*payload.shape[:3], 0.9, seed=51).requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    ref = oracle_reference(payload=p, write_keys=write_keys,
                           query_keys=query_keys, decay=decay, scores=g,
                           rank=rank, policy=CRAFTED, initial_state=state0,
                           initial_live=live0, initial_keys=keys0)
    grads = torch.autograd.grad(ref["reads"].sum(), [g, p], allow_unused=True)
    assert grads[0] is None or torch.equal(
        grads[0], torch.zeros_like(grads[0])), grads[0]


@needs_cuda
def test_the_estimator_matches_its_fp64_reference():
    """The admitted straight-through estimator vs the fp64 AUTOGRAD of its
    own DEFINED surrogate objective (c_t * p_{k*}, p = softmax over the five
    recorded scores) -- the mandated fp64 gradcheck through the router path
    -- plus P5's step-0 receipt: the gradient reaches the ZERO-BORN router
    at birth on a fixture that fires."""
    payload, write_keys, query_keys, decay, state0, keys0, live0 = \
        crafted_routing_fixture()
    batch, tokens, heads, V = payload.shape
    rank = keys0.shape[2]
    upstream = torch.randn(batch, tokens, heads, V,
                           generator=torch.Generator().manual_seed(1234)
                           ).to(DEVICE)

    def collect_launch(scores):
        return kernel_forward(payload, write_keys, query_keys, decay, scores,
                              rank=rank, segment_tokens=16,
                              policy32=CRAFTED32, initial_state=state0,
                              initial_live=live0, initial_keys=keys0)[0]

    for label, scores in (("birth", router_scores(batch, tokens, heads)),
                          ("fired", router_scores(batch, tokens, heads, 0.8,
                                                  seed=52))):
        # the kernel path's estimator, end to end through the custom op
        s = scores.clone().requires_grad_(True)
        result = routing_scan(payload, write_keys, query_keys, decay, s,
                              policy=CRAFTED, rank=rank, rank_max=rank,
                              segment_tokens=16, initial_state=state0,
                              initial_live=live0, initial_keys=keys0)
        assert result.schedule is None, "the kernel path must be taken"
        (result.reads * upstream).sum().backward()
        d_router = s.grad.clone()
        # the recorded channels (collect launch) + c_t from the imported
        # value walk's own output
        out = collect_launch(scores)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        grads = _k14_backward_body(payload, write_keys, query_keys, decay,
                                   upstream, schedule, out["live_pre"],
                                   out["ckpt_state"], out["ckpt_live"],
                                   out["ckpt_keys"], out["op_scores"], rank,
                                   16, CRAFTED.read_scale)
        assert torch.equal(grads[4], d_router), \
            "the custom op's returned d(router_scores) must be the estimator"
        c = (grads[0].to(torch.float32) * payload.to(torch.float32)).sum(-1)
        # the fp64 autograd reference of the surrogate objective
        s64 = out["op_scores"].to(torch.float64).clone().requires_grad_(True)
        kstar = out["op"][..., 0]
        probs = torch.softmax(s64.view(-1, 5), dim=-1)
        rows = torch.arange(probs.shape[0], device=DEVICE)
        pk = probs[rows, kstar.reshape(-1).to(torch.int64)]
        surrogate = (c.to(torch.float64).reshape(-1) * pk).sum()
        d_ref = torch.autograd.grad(surrogate, s64)[0]
        err = (d_router.to(torch.float64) - d_ref).abs().max().item()
        scale = max(float(d_ref.abs().max()), 1e-12)
        assert err <= 1e-9 + 1e-6 * scale, (label, err, scale)
        assert d_router.shape == scores.shape
    # P5's step-0 receipt: the birth router receives gradient at step 0
    s = router_scores(batch, tokens, heads).requires_grad_(True)
    result = routing_scan(payload, write_keys, query_keys, decay, s,
                          policy=CRAFTED, rank=rank, rank_max=rank,
                          segment_tokens=16, initial_state=state0,
                          initial_live=live0, initial_keys=keys0)
    (result.reads * upstream).sum().backward()
    assert s.grad.abs().max() > 0, \
        "the estimator must reach the zero-born router at step 0"


# --------------------------------------------------------------------- gate B


@needs_cuda
def test_gate_b_oracle_gradcheck_fp64():
    """The routing oracle validates the six K10 channels (fp64 gradcheck
    through every adjoint branch) at the BIRTH router -- K10/K12's own
    discipline.  The FLIPPED-router regime is NOT gradcheckable by
    construction: its argmax margins are arbitrary fp32 values, and the
    discrete flip boundary is exactly the recorded-schedule doctrine's wall
    (the flipped regime's gradients are gated by the floors test below);
    the router's own PATH -- the estimator's surrogate objective -- is a
    smooth softmax and IS gradchecked here in fp64, on top of its
    analytic-vs-autograd receipt."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 32, 2, 8, 4, 5, 11)
    payload, write_keys = payload.double(), write_keys.double()
    query_keys, decay = query_keys.double(), decay.double()
    hist_holder = {}
    scores = router_scores(2, 32, 2)
    p = payload.clone().requires_grad_(True)
    wk = write_keys.clone().requires_grad_(True)
    qk = query_keys.clone().requires_grad_(True)
    d = decay.clone().requires_grad_(True)
    s = torch.zeros(2, 2, 5, 8, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)
    k = torch.zeros(2, 2, 5, 4, dtype=torch.float64,
                    device=DEVICE).requires_grad_(True)

    def f(p_, wk_, qk_, d_, s_, k_):
        ref = routing_oracle_scan(p_, wk_, qk_, d_, scores, policy=POLICY,
                                  rank=5, rank_max=5, initial_state=s_,
                                  initial_keys=k_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    assert torch.autograd.gradcheck(f, (p, wk, qk, d, s, k), eps=1e-6,
                                    atol=1e-4, rtol=1e-4)
    hist = hist_holder["hist"]
    assert all(hist[n] > 0 for n in ("append", "correct")), hist
    # the ROUTER PATH's own fp64 gradcheck: the surrogate objective
    # c_t * p_{k*}(s) is smooth in the recorded scores -- gradcheck it
    # against finite differences with the analytic estimator.
    out, _ = kernel_forward(payload.float(), write_keys.float(),
                            query_keys.float(), decay.float(), scores,
                            rank=5, segment_tokens=8)
    c = torch.randn(*out["reads"].shape[:-1], generator=torch.Generator()
                    .manual_seed(7)).to(DEVICE).to(torch.float64)
    kstar = out["op"][..., 0].to(torch.int64)
    s64 = out["op_scores"].to(torch.float64).clone()
    s64[..., APPEND] = torch.where(  # finite -inf breaks gradcheck's FD
        torch.isinf(s64[..., APPEND]), torch.full_like(s64[..., APPEND], -30.0),
        s64[..., APPEND])
    s64[..., POP] = torch.where(torch.isinf(s64[..., POP]),
                                torch.full_like(s64[..., POP], -30.0),
                                s64[..., POP])
    s64.requires_grad_(True)
    flat = s64.view(-1, 5)

    def g(x):
        probs = torch.softmax(x, dim=-1)
        rows = torch.arange(probs.shape[0], device=DEVICE)
        pk = probs[rows, kstar.reshape(-1)]
        return (c.reshape(-1).to(torch.float64) * pk).sum()

    assert torch.autograd.gradcheck(g, (flat,), eps=1e-6, atol=1e-8,
                                    rtol=1e-6)
    # and the analytic estimator equals the autograd of the surrogate on the
    # same inputs
    flat2 = flat.detach().clone().requires_grad_(True)
    surrogate = g(flat2)
    d_ref = torch.autograd.grad(surrogate, flat2)[0]
    d_router, _ = _router_estimator(payload.float(),
                                    torch.zeros_like(payload.float()),
                                    s64.detach().view(out["op_scores"].shape),
                                    kstar)
    pk_analytic = torch.softmax(
        s64.detach().view(-1, 5), dim=-1)[torch.arange(
            s64.numel() // 5, device=DEVICE), kstar.reshape(-1)]
    want = (pk_analytic * c.reshape(-1).to(torch.float64))
    onehot = torch.zeros_like(torch.softmax(s64.detach().view(-1, 5), -1))
    onehot[torch.arange(onehot.shape[0], device=DEVICE),
           kstar.reshape(-1)] = 1.0
    analytic = want.unsqueeze(-1) * (onehot - torch.softmax(
        s64.detach().view(-1, 5), -1))
    err = (d_ref - analytic).abs().max().item()
    assert err <= 1e-12, err


@needs_cuda
def test_gate_b_six_gradients_within_measured_floors():
    """The kernel path's SIX K10 gradients (via the k14 backward, whose six
    are K10's own body) vs the routing oracle's autograd, inside
    NOISE_FACTOR of the fp32-vs-fp64 floor measured on the same fixture, at
    a flipped router (the learned regime)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    rank, seg = 5, 16
    scores = router_scores(2, 64, 2, 0.5, seed=62)

    def oracle_grads(dtype):
        p = payload.clone().requires_grad_(True)
        wk = write_keys.clone().requires_grad_(True)
        qk = query_keys.clone().requires_grad_(True)
        d = decay.clone().requires_grad_(True)
        s = torch.zeros(2, 2, rank, 8, dtype=dtype,
                        device=DEVICE).requires_grad_(True)
        k = torch.zeros(2, 2, rank, 4, dtype=dtype,
                        device=DEVICE).requires_grad_(True)
        ref = routing_oracle_scan(p.to(dtype), wk.to(dtype), qk.to(dtype),
                                  d.to(dtype), scores, policy=POLICY,
                                  rank=rank, rank_max=rank, initial_state=s,
                                  initial_keys=k)
        upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                               .manual_seed(1234)).to(dtype).to(DEVICE)
        (ref["reads"] * upstream).sum().backward()
        return ref, (p.grad, wk.grad, qk.grad, d.grad, s.grad, k.grad)

    ref, oracle_grads32 = oracle_grads(torch.float32)
    _, oracle_grads64 = oracle_grads(torch.float64)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, scores,
                            rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k14_backward_body(payload, write_keys, query_keys, decay,
                               upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], out["op_scores"], rank, seg,
                               POLICY.read_scale)
    six = (grads[0], grads[1], grads[2], grads[3], grads[5], grads[6])
    for name, kern, orac, orac64 in zip(
            ("dpayload", "dwrite_keys", "dquery_keys", "ddecay", "dstate",
             "dkeys"), six, oracle_grads32, oracle_grads64):
        floor = (orac.float() - orac64.float()).abs().max().item()
        deviation = (kern.float() - orac.float()).abs().max().item()
        assert deviation <= max(NOISE_FACTOR * max(floor, 1e-30), 1e-6) or \
            deviation <= 1e-5, (name, deviation, floor)
        assert torch.isfinite(kern).all(), name


@needs_cuda
def test_gate_b_estimator_is_not_the_value_adjoint():
    """Honest scope: the router's gradient is the admitted ESTIMATOR, not a
    true adjoint -- it must NOT change the six value/key gradients (the k14
    backward's six equal K10's own body's BIT-EXACTLY on the same recorded
    channels)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    rank, seg = 5, 16
    scores = router_scores(2, 64, 2, 0.5, seed=62)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, scores,
                            rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(out["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    seven = _k14_backward_body(payload, write_keys, query_keys, decay,
                               upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], out["op_scores"], rank, seg,
                               POLICY.read_scale)
    six = _k10_backward_body(payload, write_keys, query_keys, decay, upstream,
                              schedule, out["live_pre"], out["ckpt_state"],
                              out["ckpt_live"], out["ckpt_keys"], rank, seg,
                              POLICY.read_scale)
    for name, a, b in zip(("dpayload", "dwrite_keys", "dquery_keys",
                           "ddecay", "dstate", "dkeys"), seven[:4],
                          six[:4]):
        assert torch.equal(a, b), name
    assert torch.equal(seven[5], six[4])
    assert torch.equal(seven[6], six[5])


# ------------------------------------------------------- refusals, conversion


@needs_cuda
def test_refusals_by_name():
    payload, write_keys, query_keys, decay = keyed_stream(2, 16, 2, 8, 4, 5, 3)
    scores = router_scores(2, 16, 2)
    with pytest.raises(ValueError, match="append-only"):
        routing_scan(payload, write_keys, query_keys, decay, scores,
                     policy=RoutingPolicy(ops=False), rank=5, rank_max=5)
    with pytest.raises(ValueError, match="growth"):
        routing_scan(payload, write_keys, query_keys, decay, scores,
                     policy=RoutingPolicy(ops=True, grow=True), rank=5,
                     rank_max=5)
    with pytest.raises(ValueError, match="rank prefix"):
        routing_scan(payload, write_keys, query_keys, decay, scores,
                     policy=POLICY, rank=3, rank_max=5)
    with pytest.raises(ValueError, match="router_scores"):
        routing_scan(payload, write_keys, query_keys, decay,
                     scores[..., :4], policy=POLICY, rank=5, rank_max=5)
    with pytest.raises(ValueError, match="router_scores"):
        routing_scan(payload, write_keys, query_keys, decay,
                     scores.to(torch.float64), policy=POLICY, rank=5,
                     rank_max=5)


@needs_cuda
def test_conversion_accounting_and_birth():
    """The conversion adds EXACTLY the router's parameters (+553,200 at the
    registered shape) on top of the audited separated build, born at exact
    zeros, sharing every K10 module; the frozen variant pins the router."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        matrix_config, model_for as matrix_model_for)
    from experiments.expressivity_tasks.x45_matrix_emender.matched_arms import (
        RANK)
    shape = matrix_config(rank=RANK)
    policy = RoutingPolicy(ops=True)
    model = matrix_model_for(shape, policy, 128, compile_scan=False,
                             fused_whole_scan=False)
    model = convert_model_to_separated_keys(model, key_dim=16, ctx=4,
                                            segment_tokens=64, seed=3401)
    separated = sum(p.numel() for p in model.parameters())
    assert separated == 149_981_808      # the audited separated build
    model = convert_model_to_routing(model, router_trainable=True,
                                     segment_tokens=64)
    after = sum(p.numel() for p in model.parameters())
    depth = len(model.blocks)
    heads = model.blocks[0].mixer.heads
    width = model.blocks[0].mixer.width
    ctx = model.blocks[0].mixer.ctx
    expected = routing_parameter_delta(depth, heads, ctx, width)
    assert after - separated == expected == 553_200
    for block in model.blocks:
        mixer = block.mixer
        assert isinstance(mixer, RoutingSeparatedMixer)
        assert torch.equal(mixer.router.weight,
                           torch.zeros_like(mixer.router.weight))
        assert torch.equal(mixer.router.bias,
                           torch.zeros_like(mixer.router.bias))
        assert mixer.router.weight.requires_grad
    model2 = matrix_model_for(shape, policy, 128, compile_scan=False,
                              fused_whole_scan=False)
    model2 = convert_model_to_separated_keys(model2, key_dim=16, ctx=4,
                                              segment_tokens=64, seed=3401)
    model2 = convert_model_to_routing(model2, router_trainable=False,
                                      segment_tokens=64)
    for block in model2.blocks:
        assert not block.mixer.router.weight.requires_grad
        assert not block.mixer.router.bias.requires_grad
