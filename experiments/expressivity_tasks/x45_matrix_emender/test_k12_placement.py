"""K12 -- the learned placement arm, gated.

The K10 parity discipline, extended to the placement channel
(``PREREGISTRATION_PLACEMENT.md``, the supervisor's rulings cited in
``k12_placement.py``'s header):

* **Bit-equality by construction**: the K10 defined-order spec extends to the
  residue axis; the eager oracle implements it in torch and the kernels in
  Triton, so every decision channel -- slot (the PLACED slot at APPEND
  included), op, erase, live_after, the payload lane, the read WEIGHTS, the
  state, the liveness, the pre-decision liveness, the key table AND the
  residue table -- must be ``torch.equal`` against the oracle on every
  fixture and at every placement weight, including the bf16 autocast regime.
* **Birth parity (P5)**: at ``w = +0.0`` the k12 launch is BIT-EQUAL to K10's
  OWN launch on every K10 channel -- the zero-born argmax tie-break reproduces
  the first-free rule exactly -- and the frozen-weight falsifier control
  trains bit-equal to the K10 machine for as long as ``w`` stays frozen.
* **The zero-gradient finding**: the ORACLE path's structural gradient to
  ``w`` is exactly zero (asserted) -- under P2+P3 the placement effect is
  purely discrete; the kernel path's straight-through estimator is the
  admitted exception, and it must equal its DEFINED torch reference to the
  float-summation floor, with a nonzero signal on a fixture that fires.
* **The gradient gates are floor-discipline** (K10's): the fp64 oracle
  validates the six K10 channels through the placement oracle; the kernel
  path's six gradients sit within ``NOISE_FACTOR`` of the measured floor.
  The placement channel's gradient is NOT the true adjoint (the finding) and
  is gated by its reference, not by gradcheck.
* **Refusals by name**, **determinism**, and the conversion's parameter
  accounting (+48 at the registered shape).

Every test needs CUDA (Triton kernels).
"""
from __future__ import annotations

import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    APPEND, MERGE, NONE, OP_NAMES, POP, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    SeparatedSequentialMixer, _k10_backward_body, _k10_forward_launch,
    convert_model_to_separated_keys, separated_available,
    separated_sequential_oracle_scan, separated_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k12_placement import (
    PlacementSeparatedMixer, _k12_backward_body, _k12_forward_launch,
    convert_model_to_placement, placement_available, placement_oracle_scan,
    placement_parameter_delta, placement_scan)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32, tree_fold, _norm_defined, _pow2)
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
# the crafted regime: high tau_match so unmatched keys POP/MERGE, and a
# mergeable key pair -- the fixture that populates the residue channel.
CRAFTED = RoutingPolicy(ops=True, tau_match=0.95, tau_merge=0.85, tau_pop=0.10)
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


def placement_weight(heads, value):
    return torch.full((heads,), float(value), dtype=torch.float32,
                      device=DEVICE)


def kernel_forward(payload, write_keys, query_keys, decay, placement_w, rank,
                   key_dim=None, segment_tokens=64, collect=True, policy32=POL32,
                   initial_state=None, initial_live=None, initial_keys=None,
                   initial_residue=None):
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
    residue = (torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                           device=DEVICE) if initial_residue is None
              else initial_residue.contiguous())
    return _k12_forward_launch(payload, write_keys, query_keys, decay,
                               placement_w, state, live, keys, residue,
                               policy32, rank, segment_tokens, collect=collect,
                               warps=4 if rank * value_dim < 8192 else 8,
                               stages=3), state


def oracle_reference(payload, write_keys, query_keys, decay, placement_w,
                     rank, policy=POLICY, collect=True, initial_state=None,
                     initial_live=None, initial_keys=None,
                     initial_residue=None):
    return placement_oracle_scan(payload, write_keys, query_keys, decay,
                                 placement_w, policy=policy, rank=rank,
                                 rank_max=rank, collect=collect,
                                 initial_state=initial_state,
                                 initial_live=initial_live,
                                 initial_keys=initial_keys,
                                 initial_residue=initial_residue)


def assert_decisions_bit_equal(out, ref, label):
    """Gate A: every decision channel (the PLACED slot included), the payload
    lane, the weights, the state, the liveness, the pre-decision liveness, the
    key table AND the residue table bit-equal to the oracle; the read within
    the floor."""
    for name in SCHEDULE_FIELDS:
        assert torch.equal(getattr(ref["schedule"], name), out[name]), \
            (label, name)
    assert torch.equal(ref["schedule"].payload, out["payload"]), (label, "payload")
    assert torch.equal(ref["weights"], out["weights"]), (label, "weights")
    assert torch.equal(ref["state"], out["state"]), (label, "state")
    assert torch.equal(ref["live"], out["live"]), (label, "live")
    assert torch.equal(ref["live_pre"], out["live_pre"]), (label, "live_pre")
    assert torch.equal(ref["keys"], out["keys"]), (label, "keys")
    assert torch.equal(ref["residue"], out["residue"]), (label, "residue")
    read_dev = (ref["reads"] - out["reads"]).abs().max().item()
    assert read_dev <= 1e-5, (label, "read", read_dev)


def first_free_slots(out):
    """The K10 first-free rule read off the recorded channels: the lowest
    free index at each token."""
    return (~out["live_pre"]).to(torch.int64).argmax(-1)


def crafted_placement_fixture(batch=2, tokens=10, heads=2, V=8, KD=4, rank=6,
                              seed=3):
    """The residue-populating regime: a full store of distinct keys, then a
    stream of poppers (POP), merge-pair keys (MERGE kills the source row) and
    residue-matching keys (APPEND) -- deaths and appends interleave, so the
    residue channel fills and placement fires differentially."""
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
    seq = [u([1, 1, -1, -1]), u([1, 0, 0, 0]), u([0, 0, 0, 1]),
           u([0, 0, 1, 0]), u([1, 1, -1, -1]), u([0.9, 0.436, 0, 0]),
           u([0, 0, 0, 1]), u([0, 0, 1, 0]), u([1, 1, -1, -1]),
           u([1, 0, 0, 0])]
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
def test_gate_a_all_five_ops_regime_bit_equal():
    """The lane's canonical fixture at four placement weights: every channel
    bit-equal, the placed slot included."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    for w in (0.0, 0.37, -0.5, 1.0):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                placement_weight(2, w), rank=5,
                                segment_tokens=16)
        ref = oracle_reference(payload, write_keys, query_keys, decay,
                               placement_weight(2, w), 5)
        assert_decisions_bit_equal(out, ref, f"w={w}")


@needs_cuda
def test_gate_a_crafted_placement_regime():
    """Deaths populate the residue channel; placement fires differentially at
    w > 0 and reproduces first-free at w = 0; the residue records the dying
    rows' PRE-clear keys."""
    (payload, write_keys, query_keys, decay, state0, keys0,
     live0) = crafted_placement_fixture()
    rank = keys0.shape[2]
    outs = {}
    for w in (0.0, 1.0):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                placement_weight(2, w), rank=rank,
                                segment_tokens=16, policy32=CRAFTED32,
                                initial_state=state0, initial_live=live0,
                                initial_keys=keys0)
        ref = oracle_reference(payload, write_keys, query_keys, decay,
                               placement_weight(2, w), rank, policy=CRAFTED,
                               initial_state=state0, initial_live=live0,
                               initial_keys=keys0)
        assert_decisions_bit_equal(out, ref, f"crafted w={w}")
        outs[w] = out
    ops = outs[0.0]["op"][..., 0]
    hist = torch.bincount(ops.reshape(-1), minlength=len(OP_NAMES))
    assert hist[APPEND] > 0 and hist[POP] > 0 and hist[MERGE] > 0, hist.tolist()
    # the residue must record the dying rows' pre-clear keys
    residue_nonzero = int((outs[0.0]["residue"].norm(dim=-1) > 0).sum())
    assert residue_nonzero > 0
    # at w=0 the APPEND slots are exactly the first-free rule
    is_append = ops == APPEND
    assert torch.equal(outs[0.0]["slot"][..., 0][is_append],
                       first_free_slots(outs[0.0])[is_append])
    # at w=1 placement must fire somewhere the first-free rule would not
    fired = ((outs[1.0]["slot"][..., 0] != first_free_slots(outs[1.0]))
             & is_append)
    assert int(fired.sum()) > 0, "placement must fire differentially at w=1"
    # the collect receipt channels: the chosen slot's residue similarity and
    # the free-slot mean, nonzero at the firing appends
    assert float(outs[1.0]["place_sim"][is_append].abs().max()) > 0.5


@needs_cuda
def test_gate_a_birth_parity_vs_k10_launch():
    """P5's birth bar: at w = +0.0 the k12 launch is BIT-EQUAL to K10's own
    launch on every K10 channel (the residue channel is new and excluded)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 128, 2, 8, 4, 6, 91)
    out12, _ = kernel_forward(payload, write_keys, query_keys, decay,
                              placement_weight(2, 0.0), rank=5,
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
        assert torch.equal(out10[name], out12[name]), ("birth", name)
        lanes += out10[name].numel()
        flips += int((out10[name] != out12[name]).sum())
    for name in ("payload", "live_pre"):
        assert torch.equal(out10[name], out12[name]), ("birth", name)
        lanes += out10[name].numel()
        flips += int((out10[name] != out12[name]).sum())
    for name in ("weights", "state", "live", "keys", "reads"):
        assert torch.equal(out10[name], out12[name]), ("birth", name)
        lanes += out10[name].numel()
        flips += int((out10[name] != out12[name]).sum())
    assert flips == 0 and lanes > 0


@needs_cuda
def test_gate_a_frozen_w_falsifier_bit_equal_through_training():
    """The never-fires falsifier control: with w frozen at exact +0.0 the
    placement machine trains BIT-EQUAL to the K10 machine -- the losses equal
    at every step for three optimizer steps (the trajectory cannot diverge
    while w stays exactly +0.0: the forward is the same bit-exact machine and
    the frozen weight never enters the optimizer)."""
    torch.manual_seed(4242)
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
    m12 = build(PlacementSeparatedMixer,
                dict(placement_init=0.0, placement_trainable=False))
    assert torch.equal(m12.placement_weight,
                       torch.zeros_like(m12.placement_weight))
    # the shared modules must be bit-identical at birth (the same build order)
    for name in ("value", "key", "query_key", "out", "decay"):
        a = getattr(m10, name).weight
        b = getattr(m12, name).weight
        assert torch.equal(a, b), name
    opt10 = torch.optim.AdamW(m10.parameters(), lr=1e-3, fused=True)
    opt12 = torch.optim.AdamW((p for p in m12.parameters() if p.requires_grad),
                              lr=1e-3, fused=True)
    for step in range(3):
        loss10 = ((m10(x) - target) ** 2).mean()
        opt10.zero_grad(set_to_none=True)
        loss10.backward()
        torch.nn.utils.clip_grad_norm_(m10.parameters(), 1.0)
        opt10.step()
        loss12 = ((m12(x) - target) ** 2).mean()
        opt12.zero_grad(set_to_none=True)
        loss12.backward()
        torch.nn.utils.clip_grad_norm_(m12.parameters(), 1.0)
        opt12.step()
        assert loss10.item() == loss12.item(), (step, loss10.item(),
                                                loss12.item())
        assert torch.equal(m12.placement_weight,
                           torch.zeros_like(m12.placement_weight))


@needs_cuda
def test_gate_a_ragged_tokens_and_segments():
    payload, write_keys, query_keys, decay = keyed_stream(2, 51, 2, 8, 4, 5, 7)
    for seg in (8, 16, 64):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                placement_weight(2, 0.61), rank=5,
                                segment_tokens=seg)
        ref = oracle_reference(payload, write_keys, query_keys, decay,
                               placement_weight(2, 0.61), 5)
        assert_decisions_bit_equal(out, ref, f"ragged seg={seg}")


@needs_cuda
def test_gate_a_bf16_autocast_regime():
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 13)
    payload, write_keys = payload.bfloat16(), write_keys.bfloat16()
    query_keys, decay = query_keys.bfloat16(), decay.bfloat16()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                                placement_weight(2, 0.83), rank=5,
                                segment_tokens=16)
    ref = oracle_reference(payload, write_keys, query_keys, decay,
                           placement_weight(2, 0.83), 5)
    assert_decisions_bit_equal(out, ref, "bf16")


@needs_cuda
def test_gate_a_tie_break_first_index():
    """Equal placement scores (equal residues) tie-break to the lowest free
    index in BOTH implementations -- the birth construction at w > 0."""
    batch, tokens, heads, V, KD, rank = 1, 8, 1, 8, 4, 4
    g = torch.Generator().manual_seed(5)
    payload = torch.randn(batch, tokens, heads, V, generator=g).to(DEVICE)
    decay = torch.full((batch, tokens, heads), 0.9, device=DEVICE)
    # two dead rows with IDENTICAL residue keys: every score ties
    state0 = torch.zeros(batch, heads, rank, V, device=DEVICE)
    keys0 = torch.zeros(batch, heads, rank, KD, device=DEVICE)
    live0 = torch.zeros(batch, heads, rank, dtype=torch.bool, device=DEVICE)
    res0 = torch.zeros(batch, heads, rank, KD, device=DEVICE)
    res0[:, :, 2] = torch.tensor([0.6, 0.8, 0., 0.], device=DEVICE)
    res0[:, :, 3] = torch.tensor([0.6, 0.8, 0., 0.], device=DEVICE)
    write_keys = (torch.randn(batch, tokens, heads, KD, generator=g)
                  .to(DEVICE) * 2.0)
    query_keys = torch.randn(batch, tokens, heads, KD, generator=g).to(DEVICE)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                            placement_weight(1, 1.0), rank=rank,
                            segment_tokens=4, initial_state=state0,
                            initial_live=live0, initial_keys=keys0,
                            initial_residue=res0)
    ref = oracle_reference(payload, write_keys, query_keys, decay,
                           placement_weight(1, 1.0), rank,
                           initial_state=state0, initial_live=live0,
                           initial_keys=keys0, initial_residue=res0)
    assert_decisions_bit_equal(out, ref, "tie")
    is_append = out["op"][..., 0] == APPEND
    if bool(is_append.any()):
        # the first APPEND sees both equal residues live in the score: the
        # tie must resolve to the LOWER of the two tied free slots (2 < 3),
        # and both implementations agree on it (the parity assert above
        # already pins the rest of the trajectory)
        slots = out["slot"][..., 0][is_append]
        assert int(slots.reshape(-1)[0]) == 2, slots.reshape(-1).tolist()


@needs_cuda
def test_gate_a_heterogeneous_lanes_only_some_merge():
    """The multi-lane MERGE corner (found by the gate receipt's placement
    regime): when only SOME (batch, head) lanes merge at a token, the oracle
    must kill ONLY the merging lanes' source rows -- the unmasked
    ``if merge.any()`` index-put form is the latent multi-lane bug this
    module's copy fixes (K10's own oracle carries it, unreachable by its
    gate fixtures and inert in production; the finding is receipted in
    K12_PLACEMENT.md)."""
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
    # lane (b, h=0) gets the merge-pair key e1 (second = the 0.9-cos twin
    # row: MERGE); lane (b, h=1) gets an UNMATCHED key (no merge) -- the
    # heterogeneous token
    wk = torch.zeros(batch, tokens, heads, KD)
    wk[:, :, 0] = torch.tensor([1.0, 0, 0, 0])
    wk[:, :, 1] = torch.tensor([0.9, 0.436, 1.0, 0.0])   # ~orthogonal to all
    write_keys = wk.to(DEVICE)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay,
                            placement_weight(2, 0.5), rank=rank,
                            segment_tokens=4, policy32=CRAFTED32,
                            initial_state=state0, initial_live=live0,
                            initial_keys=keys0)
    ref = oracle_reference(payload, write_keys, query_keys, decay,
                            placement_weight(2, 0.5), rank, policy=CRAFTED,
                            initial_state=state0, initial_live=live0,
                            initial_keys=keys0)
    assert_decisions_bit_equal(out, ref, "heterogeneous")
    ops0 = out["op"][:, :, 0, 0]      # lane h=0: MERGE at t=0 (the pair
    # is consumed by the first merge: its source row dies, so later e1 keys
    # are plain CORRECTs on the surviving row)
    ops1 = out["op"][:, :, 1, 0]      # lane h=1: no MERGE anywhere
    assert bool((ops0[:, 0] == MERGE).all()), ops0.tolist()
    assert not bool((ops0[:, 1:] == MERGE).any()), ops0.tolist()
    assert not bool((ops1 == MERGE).any()), ops1.tolist()
    # the multi-lane receipt: the merging lane's source row dies at t=0 while
    # the non-merging lane's second-best row SURVIVES (the unmasked form
    # would kill it)
    for b in range(batch):
        second0 = int(out["slot"][b, 0, 0, 1])      # h=0's merge source
        second1 = int(out["slot"][b, 0, 1, 1])      # h=1's second-best row
        assert not bool(out["live_pre"][b, 1, 0, second0])
        target1 = int(out["slot"][b, 0, 1, 0])
        popped1 = int(out["op"][b, 0, 1, 0]) == POP
        if not (popped1 and target1 == second1):
            assert bool(out["live_pre"][b, 1, 1, second1]), \
                (b, second1, out["live_pre"][b, 1, 1].tolist())


@needs_cuda
def test_gate_a_determinism():
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 29)
    outs = [kernel_forward(payload, write_keys, query_keys, decay,
                           placement_weight(2, 0.44), rank=5,
                           segment_tokens=16)[0] for _ in range(2)]
    for name in ("slot", "op", "payload", "state", "keys", "residue",
                 "reads", "place_sim"):
        assert torch.equal(outs[0][name], outs[1][name]), name


# ------------------------------------------------- the finding and the estimator


@needs_cuda
def test_the_zero_gradient_finding():
    """THE FINDING, asserted: the ORACLE path's structural gradient to the
    placement weight is EXACTLY ZERO on a fixture where placement fires and
    the loss depends on the store -- under P2+P3 the placement effect is
    purely discrete and the recorded-schedule doctrine makes the slot a
    backward constant.  (The kernel path's straight-through estimator is the
    admitted exception; the next test gates it.)"""
    (payload, write_keys, query_keys, decay, state0, keys0,
     live0) = crafted_placement_fixture()
    rank = keys0.shape[2]
    w = placement_weight(2, 0.7).requires_grad_(True)
    p = payload.clone().requires_grad_(True)
    ref = oracle_reference(payload=p, write_keys=write_keys,
                           query_keys=query_keys, decay=decay, placement_w=w,
                           rank=rank, policy=CRAFTED, initial_state=state0,
                           initial_live=live0, initial_keys=keys0)
    grads = torch.autograd.grad(ref["reads"].sum(), [w, p], allow_unused=True)
    assert grads[0] is None or torch.equal(
        grads[0], torch.zeros_like(grads[0])), grads[0]


@needs_cuda
def test_the_estimator_matches_its_reference():
    """The admitted straight-through estimator: the kernel path's dL/dw
    equals its DEFINED torch reference (replayed from the recorded channels,
    with g_t from the oracle's own value adjoint) to the float-summation
    floor, on a fixture whose placement fires -- and the gradient is nonzero
    at the zero-born w (P5's step-0 receipt)."""
    (payload, write_keys, query_keys, decay, state0, keys0,
     live0) = crafted_placement_fixture()
    batch, tokens, heads, V = payload.shape
    rank, KD = keys0.shape[2], keys0.shape[-1]
    res0 = torch.zeros_like(keys0)
    g = torch.Generator().manual_seed(31)
    upstream = torch.randn(batch, tokens, heads, V, generator=g).to(DEVICE)

    def kernel_dw(w_value):
        w = placement_weight(heads, w_value).requires_grad_(True)
        result = placement_scan(payload, write_keys, query_keys, decay, w,
                                policy=CRAFTED, rank=rank, rank_max=rank,
                                segment_tokens=16, initial_state=state0,
                                initial_live=live0, initial_keys=keys0,
                                initial_residue=res0)
        (result.reads * upstream).sum().backward()
        return w.grad.clone()

    def reference_dw(w_value, g_chan):
        w_t = placement_weight(heads, w_value)
        out, _ = kernel_forward(payload, write_keys, query_keys, decay, w_t,
                                rank=rank, segment_tokens=16,
                                policy32=CRAFTED32, initial_state=state0,
                                initial_live=live0, initial_keys=keys0,
                                initial_residue=res0)
        KP = _pow2(KD)
        op0_all = out["op"][..., 0]
        slot0_all = out["slot"][..., 0]
        merge_all = out["op"][..., 1] == MERGE
        dw = torch.zeros(heads, device=DEVICE)
        kacc, racc = keys0.clone(), res0.clone()
        for t in range(tokens):
            live = out["live_pre"][:, t]
            free = ~live
            wk = write_keys[:, t].detach()
            if bool((op0_all[:, t] == APPEND).any()):
                nwk = _norm_defined(wk, -1, KP)
                uk = wk / nwk.clamp_min(EPS).unsqueeze(-1)
                nres = _norm_defined(racc, -1, KP)
                simres = tree_fold(uk.unsqueeze(2) * racc, -1, KP) \
                    / nres.clamp_min(EPS)
                s = w_t.view(1, -1, 1) * simres + 0.0
                s = torch.where(free, s, torch.full_like(s, float("-inf")))
                p = torch.softmax(s, dim=-1)
                slot0 = slot0_all[:, t]
                sim_c = simres.gather(-1, slot0.unsqueeze(-1)).squeeze(-1)
                p_c = p.gather(-1, slot0.unsqueeze(-1)).squeeze(-1)
                p_mean = (p * simres).sum(-1)
                term = g_chan[:, t] * p_c * (sim_c - p_mean)
                is_app = (op0_all[:, t] == APPEND).to(term.dtype)
                dw += (term * is_app).sum(0)
            for b in range(batch):
                for h in range(heads):
                    o = int(op0_all[b, t, h])
                    sl = int(slot0_all[b, t, h])
                    second = int(out["slot"][b, t, h, 1])
                    if o == POP:
                        racc[b, h, sl] = kacc[b, h, sl]
                    if bool(merge_all[b, t, h]):
                        racc[b, h, second] = kacc[b, h, second]
                    if o != NONE:
                        kacc[b, h, sl] = (torch.zeros(KD, device=DEVICE)
                                          if o == POP else wk[b, h].detach())
                    if bool(merge_all[b, t, h]):
                        kacc[b, h, second] = 0.0
        return dw

    for w_value in (0.0, 0.7):
        # g_t = <dpayload_t, p_t> from the oracle's own value adjoint
        p_o = payload.clone().requires_grad_(True)
        ref = oracle_reference(payload=p_o, write_keys=write_keys,
                               query_keys=query_keys, decay=decay,
                               placement_w=placement_weight(heads, w_value),
                               rank=rank, policy=CRAFTED,
                               initial_state=state0, initial_live=live0,
                               initial_keys=keys0)
        (ref["reads"] * upstream).sum().backward()
        g_chan = (p_o.grad * payload).sum(-1)
        kern = kernel_dw(w_value)
        want = reference_dw(w_value, g_chan)
        err = (kern - want).abs().max().item()
        scale = max(float(want.abs().max()), 1e-6)
        assert err <= 1e-4 + 1e-5 * scale, (w_value, err, scale)
    # P5's step-0 receipt: the gradient reaches the ZERO-BORN w at step 0
    assert kernel_dw(0.0).abs().max() > 1e-6, \
        "the estimator must reach the zero-born weight at step 0"
    # and the signal is nonzero at the firing weight
    assert kernel_dw(0.7).abs().max() > 1e-3


# --------------------------------------------------------------------- gate B


@needs_cuda
def test_gate_b_oracle_gradcheck_fp64():
    """The placement oracle validates the six K10 channels (fp64 gradcheck
    through every adjoint branch); the placement weight is excluded -- its
    structural gradient is the zero of the finding."""
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
        ref = placement_oracle_scan(p_, wk_, qk_, d_,
                                    torch.zeros(2, dtype=torch.float32,
                                                device=DEVICE), policy=POLICY,
                                    rank=5, rank_max=5, initial_state=s_,
                                    initial_keys=k_)
        hist_holder["hist"] = op_histogram(ref["schedule"])
        return ref["reads"], ref["state"], ref["keys"]

    assert torch.autograd.gradcheck(f, (p, wk, qk, d, s, k), eps=1e-6,
                                    atol=1e-4, rtol=1e-4)
    hist = hist_holder["hist"]
    assert all(hist[n] > 0 for n in ("append", "correct")), hist


@needs_cuda
def test_gate_b_six_gradients_within_measured_floors():
    """The kernel path's SIX K10 gradients (via the k12 backward, which is
    K10's own body) vs the placement oracle's autograd, inside NOISE_FACTOR
    of the fp32-vs-fp64 floor measured on the same fixture."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    rank, seg = 5, 16
    w = placement_weight(2, 0.5)

    def oracle_grads(dtype):
        p = payload.clone().requires_grad_(True)
        wk = write_keys.clone().requires_grad_(True)
        qk = query_keys.clone().requires_grad_(True)
        d = decay.clone().requires_grad_(True)
        s = torch.zeros(2, 2, rank, 8, dtype=dtype,
                        device=DEVICE).requires_grad_(True)
        k = torch.zeros(2, 2, rank, 4, dtype=dtype,
                        device=DEVICE).requires_grad_(True)
        ref = placement_oracle_scan(p.to(dtype), wk.to(dtype), qk.to(dtype),
                                    d.to(dtype), w, policy=POLICY,
                                    rank=rank, rank_max=rank, initial_state=s,
                                    initial_keys=k)
        upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                               .manual_seed(1234)).to(dtype).to(DEVICE)
        (ref["reads"] * upstream).sum().backward()
        return ref, (p.grad, wk.grad, qk.grad, d.grad, s.grad, k.grad)

    ref, oracle_grads32 = oracle_grads(torch.float32)
    _, oracle_grads64 = oracle_grads(torch.float64)
    out, _ = kernel_forward(payload, write_keys, query_keys, decay, w,
                            rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(ref["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    grads = _k12_backward_body(payload, write_keys, query_keys, decay, w,
                               upstream, schedule, out["live_pre"],
                               out["ckpt_state"], out["ckpt_live"],
                               out["ckpt_keys"], out["state"], out["residue"],
                               rank, seg, POLICY.read_scale)
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
    """Honest scope: the placement weight's gradient is the admitted
    ESTIMATOR, not a true adjoint -- it must NOT change the six value/key
    gradients (the k12 backward's six equal K10's own body's on the same
    recorded channels)."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 64, 2, 8, 4, 6, 23)
    rank, seg = 5, 16
    w = placement_weight(2, 0.5)
    out, state0 = kernel_forward(payload, write_keys, query_keys, decay, w,
                                  rank=rank, segment_tokens=seg)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    upstream = torch.randn(out["reads"].shape, generator=torch.Generator()
                           .manual_seed(1234)).to(DEVICE)
    twelve = _k12_backward_body(payload, write_keys, query_keys, decay, w,
                                upstream, schedule, out["live_pre"],
                                out["ckpt_state"], out["ckpt_live"],
                                out["ckpt_keys"], out["state"], out["residue"],
                                rank, seg, POLICY.read_scale)
    ten = _k10_backward_body(payload, write_keys, query_keys, decay, upstream,
                             schedule, out["live_pre"], out["ckpt_state"],
                             out["ckpt_live"], out["ckpt_keys"], rank, seg,
                             POLICY.read_scale)
    # twelve: (dpayload, dwrite_keys, dquery_keys, ddecay, dw, dstate, dkeys)
    # ten:    (dpayload, dwrite_keys, dquery_keys, ddecay, dstate, dkeys)
    for name, a, b in zip(("dpayload", "dwrite_keys", "dquery_keys",
                           "ddecay", "dstate", "dkeys"),
                          (twelve[0], twelve[1], twelve[2], twelve[3],
                           twelve[5], twelve[6]), ten):
        assert torch.equal(a, b), name


# ------------------------------------------------------ refusals and the wiring


@needs_cuda
def test_refusals_are_by_name():
    payload, write_keys, query_keys, decay = keyed_stream(2, 16, 2, 8, 4, 5, 3)
    w = placement_weight(2, 0.5)
    with pytest.raises(ValueError, match="append-only"):
        placement_scan(payload, write_keys, query_keys, decay, w,
                       policy=RoutingPolicy(ops=False), rank=5, rank_max=5)
    with pytest.raises(ValueError, match="growth"):
        placement_scan(payload, write_keys, query_keys, decay, w,
                       policy=RoutingPolicy(ops=True, grow=True), rank=5,
                       rank_max=5)
    with pytest.raises(ValueError, match="rank prefix"):
        placement_scan(payload, write_keys, query_keys, decay, w,
                       policy=POLICY, rank=5, rank_max=6)
    with pytest.raises(ValueError, match="placement_w"):
        placement_scan(payload, write_keys, query_keys, decay,
                       torch.zeros(3, device=DEVICE), policy=POLICY, rank=5,
                       rank_max=5)


@needs_cuda
def test_cpu_inputs_take_the_oracle_verbatim():
    """Non-CUDA inputs take the eager oracle (the honest hard semantics);
    the kernel path is asserted available at the registered shape by the
    probe."""
    payload, write_keys, query_keys, decay = keyed_stream(2, 16, 2, 8, 4, 5, 3)
    cpu = tuple(t.cpu() for t in (payload, write_keys, query_keys, decay))
    w = placement_weight(2, 0.5).cpu()
    result = placement_scan(cpu[0], cpu[1], cpu[2], cpu[3], w, policy=POLICY,
                            rank=5, rank_max=5)
    ref = placement_oracle_scan(cpu[0], cpu[1], cpu[2], cpu[3], w,
                                policy=POLICY, rank=5, rank_max=5,
                                collect=False)
    assert torch.equal(result.reads, ref["reads"])
    assert torch.equal(result.state, ref["state"])


@needs_cuda
def test_conversion_parameter_accounting_and_shared_modules():
    """The conversion shares K10's modules and adds exactly one placement
    weight per head per mixer (+48 at the registered depth/heads), born at
    exact zero (learned) or pinned (fixed/frozen)."""
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        matrix_config, model_for as matrix_model_for)
    from experiments.expressivity_tasks.x45_matrix_emender.matched_arms import (
        RANK)
    shape = matrix_config(rank=RANK)
    policy = RoutingPolicy(ops=True)
    model = matrix_model_for(shape, policy, 128, compile_scan=False,
                             fused_whole_scan=False)
    before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_separated_keys(model, key_dim=16, ctx=4,
                                            segment_tokens=64, seed=3401)
    separated = sum(p.numel() for p in model.parameters())
    model = convert_model_to_placement(model, placement_init=0.0,
                                       placement_trainable=True,
                                       segment_tokens=64)
    after = sum(p.numel() for p in model.parameters())
    depth, heads = len(model.blocks), model.blocks[0].mixer.heads
    assert after - separated == placement_parameter_delta(depth, heads)
    assert separated - before == 0       # the K10 swap is parameter-neutral
    for block in model.blocks:
        mixer = block.mixer
        assert isinstance(mixer, PlacementSeparatedMixer)
        assert mixer.placement_weight.shape == (heads,)
        assert torch.equal(mixer.placement_weight,
                           torch.zeros_like(mixer.placement_weight))
        assert mixer.placement_weight.requires_grad
    # the pinned/fixed conversion freezes the weight at the given value
    model2 = matrix_model_for(shape, policy, 128, compile_scan=False,
                              fused_whole_scan=False)
    model2 = convert_model_to_separated_keys(model2, key_dim=16, ctx=4,
                                             segment_tokens=64, seed=3401)
    model2 = convert_model_to_placement(model2, placement_init=0.5,
                                        placement_trainable=False,
                                        segment_tokens=64)
    for block in model2.blocks:
        assert not block.mixer.placement_weight.requires_grad
        assert torch.equal(block.mixer.placement_weight,
                          torch.full_like(block.mixer.placement_weight, 0.5))


@needs_cuda
def test_placement_available_matches_k10():
    assert placement_available(128, 128, 16) == separated_available(128, 128, 16)
    assert placement_available(128, 128, 16)
    assert not placement_available(256, 128, 16)
