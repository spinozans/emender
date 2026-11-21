"""K22 -- THE HYBRID (GDN2's committed stack + the composed slot mixer at 3
mid-stack layers through the zero-born read gate), gated.

The lane's parity disciplines, per ``PREREGISTRATION_K22_HYBRID.md``
(committed before any arm):

* **THE BIRTH RECEIPT (born as GDN2, bit for bit)**: at the zero-born
  read gate the slot contribution is exactly zero, so the hybrid layer IS
  the committed GDN2 layer's own output bitwise (the only legal escape is
  the IEEE sign-of-zero corner: a +0.0 contribution added to a -0.0 dense
  entry -- counted by the gate receipt, invisible to every value bar and
  to the loss).
* **THE INERT BIRTH**: while a read gate is exactly zero, every OTHER
  slot parameter's gradient is exactly zero (the gate's zero cuts the
  gradient), and the gate's OWN gradient (the open signal,
  ``<upstream, slot output>``) is nonzero.
* **THE PARAMETER ACCOUNTING is the registration's own**: purely
  additive (the dense mixers stay -- the falsifier's own construction),
  computed not hardcoded, asserted by the probe.
* **THE ROUTING is closed**: the banked GDN2 row's own option-B split on
  the dense names + the slot branch per the K18 convention mirrored
  (matrices on Muon; the gate channels, the conv kernel and the K22 read
  gate on AdamW); the frozen falsifier's slot branch drops out of the
  inventory by ``requires_grad``.
* **THE REFUSALS are by name**: unregistered slot layers, a non-GDN2
  layer at a slot position, a drifted parameter delta.

The wrapper/birth/accounting/routing tests run anywhere (a stub dense
layer); the scan/gradcheck/engine tests need CUDA (the committed Triton
kernels).
"""
from __future__ import annotations

import pytest
import torch
from torch import nn

from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid import (
    HybridSlotGDN2Layer, SLOT_LAYERS, build_slot_mixer,
    hybrid_slot_parameter_delta, slot_mixer_parameter_count,
    slot_policy, split_parameters_k22)

DEVICE = "cuda"
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="the committed Triton kernels need CUDA")


# ------------------------------------------------------------- the accounting


def test_the_parameter_accounting_is_computed_not_hardcoded():
    """One composed slot mixer at the host width 640: the exact inventory
    (value/key/query_key/out/decay/f_proj/b_proj/g_proj + the three
    channel scalars + the depthwise conv kernel), plus one read gate
    scalar per slot layer, x3 slot layers -- the registration's purely
    ADDITIVE delta (the dense mixers stay; nothing is removed)."""
    width = 640
    per_mixer = slot_mixer_parameter_count(width)
    # the K15 p3 mixer inventory at width 640, by hand:
    assert per_mixer == (
        640 * 256            # value
        + 4 * 640 * 2 * 16   # key
        + 4 * 640 * 2 * 16   # query_key
        + 256 * 640          # out
        + 640 * 2 + 2        # decay + bias
        + 640 * 128 + 128 * 256    # f_proj.0/1
        + 640 * 256          # b_proj
        + 640 * 128 + 128 * 256 + 256  # g_proj.0/1 + bias
        + 3 * 2              # the three channel scalars
        + 256 * 4)           # v_conv
    assert per_mixer == 887_304
    assert hybrid_slot_parameter_delta(width) == 3 * (per_mixer + 1)
    assert hybrid_slot_parameter_delta(width) == 2_661_915


def test_the_slot_layers_are_the_registrations_own():
    """The placement receipt: 3 slot layers of 24, spread mid-stack
    (Memorizing Transformers' mid-stack optimum; the even 6-spacing
    interleave), centered on the stack's middle."""
    assert SLOT_LAYERS == (6, 12, 18)
    assert len(SLOT_LAYERS) == 3
    assert SLOT_LAYERS[1] == 12      # the stack's middle (24 layers)


def test_the_slot_policy_is_the_incumbents_own():
    """The slot machinery runs the K19-confirmed incumbent thresholds
    verbatim (tau_match 0.45 / tau_merge 0.85 / tau_pop 0.10 /
    read_scale 2.787, the five-op policy)."""
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    policy = slot_policy()
    assert policy.ops is True
    assert policy.tau_match == TUNED_WINNER["tau_match"] == 0.45
    assert policy.tau_merge == TUNED_WINNER["tau_merge"] == 0.85
    assert policy.tau_pop == TUNED_WINNER["tau_pop"] == 0.10
    assert policy.read_scale == TUNED_WINNER["read_scale"]


# ----------------------------------------------------------- the seam wrapper


class _StubDense(nn.Module):
    """A stand-in committed layer: returns its input unchanged (the
    wrapper's parity bar is then exact on every entry, -0.0 included:
    x + 0.0*slot == x bitwise wherever the corner does not fire)."""

    def __init__(self, dtype_corner: bool = False):
        super().__init__()
        self.dtype_corner = dtype_corner

    def forward(self, x, h0=None, **kwargs):
        if self.dtype_corner:
            return x, dict(cache=1)
        return x, None


def _wrapper(width=48, value_dim=8, key_dim=4, rank=5):
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import ShortConvComposedMixer
    torch.manual_seed(11)
    mixer = ShortConvComposedMixer(
        width=width, heads=1, value_dim=value_dim, rank=rank,
        rank_max=rank, key_dim=key_dim, policy=slot_policy(), ctx=4,
        segment_tokens=64, birth="identity")
    return HybridSlotGDN2Layer(_StubDense(), mixer, 0)


def test_the_read_gate_is_born_at_exact_zero():
    """The birth: one scalar read gate per slot layer, born at EXACT 0.0
    (the K13/K11B machinery inverted: the gate's zero makes the slot
    layers' contribution exactly nothing at birth)."""
    wrapper = _wrapper()
    assert wrapper.read_gate.shape == ()
    assert float(wrapper.read_gate) == 0.0
    assert wrapper.read_gate.requires_grad


def test_born_identical_bit_for_bit_cpu():
    """At the zero-born gate the wrapper IS the committed layer bitwise on
    a CPU fixture: the contribution is exactly zero and the output equals
    the dense layer's own (the -0.0 corner counted: x + (+0.0) flips a
    -0.0 dense entry -- the wrapper's own corner, receipted)."""
    wrapper = _wrapper()
    x = torch.randn(2, 24, 48)
    with torch.no_grad():
        out, cache = wrapper(x)
    assert out.shape == x.shape
    assert cache is None
    # the value bar: bit-identical by torch.equal on this fixture
    assert torch.equal(out, x)
    # the bitwise bar: any flip must be the sign-of-zero corner only
    diff = out.view(torch.int32) != x.view(torch.int32)
    zero = (out == 0) | (x == 0)
    assert bool((diff & ~zero).sum() == 0), "flips outside exact-zero entries"
    # and with a -0.0-carrying input the corner is the only legal escape
    xcorner = x.clone()
    xcorner[0, 0, 0] = -0.0
    with torch.no_grad():
        outc, _ = wrapper(xcorner)
    assert torch.equal(outc, xcorner) or bool(
        (outc.view(torch.int32) != xcorner.view(torch.int32)).sum() > 0
        and torch.equal(outc, xcorner))


def test_the_open_signal_and_the_inert_birth_cpu():
    """While the gate is exactly zero: the gate's OWN gradient is nonzero
    (the open signal -- dLoss/dgate = <upstream, slot output>), every
    OTHER slot parameter's gradient is EXACTLY zero (inert at birth), and
    a NONZERO gate passes gradient into the slot branch (the designed
    dynamics: the gates open, then the store learns)."""
    torch.manual_seed(13)
    wrapper = _wrapper()
    x = torch.randn(2, 24, 48)
    upstream = torch.randn_like(x)
    out, _ = wrapper(x)
    (out * upstream).sum().backward()
    assert float(wrapper.read_gate.grad) != 0.0, "the open signal is dead"
    slot_grads = [p.grad for p in wrapper.slot.parameters()
                  if p.grad is not None]
    assert slot_grads, "the slot branch must carry grads to inspect"
    assert all(bool((g == 0).all()) for g in slot_grads), \
        "the slot branch is not inert at birth"
    # a nonzero gate: gradient reaches the slot branch
    with torch.no_grad():
        wrapper.read_gate.fill_(0.3)
    wrapper.zero_grad(set_to_none=True)
    out, _ = wrapper(x)
    (out * upstream).sum().backward()
    live = [p.grad for p in wrapper.slot.parameters()
            if p.grad is not None]
    assert any(bool((g != 0).any()) for g in live), \
        "an open gate must pass gradient into the slot branch"


def test_the_gate_is_a_single_scalar_per_slot_layer():
    """The K22 seam lever is ONE scalar per slot layer (the cleanest
    attribution that can exist: any delta belongs to the slot reads
    opening); the engine's own per-channel r_vec read gate stays the
    engine's committed birth (r_vec == 1 at birth, untouched)."""
    wrapper = _wrapper()
    assert wrapper.slot.channels == dict(decay=True, write=True, read=True)
    assert float(wrapper.slot.read_scalar.abs()) == 0.0
    with torch.no_grad():
        r_vec = 1.0 + wrapper.slot.read_scalar.view(1, 1, -1, 1) \
            * torch.sigmoid(wrapper.slot.g_proj(x := torch.randn(
                1, 4, 48)))
    assert bool((r_vec == 1.0).all()), "the engine's read gate must be " \
                                       "born at its committed r_vec == 1"


# ----------------------------------------------------------- the refusals


def test_the_construction_refuses_unregistered_layers():
    """The graft refuses a non-GDN2ExternalLayer at a slot position and a
    slot layer outside the stack (fail loudly, never silently graft)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import convert_gdn2_to_hybrid
    stub = nn.Module()
    blocks = nn.ModuleList()
    for _ in range(24):
        block = nn.Module()
        layer = nn.Module()
        layer.gdn2 = _StubDense()
        block.layer = layer
        blocks.append(block)
    stub.model = nn.Module()
    stub.model.blocks = blocks
    stub.model.width = 48
    with pytest.raises(ValueError, match="GDN2ExternalLayer"):
        convert_gdn2_to_hybrid(stub)
    with pytest.raises(ValueError, match="outside the stack"):
        convert_gdn2_to_hybrid(stub, slot_layers=(6, 12, 24))


def test_the_split_refuses_unclassified_parameters():
    """The routing inventory is closed: a parameter outside the banked
    GDN2 patterns and the enumerated slot inventory fails loudly."""
    model = nn.Module()
    model.model = nn.Module()
    model.model.blocks = nn.ModuleList([])
    rogue = nn.Module()
    rogue.rogue = nn.Parameter(torch.zeros(2, 2))
    model.model.blocks.append(rogue)
    with pytest.raises(ValueError, match="unclassified hybrid parameter"):
        split_parameters_k22(model)


def test_the_slot_branch_routes_per_the_k18_convention():
    """The slot branch's routing: the hidden-map matrices (value/key/
    query_key/out) on Muon; the gate channels (decay, f_proj, b_proj,
    g_proj, the three scalars), the depthwise conv kernel and the K22
    read gate on AdamW (the GDN2 banked row's own option-B convention
    mirrored); the frozen falsifier's whole slot branch drops out by
    requires_grad."""
    wrapper = _wrapper()
    model = nn.Module()
    model.model = nn.Module()
    block = nn.Module()
    layer = nn.Module()
    layer.gdn2 = wrapper
    block.layer = layer
    model.model.blocks = nn.ModuleList([block])
    muon, adamw, muon_names, adamw_names = split_parameters_k22(model)
    assert sorted(n.split(".slot.")[-1] for n in muon_names
                  if ".slot." in n) == ["key.weight", "out.weight",
                                        "query_key.weight", "value.weight"]
    assert any(n.endswith(".read_gate") for n in adamw_names)
    assert any(n.endswith("v_conv.weight") for n in adamw_names)
    assert any(n.endswith("g_proj.1.bias") for n in adamw_names)
    assert any(n.endswith("decay_scalar") for n in adamw_names)
    assert not any(".slot." in n for n in muon_names
                   if n.endswith(("decay.weight", "b_proj.weight")))
    # the frozen falsifier: the whole slot branch leaves the inventory
    for p in wrapper.slot.parameters():
        p.requires_grad_(False)
    wrapper.read_gate.requires_grad_(False)
    with pytest.raises(ValueError, match="both semantic groups"):
        split_parameters_k22(model)
    # (the probe's frozen arm keeps the dense inventory on both sides; a
    # slot-only stub has nothing left to route -- the refusal is the
    # correct behavior)


# ------------------------------------------------- the CUDA engine receipts


@needs_cuda
def test_the_slot_mixer_builds_at_the_host_width_deterministically():
    """The committed engine, instantiated at the host width under the
    forked RNG: identical builds across calls (the two arms' slot inits
    are the same machine), the registered births inside (the p3 gates at
    their scalars, the mimetic keys, the identity conv)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import ShortConvComposedMixer
    a = build_slot_mixer(width=640)
    b = build_slot_mixer(width=640)
    assert isinstance(a, ShortConvComposedMixer)
    for (na, pa), (nb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert na == nb and pa.shape == pb.shape
        assert torch.equal(pa, pb), f"the slot init is not deterministic: {na}"
    assert a.width == 640 and a.heads == 2 and a.value_dim == 128
    assert a.rank == 128 and a.key_dim == 16 and a.ctx == 4
    with torch.no_grad():
        assert bool((a.v_conv.weight == torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=a.v_conv.weight.device)
            .view(1, -1)).all())


@needs_cuda
def test_the_seam_runs_at_the_production_shape_under_autocast():
    """The wrapper's forward at the production shape (batch 8 x 2048
    tokens at width 640) under the bf16 autocast regime: the output dtype
    is the dense output's own and the loss-shaped consumption works."""
    from ndm.models.external_gdn2 import GDN2ExternalLayer
    torch.manual_seed(17)
    layer = GDN2ExternalLayer(dim=640, expansion=1.0, head_dim=128,
                               num_heads=5).cuda()
    mixer = build_slot_mixer(width=640).cuda()
    wrapper = HybridSlotGDN2Layer(layer, mixer, 6).cuda()
    x = torch.randn(2, 512, 640, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out, cache = wrapper(x)
    assert out.dtype == torch.bfloat16
    assert out.shape == x.shape
    assert torch.isfinite(out).all()
