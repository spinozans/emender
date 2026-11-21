"""K22 -- THE HYBRID: GDN2's committed 150M arm with 3 of its 24 mixer
positions carrying our committed composed slot mixer, born as PURE GDN2.

THE MACHINE (the operator's architecture go: 3 slot layers of 24,
mid-stack)
---------------------------------------------------------------
GDN2's committed arm (``matched_arms.build_gdn2`` verbatim: the registered
lattice_gdn2 nomination, 24 dense mixers at width 640, the banked 1.3522
machine) with the mixer positions {6, 12, 18} carrying our committed
composed slot mixer (the K15/K18 engine imported VERBATIM --
``ShortConvComposedMixer``: the K13 additive read, the K11B p3 per-channel
decay/write/read gates, the K10 separated keys at the mimetic birth, the
K18 identity-born short conv -- instantiated at the host's width 640 with
the registered store geometry: heads 2, value_dim 128, rank/rank_max 128,
key_dim 16, ctx 4, segment_tokens 64, the K19-confirmed incumbent
thresholds).

THE SEAM, settled by the non-negotiable falsifier.  The task text says
both "3 of its 24 mixers REPLACED" and "the machine is born bit-identical
to the committed GDN2 arm ... a hybrid frozen at zero-read must be
bit-identical to pure GDN2's banked trajectory FOREVER".  A literal
removal of the dense mixers at the slot layers makes the birth-parity bar
impossible BY CONSTRUCTION (a 21-mixer GDN2 is not the banked 24-mixer
machine at step 1), so the only reading that satisfies the birth
discipline is the additive one the falsifier dictates: the slot layer
KEPS GDN2's dense mixer and GAINS the slot read, added to the dense
mixer's output through a ZERO-BORN READ GATE -- the K13 additive-read
convention at the layer seam.  At birth the slot contribution is exactly
zero, the machine is bit-identical to the committed GDN2 arm, the extra
parameters are present but inert, and any measured delta belongs to the
slot reads opening.  The parameter delta is therefore purely additive and
is quoted exactly in every receipt (``hybrid_slot_parameter_delta``).

BIRTH (the K13/K11B machinery inverted)
---------------------------------------
Each slot layer's READ gate (one scalar, K22's own seam lever, named
``read_gate``) is born at EXACT zero, so ``out + gate * slot(x) == out``
bitwise (the only legal escape is the IEEE -0.0 corner -- a +0.0
contribution added to a -0.0 dense entry flips the sign of zero; counted
and receipted by the gate receipt, invisible to every value comparison
and to the loss).  The dense GDN2 mixers, their parameters, their routing
and their optimizer treatment are the banked row's own VERBATIM.  The
slot store machinery RUNS from step 1 (APPEND/CORRECT fill the store; the
ops fire inside the dense stack) but nothing reads out until the gate
opens; the gate's gradient (``dLoss/dgate = <upstream, slot output>``) is
the open signal, and while the gate is exactly zero every OTHER slot
parameter receives an exactly-zero gradient (inert at birth, receipted).

THE ARMS (the probe's own, ``k22_probe_hybrid.py``)
---------------------------------------------------
``hybrid``         born zero-read, the gates learn to open (seed 3401;
                   seed 3402 owed on any >= 0.01 win per the standing
                   rule).
``hybrid_frozen``  the falsifier: zero-read, the whole slot branch
                   frozen (``requires_grad=False``), must be
                   bit-identical to the banked GDN2 trajectory (the
                   gdn2_speedrun_rescored row's own training log) at
                   every one of its 204 steps and land at the banked
                   1.3521734476089478 post-cooldown bit for bit.

Both engines are imported verbatim and never modified:
``matched_arms.py`` (GDN2's committed build), ``k15_composition.py`` /
``k18_short_conv.py`` (the composed slot mixer), ``schedule.py`` (the
routing policy), ``muon_control.py`` (the banked GDN2 parameter routing),
``bakeoff_optimizers.py`` (HybridMuonX45), ``protocol_correction.py``
(the corrected probe loop).  This file only WRAPS: the committed
GDN2ExternalLayer becomes ``wrapper.gdn2`` and the composed mixer becomes
``wrapper.slot`` inside one new module.
"""
from __future__ import annotations

import re

import torch
from torch import nn

from .k15_composition import ComposedReadMixer
from .k10_separated_keys import (DEFAULT_CONTEXT, DEFAULT_KEY_DIM,
                                 DEFAULT_SEGMENT_TOKENS, MIMETIC_SEED)
from .k18_short_conv import (SHORT_CONV_KERNEL_SIZE, ShortConvComposedMixer)

# The placement (the registration's receipted choice): 3 slot layers
# spread mid-stack, centered on Memorizing Transformers' mid-stack optimum
# (Wu et al., arXiv:2203.08913 -- a single memory layer placed mid-stack
# beats early/late placements in a dense stack), with the spacing (every
# 6th of 24) the Jamba/Samba minority-interleave pattern at the 1:7 ratio.
SLOT_LAYERS = (6, 12, 18)

# The registered store geometry (the composed machine's own, unchanged:
# the engine is instantiated at the HOST's width, the store is not
# resized -- the registration states this).
SLOT_HEADS = 2
SLOT_VALUE_DIM = 128
SLOT_RANK = 128
SLOT_RANK_MAX = 128

# The slot machinery's dedicated init seed: the slot mixers are
# constructed AFTER the GDN2 build (whose global-RNG consumption order is
# the banked row's own, untouched), under a forked RNG seeded here, so
# the slot init is deterministic, identical across the arms, and the
# global stream is left exactly where the GDN2 build left it.
K22_SLOT_SEED = 3401

# The GDN2 layer names this lane rewraps (imported module's own classes;
# matched by isinstance, never by patching).
_GDN2_EXTERNAL_LAYER_QNAME = ("experiments.expressivity_tasks.x45_matrix"
                               "_emender")  # documentation only


def slot_policy():
    """The composed machine's banked routing policy (the K19
    protocol-confirmed incumbent thresholds), constructed verbatim from the
    tuned winner's config."""
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    return RoutingPolicy(ops=True,
                         tau_match=TUNED_WINNER["tau_match"],
                         tau_merge=TUNED_WINNER["tau_merge"],
                         tau_pop=TUNED_WINNER["tau_pop"],
                         read_scale=TUNED_WINNER["read_scale"])


def build_slot_mixer(width: int, policy=None, seed: int = K22_SLOT_SEED,
                     ctx: int = DEFAULT_CONTEXT,
                     key_dim: int = DEFAULT_KEY_DIM,
                     segment_tokens: int = DEFAULT_SEGMENT_TOKENS
                     ) -> ShortConvComposedMixer:
    """The committed composed slot mixer (the K15/K18 engine VERBATIM) at
    the host's width, under a forked RNG so the GDN2 build's global-RNG
    stream is untouched.  The mimetic key family (the K10 birth) and the
    p3 gate births are the engine's own, applied inside its constructor."""
    if policy is None:
        policy = slot_policy()
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        mixer = ShortConvComposedMixer(
            width=int(width), heads=SLOT_HEADS, value_dim=SLOT_VALUE_DIM,
            rank=SLOT_RANK, rank_max=SLOT_RANK_MAX, key_dim=int(key_dim),
            policy=policy, ctx=int(ctx), segment_tokens=int(segment_tokens),
            birth="identity")
    finally:
        torch.random.set_rng_state(state)
    return mixer


def slot_mixer_parameter_count(width: int, heads: int = SLOT_HEADS,
                               value_dim: int = SLOT_VALUE_DIM,
                               key_dim: int = DEFAULT_KEY_DIM,
                               ctx: int = DEFAULT_CONTEXT,
                               kernel_size: int = SHORT_CONV_KERNEL_SIZE
                               ) -> int:
    """The slot machinery's exact parameter count at a width, computed not
    hardcoded (the K15 p3 mixer's own inventory + the K18 conv kernel):

    * ``value``    width x heads*value_dim            (the payload path)
    * ``key``      ctx*width x heads*key_dim          (the write keys)
    * ``query_key`` ctx*width x heads*key_dim          (the read queries)
    * ``out``      heads*value_dim x width            (the read projection)
    * ``decay``    width x heads + heads              (the incumbent gate)
    * ``f_proj``   width x value_dim + value_dim x heads*value_dim
    * ``b_proj``   width x heads*value_dim             (the write channel)
    * ``g_proj``   width x value_dim + value_dim x heads*value_dim
                   + heads*value_dim bias             (the read channel)
    * ``{decay,write,read}_scalar``  heads each
    * ``v_conv``   heads*value_dim x kernel_size      (the K18 conv)
    """
    channels = heads * value_dim
    return int(
        width * channels                      # value
        + ctx * width * heads * key_dim        # key
        + ctx * width * heads * key_dim        # query_key
        + channels * width                     # out
        + width * heads + heads                # decay (+bias)
        + width * value_dim + value_dim * channels   # f_proj.0/1
        + width * channels                     # b_proj
        + width * value_dim + value_dim * channels + channels  # g_proj+bias
        + 3 * heads                            # the three channel scalars
        + channels * kernel_size)              # v_conv


def hybrid_slot_parameter_delta(width: int, slot_layers=SLOT_LAYERS) -> int:
    """The registration's exact additive parameter delta: one composed
    slot mixer per slot layer plus one zero-born read gate scalar each.
    Nothing is removed (the falsifier's own construction: the dense mixers
    stay, bit-identical at birth)."""
    return len(slot_layers) * (slot_mixer_parameter_count(width) + 1)


class HybridSlotGDN2Layer(nn.Module):
    """One hybrid slot layer: GDN2's committed ``GDN2ExternalLayer``
    (imported verbatim, held as ``self.gdn2``) with the committed composed
    slot mixer (held as ``self.slot``) added to its output through the
    ZERO-BORN read gate ``self.read_gate`` (one scalar).

    forward is the committed layer's own with ONE additive term::

        out, cache = self.gdn2(x, h0, **kwargs)
        return out + self.read_gate.to(out.dtype) * self.slot(x), cache

    At birth ``read_gate == 0`` exactly, so the layer IS the committed
    GDN2 layer bit for bit (the contribution is exactly zero; the only
    legal escape is the IEEE sign-of-zero corner, receipted by the gate
    receipt).  The gate's gradient (``<upstream, slot output>``) is the
    open signal; while the gate is exactly zero every other slot parameter
    receives an exactly-zero gradient.  The dtype cast keeps the
    contribution in the dense output's own carrier dtype under the
    production autocast regime (bf16), so the residual stream's dtype is
    the committed machine's own at birth and after."""

    def __init__(self, gdn2_layer: nn.Module, slot_mixer: nn.Module,
                 layer_index: int):
        super().__init__()
        self.gdn2 = gdn2_layer
        self.slot = slot_mixer
        self.layer_index = int(layer_index)
        self.read_gate = nn.Parameter(torch.zeros(()))

    def slot_contribution(self, x: torch.Tensor) -> torch.Tensor:
        """The gated slot read alone (the sweep's object; the wrapper's
        own expression, recomputed for the instruments)."""
        out_dtype = x.dtype
        return self.read_gate.to(out_dtype) * self.slot(x)

    def forward(self, x, h0=None, **kwargs):
        out, cache = self.gdn2(x, h0, **kwargs)
        contribution = self.read_gate.to(out.dtype) * self.slot(x)
        return out + contribution, cache


def convert_gdn2_to_hybrid(model, slot_layers=SLOT_LAYERS, policy=None,
                           seed: int = K22_SLOT_SEED):
    """Wrap the committed GDN2 arm's mixer layers at the slot positions
    with :class:`HybridSlotGDN2Layer`.  The GDN2 build is the caller's own
    (``matched_arms.build_gdn2`` verbatim, its RNG consumption order
    untouched); every carried module is carried over UNTOUCHED (the wrapper
    HOLDS the committed ``GDN2ExternalLayer``; nothing inside it changes),
    the slot mixers are built under a forked RNG, and the parameter delta
    is asserted against the registration's computed accounting.  Returns
    the same model object."""
    from ndm.models.external_gdn2 import GDN2ExternalLayer
    blocks = model.model.blocks
    depth = len(blocks)
    for index in slot_layers:
        if not 0 <= index < depth:
            raise ValueError(f"slot layer {index} outside the stack's "
                             f"0..{depth - 1}")
    width = int(getattr(model.model, "width"))
    parameters_before = sum(p.numel() for p in model.parameters())
    for index in slot_layers:
        layer = blocks[index].layer.gdn2
        if not isinstance(layer, GDN2ExternalLayer):
            raise ValueError(
                f"block {index}.layer.gdn2 must be the committed "
                f"GDN2ExternalLayer; got {type(layer).__name__}")
        if int(layer.dim) != width:
            raise ValueError(f"block {index} carries width {layer.dim}; "
                             f"the model's registered width is {width}")
        mixer = build_slot_mixer(width, policy=policy, seed=seed)
        device = next(layer.parameters()).device
        wrapper = HybridSlotGDN2Layer(layer, mixer, index).to(device)
        blocks[index].layer.gdn2 = wrapper
    parameters_after = sum(p.numel() for p in model.parameters())
    added = parameters_after - parameters_before
    expected = hybrid_slot_parameter_delta(width, slot_layers)
    if added != expected:
        raise ValueError(f"the hybrid's parameter delta must be the "
                         f"registration's {expected}; got {added}")
    return model


def hybrid_slot_layers(model) -> tuple:
    """The model's slot layer indices, read back from the wrappers (the
    probe asserts this equals the registration's SLOT_LAYERS)."""
    found = tuple(block_index
                  for block_index, block in enumerate(model.model.blocks)
                  if isinstance(block.layer.gdn2, HybridSlotGDN2Layer))
    if found != tuple(SLOT_LAYERS):
        raise ValueError(f"the hybrid's slot layers drifted: {found} != "
                         f"the registered {SLOT_LAYERS}")
    return found


def freeze_slot_branch(model) -> int:
    """The falsifier's freeze: every slot parameter (the composed mixer's
    AND the read gate) carries ``requires_grad=False`` -- the slot branch
    then sits outside the optimizer, outside the grad clip and outside
    the EMA by construction, and the split equals the banked GDN2 row's
    own exactly (the K18 frozen-control convention).  Returns the number
    of frozen tensors."""
    frozen = 0
    for block_index in hybrid_slot_layers(model):
        wrapper = model.model.blocks[block_index].layer.gdn2
        for param in wrapper.slot.parameters():
            param.requires_grad_(False)
            frozen += 1
        wrapper.read_gate.requires_grad_(False)
        frozen += 1
    return frozen


# --------------------------------------------------- the optimizer routing

# The slot-layer dense-name normalization: wrapping deepens the committed
# core's path by ONE ``.gdn2`` level at the slot layers
# (``...layer.gdn2.gdn2.gdn2.q_proj.weight``); the banked routing's own
# name patterns classify the NORMALIZED name.  Everything else about the
# dense inventory is the banked row's own.
def _slot_normalizer(slot_layers=SLOT_LAYERS):
    indices = "|".join(str(int(i)) for i in slot_layers)
    pattern = re.compile(
        r"^(?P<prefix>(?:model\.)?)blocks\.(?P<index>" + indices
        + r")\.layer\.gdn2\.gdn2\.gdn2\.(?P<rest>.*)$")

    def normalize(name: str) -> str:
        match = pattern.fullmatch(name)
        if match is None:
            return name
        return (f"{match.group('prefix')}blocks.{match.group('index')}"
                f".layer.gdn2.gdn2.{match.group('rest')}")
    return normalize


# The slot branch's own inventory, ENUMERATED (a drift fails loudly, the
# muon_control convention).  ``mixer`` tensors route per the K18/K13
# convention mirrored onto GDN2's banked option B: the hidden-map matrices
# (value/key/query_key/out) to Muon, every gate channel (decay, f_proj,
# b_proj, g_proj, the three scalars), the depthwise conv kernel and the
# K22 read gate to AdamW.
_SLOT_MATRIX = re.compile(
    r"(?:model\.)?blocks\.\d+\.layer\.gdn2\.slot\."
    r"(value|key|query_key|out)\.weight$")
_SLOT_ADAMW = re.compile(
    r"(?:model\.)?blocks\.\d+\.layer\.gdn2\.slot\."
    r"(decay\.weight|decay\.bias|f_proj\.[01]\.weight|b_proj\.weight"
    r"|g_proj\.[01]\.weight|g_proj\.1\.bias|decay_scalar|write_scalar"
    r"|read_scalar|v_conv\.weight)$")
_SLOT_GATE = re.compile(
    r"(?:model\.)?blocks\.\d+\.layer\.gdn2\.read_gate$")


def split_parameters_k22(model, frozen: bool = False):
    """The banked GDN2 row's own routing (option B, gates_on_adamw=True --
    the 1.3522 machine's own) applied to the hybrid's dense inventory via
    the slot-layer name normalization, PLUS the slot branch routed per the
    K18 convention mirrored (matrices to Muon; the gate channels, the
    conv kernel and the K22 read gate to AdamW).  A frozen falsifier's
    slot branch carries ``requires_grad=False`` and is excluded by the
    ``requires_grad`` filter -- the split then equals the banked GDN2
    row's own exactly (the falsifier's own plumbing receipt).  Fails
    loudly on ANY parameter outside the two enumerated inventories."""
    from experiments.expressivity_tasks.x45_matrix_emender.muon_control \
        import (_GDN2_CORE, _GDN2_EMBED_HEAD, _GDN2_GATE_PROJ, _GDN2_MLP,
                _GDN2_NONMATRIX)
    normalize = _slot_normalizer()
    muon, adamw = [], []
    storage = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.device.type != "meta":
            key = (param.device, param.untyped_storage().data_ptr())
            if key in storage:
                raise ValueError(f"distinct parameters share storage: {name}")
            storage.add(key)
        if _SLOT_GATE.fullmatch(name) is not None:
            kind = "slot_read_gate"
        elif _SLOT_MATRIX.fullmatch(name) is not None:
            kind = "slot_matrix"
        elif _SLOT_ADAMW.fullmatch(name) is not None:
            kind = "slot_gate_channel"
        else:
            dense_name = normalize(name)
            if any(r.fullmatch(dense_name) for r in _GDN2_EMBED_HEAD) \
                    or _GDN2_CORE.fullmatch(dense_name) is not None \
                    or _GDN2_MLP.fullmatch(dense_name) is not None:
                kind = "dense_matrix"
            elif _GDN2_GATE_PROJ.fullmatch(dense_name) is not None:
                kind = "dense_gate_proj"          # option B: AdamW
            elif _GDN2_NONMATRIX.fullmatch(dense_name) is not None:
                kind = "dense_nonmatrix"          # AdamW
            else:
                raise ValueError(
                    f"unclassified hybrid parameter: {name} (both "
                    f"inventories are enumerated; a drift must fail "
                    f"loudly, not silently route)")
        if kind in ("dense_matrix", "slot_matrix"):
            if param.ndim != 2:
                raise ValueError(f"Muon-owned parameter is not a full 2D "
                                 f"matrix: {name}")
            muon.append(name)
        else:
            adamw.append(name)
    if not muon or not adamw:
        raise ValueError("the hybrid requires both semantic groups "
                         "non-empty")
    params = dict(model.named_parameters())
    return ([params[n] for n in muon], [params[n] for n in adamw],
            muon, adamw)


def _routing_cross_check_k22(model, seed: int) -> dict:
    """The no-drift receipt: a pure-GDN2 twin of the SAME build must route
    EXACTLY as the banked row's own split routes it; the hybrid model's
    DENSE names (after the slot-layer normalization) must equal that split
    exactly, and the slot side must equal exactly the enumerated slot
    inventory (matrices on Muon; the gate channels, the conv kernel and
    the read gate on AdamW).  The frozen falsifier's slot branch is
    excluded by ``requires_grad`` by construction."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender import (
        matched_arms)
    from experiments.expressivity_tasks.x45_matrix_emender.muon_control \
        import split_parameters_gdn2
    hybrid_muon, hybrid_adamw, muon_names, adamw_names = \
        split_parameters_k22(model)
    normalize = _slot_normalizer()
    dense_muon = sorted(normalize(n) for n in muon_names
                        if ".slot." not in n and not n.endswith(".read_gate"))
    dense_adamw = sorted(normalize(n) for n in adamw_names
                         if ".slot." not in n and not n.endswith(".read_gate"))
    twin, _cfg, _summary = matched_arms.build_gdn2(seed)
    _t_muon, _t_adamw, twin_muon, twin_adamw = split_parameters_gdn2(
        twin, gates_on_adamw=True)
    del twin, _t_muon, _t_adamw
    torch.cuda.empty_cache()
    if dense_muon != sorted(twin_muon):
        raise ValueError("the hybrid's dense Muon group (normalized) "
                         "drifted from the banked GDN2 row's own split")
    if dense_adamw != sorted(twin_adamw):
        raise ValueError("the hybrid's dense AdamW group (normalized) "
                         "drifted from the banked GDN2 row's own split")
    prefix = "model." if any(n.startswith("model.")
                              for n in muon_names) else ""
    expected_slot_muon = sorted(
        f"{prefix}blocks.{i}.layer.gdn2.slot.{name}.weight"
        for i in SLOT_LAYERS
        for name in ("value", "key", "query_key", "out"))
    actual_slot_muon = sorted(n for n in muon_names if ".slot." in n)
    if actual_slot_muon != expected_slot_muon:
        raise ValueError(f"the slot Muon group drifted: {actual_slot_muon}")
    slot_adamw = sorted(n for n in adamw_names
                        if ".slot." in n or n.endswith(".read_gate"))
    expected_slot_adamw = sorted(
        [f"{prefix}blocks.{i}.layer.gdn2.slot.{name}"
         for i in SLOT_LAYERS
         for name in ("decay.weight", "decay.bias", "f_proj.0.weight",
                      "f_proj.1.weight", "b_proj.weight",
                      "g_proj.0.weight", "g_proj.1.weight",
                      "g_proj.1.bias", "decay_scalar", "write_scalar",
                      "read_scalar", "v_conv.weight")]
        + [f"{prefix}blocks.{i}.layer.gdn2.read_gate"
           for i in SLOT_LAYERS])
    if slot_adamw != expected_slot_adamw:
        raise ValueError("the slot AdamW group drifted from the enumerated "
                         "slot inventory")
    return dict(
        muon_tensors=len(muon_names), adamw_tensors=len(adamw_names),
        dense_muon=len(dense_muon), dense_adamw=len(dense_adamw),
        slot_muon=len(actual_slot_muon), slot_adamw=len(slot_adamw),
        note="the banked GDN2 row's own option-B split (the 1.3522 "
             "machine's), verified equal on the normalized dense names "
             "against a pure-GDN2 twin's own split; the slot branch per "
             "the K18 convention mirrored: matrices on Muon, the gate "
             "channels + the conv kernel + the K22 read gate on AdamW")


__all__ = [
    "SLOT_LAYERS", "SLOT_HEADS", "SLOT_VALUE_DIM", "SLOT_RANK",
    "SLOT_RANK_MAX", "K22_SLOT_SEED", "MIMETIC_SEED",
    "HybridSlotGDN2Layer", "build_slot_mixer", "convert_gdn2_to_hybrid",
    "hybrid_slot_layers", "freeze_slot_branch", "slot_policy",
    "slot_mixer_parameter_count", "hybrid_slot_parameter_delta",
    "split_parameters_k22", "_routing_cross_check_k22",
    "ComposedReadMixer", "ShortConvComposedMixer", "DEFAULT_KEY_DIM",
    "DEFAULT_CONTEXT", "DEFAULT_SEGMENT_TOKENS",
]
