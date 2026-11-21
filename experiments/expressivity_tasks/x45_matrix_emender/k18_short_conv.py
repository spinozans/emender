"""K18 -- THE SHORT-CONV TRANSPLANT: GDN2's registered short depthwise
convolution (conv_size=4, default ON), on the COMMITTED composed machine.

Why this file exists
--------------------
GDN2's registered block carries a short depthwise convolution
(``ShortConvolution(hidden_size, kernel_size=conv_size=4, bias=False,
activation='silu')``) applied to each of the q/k/v projections' outputs,
default ON (``/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2.py``); the
delta-rule family's own ablations say the short conv is load-bearing
locally (published with/without deltas span 0.05-13.5 ppl across the
lineage -- heterogeneous, never zero).  Our comparison machine has NEVER
had one: no convolution exists anywhere in the K10/K15 mixer lineage.
Every transplant this session ran machine-without-conv vs
machine-with-conv, so the missing-component candidate for the composed
machine's remaining ~0.03 vs GDN2 now has published backing.

THE ESCALATED PLACEMENT (approved by the supervisor, 2026-11-26)
------------------------------------------------------------------
GDN2's conv sits POST-projection, per side: ``q_conv1d(q_proj(x))``,
``k_conv1d(k_proj(x))``, ``v_conv1d(v_proj(x))`` -- three separate
depthwise convs, NOT a conv on the mixer input before the projections.
The K18 transplant is the PAYLOAD side only -- ``v_conv1d``'s exact
functional analog: ``payload = v_conv(value(x))``.  The asymmetry
receipt that justifies the single-side transplant: our key/query
projections already consume a 4-token causal context window
(``DEFAULT_CONTEXT=4`` -- exactly the conv's receptive-field width), so
the q/k side already has the locality GDN2's q/k convs buy, while the
PAYLOAD path has NO temporal receptive field at all (``value(x)`` sees
the current token only).  The payload conv is therefore not a shortcut
of GDN2's triple; it is the faithful functional analog that COMPLETES
the receptive-field parity GDN2 has on all three sides.  The q/k-conv
variant (with P2 re-registered as decisions conv-influenced) is a named
follow-up, not this lane.

BIRTH (the K15 birth discipline, escalated and approved)
-------------------------------------------------------
The conv is born at the IDENTITY KERNEL in the STANDARD nn.Conv1d/fla
tap layout: weight ``[0, 0, 0, 1]`` per channel -- the LAST tap multiplies
the current token (tap 0 is the oldest, ``x[t-3]``); this is the layout
fla's ``causal_conv1d`` and ``nn.Conv1d`` both use, so the learned
kernels stay directly comparable to GDN2's and to the literature's.  At
that birth the conv is EXACTLY the pass-through (1.0*x is exact in every
dtype; the zero taps add +-0.0), so the machine is BORN AS THE COMPOSED
MACHINE bit for bit (the 0-flip bar), and the frozen-conv falsifier
control must stay bit-identical forever.  The fused SiLU of GDN2's
``ShortConvolution(activation='silu')`` is NOT transplanted (escalated,
approved): the silu is a standing, pre-existing cross-machine mismatch
(ours never silus the payload; GDN2 silus q/k/v even with the conv OFF,
so the conv's own ablated delta in GDN2's family is the TEMPORAL FILTER,
and ours isolates the same object).  A conv+silu arm would break the
0-flip birth bar by construction and confound the conv's delta with the
silu's; it is a named follow-up with its own registration.

THE VALUE-CHANNEL CLAIM (P1/P2/P3 hold trivially)
------------------------------------------------
The conv output feeds the store's payload only.  No conv value feeds op
selection, thresholds, routing similarity or liveness (the decisions
read the KEY channel alone -- the standing detached-key doctrine), so the
discrete op sequence is bit-invariant to every conv value at fixed keys
(the 0-flip receipt), and the scan's every committed engine is imported
UNTOUCHED: the conv sits in the mixer's input path BEFORE the unchanged
``composed_scan`` opaque op pair, so autograd carries the conv's own
adjoint (no new forward or backward kernel exists in this lane) and the
fp64 gradcheck goes through conv + the composed oracle.

The kernels (the committed engines imported; ``k10_*``/``k11b_*``/
``k13_*``/``k15_*`` sources are imported, never edited)
--------------------------------------------------------------------------------
``composed_scan`` / ``composed_oracle_scan``
                            IMPORTED VERBATIM (the K15 composed walk: the
                            additive read under the per-channel decay/write
                            gates; every committed kernel inside it).  The
                            conv changes only the PAYLOAD tensor the mixer
                            hands the scan.
``short_conv1d_causal``    THE ONE NEW FUNCTION (this file): the causal
                            depthwise conv as an explicit, deterministic
                            tap sum -- exactly ``nn.Conv1d(C, C, 4,
                            groups=C, bias=False)`` with left zero-padding
                            (fla's ``ShortConvolution`` semantics), stated
                            in the standard tap layout.  Realized as the
                            explicit tap sum (not cuDNN) so (a) the birth
                            is bit-exact and deterministic on every
                            backend, (b) the fp64 gradcheck can go through
                            the conv in fp64, and (c) the falsifier's
                            full-trajectory bit-identity is not hostage to
                            cuDNN's nondeterministic backward.  Equivalence
                            to ``F.conv1d`` is receipted by the suite.

Doctrine, unchanged: decisions are recorded constants in the backward.
The conv WEIGHT and the payload carry gradient to their inputs through
autograd; the gradient gates are floor-discipline (the fp64 gradcheck of
the oracle THROUGH THE CONV PATH first, fp32-vs-fp64 floors,
``NOISE_FACTOR`` x), stated, not hidden.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .k15_composition import (ComposedReadMixer, composed_parameter_delta,
                              composed_scan,
                              convert_model_to_composition)
from .k10_separated_keys import (DEFAULT_CONTEXT, DEFAULT_KEY_DIM,
                                 DEFAULT_SEGMENT_TOKENS)
from .key_separation_probe import causal_context

# GDN2's registered short-convolution shape (lit_gpt/gdn2.py: conv_size=4,
# conv_bias=False, use_short_conv=True by default).
SHORT_CONV_KERNEL_SIZE = 4

# The registered births, in the STANDARD nn.Conv1d/fla tap layout
# (tap j multiplies x[t - (K-1) + j]; the LAST tap is the current token).
# IDENTITY: [0, 0, 0, 1] -- the position-K-1 pass-through, exactly identity.
# SHIFT (the fixed non-identity control): [0, 0, 0.5, 0.5] -- the average
# of the current and the previous token (a frozen token-shift, never
# learned; separates "any local mixing" from "a learned filter").
IDENTITY_SHORT_CONV_BIRTH = (0.0, 0.0, 0.0, 1.0)
SHIFT_SHORT_CONV_BIRTH = (0.0, 0.0, 0.5, 0.5)

SHORT_CONV_BIRTHS = {"identity": IDENTITY_SHORT_CONV_BIRTH,
                    "shift": SHIFT_SHORT_CONV_BIRTH}


def short_conv1d_causal(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The causal depthwise short convolution as an explicit tap sum.

    ``x`` is ``(B, T, C)``; ``weight`` is ``(C, K)`` in the STANDARD
    nn.Conv1d layout: ``out[t, c] = sum_j weight[c, j] * x[t-(K-1)+j, c]``
    (the LAST tap multiplies the current token; the ``K-1`` taps before
    t are causal left zero-padding).  Exactly equivalent to
    ``F.conv1d(F.pad(x, (K-1, 0)).transpose(1, 2), w.view(C, 1, K),
    groups=C).transpose(1, 2)`` -- receipted by the suite -- evaluated as
    the explicit deterministic sum so the identity birth is bit-exact on
    every backend, the fp64 gradcheck can pass through it, and the frozen
    falsifier is not hostage to cuDNN's nondeterministic backward.  The
    current tap is accumulated FIRST (at the identity birth the sum is
    then exactly ``1.0*x + 0.0*...`` in every dtype; the only legal
    escape is the IEEE -0.0 corner -- sign-of-zero flips on exact-zero
    entries -- counted and receipted by the suite)."""
    batch, tokens, channels = x.shape
    if weight.shape[0] != channels:
        raise ValueError(f"the conv weight's channel axis must be the "
                         f"payload's {channels}; got {tuple(weight.shape)}")
    kernel = weight.shape[1]
    if tokens < 1:
        raise ValueError("the conv needs at least one token")
    w = weight.to(x.dtype)
    padded = F.pad(x, (0, 0, kernel - 1, 0))          # (B, T+K-1, C)
    # the current tap first: at the identity birth this is 1.0*x, exact.
    out = w[:, kernel - 1] * padded[:, kernel - 1: kernel - 1 + tokens]
    for j in range(kernel - 2, -1, -1):
        out = out + w[:, j] * padded[:, j: j + tokens]
    return out


class IdentityBornShortConv(nn.Module):
    """The K18 short convolution: one depthwise kernel per payload channel
    (GDN2's ``v_conv1d`` analog: ``ShortConvolution(hidden, 4, bias=False)``
    with the fused silu REMOVED per the escalated registration), weight
    ``(channels, K)`` in the standard nn.Conv1d tap layout, born at the
    registered birth (``identity``: ``[0, 0, 0, 1]`` -- exactly the
    pass-through, so the mixer is BORN AS THE COMPOSED MACHINE bit for
    bit; ``shift``: ``[0, 0, 0.5, 0.5]`` -- the fixed non-identity
    control, frozen by the probe, never learned)."""

    def __init__(self, channels: int,
                 kernel_size: int = SHORT_CONV_KERNEL_SIZE,
                 birth: str = "identity"):
        super().__init__()
        if birth not in SHORT_CONV_BIRTHS:
            raise ValueError(f"unregistered conv birth {birth!r} (expected "
                             f"one of {sorted(SHORT_CONV_BIRTHS)})")
        if kernel_size != SHORT_CONV_KERNEL_SIZE:
            raise ValueError(f"the registration's kernel is "
                             f"{SHORT_CONV_KERNEL_SIZE} (GDN2's conv_size); "
                             f"got {kernel_size}")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.birth = str(birth)
        self.weight = nn.Parameter(torch.empty(self.channels,
                                                self.kernel_size))
        self.apply_registered_birth()

    def apply_registered_birth(self):
        """Set the registered birth kernel: per channel, the registered tap
        values (``identity``: the last tap 1, the rest 0 -- the exact
        pass-through; ``shift``: taps 2/3 at 0.5 -- the fixed average of
        the current and previous token)."""
        with torch.no_grad():
            taps = torch.tensor(SHORT_CONV_BIRTHS[self.birth],
                                dtype=self.weight.dtype)
            self.weight.copy_(taps.unsqueeze(0).expand(self.channels,
                                                       self.kernel_size))
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return short_conv1d_causal(x, self.weight)


def short_conv_parameter_delta(depth: int, width: int, heads: int,
                                value_dim: int,
                                kernel_size: int = SHORT_CONV_KERNEL_SIZE
                                ) -> int:
    """The registration's parameter accounting, computed not hardcoded: the
    composed machine's own delta (the committed k15 accounting) plus ONE
    depthwise kernel of ``heads*value_dim`` channels x ``kernel_size``
    taps per mixer (GDN2's own ``v_conv1d`` shape).  ``width`` is carried
    for signature parity with the committed accounting helpers; the
    depthwise kernel does not see it."""
    composed = composed_parameter_delta(depth, width, heads, value_dim)
    return composed + depth * heads * value_dim * int(kernel_size)


def conv_payload(mixer, x: torch.Tensor) -> torch.Tensor:
    """The conv machine's payload channel at the mixer's own inputs: the
    CONVED value projection ``(b, T, heads, value_dim)`` -- the probe's
    receipts recompute the production expression through this helper so
    the instrument sees the machine the arm trains on."""
    batch, tokens, _ = x.shape
    return short_conv1d_causal(mixer.value(x),
                               mixer.v_conv.weight).view(
        batch, tokens, mixer.heads, mixer.value_dim)


class ShortConvComposedMixer(ComposedReadMixer):
    """The K18 mixer: the K15 composed mixer (the same projections, the
    same per-channel gate channels, the same committed ``channel_decay``
    birth, the additive read) with GDN2's short depthwise convolution
    transplanted onto the PAYLOAD path -- ``payload = v_conv(value(x))``
    -- at GDN2's own position (post-projection, pre-recurrence).  At the
    identity birth the conv is EXACTLY the pass-through (the last tap 1,
    the rest 0 -- the standard layout), so the mixer IS the composed
    machine bit for bit and the discrete schedule is the banked winner's
    own; the conv then LEARNS to deviate (a value-channel change: no conv
    value ever feeds op selection)."""

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy, ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                 birth: str = "identity"):
        super().__init__(width, heads, value_dim, rank, rank_max, key_dim,
                        policy, ctx=ctx, segment_tokens=segment_tokens)
        self.v_conv = IdentityBornShortConv(heads * value_dim,
                                            SHORT_CONV_KERNEL_SIZE,
                                            birth=birth)

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        context = causal_context(x, self.ctx)          # (b, T, ctx, width)
        context = context.reshape(batch, tokens, self.ctx * self.width)
        # THE TRANSPLANT: GDN2's v_conv1d position -- the conv sits on the
        # value projection's output, before the store sees the payload.
        payload = short_conv1d_causal(self.value(x),
                                      self.v_conv.weight).view(
            batch, tokens, self.heads, self.value_dim)
        write_keys = self.key(context).view(batch, tokens, self.heads,
                                            self.key_dim)
        query_keys = self.query_key(context).view(batch, tokens, self.heads,
                                                  self.key_dim)
        lam_vec, _lam_inc = self.channel_decay(x)
        gated = self.channels["write"]
        if gated:
            gates = 1.0 - self.write_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(self.b_proj(x)).view(
                    batch, tokens, self.heads, self.value_dim)
        else:
            # a 4-d dummy so the kernel's stride arguments keep their arity
            # (the tensor is never loaded: GATED=0)
            gates = payload.new_ones((1, 1, 1, 1))
        result = composed_scan(payload, write_keys, query_keys, lam_vec,
                               gates, policy=self.policy, gated=gated,
                               rank=self.rank, rank_max=self.rank_max,
                               segment_tokens=self.segment_tokens)
        reads = result.reads
        if self.channels["read"]:
            r_vec = 1.0 + self.read_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(self.g_proj(x)).view(
                    batch, tokens, self.heads, self.value_dim)
            reads = reads * r_vec
        return (self.out(reads.reshape(batch, tokens,
                                       self.heads * self.value_dim)),
                result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mix(x)
        return out


def convert_model_to_short_conv(model,
                                segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                                birth: str = "identity"):
    """Swap every ``ComposedReadMixer`` for the short-conv one, sharing
    parameters (the K10/K11/K11B/K13/K15/K17 conversion pattern: the
    ``nn.Linear`` modules the optimizer already saw are the very same
    objects), adding the depthwise short conv at the registered birth.
    Every carried module is carried over UNTOUCHED, so at the identity
    birth the converted model IS the K15 composed machine bit for bit
    (the conv is exactly the pass-through) and the discrete schedule is
    the banked winner's bit for bit at fixed inputs.  ``k15_composition.py``
    is not edited; this builds on the K15 conversion's output.  The
    parameter DELTA is the registration's (the composed machine's own
    8,657,040 + the conv kernels' 24,576 at depth 24; stated, asserted
    by the probe, not hidden)."""
    if birth not in SHORT_CONV_BIRTHS:
        raise ValueError(f"unregistered conv birth {birth!r}")
    for block in model.blocks:
        mixer = block.mixer
        if not isinstance(mixer, ComposedReadMixer):
            raise ValueError("convert_model_to_short_conv expects a "
                             "K15-composed model (run "
                             "convert_model_to_composition first); got "
                             f"{type(mixer).__name__}")
        sep = ShortConvComposedMixer.__new__(ShortConvComposedMixer)
        torch.nn.Module.__init__(sep)
        sep.__dict__.update(mixer.__dict__)
        sep._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        sep.segment_tokens = int(segment_tokens)
        sep.width = int(mixer.value.weight.shape[1])
        heads = int(mixer.heads)
        value_dim = int(mixer.value_dim)
        sep.v_conv = IdentityBornShortConv(heads * value_dim,
                                           SHORT_CONV_KERNEL_SIZE,
                                           birth=birth)
        block.mixer = sep
    return model


__all__ = ["ShortConvComposedMixer", "IdentityBornShortConv",
           "convert_model_to_short_conv", "short_conv_parameter_delta",
           "short_conv1d_causal", "conv_payload", "SHORT_CONV_KERNEL_SIZE",
           "IDENTITY_SHORT_CONV_BIRTH", "SHIFT_SHORT_CONV_BIRTH",
           "SHORT_CONV_BIRTHS", "DEFAULT_KEY_DIM", "DEFAULT_CONTEXT",
           "DEFAULT_SEGMENT_TOKENS", "ComposedReadMixer",
           "convert_model_to_composition", "composed_parameter_delta"]
