"""TypedHeadMixtureLayer — a horizontal population of NATIVE recurrent head types.

This is the `typed-gdn-2-head` experiment: rather than approximating every
capability as an operating point ("corner") of one unified matrix-recurrence cell
(E98), instantiate a *heterogeneous* population of genuinely-different update
rules side-by-side in the SAME layer. The motivation is e98-sixth-corner: the
unified cell could NOT reproduce GDN's MQAR recall as a placed corner (gated-delta
preset 0.171 vs native GDN 0.951) — GDN recall looked architectural, not a knob.
So here GDN-2 is a FIRST-CLASS head type with its own native delta-memory kernel,
not an E98 knob.

Five native head types live in one layer:

    gdn2_recall : real Gated-DeltaNet-2 delta-memory heads (FLA chunked gated
                  delta-rule kernel, allow_neg_eigval=True == the GDN-2 negative
                  along-key eigenvalue for tracking). Matrix state N=32 per head.
                  This is the recall/associative-memory workhorse.
    e97_track   : E97 split-gated reflection/tracking heads (UnifiedCell 'track'
                  corner WITH the split gate => the validated E97 recurrence).
    count       : pure-integrator heads (UnifiedCell 'count' corner, lambda=1).
    latch       : bistable +/-1 latch heads (UnifiedCell 'latch' corner, tanh).
    nonlin      : iterated-nonlinear-map / state-state heads (UnifiedCell 'nonlin'
                  corner, tanh, state-nonlinear phi).

Composition: the four E98-native corner types share ONE UnifiedCellLayer running
in `fixed_pop` mode (per-head knobs are FROZEN buffers at their corner — the head
"personalities" do not train; only the q/k/v/o projections do). The gdn2_recall
heads run a separate native FLA GatedDeltaNet sub-block. Both sub-blocks map
dim->dim with their own readout, and the layer output is their SUM into the shared
residual stream — i.e. one layer holding two native pathways, sized by the head
allocation.

Deterministic instantiation from type logits (the CMA search variable):
    1. CMA proposes 5 unconstrained logits, one per type.
    2. softmax -> desired type fractions.
    3. largest-remainder rounding -> integer head counts that sum to n_heads.
    4. deterministic (NOT stochastic) allocation; a type may receive zero heads
       and that is reported honestly (no floor is imposed here).

Head shapes are MATCHED: every head (GDN or unified) carries an N=V=32 state and
contributes 32 readout dims to its sub-block, so raw type fractions are directly
comparable. The only per-head parameter asymmetry is the unified split-gate
(erase+value-write projections) vs the GDN short-conv+decay/beta projections;
this is <=~1.4x and is documented in the report. Total model params are matched
across configs by deriving `dim` to a target budget in the CMA driver.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unified_cell import UnifiedCellLayer

try:
    from fla.layers import GatedDeltaNet as _FLAGatedDeltaNet
    _FLA_OK = True
except ImportError:  # pragma: no cover - exercised only without FLA
    _FLAGatedDeltaNet = None
    _FLA_OK = False


# Canonical type order. Index 0 is the native GDN head; 1..4 are the four
# UnifiedCell corner personalities, IN THE SAME ORDER as SPREAD_CORNERS_4 so the
# unified sub-block's corner_mixture is just counts[1:5].
TYPE_NAMES: List[str] = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin']
UNIFIED_CORNER_ORDER = ['track', 'count', 'latch', 'nonlin']  # == SPREAD_CORNERS_4


def largest_remainder_counts(n_heads: int, fractions: List[float]) -> List[int]:
    """Deterministically allocate `n_heads` across len(fractions) types by largest
    remainder. fractions need not be normalized; non-positive entries get 0.

    NO coverage floor: a type with a tiny fraction can legitimately get 0 heads,
    which the experiment reports honestly. Always sums to exactly n_heads.
    """
    fr = [max(0.0, float(f)) for f in fractions]
    s = sum(fr)
    if s <= 0:
        # degenerate -> uniform
        fr = [1.0] * len(fr)
        s = float(len(fr))
    fr = [f / s for f in fr]
    raw = [f * n_heads for f in fr]
    counts = [int(r) for r in raw]
    deficit = n_heads - sum(counts)
    if deficit > 0:
        order = sorted(range(len(fr)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for j in range(deficit):
            counts[order[j % len(fr)]] += 1
    elif deficit < 0:  # pragma: no cover - int() never over-allocates
        for _ in range(-deficit):
            i = max(range(len(fr)), key=lambda i: counts[i])
            counts[i] -= 1
    return counts


def allocate_types(n_heads: int, head_type_logits: List[float]) -> dict:
    """softmax(logits) -> fractions -> largest-remainder integer counts.

    Returns a dict with the per-type fractions and integer counts (keyed by
    TYPE_NAMES) plus the raw GDN / unified split.
    """
    logits = torch.tensor([float(x) for x in head_type_logits], dtype=torch.float64)
    if logits.numel() != len(TYPE_NAMES):
        raise ValueError(
            f"head_type_logits must have {len(TYPE_NAMES)} entries "
            f"({TYPE_NAMES}), got {logits.numel()}")
    fracs = torch.softmax(logits, dim=0).tolist()
    counts = largest_remainder_counts(n_heads, fracs)
    return {
        'type_names': list(TYPE_NAMES),
        'fractions': {TYPE_NAMES[i]: float(fracs[i]) for i in range(len(TYPE_NAMES))},
        'counts': {TYPE_NAMES[i]: int(counts[i]) for i in range(len(TYPE_NAMES))},
        'n_gdn': int(counts[0]),
        'n_unified': int(sum(counts[1:])),
        'unified_counts': [int(c) for c in counts[1:]],  # [track,count,latch,nonlin]
    }


class TypedHeadMixtureLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        n_state: int = 32,
        n_heads: int = 48,
        expansion: float = 1.0,
        head_type_logits: Optional[List[float]] = None,
        # frozen global knobs (NOT tuned per-task; from the cma-capability winner so
        # the placed corners are exactly the validated operating points):
        lam_max: float = 1.585,
        beta_max: float = 2.747,
        igain_max: float = 2.0,
        # GDN-2 native head settings (known-good; frozen):
        gdn_allow_neg_eigval: bool = True,
        gdn_use_conv: bool = True,
        gdn_conv_size: int = 4,
        use_gate: bool = True,
        gate_activation: str = 'silu',
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        if head_type_logits is None:
            # balanced default: equal logits -> ~uniform across the 5 types
            head_type_logits = [0.0] * len(TYPE_NAMES)
        self.dim = dim
        self.n_state = int(n_state)
        self.n_heads = int(n_heads)
        self.expansion = expansion
        self.head_type_logits = [float(x) for x in head_type_logits]

        alloc = allocate_types(self.n_heads, self.head_type_logits)
        self.alloc = alloc
        n_gdn = alloc['n_gdn']
        n_unified = alloc['n_unified']
        unified_counts = alloc['unified_counts']

        # --- native GDN-2 sub-block (recall / associative memory) ---
        self.gdn = None
        if n_gdn > 0:
            if not _FLA_OK:
                raise ImportError(
                    "TypedHeadMixtureLayer needs flash-linear-attention for the "
                    "native GDN-2 heads. pip install flash-linear-attention")
            # FLA GatedDeltaNet: hidden_size==model dim; the head space is
            # num_heads*head_dim, independent of dim. allow_neg_eigval=True is the
            # GDN-2 negative along-key eigenvalue (Grazzi-2025 / DeltaProduct
            # tracking). head_dim=n_state=32 matches the unified heads.
            self.gdn = _FLAGatedDeltaNet(
                hidden_size=dim,
                expand_v=expansion,
                head_dim=self.n_state,
                num_heads=n_gdn,
                use_gate=use_gate,
                use_short_conv=gdn_use_conv,
                conv_size=gdn_conv_size,
                allow_neg_eigval=gdn_allow_neg_eigval,
                mode='chunk',
                layer_idx=0,
            )

        # --- E98-native corner sub-block (track/count/latch/nonlin) ---
        # fixed_pop => per-head knobs are FROZEN buffers at their corner; split_gate
        # on so the 'track' heads are the validated E97 reflection recurrence.
        self.unified = None
        if n_unified > 0:
            s = float(sum(unified_counts))
            mixture = [c / s for c in unified_counts]  # exact fractions -> exact counts
            self.unified = UnifiedCellLayer(
                dim=dim,
                n_state=self.n_state,
                n_heads=n_unified,
                expansion=expansion,
                knob_mode='fixed_pop',
                n_spread_corners=4,
                corner_mixture=mixture,
                split_gate=True,
                lam_max=lam_max,
                beta_max=beta_max,
                igain_max=igain_max,
                head_norm=True,
                use_gate=use_gate,
                gate_activation=gate_activation,
                dropout=dropout,
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def head_alloc(self) -> dict:
        """Allocation metadata for logging (fractions + integer counts per type)."""
        return dict(self.alloc)

    def forward(self, x: torch.Tensor):
        # x: [B, T, dim]. Both sub-blocks map dim->dim; the layer output is their
        # sum into the shared residual stream.
        out = None
        if self.gdn is not None:
            g_out = self.gdn(x, use_cache=False)[0]
            out = g_out if out is None else out + g_out
        if self.unified is not None:
            u_out = self.unified(x)
            out = u_out if out is None else out + u_out
        if out is None:  # pragma: no cover - n_heads>0 guarantees a block
            out = torch.zeros_like(x)
        return self.dropout(out)

    def extra_repr(self):
        c = self.alloc['counts']
        return (f"dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, "
                f"counts={c}")
