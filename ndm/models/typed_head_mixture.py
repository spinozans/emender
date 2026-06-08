"""TypedHeadMixtureLayer — a horizontal population of NATIVE recurrent head types.

This is the `typed-gdn-2-head` experiment: rather than approximating every
capability as an operating point ("corner") of one unified matrix-recurrence cell
(E98), instantiate a *heterogeneous* population of genuinely-different update
rules side-by-side in the SAME layer. The motivation is e98-sixth-corner: the
unified cell could NOT reproduce GDN's MQAR recall as a placed corner (gated-delta
preset 0.171 vs native GDN 0.951) — GDN recall looked architectural, not a knob.
So here GDN-2 is a FIRST-CLASS head type with its own native delta-memory kernel,
not an E98 knob.

Six native head types live in one layer:

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
    gdn2_nonlin_shell : native GDN-2 delta-memory plumbing (same FLA projections
                  /short-conv/gate as gdn2_recall) WITH a bounded nonlinear-in-time
                  state map fused into the chunked scan (GDN2NonlinShellLayer). This
                  is the §3 fairness head: it isolates the nonlinearity-in-time
                  itself from the external UnifiedCell plumbing of `nonlin`.

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
from .e88_fla_hybrid import E88FLAHybrid

try:
    from fla.layers import GatedDeltaNet as _FLAGatedDeltaNet
    _FLA_OK = True
except ImportError:  # pragma: no cover - exercised only without FLA
    _FLAGatedDeltaNet = None
    _FLA_OK = False


# Canonical type order. Index 0 is the native GDN head; 1..4 are the four
# UnifiedCell corner personalities, IN THE SAME ORDER as SPREAD_CORNERS_4 so the
# unified sub-block's corner_mixture is just counts[1:5]; index 5 is the native
# GDN-2 shell with a fused nonlinear-in-time state (gdn2_nonlin_shell); indices
# 6,7 are the FUSED E97 split-edit heads (e97-raw / e97-delta) — a genuine E97
# split-edit recurrence run over its allocated head subset on the bf16 Triton
# kernel (use_split_edit=True, raw_write True/False, state tanh), exactly the
# validated E97 LM cell, NOT an approximated UnifiedCell corner. These close the
# gap that forced the inferior interleaved-layer fallback (task e97-heads-in):
#   e97_raw   : E97 split-edit + RAW WRITE (write v, drop the delta read term).
#               #1 cell on the 1.3B LM CMA leaderboard (avg-loss 5.9511).
#   e97_delta : E97 split-edit + DELTA correction (write v - S@k). The plain E97.
# Both run the FUSED split-edit Triton fwd+bwd kernel (the use_triton path,
# commit 4db8099), parity-verified vs the eager reference at bf16.
#
# Backward compatibility: callers that predate later types pass 5 (pre-shell) or
# 6 (pre-e97) logits. Those are right-padded with -inf for the new trailing
# slots, which softmax maps to 0 heads, reproducing the legacy allocation
# EXACTLY (see allocate_types). A 7-vector is rejected (no historical contract).
TYPE_NAMES: List[str] = ['gdn2_recall', 'e97_track', 'count', 'latch', 'nonlin',
                         'gdn2_nonlin_shell', 'e97_raw', 'e97_delta']
LEGACY_N_TYPES = 5  # the original 5-type contract; trailing types padded off
# accepted legacy logit-vector lengths; each is right-padded with -inf to the
# full TYPE_NAMES width so softmax assigns the new trailing types 0 heads.
ACCEPTED_LEGACY_LENS = (5, 6)
UNIFIED_CORNER_ORDER = ['track', 'count', 'latch', 'nonlin']  # == SPREAD_CORNERS_4
# the unified corner types occupy indices 1..4 (between gdn2_recall and the shell)
UNIFIED_SLICE = slice(1, 5)


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
    raw = [float(x) for x in head_type_logits]
    if len(raw) in ACCEPTED_LEGACY_LENS:
        # legacy call (5-type pre-shell, or 6-type pre-e97): right-pad the new
        # trailing slots OFF (softmax(-inf)==0) so the historical types keep their
        # EXACT softmax and the new types are allocated 0 heads.
        raw = raw + [float('-inf')] * (len(TYPE_NAMES) - len(raw))
    if len(raw) != len(TYPE_NAMES):
        raise ValueError(
            f"head_type_logits must have {len(TYPE_NAMES)} entries "
            f"({TYPE_NAMES}) or one of {ACCEPTED_LEGACY_LENS} (legacy, trailing "
            f"types padded off), got {len(raw)}")
    logits = torch.tensor(raw, dtype=torch.float64)
    fracs = torch.softmax(logits, dim=0).tolist()
    counts = largest_remainder_counts(n_heads, fracs)
    unified_counts = [int(c) for c in counts[UNIFIED_SLICE]]  # [track,count,latch,nonlin]
    return {
        'type_names': list(TYPE_NAMES),
        'fractions': {TYPE_NAMES[i]: float(fracs[i]) for i in range(len(TYPE_NAMES))},
        'counts': {TYPE_NAMES[i]: int(counts[i]) for i in range(len(TYPE_NAMES))},
        'n_gdn': int(counts[0]),
        'n_unified': int(sum(unified_counts)),
        'unified_counts': unified_counts,
        'n_shell': int(counts[5]),
        'n_e97_raw': int(counts[6]),
        'n_e97_delta': int(counts[7]),
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
        # native GDN-2 shell w/ fused nonlinear-in-time state (the 6th head type):
        shell_state_nonlin: str = 'tanh',
        shell_state_chunk: int = 64,
        # FUSED E97 split-edit heads (7th/8th head types: e97_raw, e97_delta).
        # use_triton_e97=True routes them through the bf16 split-edit Triton
        # fwd/bwd kernel (commit 4db8099) — the ONLY fused path for split-edit /
        # raw-write (the CUDA register-owned kernel rejects both). state_activation
        # 'tanh' matches the validated E97 LM cell (kernel-compatible: tanh/identity
        # only). cast_recurrent_bf16 feeds these heads bf16 under autocast/fp32 so
        # the fused gate (which requires x.dtype==bfloat16) actually engages instead
        # of silently falling back to the eager T-scan — the bug that bit
        # wire-fused-e97. With it on, e97 heads are NEVER eager during training.
        use_triton_e97: bool = True,
        cast_recurrent_bf16: bool = True,
        e97_state_nonlin: str = 'tanh',
        # Route the e97_delta head through the chunked-parallel fwd+bwd Triton
        # kernel (GDN-2-class throughput; flips the within-layer NO-GO). Engages
        # only for LINEAR-state e97_delta (per-step tanh is non-chunkable); with
        # e97_state_nonlin='tanh' the head keeps the sequential split-edit kernel.
        use_chunked_e97_delta: bool = True,
        e97_chunk_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        if head_type_logits is None:
            # balanced default: equal logits -> ~uniform across the 6 types
            head_type_logits = [0.0] * len(TYPE_NAMES)
        self.dim = dim
        self.n_state = int(n_state)
        self.n_heads = int(n_heads)
        self.expansion = expansion
        self.head_type_logits = [float(x) for x in head_type_logits]

        self.use_triton_e97 = bool(use_triton_e97)
        self.cast_recurrent_bf16 = bool(cast_recurrent_bf16)
        self.e97_state_nonlin = str(e97_state_nonlin)
        self.use_chunked_e97_delta = bool(use_chunked_e97_delta)
        self.e97_chunk_size = int(e97_chunk_size)

        alloc = allocate_types(self.n_heads, self.head_type_logits)
        self.alloc = alloc
        n_gdn = alloc['n_gdn']
        n_unified = alloc['n_unified']
        unified_counts = alloc['unified_counts']
        n_shell = alloc['n_shell']
        n_e97_raw = alloc['n_e97_raw']
        n_e97_delta = alloc['n_e97_delta']

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

        # --- native GDN-2 shell sub-block (nonlinear-in-time state) ---
        # Same native delta-memory plumbing as gdn2_recall but with a bounded
        # nonlinear-in-time state map fused into the chunked scan (the §3 head
        # under fairness test). dim->dim like the other sub-blocks; summed in.
        self.shell = None
        if n_shell > 0:
            from .gdn2_nonlin_shell import GDN2NonlinShellLayer
            self.shell = GDN2NonlinShellLayer(
                dim=dim,
                n_state=self.n_state,
                n_heads=n_shell,
                expansion=expansion,
                state_nonlin=shell_state_nonlin,
                state_chunk=shell_state_chunk,
                gdn_allow_neg_eigval=gdn_allow_neg_eigval,
                gdn_use_conv=gdn_use_conv,
                gdn_conv_size=gdn_conv_size,
                use_gate=use_gate,
                dropout=dropout,
            )

        # --- FUSED E97 split-edit sub-blocks (e97_raw / e97_delta) ---
        # Each is a genuine E88FLAHybrid running the validated E97 split-edit
        # recurrence over its allocated head subset, on the fused bf16 Triton
        # kernel. raw_write distinguishes the two: True == raw write (the 1.3B
        # leaderboard winner), False == delta read-modify-write (plain E97). Same
        # n_state / use_gate / silu gate as the other sub-blocks so head shapes
        # stay matched; dim->dim, summed into the shared residual.
        #   use_gate + gate_activation='silu' are REQUIRED for the fused split-edit
        #   dispatch (_use_optimized gate in E88FLAHybrid.forward); the head-type
        #   contract here always carries them so the kernel engages.
        e97_common = dict(
            dim=dim,
            n_state=self.n_state,
            expansion=expansion,
            use_split_edit=True,
            state_activation=self.e97_state_nonlin,
            use_gate=True,             # required by the fused split-edit dispatch
            gate_activation='silu',    # required by the fused split-edit dispatch
            use_triton=self.use_triton_e97,
            dropout=dropout,
        )
        self.e97_raw = None
        if n_e97_raw > 0:
            self.e97_raw = E88FLAHybrid(n_heads=n_e97_raw, raw_write=True, **e97_common)
        self.e97_delta = None
        if n_e97_delta > 0:
            self.e97_delta = E88FLAHybrid(
                n_heads=n_e97_delta, raw_write=False,
                use_chunked_e97=self.use_chunked_e97_delta,
                e97_chunk_size=self.e97_chunk_size,
                **e97_common,
            )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _run_e97(self, block: "E88FLAHybrid", x: torch.Tensor) -> torch.Tensor:
        """Run a fused E97 split-edit sub-block, guaranteeing the fused kernel
        engages (no silent eager fallback) when use_triton is requested.

        The split-edit/raw-write fused Triton fwd+bwd kernel dispatches only when
        the input is bf16 (plus use_gate/silu/training, set at construction). The
        residual stream is fp32 under autocast (RMSNorm emits fp32) and fp32 in the
        typed-gdn2-lm sanity dtype, so without an explicit cast the kernel gate
        fails and the recurrence silently runs the eager T-scan — the wire-fused-e97
        bug. We cast the sub-block input to bf16 and, during training, FAIL LOUDLY
        if the fused path still could not engage rather than degrade silently.
        """
        xin = x
        if block.use_triton and self.cast_recurrent_bf16 and xin.is_cuda \
                and xin.dtype == torch.float32:
            xin = xin.to(torch.bfloat16)
        if block.use_triton and block.training and xin.is_cuda \
                and xin.dtype != torch.bfloat16:
            raise RuntimeError(
                "Fused E97 split-edit head requires bf16 input to engage the Triton "
                f"kernel (got {xin.dtype}); refusing to silently fall back to the eager "
                "T-scan. Enable cast_recurrent_bf16 or feed the layer bf16.")
        # The fused split-edit kernel's sparse-checkpoint forward requires
        # T % checkpoint_interval == 0. The LM next-token path feeds T-1 timesteps,
        # so an aligned context (512/1024) arrives here unaligned (511/1023). The
        # recurrence is strictly causal, so zero-padding the END of the sequence to
        # the next multiple and truncating back is EXACT — appended future steps
        # cannot change earlier outputs. This keeps every fused E97 head on the
        # kernel for ANY sequence length instead of crashing or going eager.
        T = xin.shape[1]
        pad = 0
        if block.use_triton:
            ckpt = int(getattr(block, 'checkpoint_interval', 16) or 16)
            pad = (-T) % ckpt
        if pad:
            xin = F.pad(xin, (0, 0, 0, pad))
        out = block(xin)[0]
        if pad:
            out = out[:, :T]
        return out.to(x.dtype)

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
        if self.shell is not None:
            s_out = self.shell(x)
            out = s_out if out is None else out + s_out
        if self.e97_raw is not None:
            r_out = self._run_e97(self.e97_raw, x)
            out = r_out if out is None else out + r_out
        if self.e97_delta is not None:
            d_out = self._run_e97(self.e97_delta, x)
            out = d_out if out is None else out + d_out
        if out is None:  # pragma: no cover - n_heads>0 guarantees a block
            out = torch.zeros_like(x)
        return self.dropout(out)

    def extra_repr(self):
        c = self.alloc['counts']
        return (f"dim={self.dim}, n_heads={self.n_heads}, n_state={self.n_state}, "
                f"counts={c}")
