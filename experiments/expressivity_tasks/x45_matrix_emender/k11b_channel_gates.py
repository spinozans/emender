"""K11B -- PER-CHANNEL GATES on the emender: GDN2's gate granularity
(per-channel, per-token, input-dependent DECAY / write admission / read-out)
through the existing recorded-schedule channels, decisions inviolable.

Why this file exists
--------------------
K11 tested the per-head SCALAR write-strength + read-out gate surface and
receipted the honest null (``RESULT_LEARNED_GATES.md``: "the per-head scalar
gate surface does not own the residual").  The op-set audit's table
(``RESULT_OP_SET_AUDIT.md`` 1.4) names the gap this file attacks: our DECAY
is "the strictly poorer instance -- per-head scalar vs GDN2's per-channel
input-driven".  GDN2's actual gates are PER-CHANNEL, PER-TOKEN,
INPUT-DEPENDENT (``/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2.py``):
``f_proj`` a small MLP to a per-token decay pre-activation, ``b_proj`` the
per-token write/erase gate, ``g_proj`` the per-token read gate -- the bulk of
its 27.4M gate parameters.  The per-channel form on OUR machine was
untested; this lane is it.

The semantics (``PREREGISTRATION_CHANNEL_GATES.md``, committed before any
probe measurement; the supervisor's rulings cited inline)
--------------------------------------------------------------------------------
* **Arm P1 -- per-channel input-dependent DECAY** (the f_proj analog, the
  audit's own "strictly poorer" row): a per-token ``d_v``-dim decay VECTOR
  replacing the per-head scalar, consumed through the same DECAY channel
  (values only, keys never decayed):
  ``lam_vec = clamp(lam_inc * (1 + m_f*sigmoid(f_proj(x))), DECAY_FLOOR,
  1-1e-6)`` with ``lam_inc = sigmoid(decay(x)).clamp(...)`` the incumbent's
  REGISTERED per-head scalar projection, kept verbatim and still trained;
  ``f_proj`` GDN2's own shape transplanted (Sequential(Linear(width, d_v,
  bias=False), Linear(d_v, heads*d_v, bias=False)), all weights born at
  exact zero) and ``m_f`` a per-head scalar born at EXACT zero.  At birth
  the factor is exactly 1.0, so ``lam_vec`` is BIT-EQUAL to the registered
  scalar on every channel (IEEE mul-by-1.0 is exact; clamp of an in-range
  value is the identity) -- the arm is born as the banked winner.
  SUPERVISOR RULING (approved): the per-channel input-DEPENDENCE and channel
  width are transplanted whole; the ABSOLUTE form is a deviation around the
  registered scalar because mimetic-neutral birth requires it (GDN2 has no
  incumbent scalar to be born equal to).  The axis reading is d_v (the VALUE
  channel) per the task text: GDN2's f_proj decays its FIXED key axis, and
  our machine has no fixed address axis -- slots are content-addressed and
  dynamic -- so the only axis a learned projection can transplant onto
  whole is the value channel.  The rejected strict analog -- PER-SLOT
  (address-axis) decay, "decay-by-address-match", the write key's similarity
  to each slot entering the decay -- is the registration's NAMED FOLLOW-UP
  (not built in this lane): it is a genuinely different mechanism, not a
  projection transplant.
* **Arm P2 -- + per-channel write admission** (the b_proj analog at
  CORRECT's blend / APPEND's admission): ``g_vec = 1 - m_w*sigmoid(b_proj(x))``
  per (token, head, channel) with ``b_proj = Linear(width, heads*d_v,
  bias=False)`` zero-born (GDN2's b_proj is a single bias-free Linear) and
  ``m_w`` per-head born at exact zero.  APPEND admits ``g_vec (.) p`` (the
  Lean APPEND's ``gate*value``), CORRECT blends
  ``g_vec (.) p + (1-g_vec) (.) (lam_vec (.) v_pre)`` per channel (the Lean
  MERGE arithmetic at the CORRECT write enable); MERGE/POP/NONE stay
  VERBATIM.  Born at exact 1 per channel.
* **Arm P3 -- + per-channel read gates** (the g_proj analog), combined:
  ``r_vec = 1 + m_o*sigmoid(g_proj(x))`` with ``g_proj =
  Sequential(Linear(width, d_v, bias=False), Linear(d_v, heads*d_v,
  bias=True))`` zero-born (GDN2's g_proj shape), applied
  ``reads' = reads (.) r_vec`` in the mixer before the output projection
  (outside the scan; K11's read-out convention).  Born at exact 1.
* **P2 (decisions inviolable), verbatim**: no gate value feeds op selection.
  The decision chain is the K10 verbatim on the detached key channel; the
  gates scale payload magnitudes, the DECAY channel and read outputs ONLY.
  The discrete op sequence is bit-invariant to every gate value at fixed
  inputs (the 0-flip receipt).
* **P5 (mimetic-neutral birth)**: every channel carries a per-head zero-born
  scalar over zero-born projections, so at birth ``lam_vec == lam_inc`` per
  channel, ``g_vec == 1`` per channel and ``r_vec == 1`` per channel
  exactly -- the arm is born as the banked winner bit for bit, and the gates
  grow by training.  The gradient reaches the zero-born scalar at step 0
  (sigmoid(0) = 0.5); the projections wake one step later (the K11 two-step
  wake-up pattern -- the step-0 gradient to f_proj's weights is exactly 0
  while m_f == 0; receipted so the step-0 zero is not read as a dead
  channel).

The arithmetic spec is K10/K11's verbatim (defined-order tree folds, fp32,
correctly-rounded IEEE ops, libdevice exp, ``mul.rn``/``add.rn``/``sub.rn``)
with the DECAY channel and the gated lane-0 arithmetic elementwise in the
value channel: ``acc = mul.rn(acc, lam_vec)`` broadcast per channel, and the
CORRECT blend's pre-decay form
``g_vec*p + (1-g_vec)*(lam_vec*v_pre)`` -- at birth both degenerate to the
incumbent's arithmetic bit for bit.  The recorded schedule is K10's own
2-lane layout (no dense lane: K11's G2 receipted the dense channel's null
separately); the recorded lane-0 payload is the BLENDED lane value, an
ordinary scheduled write.

The kernels (K10/K11's walks, extended; ``k10_*``/``k11_*``/``k6_*``/
``k1_*`` sources are imported, never edited)
--------------------------------------------------------------------------------
``_k11b_seq_fwd_kernel``     K11's persistent whole-window walk with the
                             vector DECAY load and the vector write gate
                             (``GATED`` constexpr), the 2-lane schedule.
``_k11b_traj_kernel``        the segment value-trajectory replay at K4's own
                             defined-order spec with the PER-CHANNEL decay
                             load -- K4's imported kernel takes per-head
                             SCALAR decay only, the engine limitation the
                             supervisor ruled on; this lane-specific kernel
                             carries it (K10's bwd-prep and K11's key-traj
                             are imported verbatim: keys are not decayed
                             and the prep is gate-agnostic).
``_k11_bwd_prep_kernel``    IMPORTED VERBATIM (read weights recomputed from
                             the trajectory -- gate-agnostic).
``_k11b_bwd_value_walk_kernel`` K11's reverse value walk plus the
                             per-channel gate adjoint (``dpayload =
                             g_vec*dpay0``, ``dgate = dpay0*(p - v_dec)``
                             per channel, the ``(1-g_vec)`` old-row path at
                             CORRECT) and the per-channel DECAY adjoint
                             (``dlam[c] = <ds[:,c], S_t[:,c]>`` per channel,
                             ``ds <- lam_vec (.) ds``).
``_k11_key_traj_kernel``    IMPORTED VERBATIM (the key lane lives on lanes
                             0/1; keys are never decayed).
``_k11_bwd_key_walk_kernel`` IMPORTED VERBATIM (the key-table adjoint,
                             stride-generalized in K11 already).

Doctrine, unchanged: decisions are recorded constants in the backward.  The
read weights, payload lanes, the DECAY channel, the gates and the KEY
CHANNEL carry gradient to their inputs; the cross-axis reductions are
autograd-ordered on the oracle side, so the gradient gates are
floor-discipline (fp64 gradcheck of the oracle first, fp32-vs-fp64 floors,
``NOISE_FACTOR`` x), stated, not hidden.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule)
from .k1_schedule_scan import NONE_CODE, _add_rn, _mul_rn, _sub_rn
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                   _log2_up, _norm_defined, _pow2,
                                   _round_f32, _seq_warps, _sqrt_rn32,
                                   tree_fold)
from .k10_separated_keys import (_key_selection, _k10_bwd_prep_kernel,
                                 DEFAULT_CONTEXT, DEFAULT_KEY_DIM,
                                 DEFAULT_SEGMENT_TOKENS,
                                 separated_available, _split_fold_axis2)
from .k11_learned_gates import (_k11_bwd_key_walk_kernel,
                                _k11_key_traj_kernel)
from .key_separation_probe import causal_context
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

# The recorded schedule's lane count for every k11b scan: K10's own 2-lane
# layout (the main write and the MERGE-source close).  K11B carries NO dense
# lane -- K11's G2 arm receipted the dense-write channel's null separately.
LANES = 2
# The backward's walk blocking (K10/K11's own constants).
BT_BWD = 16
BV_BWD = 16
# The arms (the registration's 1, cumulative staging): which gate channels
# each arm carries.  The per-channel DECAY channel is in every arm (the
# audit's "strictly poorer instance" row is ARM 1, the priority).
ARMS = ("p1", "p2", "p3")
ARM_CHANNELS = {
    "p1": dict(decay=True, write=False, read=False),
    "p2": dict(decay=True, write=True, read=False),
    "p3": dict(decay=True, write=True, read=True),
}


def arm_channels(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(f"unregistered arm: {arm!r} (expected one of {ARMS})")
    return dict(ARM_CHANNELS[arm])


# ------------------------------------------------------------- the oracle


def channel_gated_oracle_scan(payload, write_keys, query_keys, decay, gates,
                              *, policy: RoutingPolicy, gated: bool = True,
                              rank=None, rank_max=None, initial_state=None,
                              initial_live=None, initial_keys=None,
                              collect=True):
    """The per-channel-gate arm's eager per-token oracle: the parity fixed
    point, differentiable through the key channel, the per-channel DECAY
    vector, the per-channel write gate, and the value state.

    K10's oracle walk (decisions from the DETACHED key channel, read weights
    from the ATTACHED state and key table, the MERGE blend from the
    PRE-decay rows, DECAY then lanes in the registered order) with the two
    per-channel extensions: the DECAY channel is a per-token ``d_v``-dim
    VECTOR (``state <- state * lam_vec`` per channel), and the gated lane-0
    arithmetic (APPEND admits ``g_vec (.) p``, CORRECT blends
    ``g_vec (.) p + (1-g_vec) (.) (lam_vec (.) v_pre)`` per channel).  The
    recorded schedule carries TWO lanes; the lane-0 payload field is the
    blended lane value, so the replay consumes it verbatim.
    """
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if decay.shape != payload.shape:
        raise ValueError(f"the per-channel decay must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(decay.shape)}")
    if gated and gates.shape != payload.shape:
        raise ValueError(f"the per-channel gates must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(gates.shape)}")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the "
                          "channel-gate registration; it is a chunk-dial "
                          "control")
    if policy.grow:
        raise ValueError("growth is refused by the channel-gate "
                          "registration: it is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the channel-gate registration requires rank == "
                         "rank_max (the registered shape); a rank prefix is "
                         "refused by name")
    device = payload.device
    compute_dtype = torch.promote_types(_fp32(payload).dtype, torch.float32)
    state = (torch.zeros(batch, heads, rank, value_dim, dtype=compute_dtype,
                         device=device) if initial_state is None
             else initial_state)
    live = (torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
            if initial_live is None else initial_live.clone())
    keys = (torch.zeros(batch, heads, rank, key_dim, dtype=compute_dtype,
                        device=device) if initial_keys is None
            else initial_keys)
    p32 = _fp32(payload)
    wk32 = _fp32(write_keys)
    qk32 = _fp32(query_keys)
    d32 = _fp32(decay)
    g32 = _fp32(gates)
    RP, VP, KP = _pow2(rank), _pow2(value_dim), _pow2(key_dim)
    tau_match, tau_merge, tau_pop = (_round_f32(policy.tau_match),
                                     _round_f32(policy.tau_merge),
                                     _round_f32(policy.tau_pop))
    tie = _round_f32(policy.tie_break)
    scale = _round_f32(policy.read_scale)
    lanes = LANES
    reads = torch.empty(batch, tokens, heads, value_dim, dtype=compute_dtype,
                        device=device)
    fields = dict(
        slot=torch.zeros(batch, tokens, heads, lanes, dtype=torch.int64,
                         device=device),
        erase=torch.zeros(batch, tokens, heads, lanes, dtype=torch.bool,
                          device=device),
        payload=torch.zeros(batch, tokens, heads, lanes, value_dim,
                            dtype=compute_dtype, device=device),
        op=torch.zeros(batch, tokens, heads, lanes, dtype=torch.int64,
                       device=device),
        live_after=torch.zeros(batch, tokens, heads, lanes, dtype=torch.bool,
                               device=device))
    weights_out = (torch.zeros(batch, tokens, heads, rank,
                               dtype=compute_dtype, device=device)
                   if collect else None)
    live_pre_out = (torch.zeros(batch, tokens, heads, rank, dtype=torch.bool,
                                device=device) if collect else None)
    b_index = torch.arange(batch, device=device)[:, None]
    h_index = torch.arange(heads, device=device)[None, :]
    for t in range(tokens):
        if collect:
            live_pre_out[:, t] = live
        p_t, wk_t, qk_t = p32[:, t], wk32[:, t], qk32[:, t]
        lam_t = d32[:, t]                                 # (b, h, V)
        # ---- the read: certified weights on the KEY channel (K10 verbatim).
        nr = _norm_defined(state, -1, VP)
        nk = _norm_defined(keys, -1, KP)
        nqk = _norm_defined(qk_t, -1, KP)
        uqk = qk_t / nqk.clamp_min(EPS).unsqueeze(-1)
        scores = (scale * tree_fold(uqk.unsqueeze(2) * keys, -1, KP)) \
            / nk.clamp_min(EPS)
        live_f = live.to(compute_dtype)
        any_live = live.any(-1, keepdim=True)
        masked = scores.masked_fill(~live, float("-inf"))
        safe = torch.where(any_live, masked, torch.zeros_like(masked))
        pmax = safe.max(-1).values
        expo = torch.exp(safe - pmax.unsqueeze(-1))
        denom = tree_fold(expo * live_f, -1, RP)
        any_live_flat = live.any(-1)
        denom_safe = torch.where(any_live_flat, denom, torch.ones_like(denom))
        probs = torch.where(any_live, expo / denom_safe.unsqueeze(-1),
                            torch.zeros_like(expo))
        w = probs * nr * live_f
        reads[:, t] = tree_fold(w.unsqueeze(-1) * state, -2, RP)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the registered chain on the DETACHED key channel
        # (P2: no gate value feeds op selection).
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)
        sim = sim.masked_fill(~live, -2.0)
        sel = _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie,
                             rank, RP)
        # ---- the MERGE blend from the PRE-decay rows (K10 verbatim).
        strength = _norm_defined(state.detach(), -1, VP) * live.to(compute_dtype)
        blend = None
        if bool(sel["merge"].any()):
            rows = state[b_index.expand(batch, heads),
                         h_index.expand(batch, heads), sel["best_index"]]
            rows2 = state[b_index.expand(batch, heads),
                          h_index.expand(batch, heads), sel["second_index"]]
            fw = strength.gather(-1, sel["best_index"].unsqueeze(-1)
                                 ).clamp_min(EPS).squeeze(-1)
            sw = strength.gather(-1, sel["second_index"].unsqueeze(-1)
                                 ).clamp_min(EPS).squeeze(-1)
            total = fw + sw
            blend = ((fw / total).detach().unsqueeze(-1) * rows
                     + (sw / total).detach().unsqueeze(-1) * rows2)
        # ---- DECAY (values only, PER CHANNEL), then the lane-0 value (K10's
        # registered order).  The CORRECT blend is the PRE-decay form
        # g*p + (1-g)*(lam*v_pre) per channel -- the identical expression the
        # kernel implements, so the parity bar is torch.equal; at the birth
        # (g == 1 per channel, lam_vec == the registered scalar) it
        # degenerates to the incumbent's arithmetic bit for bit.
        rows_pre = state[b_index.expand(batch, heads),
                         h_index.expand(batch, heads), sel["slot0"]]
        state = state * lam_t.unsqueeze(2)                # (b,h,R,V)*(b,h,1,V)
        rows_at = state[b_index.expand(batch, heads),
                        h_index.expand(batch, heads), sel["slot0"]]
        if gated:
            gate_t = g32[:, t]                            # (b, h, V)
            blend_correct = gate_t * p_t + (1.0 - gate_t) * (lam_t * rows_pre)
            admission = gate_t * p_t
            lane0 = torch.where(sel["correct"].unsqueeze(-1), blend_correct,
                                torch.where(sel["carries"].unsqueeze(-1),
                                            admission,
                                            torch.zeros_like(p_t)))
        else:
            lane0 = torch.where(sel["carries"].unsqueeze(-1), p_t,
                                torch.zeros_like(p_t))
        if blend is not None:
            lane0 = torch.where(sel["merge"].unsqueeze(-1), blend, lane0)
        # ---- the key lane: K10 verbatim.
        key_lane = torch.where(sel["op0"].eq(POP).unsqueeze(-1),
                               torch.zeros_like(wk_t), wk_t)
        # ---- record lanes 0/1 (K10's 2-lane layout) and apply.
        fields["slot"][:, t, :, 0] = sel["slot0"]
        fields["slot"][:, t, :, 1] = sel["second_index"]
        fields["op"][:, t, :, 0] = sel["op0"]
        fields["op"][:, t, :, 1] = torch.where(sel["merge"],
                                               torch.full_like(sel["op0"], MERGE),
                                               torch.full_like(sel["op0"], NONE))
        fields["erase"][:, t, :, 0] = sel["erase0"]
        fields["erase"][:, t, :, 1] = sel["merge"]
        fields["live_after"][:, t, :, 0] = sel["live0"]
        fields["live_after"][:, t, :, 1] = False
        fields["payload"][:, t, :, 0, :] = lane0
        written = torch.where(sel["erase0"].unsqueeze(-1), lane0,
                              rows_at + lane0)
        state = state.index_put((b_index.expand(batch, heads),
                                 h_index.expand(batch, heads),
                                 sel["slot0"]), written)
        keys_at = keys[b_index.expand(batch, heads),
                       h_index.expand(batch, heads), sel["slot0"]]
        keys = keys.index_put(
            (b_index.expand(batch, heads),
             h_index.expand(batch, heads),
             sel["slot0"]),
            torch.where(sel["op0"].ne(NONE).unsqueeze(-1), key_lane, keys_at))
        live = live.index_put((b_index.expand(batch, heads),
                               h_index.expand(batch, heads),
                               sel["slot0"]),
                              torch.where(sel["op0"].ne(NONE), sel["live0"],
                                          live.gather(-1,
                                                      sel["slot0"].unsqueeze(-1)
                                                      ).squeeze(-1)))
        # the MERGE-source close, guarded PER LANE (the K12-flagged
        # multi-lane corner of the K10/K11 oracle -- index-putting zeros at
        # second_index for EVERY (batch, head) lane whenever ANY lane
        # merges -- is NOT copied: this is a new file and the known bug is
        # fixed at birth, matching the kernel's per-lane guard
        # ``hit2 = merge & (second == ranks)`` bit for bit; the gate-E
        # analog receipts oracle-vs-kernel agreement on a crafted
        # heterogeneous MERGE fixture, where the K10/K11 oracle diverges
        # from its own kernel)
        if bool(sel["merge"].any()):
            second_rows = state[b_index.expand(batch, heads),
                                 h_index.expand(batch, heads),
                                 sel["second_index"]]
            state = state.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"].unsqueeze(-1),
                            torch.zeros(batch, heads, value_dim,
                                        dtype=compute_dtype, device=device),
                            second_rows))
            second_keys = keys[b_index.expand(batch, heads),
                                h_index.expand(batch, heads),
                                sel["second_index"]]
            keys = keys.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"].unsqueeze(-1),
                            torch.zeros(batch, heads, key_dim,
                                        dtype=compute_dtype, device=device),
                            second_keys))
            second_live = live.gather(-1, sel["second_index"].unsqueeze(-1)
                                       ).squeeze(-1)
            live = live.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["merge"], torch.zeros(batch, heads,
                                                      dtype=torch.bool,
                                                      device=live.device),
                            second_live))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, keys=keys,
                schedule=schedule, weights=weights_out, live_pre=live_pre_out,
                rank=int(rank))


# ------------------------------------------------------------- the forward kernel

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k11b_seq_fwd_kernel(
    payload, write_keys, query_keys, decay, gates, state_in, live_in, keys_in,
    reads, state_out, live_out, keys_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, ckpt_keys, weights_out,
    T, R, ROWS, SEG, KD,
    pbs, pts, phs, pvs,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    cbs, cts, chs, cvs,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, L: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr, LOGK: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    GATED: tl.constexpr,
    STAGES: tl.constexpr,
):
    """One program per (batch*head) row: K11's whole-window walk with the
    PER-CHANNEL decay load (``acc = mul.rn(acc, lam_vec)`` broadcast per
    value channel), the per-channel write gate at the lane-0 arithmetic, and
    K10's own 2-lane recorded schedule.  The value tile ``acc`` and the key
    tile ``kacc`` are register-resident for the whole window; per token the
    K10-verbatim read/decision/blend chain, then DECAY (per channel), the
    gated lane 0 and the MERGE-source close."""
    row = tl.program_id(0)
    batch_index = row // H
    head_index = row % H

    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = tl.arange(0, VP)
    keep_v = values < V
    kdims = tl.arange(0, KP)
    keep_k = kdims < KD
    keep = keep_r[:, None] & keep_v[None, :]
    keep_rk = keep_r[:, None] & keep_k[None, :]

    acc = tl.load(state_in + row.to(tl.int64) * R * V
                  + ranks[:, None] * V + values[None, :], mask=keep, other=0.0)
    kacc = tl.load(keys_in + row.to(tl.int64) * R * KD
                   + ranks[:, None] * KD + kdims[None, :], mask=keep_rk,
                   other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0

    payload_row = payload + batch_index * pbs + head_index * phs
    wk_row = write_keys + batch_index * wbs + head_index * whs
    qk_row = query_keys + batch_index * qbs + head_index * qhs
    decay_row = decay + batch_index * cbs + head_index * chs
    gates_row = gates + batch_index * gbs + head_index * ghs
    reads_base = reads + batch_index.to(tl.int64) * T * H * V + head_index * V
    live_pre_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                     + head_index * R)
    slot_base = slot_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    op_base = op_out + batch_index.to(tl.int64) * T * H * L + head_index * L
    erase_base = (erase_out + batch_index.to(tl.int64) * T * H * L
                  + head_index * L)
    live_after_base = (live_after_out + batch_index.to(tl.int64) * T * H * L
                       + head_index * L)
    payload_base = (payload_out + batch_index.to(tl.int64) * T * H * L * V
                    + head_index * L * V)
    weights_base = (weights_out + batch_index.to(tl.int64) * T * H * R
                    + head_index * R)

    for t in tl.range(0, T, num_stages=STAGES):
        # ---- the state and key table as this token's read and decisions see
        # them (pre-DECAY, pre-lanes).
        tl.store(live_pre_base + t * H * R + ranks, live.to(tl.int8),
                 mask=keep_r)
        if t % SEG == 0:
            seg = t // SEG
            ck = (seg * ROWS + row).to(tl.int64)
            tl.store(ckpt_state + ck * R * V + ranks[:, None] * V
                     + values[None, :], acc, mask=keep)
            tl.store(ckpt_live + ck * R + ranks, live.to(tl.int8),
                     mask=keep_r)
            tl.store(ckpt_keys + ck * R * KD + ranks[:, None] * KD
                     + kdims[None, :], kacc, mask=keep_rk)
        p = tl.load(payload_row + t * pts + values * pvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        wk = tl.load(wk_row + t * wts + kdims * wks, mask=keep_k,
                     other=0.0).to(tl.float32)
        qk = tl.load(qk_row + t * qts + kdims * qks, mask=keep_k,
                     other=0.0).to(tl.float32)
        # the PER-CHANNEL decay vector (the registered scalar's replacement)
        lam = tl.load(decay_row + t * cts + values * cvs, mask=keep_v,
                      other=0.0).to(tl.float32)

        # ---- the value norms and the read weights on the KEY channel (K10).
        nr = tl.reshape(_sqrt_rn32(_fold_axis2(_mul_rn(acc, acc), VP, LOGV)),
                        (RP,))
        strength = _mul_rn(nr, tl.where(live & keep_r, 1.0, 0.0))
        nk = tl.reshape(_sqrt_rn32(_split_fold_axis2(_mul_rn(kacc, kacc), KP,
                                                     LOGK)), (RP,))
        nqk = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(qk, qk), KP, LOGK)), (1,))
        uqk = _div_rn32(qk, tl.broadcast_to(tl.maximum(nqk, _EPS), (KP,)))
        scores = _div_rn32(
            _mul_rn(tl.reshape(_split_fold_axis2(_mul_rn(uqk[None, :], kacc),
                                                 KP, LOGK), (RP,)), SCALE),
            tl.maximum(nk, _EPS))
        live_r = live & keep_r
        live_any = tl.sum(live_r.to(tl.int32), 0) > 0
        masked = tl.where(live_r, scores, float('-inf'))
        safe = tl.where(live_any, masked, 0.0)
        pmax = tl.max(safe, 0)
        expo = libdevice.exp(safe - pmax)
        denom = tl.reshape(_fold_vec(tl.where(live_r, expo, 0.0), RP, LOGR),
                           (1,))
        probs = tl.where(live_any,
                         _div_rn32(expo, tl.broadcast_to(denom, (RP,))), 0.0)
        w = _mul_rn(probs, strength)
        read_o = tl.sum(_mul_rn(w[:, None], acc), 0)
        tl.store(reads_base + t * H * V + values, read_o, mask=keep_v)
        if COLLECT:
            tl.store(weights_base + t * H * R + ranks, w, mask=keep_r)

        # ---- the decisions: the registered selection chain on the KEY
        # channel (K10 verbatim -- P2: no gate value feeds this chain).
        nwk = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(wk, wk), KP, LOGK)), (1,))
        uk = _div_rn32(wk, tl.broadcast_to(tl.maximum(nwk, _EPS), (KP,)))
        sim = _div_rn32(
            tl.reshape(_split_fold_axis2(_mul_rn(uk[None, :], kacc), KP, LOGK),
                       (RP,)),
            tl.maximum(nk, _EPS))
        sim = tl.where(live_r, sim, -2.0)
        penalty = _mul_rn(ranks.to(tl.float32), TIE)
        score = tl.where(keep_r, sim - penalty, -3.0)
        best_top = tl.max(score, 0)
        best = tl.min(tl.where(score == best_top, ranks, _BIG_INDEX), 0)
        best_value = tl.sum(tl.where(ranks == best, sim, 0.0), 0)
        second_score = tl.where(ranks == best, -3.0, score)
        second_top = tl.max(second_score, 0)
        second = tl.min(tl.where(second_score == second_top, ranks,
                                 _BIG_INDEX), 0)
        second_value = tl.sum(tl.where(ranks == second, sim, 0.0), 0)
        worst_score = tl.where(keep_r, score, float('inf'))
        worst_top = tl.min(worst_score, 0)
        worst = tl.min(tl.where(worst_score == worst_top, ranks, _BIG_INDEX),
                       0)
        worst_value = tl.sum(tl.where(ranks == worst, sim, 0.0), 0)

        correct = best_value >= TAU_MATCH
        merge = correct & (second_value >= TAU_MERGE)
        free = keep_r & (~live)
        free_count = tl.sum(free.to(tl.int32), 0)
        need = ~correct
        appendable = need & (free_count > 0)
        free_found = tl.min(tl.where(free, ranks, _BIG_INDEX), 0)
        free_slot = tl.where(free_found < _BIG_INDEX, free_found, 0)
        full = free_count <= 0
        pop = (~correct) & (~appendable) & full & (worst_value <= TAU_POP) \
            & live_any
        op0 = tl.where(correct, tl.where(merge, _MERGE_CODE, _CORRECT_CODE),
                       tl.where(appendable, _APPEND_CODE,
                                tl.where(pop, _POP_CODE, _NONE_CODE)))
        slot0 = tl.where(correct, best,
                         tl.where(appendable, free_slot, worst))
        erase0 = (op0 == _CORRECT_CODE) | (op0 == _MERGE_CODE) \
            | (op0 == _POP_CODE)
        live0 = op0 != _POP_CODE
        # ---- the MERGE blend from the PRE-decay rows (K10 verbatim), and
        # the gated lane 0.  The CORRECT blend is the PRE-decay per-channel
        # form  g*p + (1-g)*(lam*v_pre)  -- the same value as
        # v_dec + g*(p - v_dec) in exact arithmetic with v_dec = lam*v_pre,
        # so the register walk keeps K10's registered lane order (the blend
        # chain before DECAY, the lanes after) AND the unity-gate birth is
        # the incumbent's arithmetic bit for bit (g = 1 per channel gives
        # add_rn(p, +-0), the -0.0 corner K11 receipted; the lam_vec birth
        # equals the registered scalar per channel, so the DECAY is the
        # incumbent's per element).
        fw = tl.maximum(tl.sum(tl.where(ranks == best, strength, 0.0), 0),
                        _EPS)
        sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0), 0),
                        _EPS)
        total = fw + sw
        a_mix = _div_rn32(fw, total)
        b_mix = _div_rn32(sw, total)
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        hit = keep_r & (slot0 == ranks)
        if merge:
            coef = tl.where(ranks == best, a_mix,
                            tl.where(ranks == second, b_mix, 0.0))
            blend = tl.sum(_mul_rn(coef[:, None], acc), 0)
            lane0 = blend
        else:
            if GATED:
                gt = tl.load(gates_row + t * gts + values * gvs, mask=keep_v,
                             other=0.0).to(tl.float32)
                if op0 == _CORRECT_CODE:
                    # the delta-rule blend in the pre-decay form, per channel
                    old_pre = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)
                    lane0 = _add_rn(
                        _mul_rn(gt, p),
                        _mul_rn(_sub_rn(1.0, gt), _mul_rn(lam, old_pre)))
                else:
                    # the admission form (gate*value) on the carrying ops --
                    # at a free row the old content is exact zero, so this
                    # is g*p on a fresh row; POP/NONE carry no content
                    lane0 = tl.where(carries, _mul_rn(gt, p), 0.0)
            else:
                lane0 = tl.where(carries, p, 0.0)

        # ---- apply: DECAY (values only, PER CHANNEL), then the value lanes
        # in order (K10 verbatim), then the key lane.
        acc = _mul_rn(acc, tl.broadcast_to(lam[None, :], (RP, VP)))
        old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)

        # ---- record the decision channel (the 2-lane (b, T, h, L) layout).
        tl.store(slot_base + t * H * L + 0, slot0.to(tl.int64))
        tl.store(slot_base + t * H * L + 1, second.to(tl.int64))
        tl.store(op_base + t * H * L + 0, op0.to(tl.int64))
        tl.store(op_base + t * H * L + 1,
                 tl.where(merge, _MERGE_CODE, _NONE_CODE).to(tl.int64))
        tl.store(erase_base + t * H * L + 0, erase0.to(tl.int8))
        tl.store(erase_base + t * H * L + 1, merge.to(tl.int8))
        tl.store(live_after_base + t * H * L + 0, live0.to(tl.int8))
        tl.store(live_after_base + t * H * L + 1,
                 tl.zeros((), dtype=tl.int8))
        tl.store(payload_base + t * H * L * V + 0 * V + values, lane0,
                 mask=keep_v)
        tl.store(payload_base + t * H * L * V + 1 * V + values,
                 tl.zeros((VP,), dtype=tl.float32), mask=keep_v)

        # ---- apply lane 0, then the MERGE-source close (K10 verbatim).
        if erase0:
            new_row = lane0
        else:
            new_row = _add_rn(old_row, lane0)
        acc = tl.where(hit[:, None], new_row[None, :], acc)
        live = tl.where(hit, live0, live)
        hit2 = merge & keep_r & (second == ranks)
        acc = tl.where(hit2[:, None], 0.0, acc)
        live = tl.where(hit2, False, live)
        # ---- the key lane: K10 verbatim (gates never touch it).
        carries_key = op0 != _NONE_CODE
        pop_gate = tl.where(op0 == _POP_CODE, 0.0, 1.0)
        key_lane = _mul_rn(wk, pop_gate)
        hitk = carries_key & keep_r & (slot0 == ranks)
        kacc = tl.where(hitk[:, None], key_lane[None, :], kacc)
        hit2k = merge & keep_r & (second == ranks)
        kacc = tl.where(hit2k[:, None], 0.0, kacc)

    tl.store(state_out + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], acc, mask=keep)
    tl.store(live_out + row * R + ranks, live.to(tl.int8), mask=keep_r)
    tl.store(keys_out + row.to(tl.int64) * R * KD + ranks[:, None] * KD
             + kdims[None, :], kacc, mask=keep_rk)


# ------------------------------------------------------------ the forward wrapper


def _check_channel_shapes(payload, write_keys, query_keys, decay, gates,
                          state, live, keys, rank, segment_tokens, gated):
    """Refuse anything the per-channel path cannot express, before any
    launch."""
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda):
        raise ValueError("the channel-gated scan is a device kernel; inputs "
                          "must be on CUDA")
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape != (batch, tokens, heads, key_dim):
        raise ValueError(f"keys must be (batch, tokens, heads, key_dim); got "
                         f"{tuple(write_keys.shape)}")
    if decay.shape != payload.shape:
        raise ValueError(f"the per-channel decay must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(decay.shape)}")
    if gated and gates.shape != payload.shape:
        raise ValueError(f"the per-channel gates must share the payload's "
                         f"shape {tuple(payload.shape)}; got "
                         f"{tuple(gates.shape)}")
    if state.shape != (batch, heads, rank, value_dim):
        raise ValueError(f"state {tuple(state.shape)} is not (batch, heads, "
                         f"rank, value_dim) = {(batch, heads, rank, value_dim)}")
    if keys.shape != (batch, heads, rank, key_dim):
        raise ValueError(f"keys {tuple(keys.shape)} is not (batch, heads, "
                          f"rank, key_dim) = {(batch, heads, rank, key_dim)}")
    if live.shape != (batch, heads, rank) or live.dtype != torch.bool:
        raise ValueError("live must be a bool (batch, heads, rank) tensor")
    for name, tensor in (("state", state), ("keys", keys)):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be fp32; the algebra is fp32")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    if tokens < 1 or segment_tokens < 1:
        raise ValueError("tokens and segment_tokens must be positive")


def _k11b_forward_launch(payload, write_keys, query_keys, decay, gates,
                         state, live, keys, policy_f32, rank, segment_tokens,
                         gated, collect, warps, stages):
    """The persistent walk: one launch, every output allocated contiguous."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    lanes = LANES
    reads = torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device)
    out_state = torch.empty_like(state)
    out_live = torch.empty_like(live)
    out_keys = torch.empty_like(keys)
    shape = (batch, tokens, heads, lanes)
    slot = torch.empty(shape, dtype=torch.int64, device=device)
    op = torch.empty(shape, dtype=torch.int64, device=device)
    erase = torch.empty(shape, dtype=torch.bool, device=device)
    live_after = torch.empty(shape, dtype=torch.bool, device=device)
    payload_field = torch.empty((*shape, value_dim), dtype=torch.float32,
                                device=device)
    live_pre = torch.empty((batch, tokens, heads, rank), dtype=torch.bool,
                           device=device)
    nseg = -(-tokens // int(segment_tokens))
    ckpt_state = torch.empty((nseg, batch, heads, rank, value_dim),
                             dtype=torch.float32, device=device)
    ckpt_live = torch.empty((nseg, batch, heads, rank), dtype=torch.bool,
                            device=device)
    ckpt_keys = torch.empty((nseg, batch, heads, rank, key_dim),
                            dtype=torch.float32, device=device)
    weights = (torch.empty((batch, tokens, heads, rank), dtype=torch.float32,
                           device=device) if collect
              else torch.empty((1,), dtype=torch.float32, device=device))
    tau_match, tau_merge, tau_pop, tie, scale = policy_f32
    handle = _k11b_seq_fwd_kernel[(batch * heads,)](
        payload, write_keys, query_keys, decay, gates, state,
        live.view(torch.int8), keys,
        reads, out_state, out_live.view(torch.int8), out_keys,
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), ckpt_keys, weights,
        tokens, rank, batch * heads, int(segment_tokens), key_dim,
        *payload.stride(), *write_keys.stride(), *query_keys.stride(),
        *decay.stride(), *gates.stride(),
        H=heads, V=value_dim, VP=_pow2(value_dim), KP=_pow2(key_dim),
        RP=_pow2(rank), L=lanes,
        LOGV=_log2_up(_pow2(value_dim)), LOGR=_log2_up(_pow2(rank)),
        LOGK=_log2_up(_pow2(key_dim)),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0,
        GATED=1 if gated else 0,
        STAGES=stages, num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, keys=out_keys,
                slot=slot, op=op, erase=erase, live_after=live_after,
                payload=payload_field, live_pre=live_pre,
                ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                ckpt_keys=ckpt_keys, weights=weights, kernel=handle)


# ------------------------------------------- the per-channel trajectory replay


@triton.jit(do_not_specialize=['T', 'R'])
def _k11b_traj_kernel(
    state_in, live_in, decay, slot, op, erase, live_after, payload,
    reads, state_out, live_out, trajectory,
    T, R,
    dbs, dhs, dt, dv,
    sbs, st, sh, sl,
    pbs, pst, psh, psl,
    H: tl.constexpr, L: tl.constexpr, V: tl.constexpr,
    RP: tl.constexpr, BV: tl.constexpr, STAGES: tl.constexpr,
):
    """K4's persistent block scan + trajectory, with the PER-CHANNEL decay
    load -- the engine limitation the supervisor ruled on: K4's imported
    kernel takes per-head SCALAR decay only, so this lane-specific kernel
    carries the per-channel form at K4's own defined-order spec (the walk
    verbatim: read, DECAY, every lane in order, ``mul.rn``/``add.rn`` so no
    FMA contraction can move it; one extra ``tl.store`` per token writes the
    pre-decay state the backward's DECAY and read adjoints need)."""
    row = tl.program_id(0)
    block = tl.program_id(1)
    batch_index = row // H
    head_index = row % H

    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = block * BV + tl.arange(0, BV)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    state_base = row * R * V
    acc = tl.load(state_in + state_base + ranks[:, None] * V + values[None, :],
                  mask=keep, other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0

    decay_row = decay + batch_index * dbs + head_index * dhs
    payload_row = payload + batch_index * pbs + head_index * psh
    traj_row = trajectory + row.to(tl.int64) * T * R * V

    for t in tl.range(0, T, num_stages=STAGES):
        tl.store(traj_row + t * R * V + ranks[:, None] * V
                 + values[None, :], acc, mask=keep)
        lam = tl.load(decay_row + t * dt + values * dv, mask=keep_v,
                      other=0.0)
        acc = _mul_rn(acc, tl.broadcast_to(lam[None, :], (RP, BV)))
        for lane in tl.static_range(L):
            channel = batch_index * sbs + t * st + head_index * sh + lane * sl
            if tl.load(op + channel) != NONE_CODE:
                target = tl.load(slot + channel).to(tl.int32)
                hit = keep_r & (target == ranks)
                written = tl.load(payload_row + t * pst + lane * psl + values,
                                  mask=keep_v, other=0.0)
                tile = tl.broadcast_to(written[None, :], (RP, BV))
                if tl.load(erase + channel) != 0:
                    acc = tl.where(hit[:, None], tile, acc)
                else:
                    acc = tl.where(hit[:, None], _add_rn(acc, tile), acc)
                live = tl.where(hit, tl.load(live_after + channel) != 0, live)

    tl.store(state_out + state_base + ranks[:, None] * V + values[None, :], acc,
             mask=keep)
    if block == 0:
        tl.store(live_out + row * R + ranks, live.to(tl.int8), mask=keep_r)


def channel_persistent_trajectory(state: torch.Tensor, live: torch.Tensor,
                                   decay: torch.Tensor,
                                   schedule: MatrixSchedule,
                                   block: int | None = None,
                                   warps: int | None = None, stages: int = 3):
    """The per-channel-decay segment replay: K4's
    ``persistent_scan_trajectory`` with the DECAY channel a per-token
    ``d_v``-dim vector ``(batch, heads, tokens, value_dim)``.  Returns
    ``(reads, state, live, trajectory)`` with ``trajectory``
    ``(batch, heads, tokens, rank, value_dim)``: the state as each token's
    read saw it (pre-DECAY) -- the backward's input, at K4's own memory cost.
    """
    from .hostloop_free import (check_block_shapes, value_block, warp_count,
                                MIN_VALUE_BLOCK)
    if not (state.is_cuda and live.is_cuda and decay.is_cuda):
        raise ValueError("the replay is a device kernel; inputs must be CUDA")
    batch, heads, rank, value_dim = state.shape
    tokens = schedule.tokens
    if decay.shape != (batch, heads, tokens, value_dim):
        raise ValueError(f"decay {tuple(decay.shape)} is not (batch, heads, "
                         f"tokens, value_dim) = "
                         f"{(batch, heads, tokens, value_dim)}")
    if state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError("state must be contiguous fp32")
    block = value_block(rank, value_dim) if block is None else int(block)
    if block < MIN_VALUE_BLOCK or block & (block - 1):
        raise ValueError(f"the value block must be a power of two >= "
                         f"{MIN_VALUE_BLOCK}")
    device = state.device
    reads = torch.empty((batch, heads, tokens, value_dim), dtype=torch.float32,
                        device=device)
    out_state = torch.empty_like(state)
    out_live = torch.empty_like(live)
    trajectory = torch.empty((batch, heads, tokens, rank, value_dim),
                             dtype=torch.float32, device=device)
    grid = (batch * heads, triton.cdiv(value_dim, block))
    tile = triton.next_power_of_2(rank) * block
    warps = warp_count(tile) if warps is None else int(warps)
    _k11b_traj_kernel[grid](
        state, live.view(torch.int8), decay,
        schedule.slot, schedule.op, schedule.erase.view(torch.int8),
        schedule.live_after.view(torch.int8), schedule.payload,
        reads, out_state, out_live.view(torch.int8), trajectory,
        tokens, rank,
        *decay.stride(),
        *schedule.slot.stride(), *schedule.payload.stride()[:4],
        H=heads, L=schedule.lanes, V=value_dim,
        RP=triton.next_power_of_2(rank), BV=block, STAGES=stages,
        num_warps=warps)
    return reads, out_state, out_live, trajectory


# --------------------------------------------------------- the backward kernels


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k11b_bwd_value_walk_kernel(
    traj, payload, gates, dout, decay32, live_pre, carry_in,
    nr_ch, w_ch, dstrength_ch,
    slot, op, erase,
    dpayload, dlam_ch, dgate_ch, dstate_seg,
    SEGT, T, T0, R,
    pbs, pts, phs, pvs,
    gbs, gts, ghs, gvs,
    xbs, xts, xhs, xvs,
    sbs, sts, shs, sls,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    LOGV: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    GATED: tl.constexpr,
):
    """K11's value-blocked reverse walk with the PER-CHANNEL adjoints.
    Lanes last first: the MERGE-source zeroing, then lane 0 -- under the
    per-channel gate ``dpayload = g_vec*dpay0``,
    ``dgate[c] = dpay0[c]*(p[c] - v_dec[c])`` per channel, and the CORRECT
    blend's old-row path ``dS_dec := (1-g_vec)*dS`` per channel -- then the
    PER-CHANNEL DECAY adjoint (``dlam[c] = <ds[:,c], S_t[:,c]>``, then
    ``ds <- lam_vec (.) ds``), the MERGE-blend scatter, and the read adjoint
    (K10 verbatim).  ``dlam_ch``/``dgate_ch`` are per-(row, token, channel)
    buffers; each value-block program writes its own disjoint channel slice.
    """
    row = tl.program_id(0)
    vblock = tl.program_id(1)
    batch_index = row // H
    head_index = row % H
    ranks = tl.arange(0, RP)
    keep_r = ranks < R
    values = vblock * BV + tl.arange(0, BV)
    keep_v = values < V
    keep = keep_r[:, None] & keep_v[None, :]

    ds = tl.load(carry_in + row.to(tl.int64) * R * V + ranks[:, None] * V
                 + values[None, :], mask=keep, other=0.0)
    traj_base = traj + row.to(tl.int64) * SEGT * R * V
    payload_row = payload + batch_index * pbs + head_index * phs
    gates_row = gates + batch_index * gbs + head_index * ghs
    dout_row = dout + batch_index.to(tl.int64) * xbs + head_index * xhs
    # decay32 is the contiguous (rows, T, V) view of the per-channel decay
    decay_row = decay32 + row.to(tl.int64) * T * V
    live_base = (live_pre + batch_index.to(tl.int64) * T * H * R
                 + head_index * R)
    nr_row = nr_ch + row.to(tl.int64) * SEGT * R
    w_row = w_ch + row.to(tl.int64) * SEGT * R
    dst_row = dstrength_ch + row.to(tl.int64) * SEGT * R
    dp_row = dpayload + batch_index.to(tl.int64) * T * H * V + head_index * V
    ddl_row = dlam_ch + row.to(tl.int64) * T * V
    dgate_row = dgate_ch + row.to(tl.int64) * T * V
    channel_base = batch_index.to(tl.int64) * sbs + head_index * shs

    for i in tl.range(0, SEGT):
        tt = SEGT - 1 - i
        t = T0 + tt
        op0 = tl.load(op + channel_base + t * sts + 0 * sls)
        slot0 = tl.load(slot + channel_base + t * sts + 0 * sls).to(tl.int32)
        second = tl.load(slot + channel_base + t * sts + 1 * sls).to(tl.int32)
        erase0 = tl.load(erase + channel_base + t * sts + 0 * sls) != 0
        merge = tl.load(op + channel_base + t * sts + 1 * sls) != _NONE_CODE
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        lam = tl.load(decay_row + t * V + values, mask=keep_v, other=0.0)
        live = tl.load(live_base + t * H * R + ranks, mask=keep_r,
                       other=0) != 0
        acc = tl.load(traj_base + tt * R * V + ranks[:, None] * V
                      + values[None, :], mask=keep, other=0.0)
        nr = tl.load(nr_row + tt * R + ranks, mask=keep_r, other=0.0)
        w = tl.load(w_row + tt * R + ranks, mask=keep_r, other=0.0)
        dstrength = tl.load(dst_row + tt * R + ranks, mask=keep_r, other=0.0)
        g = tl.load(dout_row + t * xts + values * xvs, mask=keep_v,
                    other=0.0)

        # 1. the lanes, last lane first (the exact reverse of the forward).
        # 1a. lane 1 (the MERGE source close): the source row's path
        #     terminates.
        hit2 = merge & keep_r & (second == ranks)
        if merge:
            ds = tl.where(hit2[:, None], 0.0, ds)
        # 1b. lane 0.
        hit = keep_r & (slot0 == ranks)
        dpay0 = tl.sum(tl.where(hit[:, None], ds, 0.0), 0)
        if GATED:
            gt = tl.load(gates_row + t * gts + values * gvs, mask=keep_v,
                         other=0.0).to(tl.float32)
            if carries:
                # the admitted payload's adjoint: dpayload = g_vec * dpay0
                dpay = _mul_rn(gt, dpay0)
                # the gate's adjoint: dgate[c] = dpay0[c]*(p[c] - v_dec[c])
                # with v_dec the POST-decay row the forward blended against
                # (traj is the PRE-decay state; the forward's acc was
                # mul_rn(acc, lam_vec))
                p_v = tl.load(payload_row + t * pts + values * pvs,
                              mask=keep_v, other=0.0).to(tl.float32)
                old_pre = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)
                v_dec = _mul_rn(old_pre, lam)
                dgate_v = _mul_rn(dpay0, _sub_rn(p_v, v_dec))
            else:
                dpay = tl.zeros((BV,), dtype=tl.float32)
                dgate_v = tl.zeros((BV,), dtype=tl.float32)
            if erase0:
                if op0 == _CORRECT_CODE:
                    # the blend's old-row path: dS_dec := (1-g_vec) * dS
                    ds = tl.where(hit[:, None],
                                  _mul_rn(_sub_rn(1.0, gt), ds), ds)
                else:
                    # POP/MERGE: the incoming content is discarded
                    ds = tl.where(hit[:, None], 0.0, ds)
            tl.store(dgate_row + t * V + values, dgate_v, mask=keep_v)
        else:
            dpay = tl.where(carries, dpay0, 0.0)
            if erase0:
                ds = tl.where(hit[:, None], 0.0, ds)
        tl.store(dp_row + t * (H * V) + values,
                 dpay.to(dpayload.dtype.element_ty), mask=keep_v)
        # 2. the PER-CHANNEL DECAY adjoint: dlam[c] = <ds[:,c], S_t[:,c]>,
        #    then ds <- lam_vec (.) ds.
        dlam_v = tl.sum(_mul_rn(ds, acc), 0)
        tl.store(ddl_row + t * V + values, dlam_v, mask=keep_v)
        ds = _mul_rn(ds, tl.broadcast_to(lam[None, :], (RP, BV)))
        # 3. the MERGE-blend scatter against the pre-decay rows (K10 verbatim).
        if merge:
            strength = _mul_rn(nr, tl.where(live, 1.0, 0.0))
            fw = tl.maximum(tl.sum(tl.where(ranks == slot0, strength, 0.0),
                                    0), _EPS)
            sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0),
                                   0), _EPS)
            total = fw + sw
            a_mix = _div_rn32(fw, total)
            b_mix = _div_rn32(sw, total)
            ds = tl.where(hit[:, None],
                          _add_rn(ds, _mul_rn(a_mix, dpay0)[None, :]), ds)
            ds = tl.where(hit2[:, None],
                          _add_rn(ds, _mul_rn(b_mix, dpay0)[None, :]), ds)
        # 4. the read adjoint's value terms: the direct term and the strength
        #    path only (the score path lives on the key channel -- K10).
        ds = _add_rn(ds, _mul_rn(w[:, None], g[None, :]))
        dnr = _mul_rn(dstrength, tl.where(live, 1.0, 0.0))
        unit_r = tl.where(nr[:, None] > 0, _div_rn32(acc, nr[:, None]), 0.0)
        ds = _add_rn(ds, _mul_rn(dnr[:, None], unit_r))

    tl.store(dstate_seg + row.to(tl.int64) * R * V + ranks[:, None] * V
             + values[None, :], ds, mask=keep)


# ------------------------------------------------- the backward body


def _k11b_backward_body(payload, write_keys, query_keys, decay, gates,
                        dreads, schedule, live_pre, ckpt_state, ckpt_live,
                        ckpt_keys, rank, segment_tokens, scale, gated):
    """The whole window's adjoint: segments in reverse, each recomputed from
    its entry checkpoints (the gate-C doctrine) -- the PER-CHANNEL value
    replay (the lane-specific kernel), the key replay (K11's, imported), K10's
    token-parallel prep (imported verbatim), then the two reverse walks
    (the per-channel value walk; K11's key walk, imported)."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    R, SEG = int(rank), int(segment_tokens)
    rows = batch * heads
    # the contiguous (rows, T, V) per-channel decay view the walk indexes
    decay32 = (decay.to(torch.float32).permute(0, 2, 1, 3).contiguous()
               .view(rows, tokens, value_dim))
    dreads_c = dreads.contiguous()
    dpayload = torch.empty_like(payload)
    dwrite_keys = torch.empty_like(write_keys)
    dquery_keys = torch.empty_like(query_keys)
    ddecay_ch = torch.zeros((rows, tokens, value_dim), dtype=torch.float32,
                            device=device)
    dgates_ch = (torch.zeros((rows, tokens, value_dim), dtype=torch.float32,
                             device=device) if gated
                 else torch.zeros((rows, tokens, value_dim),
                                  dtype=torch.float32, device=device))
    RP, VP, KP = _pow2(R), _pow2(value_dim), _pow2(key_dim)
    LOGV, LOGR, LOGK = _log2_up(VP), _log2_up(RP), _log2_up(KP)
    nseg = -(-tokens // SEG)
    carry = torch.zeros(batch, heads, R, value_dim, dtype=torch.float32,
                        device=device)
    carry_keys = torch.zeros(batch, heads, R, key_dim, dtype=torch.float32,
                             device=device)
    nblocks = triton.cdiv(value_dim, BV_BWD)
    for seg in reversed(range(nseg)):
        start = seg * SEG
        stop = min(start + SEG, tokens)
        segt = stop - start
        # 1. the segment's value trajectory (the per-channel replay).
        _, _, _, traj = channel_persistent_trajectory(
            ckpt_state[seg], ckpt_live[seg],
            decay32.view(batch, heads, tokens, value_dim)[:, :, start:stop],
            schedule.chunk(start, stop))
        # 2. the segment's key trajectory (the key-lane replay, imported).
        ktraj = torch.empty((rows, segt, R, key_dim), dtype=torch.float32,
                            device=device)
        _k11_key_traj_kernel[(rows,)](
            ckpt_keys[seg], write_keys, schedule.slot, schedule.op, ktraj,
            segt, tokens, start, R, key_dim,
            *schedule.slot.stride(),
            *write_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        # 3. prep: the per-token weights and adjoint coefficients (K10's
        # kernel, imported verbatim -- it recomputes the read weights from the
        # trajectory and the key channel, gate-agnostic).
        nr_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        nk_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        w_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dscores_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                 device=device)
        dstrength_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                   device=device)
        z_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dc_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dnk_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                             device=device)
        scal = torch.empty((rows, segt, 2), dtype=torch.float32, device=device)
        grid = (rows, triton.cdiv(segt, BT_BWD))
        _k10_bwd_prep_kernel[grid](
            traj, ktraj, query_keys, dreads_c, live_pre.view(torch.int8),
            nr_ch, nk_ch, w_ch, dscores_ch, dstrength_ch, z_ch, dc_ch,
            dnk_ch, scal,
            segt, tokens, start, R, key_dim,
            *query_keys.stride(), *dreads_c.stride(),
            H=heads, V=value_dim, VP=VP, KP=KP, RP=RP, LOGV=LOGV, LOGR=LOGR,
            LOGK=LOGK, BT=BT_BWD, SCALE=_round_f32(scale), num_warps=8)
        # 4. the value walk (value-blocked reverse adjoint, per-channel).
        dstate_seg = torch.empty((batch, heads, R, value_dim),
                                 dtype=torch.float32, device=device)
        _k11b_bwd_value_walk_kernel[(rows, nblocks)](
            traj, payload, gates, dreads_c, decay32, live_pre.view(torch.int8),
            carry,
            nr_ch, w_ch, dstrength_ch,
            schedule.slot, schedule.op, schedule.erase.view(torch.int8),
            dpayload, ddecay_ch, dgates_ch, dstate_seg,
            segt, tokens, start, R,
            *payload.stride(), *gates.stride(), *dreads_c.stride(),
            *schedule.slot.stride(),
            H=heads, V=value_dim, VP=VP, RP=RP, LOGV=LOGV, BT=BT_BWD,
            BV=BV_BWD, GATED=1 if gated else 0,
            num_warps=4)
        # 5. the key walk (the key-table adjoint tile; K11's, imported).
        dkeys_seg = torch.empty((batch, heads, R, key_dim),
                                dtype=torch.float32, device=device)
        _k11_bwd_key_walk_kernel[(rows,)](
            ktraj, write_keys, query_keys, carry_keys,
            dc_ch, dnk_ch, nk_ch, scal,
            schedule.slot, schedule.op,
            dwrite_keys, dquery_keys, dkeys_seg,
            segt, tokens, start, R, key_dim,
            *schedule.slot.stride(), *write_keys.stride(),
            *query_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        carry = dstate_seg
        carry_keys = dkeys_seg
    dstate = carry
    dkeys = carry_keys
    ddecay = ddecay_ch.view(batch, heads, tokens, value_dim).permute(0, 2, 1, 3)
    dgates = dgates_ch.view(batch, heads, tokens, value_dim).permute(0, 2, 1, 3)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dgates, dstate,
            dkeys)


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::k11b_scan_fwd", mutates_args=())
def _k11b_scan_fwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor,
    state: torch.Tensor, live: torch.Tensor, keys: torch.Tensor, rank: int,
    segment_tokens: int, tau_match: float, tau_merge: float, tau_pop: float,
    tie_break: float, scale: float, gated: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """The whole (layer, window) per-channel-gated scan as ONE opaque op: the
    persistent walk's every channel as its outputs, so autograd itself keeps
    the backward's saves alive (the K4/K6/K10/K11 hosting pattern)."""
    _check_channel_shapes(payload, write_keys, query_keys, decay, gates,
                          state, live, keys, int(rank), int(segment_tokens),
                          bool(gated))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k11b_forward_launch(payload, write_keys, query_keys, decay, gates,
                               state, live, keys, pol32, int(rank),
                               int(segment_tokens), bool(gated),
                               collect=False,
                               warps=_seq_warps(int(rank), payload.shape[-1]),
                               stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"])


@_k11b_scan_fwd_op.register_fake
def _(payload, write_keys, query_keys, decay, gates, state, live, keys,
      rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale,
      gated):
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    nseg = -(-tokens // int(segment_tokens))
    shape = (batch, tokens, heads, LANES)
    return (torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty_like(state),
            torch.empty_like(live),
            torch.empty_like(keys),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.int64, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty(shape, dtype=torch.bool, device=device),
            torch.empty((*shape, value_dim), dtype=torch.float32, device=device),
            torch.empty((batch, tokens, heads, rank), dtype=torch.bool,
                        device=device),
            torch.empty((nseg, batch, heads, rank, value_dim),
                        dtype=torch.float32, device=device),
            torch.empty((nseg, batch, heads, rank), dtype=torch.bool,
                        device=device),
            torch.empty((nseg, batch, heads, rank, key_dim),
                        dtype=torch.float32, device=device))


def _k11b_scan_setup(ctx, inputs, output):
    (payload, write_keys, query_keys, decay, gates, state, live, keys,
     rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale,
     gated) = inputs
    ctx.save_for_backward(payload, write_keys, query_keys, decay, gates,
                          state, live, keys, *output[4:])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)
    ctx.gated = bool(gated)


@torch.library.custom_op("x45::k11b_scan_bwd", mutates_args=())
def _k11b_scan_bwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor,
    dreads: torch.Tensor, slot: torch.Tensor, op: torch.Tensor,
    erase: torch.Tensor, live_after: torch.Tensor, payload_field: torch.Tensor,
    live_pre: torch.Tensor, ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    ckpt_keys: torch.Tensor, rank: int, segment_tokens: int, scale: float,
    gated: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: the segment loop
    (per-channel value replay -> key replay -> prep -> both walks) runs
    inside, decisions as recorded constants (the standing doctrine)."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k11b_backward_body(payload, write_keys, query_keys, decay, gates,
                               dreads, schedule, live_pre, ckpt_state,
                               ckpt_live, ckpt_keys, int(rank),
                               int(segment_tokens), float(scale), bool(gated))


@_k11b_scan_bwd_op.register_fake
def _(payload, write_keys, query_keys, decay, gates, dreads, slot, op, erase,
      live_after, payload_field, live_pre, ckpt_state, ckpt_live, ckpt_keys,
      rank, segment_tokens, scale, gated):
    device = payload.device
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    return (torch.empty_like(payload),
            torch.empty_like(write_keys),
            torch.empty_like(query_keys),
            torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, tokens, heads, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, key_dim), dtype=torch.float32,
                        device=device))


def _k11b_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys):
    """The adjoint: dpayload, dwrite_keys, dquery_keys, ddecay (per channel),
    dgates (per channel), dstate, dkeys; the decision channel and the
    liveness carry no gradient by construction."""
    (payload, write_keys, query_keys, decay, gates, state, live, keys,
     slot, op, erase, live_after, payload_field, live_pre, ckpt_state,
     ckpt_live, ckpt_keys) = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    (dpayload, dwrite_keys, dquery_keys, ddecay, dgates, dstate,
     dkeys) = _k11b_scan_bwd_op(payload, write_keys, query_keys, decay, gates,
                                dreads, slot, op, erase, live_after,
                                payload_field, live_pre, ckpt_state,
                                ckpt_live, ckpt_keys, ctx.rank,
                                ctx.segment_tokens, ctx.scale, ctx.gated)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, dgates,
            dstate, None, dkeys, None, None, None, None, None, None, None,
            None)


torch.library.register_autograd("x45::k11b_scan_fwd", _k11b_scan_backward,
                                setup_context=_k11b_scan_setup)


# --------------------------------------------------------------- the wrapper


def channel_gated_scan(payload, write_keys, query_keys, decay, gates, *,
                       policy: RoutingPolicy, gated: bool = True,
                       rank=None, rank_max=None,
                       segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                       initial_state=None, initial_live=None,
                       initial_keys=None, collect: bool = False) -> RoutedResult:
    """One whole (layer, window) per-channel-gated scan at PER-TOKEN
    granularity.

    The K10 wrapper's drop-in for the channel-gate arms: the registered
    five-op policy evaluated against the CURRENT state and key table at
    every token, addressing on the key channel, with the per-channel DECAY
    vector and the per-channel write gate as schedule data.  The kernel
    path is ONE opaque op pair per scan; every refusal takes the EAGER
    ORACLE -- the same semantics, the parity fixed point -- never a silent
    change:

    * ``policy.ops=False`` (the append-only control) and ``policy.grow`` are
      refused outright (chunk-dial concepts); a rank prefix is refused by
      name;
    * a store the persistent walk cannot hold (``separated_available``), or
      non-CUDA inputs, take the oracle verbatim.
    """
    batch, tokens, heads, value_dim = payload.shape
    device = payload.device
    key_dim = write_keys.shape[-1]
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    rank, rank_max = int(rank), int(rank_max)
    if not policy.ops:
        raise ValueError("the append-only control is refused by the "
                         "channel-gate registration; it is a chunk-dial "
                         "control")
    if policy.grow:
        raise ValueError("growth is refused by the channel-gate "
                         "registration: growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the channel-gate registration requires rank == "
                         "rank_max (the registered shape); a rank prefix is "
                         "refused by name")
    state = (torch.zeros(batch, heads, rank_max, value_dim,
                         dtype=torch.float32, device=device)
             if initial_state is None else initial_state)
    live = (torch.zeros(batch, heads, rank_max, dtype=torch.bool, device=device)
            if initial_live is None else initial_live)
    keys = (torch.zeros(batch, heads, rank_max, key_dim, dtype=torch.float32,
                        device=device) if initial_keys is None
            else initial_keys)
    if collect:
        pol32 = (_round_f32(policy.tau_match), _round_f32(policy.tau_merge),
                 _round_f32(policy.tau_pop), _round_f32(policy.tie_break),
                 _round_f32(policy.read_scale))
        with torch.no_grad():
            out = _k11b_forward_launch(
                payload, write_keys, query_keys, decay, gates, state, live,
                keys, pol32, rank, segment_tokens, gated, collect=True,
                warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                            ScanStats(), schedule, out["weights"])
    if (payload.is_cuda and separated_available(rank, value_dim, key_dim)):
        reads, state_out, live_out, _keys, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live, _ckpt_keys = \
            _k11b_scan_fwd_op(payload, write_keys, query_keys, decay, gates,
                              state, live, keys, rank,
                              int(segment_tokens), policy.tau_match,
                              policy.tau_merge, policy.tau_pop,
                              policy.tie_break, policy.read_scale,
                              bool(gated))
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same semantics, the parity fixed point
    ref = channel_gated_oracle_scan(payload, write_keys, query_keys, decay,
                                    gates, policy=policy, gated=gated,
                                    rank=rank, rank_max=rank_max,
                                    initial_state=state, initial_live=live,
                                    initial_keys=keys, collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class ChannelGatedSeparatedMixer(nn.Module):
    """The K11B mixer: the K10 separated mixer's projections and DECAY gate
    with the per-channel gate channels added, all born mimetic-neutral (P5:
    every channel carries a per-head zero-born scalar over zero-born
    projections, so at birth the mixer IS the banked winner exactly -- the
    per-channel decay vector equals the registered scalar on every channel,
    the write gate is exactly 1 per channel, the read gate exactly 1).  The
    arm selects which channels exist:

    * ``decay``  the per-channel input-dependent DECAY (the f_proj analog):
                 ``lam_vec = clamp(lam_inc * (1 + m_f*sigmoid(f_proj(x))),
                 DECAY_FLOOR, 1-1e-6)`` per (token, head, channel).
    * ``write``  the per-channel write-admission gate (the b_proj analog):
                 ``g_vec = 1 - m_w*sigmoid(b_proj(x))`` -- APPEND admits
                 ``g_vec (.) p``, CORRECT blends
                 ``g_vec (.) p + (1-g_vec) (.) (lam_vec (.) v_pre)``.
    * ``read``   the per-channel read gate (the g_proj analog):
                 ``r_vec = 1 + m_o*sigmoid(g_proj(x))``, applied
                 ``reads (.) r_vec`` before the out projection.
    """

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 arm: str = "p1", ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
        super().__init__()
        self.heads, self.value_dim = heads, value_dim
        self.rank, self.rank_max = rank, rank_max
        self.policy = policy
        self.key_dim, self.ctx = int(key_dim), int(ctx)
        self.segment_tokens = int(segment_tokens)
        self.width = width
        self.arm = arm
        self.channels = arm_channels(arm)
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.query_key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))
        # the per-channel DECAY channel (GDN2's f_proj shape, transplanted)
        self.f_proj = nn.Sequential(
            nn.Linear(width, value_dim, bias=False),
            nn.Linear(value_dim, heads * value_dim, bias=False))
        self.decay_scalar = nn.Parameter(torch.zeros(heads))
        if self.channels["write"]:
            self.b_proj = nn.Linear(width, heads * value_dim, bias=False)
            self.write_scalar = nn.Parameter(torch.zeros(heads))
        if self.channels["read"]:
            self.g_proj = nn.Sequential(
                nn.Linear(width, value_dim, bias=False),
                nn.Linear(value_dim, heads * value_dim, bias=True))
            self.read_scalar = nn.Parameter(torch.zeros(heads))
        self.apply_gate_birth()

    def apply_gate_birth(self):
        """P5's mimetic-neutral birth: every gate projection is born at ZERO
        (weight and bias) and every channel's per-head scalar at EXACT zero,
        so ``lam_vec == lam_inc`` per channel, ``g_vec == 1`` per channel and
        ``r_vec == 1`` at step 0 -- the arm is born as the banked winner,
        bit for bit on the discrete schedule, and the gates grow by training
        (the gradient reaches the scalar at step 0 through sigmoid(0) = 0.5;
        the projections wake one step later -- the K11 two-step wake-up
        pattern, receipted so the step-0 zero is not read as a dead
        channel)."""
        with torch.no_grad():
            for name in ("f_proj", "b_proj", "g_proj"):
                projection = getattr(self, name, None)
                if projection is not None:
                    for module in projection.modules():
                        if isinstance(module, nn.Linear):
                            module.weight.zero_()
                            if module.bias is not None:
                                module.bias.zero_()
            for name in ("decay_scalar", "write_scalar", "read_scalar"):
                scalar = getattr(self, name, None)
                if scalar is not None:
                    scalar.zero_()

    def channel_decay(self, x: torch.Tensor):
        """The per-channel DECAY vector and the registered scalar it wraps:
        ``lam_vec = clamp(lam_inc * (1 + m_f*sigmoid(f_proj(x))), floor,
        1-1e-6)`` -- at birth exactly ``lam_inc`` on every channel.

        THE CLAMP BOUNDS ACT IN THE INCUMBENT'S CARRIER DTYPE (the
        falsifier's corner, receipt 57e1f942, fixed per the supervisor's R1
        ruling): under the production autocast regime ``lam_inc`` is bf16,
        in which the registered bound ``1-1e-6`` ROUNDS TO EXACTLY 1.0 --
        the incumbent's own effective ceiling, its expression rounding the
        same way -- while a naive fp32 clamp of the promoted product would
        preserve ``0.999999`` and break the mimetic-neutral birth at every
        sigmoid-saturating position (one such position in 32,768 at the
        probe's layer 4 amplified to a 6.6e-5 step-1 loss delta). The
        bounds below are the carrier's own rounding of the registered
        constants, so at birth ``lam_vec`` is bit-equal to the registered
        scalar on every channel, corners included; beyond birth the
        effective ceiling under autocast is the incumbent's own 1.0 (the
        disclosed deviation), and in fp32 contexts the paper bound
        survives."""
        lam_inc = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR,
                                                     max=1.0 - 1e-6)
        f = torch.sigmoid(self.f_proj(x)).view(
            x.shape[0], x.shape[1], self.heads, self.value_dim)
        factor = 1.0 + self.decay_scalar.view(1, 1, -1, 1) * f
        if lam_inc.dtype == torch.bfloat16:
            floor = float(torch.tensor(DECAY_FLOOR, dtype=torch.bfloat16))
            ceiling = float(torch.tensor(1.0 - 1e-6, dtype=torch.bfloat16))
        else:
            floor, ceiling = DECAY_FLOOR, 1.0 - 1e-6
        lam_vec = torch.clamp(lam_inc.unsqueeze(-1) * factor,
                              min=floor, max=ceiling)
        return lam_vec, lam_inc

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        context = causal_context(x, self.ctx)          # (b, T, ctx, width)
        context = context.reshape(batch, tokens, self.ctx * self.width)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
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
        result = channel_gated_scan(payload, write_keys, query_keys, lam_vec,
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


def convert_model_to_channel_gates(model, arm: str = "p1",
                                  segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``SeparatedSequentialMixer`` for the per-channel-gated one,
    sharing parameters (the K10/K11 conversion pattern: the ``nn.Linear``
    modules the optimizer already saw are the very same objects), adding the
    arm's gate channels at the mimetic-neutral birth.  The key/query-key
    projections and every value-side module are carried over UNTOUCHED, so at
    birth the converted model's discrete schedule is the banked winner's bit
    for bit.  ``k10_separated_keys.py`` and ``k11_learned_gates.py`` are not
    edited; this builds on the K10 conversion's output.  The parameter DELTA
    is the registration's 1.5 (stated, asserted by the probe, not hidden)."""
    channels = arm_channels(arm)
    for block in model.blocks:
        mixer = block.mixer
        sep = ChannelGatedSeparatedMixer.__new__(ChannelGatedSeparatedMixer)
        torch.nn.Module.__init__(sep)
        sep.__dict__.update(mixer.__dict__)
        sep._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        sep.arm = arm
        sep.channels = channels
        sep.segment_tokens = int(segment_tokens)
        sep.width = int(mixer.value.weight.shape[1])
        value_dim = int(mixer.value_dim)
        heads = int(mixer.heads)
        # the per-channel DECAY channel (every arm)
        sep.f_proj = nn.Sequential(
            nn.Linear(sep.width, value_dim, bias=False),
            nn.Linear(value_dim, heads * value_dim, bias=False))
        sep.decay_scalar = nn.Parameter(torch.zeros(heads))
        if channels["write"]:
            sep.b_proj = nn.Linear(sep.width, heads * value_dim, bias=False)
            sep.write_scalar = nn.Parameter(torch.zeros(heads))
        if channels["read"]:
            sep.g_proj = nn.Sequential(
                nn.Linear(sep.width, value_dim, bias=False),
                nn.Linear(value_dim, heads * value_dim, bias=True))
            sep.read_scalar = nn.Parameter(torch.zeros(heads))
        sep.apply_gate_birth()
        block.mixer = sep
    return model


def arm_parameter_delta(arm: str, depth: int, width: int, heads: int,
                        value_dim: int) -> int:
    """The registration's 1.5 parameter accounting, computed not hardcoded:
    per channel, GDN2's own projection shapes transplanted -- the decay
    channel's f_proj (Sequential(width -> d_v -> heads*d_v), bias-free) plus
    one per-head zero-born scalar; the write channel's b_proj (a single
    bias-free Linear(width -> heads*d_v), GDN2's own form); the read
    channel's g_proj (Sequential(width -> d_v -> heads*d_v), the second
    layer with bias, GDN2's own form)."""
    channels = arm_channels(arm)
    per_channel = {"decay": (width * value_dim
                             + value_dim * heads * value_dim + heads),
                   "write": (width * heads * value_dim + heads),
                   "read": (width * value_dim
                            + value_dim * heads * value_dim
                            + heads * value_dim + heads)}
    return depth * sum(per_channel[k] for k in ("decay", "write", "read")
                       if channels[k])


__all__ = ["ARMS", "ARM_CHANNELS", "LANES", "ChannelGatedSeparatedMixer",
           "arm_channels", "arm_parameter_delta",
           "convert_model_to_channel_gates", "channel_gated_scan",
           "channel_gated_oracle_scan", "channel_persistent_trajectory",
           "separated_available", "_k11b_forward_launch", "_k11b_backward_body",
           "RoutedResult", "ScanStats", "MatrixSchedule", "RoutingPolicy",
           "EPS"]
