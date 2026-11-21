"""K17 -- THE WRITE FORM UNDER THE ADDITIVE READ: the dense gated write
(the b_proj-delta form) on the COMMITTED composed machine.

Why this file exists
--------------------
The K15 adjudication (the supervisor's ruling of 2026-11-24) named the
WRITE FORM (event writes vs dense superposition) as the LAST structural
candidate owner of the composed machine's remaining ~0.03 vs GDN2; the K16
config bracket discharged the config alibi and left the residual UNCHANGED.
The morning's G2 arm HURT on the SOFTMAX machine (+0.007): the per-head
dense-write channel did not compose with the COMPETITIVE read.  The machine
has since changed fundamentally -- the K13 ADDITIVE read (no simplex, no
competition, magnitude rides the rows) under the K11B p3 per-channel gate
stack -- and the read/write interface is different now.  This lane is G2's
dense-write form REBORN on that interface, at the composed machine's own
per-channel granularity (the K11B registration's named follow-up 7.2:
"a per-channel dense deposit -- named, not run").

The semantics (``PREREGISTRATION_K17_WRITE_FORM.md``, committed BEFORE any
probe measurement)
--------------------------------------------------------------------------------
* **The machine is the COMMITTED K15 composition, VERBATIM** (imported,
  never modified): the banked winner build + the K13 additive read under
  the K11B p3 gate stack, every committed kernel imported as-is.
* **The dense channel (the b_proj-delta form)**: every token writes a small
  learned-gated contribution to its BEST-MATCH LIVE ROW, as recorded
  schedule lane 2, while the five ops keep managing structure::

      dense_t = m_d[h] * sigmoid(d_proj(x)) (.) p_t      # per channel

  with ``d_proj = Linear(width, heads*d_v, bias=False)`` -- the b_proj form
  (GDN2's own single bias-free write-gate Linear shape) -- born at exact
  ZERO weight and ``m_d`` a per-head scalar born at EXACT zero (the
  committed zero-born gate conventions): at birth ``dense == 0`` exactly and
  the machine is BORN AS THE COMPOSED MACHINE bit for bit.
* **P3 verbatim** (the target rule): the deposit goes to the BEST-MATCH
  LIVE ROW at decision time (argmax over live, regardless of
  ``tau_match``); if NO live row exists at decision time the deposit goes
  nowhere (writing a free slot would be an APPEND event and change the
  decision sequence); and the target row must STILL BE LIVE after lanes
  0/1 (a POP can kill it in-token).  The deposit fires on the LIVE GUARDS
  ALONE (never on its value -- a value-dependent no-op rule would make the
  recorded op code a function of the deposit and cut the birth gradient).
* **P2 verbatim**: the dense deposit rides the VALUE channels only; no
  deposit value feeds op selection, thresholds, routing similarity or
  liveness.  The discrete op sequence is bit-invariant to EVERY deposit
  value at fixed inputs (the 0-flip receipt).
* **Lane 2 is ordinary schedule data**: ``op=APPEND`` (additive),
  ``slot=best``, ``erase=False``, ``live_after=True``, ``payload=dense_t``
  (zeroed when the guards refuse) -- the imported lane-generic per-channel
  trajectory replay consumes it verbatim, and at the zero birth the
  deposit is the IEEE identity (``x + +-0 = x``; the -0.0 corner is
  receipted by the suite, the K11 precedent).

The kernels (the committed engines imported; ``k10_*``/``k11b_*``/``k13_*``
/``k15_*`` sources are imported, never edited)
--------------------------------------------------------------------------------
``_k17_seq_fwd_kernel``     THE ONE NEW KERNEL: the K15 composed walk (K11B's
                            persistent whole-window walk with K13's additive
                            read block) plus the dense lane 2 -- the load,
                            the record, and the guarded additive application
                            (K11's unconditional-select form: no runtime
                            branch reassigning the loop-carried tile).
``_k17_bwd_value_walk_kernel`` K11B's per-channel value walk plus the dense
                            lane's adjoint (``ddense = dS[best]``; ``ds``
                            passes through -- the deposit is additive).
``_k11b_traj_kernel``      IMPORTED VERBATIM via ``channel_persistent_trajectory``
                            (lane-generic: the 3-lane schedule replays bit
                            for bit, dense lane included).
``_k11_key_traj_kernel``   IMPORTED VERBATIM (K11's lane-STRIDE-generic key
                            replay -- K10's own hardcodes the 2-lane layout;
                            the dense machine's schedule carries THREE lanes;
                            the key lane lives on lanes 0/1).
``_k13_bwd_prep_kernel``   IMPORTED VERBATIM (form-agnostic: the read weights
                            recomputed from the trajectory; the strength
                            channel EXACT ZEROS).
``_k11_bwd_key_walk_kernel`` IMPORTED VERBATIM (K11's stride-generalized key
                            walk; the key channel is untouched by gates and
                            deposits).

Doctrine, unchanged: decisions are recorded constants in the backward.  The
read weights (raw scores), payload lanes, the PER-CHANNEL DECAY vector, the
PER-CHANNEL write gate, the DENSE DEPOSIT and the KEY CHANNEL carry
gradient to their inputs; the gradient gates are floor-discipline (the fp64
gradcheck of the oracle THROUGH THE DENSE-WRITE PATH first, fp32-vs-fp64
floors, ``NOISE_FACTOR`` x), stated, not hidden.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
import triton
import triton.language as tl

from .algebra import (APPEND, CORRECT, MERGE, NONE, POP, MatrixSchedule)
from .k1_schedule_scan import _add_rn, _mul_rn, _sub_rn
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                   _log2_up, _norm_defined, _pow2,
                                   _round_f32, _seq_warps, _sqrt_rn32,
                                   tree_fold)
from .k10_separated_keys import (_key_selection, DEFAULT_CONTEXT,
                                 DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS,
                                 SeparatedSequentialMixer,
                                 _split_fold_axis2,
                                 separated_available)
from .k11b_channel_gates import (ChannelGatedSeparatedMixer,
                                 _k11b_bwd_value_walk_kernel,
                                 arm_channels, arm_parameter_delta,
                                 channel_persistent_trajectory)
from .k13_additive_read import (BT_BWD, BV_BWD, _k13_bwd_prep_kernel)
from .k11_learned_gates import (_k11_bwd_key_walk_kernel,
                               _k11_key_traj_kernel)
from .k15_composition import (ComposedReadMixer, composed_oracle_scan,
                              composed_parameter_delta, composed_scan,
                              convert_model_to_composition)
from .key_separation_probe import causal_context
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))

# The recorded schedule's lane count for every k17 scan: lanes 0/1 are the
# composed machine's own (K10's main write + the MERGE-source close), lane 2
# is the dense deposit (K11's layout, carried to the composed machine).
LANES = 3


def dense_parameter_delta(depth: int, width: int, heads: int,
                          value_dim: int) -> int:
    """The registration's parameter accounting, computed not hardcoded: the
    dense channel is the b_proj form -- one ``nn.Linear(width, heads *
    value_dim, bias=False)`` plus one per-head zero-born scalar -- per
    mixer, ON TOP of the composed machine's own p3 stack (the committed k15
    accounting)."""
    composed = composed_parameter_delta(depth, width, heads, value_dim)
    per_mixer = width * heads * value_dim + heads
    return composed + depth * per_mixer


# ------------------------------------------------------------- the oracle


def dense_oracle_scan(payload, dense, write_keys, query_keys, decay, gates, *,
                      policy: RoutingPolicy, gated: bool = True,
                      use_dense: bool = True, rank=None, rank_max=None,
                      initial_state=None, initial_live=None, initial_keys=None,
                      collect=True):
    """The dense machine's eager per-token oracle: the parity fixed point,
    differentiable through the key channel, the per-channel DECAY vector,
    the per-channel write gate, the DENSE DEPOSIT and the additive read.

    The K15 composed oracle walk (decisions from the DETACHED key channel,
    K13's additive read block, the per-channel DECAY and the gated lane-0
    arithmetic, the per-lane-guarded MERGE-source close) extended with the
    dense lane (K11 G2's semantics at the per-channel granularity): the
    deposit targets the BEST-MATCH LIVE ROW at decision time, fires on the
    live guards alone, and lands after lanes 0/1 -- recorded as lane 2."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if use_dense and dense.shape != payload.shape:
        raise ValueError("the dense deposit must share the payload's shape")
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
                         "write-form registration; it is a chunk-dial "
                         "control")
    if policy.grow:
        raise ValueError("growth is refused by the write-form registration: "
                         "growth is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the write-form registration requires rank == "
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
    dn32 = _fp32(dense)
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
        dn_t = dn32[:, t] if use_dense else None
        lam_t = d32[:, t]                                 # (b, h, V)
        # ---- the read (K13's block, VERBATIM): the raw banked score,
        # live-gated -- NO softmax, NO strength.
        nk = _norm_defined(keys, -1, KP)                       # (b, h, rank)
        nqk = _norm_defined(qk_t, -1, KP)                      # (b, h)
        uqk = qk_t / nqk.clamp_min(EPS).unsqueeze(-1)
        scores = (scale * tree_fold(uqk.unsqueeze(2) * keys, -1, KP)) \
            / nk.clamp_min(EPS)
        w = scores.masked_fill(~live, 0.0)
        reads[:, t] = tree_fold(w.unsqueeze(-1) * state, -2, RP)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the registered chain on the DETACHED key channel
        # (K10 verbatim -- P2: no gate value, no deposit feeds op selection).
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)                               # (b, h, rank)
        sim = sim.masked_fill(~live, -2.0)
        sel = _key_selection(sim, live, tau_match, tau_merge, tau_pop, tie,
                             rank, RP)
        any_live_pre = live.any(-1)                       # (b, h): P3's guard
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
        # ---- DECAY (values only, PER CHANNEL), then the gated lane-0 value
        # (K11B verbatim: the pre-decay per-channel CORRECT form).
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
        # ---- record lanes 0/1 (the composed machine's own) and apply.
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
        # the MERGE-source close, guarded PER LANE (K11B's committed fix)
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
        # ---- lane 2: the dense deposit (P3 verbatim).  The target is the
        # BEST-MATCH LIVE ROW at decision time; if no live row existed at
        # decision time the deposit goes nowhere; the row must also still be
        # live after lanes 0/1 (a POP can kill it in-token).  The deposit's
        # VALUE is the gate's output (born at exact zero -- the IEEE
        # identity, the -0.0 corner receipted by the tests), so the lane
        # fires on the live guards alone and the gradient reaches the
        # zero-born scalar at step 0.
        fire = torch.zeros(batch, heads, dtype=torch.bool, device=device)
        if use_dense:
            live_best = live.gather(-1, sel["best_index"].unsqueeze(-1)
                                    ).squeeze(-1)
            fire = any_live_pre & live_best
        fields["slot"][:, t, :, 2] = sel["best_index"]
        fields["op"][:, t, :, 2] = torch.where(
            fire, torch.full_like(sel["op0"], APPEND),
            torch.full_like(sel["op0"], NONE))
        fields["erase"][:, t, :, 2] = False
        fields["live_after"][:, t, :, 2] = True
        if use_dense:
            fields["payload"][:, t, :, 2, :] = torch.where(
                fire.unsqueeze(-1), dn_t, torch.zeros_like(dn_t))
        if bool(fire.any()):
            rows_best = state[b_index.expand(batch, heads),
                              h_index.expand(batch, heads),
                              sel["best_index"]]
            state = state.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads),
                 sel["best_index"]),
                torch.where(fire.unsqueeze(-1), rows_best + dn_t, rows_best))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, keys=keys,
                schedule=schedule, weights=weights_out, live_pre=live_pre_out,
                rank=int(rank))


# ------------------------------------------------------------- the forward kernel


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k17_seq_fwd_kernel(
    payload, dense, write_keys, query_keys, decay, gates, state_in, live_in,
    keys_in,
    reads, state_out, live_out, keys_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, ckpt_keys, weights_out,
    T, R, ROWS, SEG, KD,
    pbs, pts, phs, pvs,
    dbs, dts, dhs, dvs,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    cbs, cts, chs, cvs,
    gbs, gts, ghs, gvs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, L: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr, LOGK: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    GATED: tl.constexpr, DENSE: tl.constexpr,
    STAGES: tl.constexpr,
):
    """THE ONE NEW KERNEL of the write-form lane: the K15 composed walk
    (K11B's persistent whole-window walk with K13's additive read block)
    plus the dense lane 2 -- K11 G2's dense block, carried to the composed
    machine (the per-channel DECAY load and the gated lane-0 arithmetic are
    K11B's verbatim).  Per token: the additive read, the KEY-channel
    decisions, the MERGE blend, the PER-CHANNEL DECAY, the gated lane 0,
    the key lane, then the guarded dense deposit at the best-match live
    row."""
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
    dense_row = dense + batch_index * dbs + head_index * dhs
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
        # the PER-CHANNEL decay vector (K11B's registered replacement)
        lam = tl.load(decay_row + t * cts + values * cvs, mask=keep_v,
                      other=0.0).to(tl.float32)

        # ---- the value norms (the MERGE blend's registered detached
        # strengths) and the ADDITIVE read weights on the KEY channel.
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
        # K13: the additive read -- the raw score, live-gated.  NO softmax,
        # NO strength factor.  An all-dead store reads EXACTLY zero.
        w = tl.where(live_r, scores, 0.0)
        # the read: the K6 floor channel (a native rank-axis sum).
        read_o = tl.sum(_mul_rn(w[:, None], acc), 0)
        tl.store(reads_base + t * H * V + values, read_o, mask=keep_v)
        if COLLECT:
            tl.store(weights_base + t * H * R + ranks, w, mask=keep_r)

        # ---- the decisions: the registered selection chain on the KEY
        # channel (K10 verbatim -- P2: no gate value, no deposit feeds it).
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
        # the gated lane 0 (K11B verbatim).
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
                    # the admission form (gate*value) on the carrying ops
                    lane0 = tl.where(carries, _mul_rn(gt, p), 0.0)
            else:
                lane0 = tl.where(carries, p, 0.0)

        # ---- apply: DECAY (values only, PER CHANNEL), then the value lanes
        # in order (K10 verbatim), then the key lane.
        acc = _mul_rn(acc, tl.broadcast_to(lam[None, :], (RP, VP)))
        old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)

        # ---- record the decision channel (the 3-lane (b, T, h, L) layout).
        tl.store(slot_base + t * H * L + 0, slot0.to(tl.int64))
        tl.store(slot_base + t * H * L + 1, second.to(tl.int64))
        tl.store(slot_base + t * H * L + 2, best.to(tl.int64))
        tl.store(op_base + t * H * L + 0, op0.to(tl.int64))
        tl.store(op_base + t * H * L + 1,
                 tl.where(merge, _MERGE_CODE, _NONE_CODE).to(tl.int64))
        tl.store(erase_base + t * H * L + 0, erase0.to(tl.int8))
        tl.store(erase_base + t * H * L + 1, merge.to(tl.int8))
        tl.store(erase_base + t * H * L + 2,
                 tl.zeros((), dtype=tl.int8))
        tl.store(live_after_base + t * H * L + 0, live0.to(tl.int8))
        tl.store(live_after_base + t * H * L + 1,
                 tl.zeros((), dtype=tl.int8))
        tl.store(live_after_base + t * H * L + 2,
                 tl.full((), 1, tl.int8))
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
        # ---- lane 2: the dense deposit (P3 verbatim).  Target the
        # BEST-MATCH LIVE ROW (argmax over live at decision time, `best`);
        # zero if no live row existed at decision time; the row must still
        # be live after lanes 0/1 (a POP can kill it in-token).  The
        # deposit's VALUE is the gate's output -- born at exact zero, the
        # IEEE identity (x + +-0 = x, the -0.0 corner receipted by the
        # tests) -- so the lane fires on the live guards alone and the
        # gradient reaches the zero-born scalar at step 0 (a
        # value-dependent no-op rule would make the recorded op code a
        # function of the deposit and cut the birth gradient -- the
        # dead-arm failure the registration refuses).
        if DENSE:
            dn = tl.load(dense_row + t * dts + values * dvs, mask=keep_v,
                         other=0.0).to(tl.float32)
            live_best = tl.sum(tl.where(keep_r & (best == ranks),
                                        live.to(tl.float32), 0.0), 0) > 0.0
            fire = live_any & live_best
            hitd = fire & keep_r & (best == ranks)
            # the application is an UNCONDITIONAL select (no runtime branch
            # reassigning the loop-carried tile: the pipeliner miscompiles
            # that -- K11's receipt, carried verbatim)
            old2 = tl.sum(tl.where(hitd[:, None], acc, 0.0), 0)
            acc = tl.where(hitd[:, None],
                           _add_rn(old2, dn)[None, :], acc)
            tl.store(op_base + t * H * L + 2,
                     tl.where(fire, _APPEND_CODE, _NONE_CODE).to(tl.int64))
            tl.store(payload_base + t * H * L * V + 2 * V + values,
                     tl.where(fire, dn, tl.zeros((VP,), dtype=tl.float32)),
                     mask=keep_v)
        else:
            tl.store(op_base + t * H * L + 2,
                     tl.full((), _NONE_CODE, tl.int64))
            tl.store(payload_base + t * H * L * V + 2 * V + values,
                     tl.zeros((VP,), dtype=tl.float32), mask=keep_v)
        # ---- the key lane: K10 verbatim (deposits never touch it).
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


def _check_dense_shapes(payload, dense, write_keys, query_keys, decay, gates,
                        state, live, keys, rank, segment_tokens, gated,
                        use_dense):
    """Refuse anything the dense-composed path cannot express, before any
    launch (K15's checks verbatim, the dense channel added)."""
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda):
        raise ValueError("the dense-composed scan is a device kernel; "
                         "inputs must be on CUDA")
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape != (batch, tokens, heads, key_dim):
        raise ValueError(f"keys must be (batch, tokens, heads, key_dim); got "
                         f"{tuple(write_keys.shape)}")
    if use_dense and dense.shape != payload.shape:
        raise ValueError(f"the dense deposit must share the payload's shape "
                         f"{tuple(payload.shape)}; got {tuple(dense.shape)}")
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


def _k17_forward_launch(payload, dense, write_keys, query_keys, decay, gates,
                        state, live, keys, policy_f32, rank, segment_tokens,
                        gated, use_dense, collect, warps, stages):
    """The persistent walk: one launch, every output allocated contiguous
    (K15's launch verbatim, the dense pointer and the 3-lane layout added)."""
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
    handle = _k17_seq_fwd_kernel[(batch * heads,)](
        payload, dense, write_keys, query_keys, decay, gates, state,
        live.view(torch.int8), keys,
        reads, out_state, out_live.view(torch.int8), out_keys,
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), ckpt_keys, weights,
        tokens, rank, batch * heads, int(segment_tokens), key_dim,
        *payload.stride(), *dense.stride(), *write_keys.stride(),
        *query_keys.stride(), *decay.stride(), *gates.stride(),
        H=heads, V=value_dim, VP=_pow2(value_dim), KP=_pow2(key_dim),
        RP=_pow2(rank), L=lanes,
        LOGV=_log2_up(_pow2(value_dim)), LOGR=_log2_up(_pow2(rank)),
        LOGK=_log2_up(_pow2(key_dim)),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0,
        GATED=1 if gated else 0, DENSE=1 if use_dense else 0,
        STAGES=stages, num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, keys=out_keys,
                slot=slot, op=op, erase=erase, live_after=live_after,
                payload=payload_field, live_pre=live_pre,
                ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                ckpt_keys=ckpt_keys, weights=weights, kernel=handle)


# --------------------------------------------------------- the backward kernels


@triton.jit(do_not_specialize=['SEGT', 'T', 'T0'])
def _k17_bwd_value_walk_kernel(
    traj, payload, gates, dout, decay32, live_pre, carry_in,
    nr_ch, w_ch, dstrength_ch,
    slot, op, erase,
    dpayload, ddense, dlam_ch, dgate_ch, dstate_seg,
    SEGT, T, T0, R,
    pbs, pts, phs, pvs,
    gbs, gts, ghs, gvs,
    xbs, xts, xhs, xvs,
    sbs, sts, shs, sls,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, RP: tl.constexpr,
    LOGV: tl.constexpr, BT: tl.constexpr, BV: tl.constexpr,
    GATED: tl.constexpr, DENSE: tl.constexpr,
):
    """K11B's value-blocked reverse walk plus the DENSE lane's adjoint.
    Lanes last first: the dense lane's ``ddense = dS[best]`` (additive,
    ``ds`` passes through), then the MERGE-source zeroing, then lane 0 --
    under the per-channel gate ``dpayload = g_vec*dpay0``,
    ``dgate[c] = dpay0[c]*(p[c] - v_dec[c])`` and the CORRECT blend's
    old-row path -- then the PER-CHANNEL DECAY adjoint, the MERGE-blend
    scatter, and the read adjoint (K15 verbatim).  ``dlam_ch``/``dgate_ch``
    are per-(row, token, channel) buffers; each value-block program writes
    its own disjoint channel slice."""
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
    dd_row = ddense + batch_index.to(tl.int64) * T * H * V + head_index * V
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
        # 1a. lane 2 (the dense deposit): additive -- the full incoming
        #     adjoint at the best row IS ddense; ds passes through.
        if DENSE:
            op2 = tl.load(op + channel_base + t * sts + 2 * sls)
            slot2 = tl.load(slot + channel_base + t * sts + 2 * sls
                            ).to(tl.int32)
            if op2 != _NONE_CODE:
                hitd = keep_r & (slot2 == ranks)
                ddense_v = tl.sum(tl.where(hitd[:, None], ds, 0.0), 0)
            else:
                ddense_v = tl.zeros((BV,), dtype=tl.float32)
            tl.store(dd_row + t * (H * V) + values,
                     ddense_v.to(ddense.dtype.element_ty), mask=keep_v)
        # 1b. lane 1 (the MERGE source close): the source row's path
        #     terminates.
        hit2 = merge & keep_r & (second == ranks)
        if merge:
            ds = tl.where(hit2[:, None], 0.0, ds)
        # 1c. lane 0.
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


# --------------------------------------------------- the backward body


def _k17_backward_body(payload, dense, write_keys, query_keys, decay, gates,
                       dreads, schedule, live_pre, ckpt_state, ckpt_live,
                       ckpt_keys, rank, segment_tokens, scale, gated,
                       use_dense=True):
    """The whole window's adjoint: the K15 composed segment loop with the
    dense lane -- the per-channel value replay (K11B's, imported, LANE-
    GENERIC: the 3-lane schedule replays verbatim), the key replay (K10's,
    imported), the ADDITIVE prep (K13's, imported verbatim), the PER-CHANNEL
    value walk (the K17 walk: K11B's plus the dense lane's ``ddense = dS
    [best]``), and the key walk (K10's, imported).  Decisions are recorded
    constants (the standing doctrine)."""
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
    ddense = torch.empty_like(dense) if use_dense \
        else torch.zeros_like(dense)
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
        # 1. the segment's value trajectory (K11B's per-channel replay,
        #    imported verbatim -- lane-generic, the dense lane included).
        _, _, _, traj = channel_persistent_trajectory(
            ckpt_state[seg], ckpt_live[seg],
            decay32.view(batch, heads, tokens, value_dim)[:, :, start:stop],
            schedule.chunk(start, stop))
        # 2. the segment's key trajectory (the key-lane replay: K11's
        #    lane-STRIDE-generic kernel, imported verbatim -- K10's own
        #    hardcodes the 2-lane layout, and the dense machine's recorded
        #    schedule carries THREE lanes; the key lane lives on lanes 0/1).
        ktraj = torch.empty((rows, segt, R, key_dim), dtype=torch.float32,
                            device=device)
        _k11_key_traj_kernel[(rows,)](
            ckpt_keys[seg], write_keys, schedule.slot, schedule.op, ktraj,
            segt, tokens, start, R, key_dim,
            *schedule.slot.stride(),
            *write_keys.stride(),
            H=heads, KP=KP, RP=RP, num_warps=4)
        # 3. prep: the per-token weights and adjoint coefficients, both sides
        #    (the K13 ADDITIVE form, imported verbatim; dstrength is EXACT
        #    ZEROS).
        nr_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        nk_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        w_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dscores_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                                 device=device)
        dstrength_ch = torch.zeros((rows, segt, R), dtype=torch.float32,
                                   device=device)
        z_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dc_ch = torch.empty((rows, segt, R), dtype=torch.float32, device=device)
        dnk_ch = torch.empty((rows, segt, R), dtype=torch.float32,
                             device=device)
        scal = torch.empty((rows, segt, 2), dtype=torch.float32, device=device)
        grid = (rows, triton.cdiv(segt, BT_BWD))
        _k13_bwd_prep_kernel[grid](
            traj, ktraj, query_keys, dreads_c, live_pre.view(torch.int8),
            nr_ch, nk_ch, w_ch, dscores_ch, dstrength_ch, z_ch, dc_ch,
            dnk_ch, scal,
            segt, tokens, start, R, key_dim,
            *query_keys.stride(), *dreads_c.stride(),
            H=heads, V=value_dim, VP=VP, KP=KP, RP=RP, LOGV=LOGV, LOGR=LOGR,
            LOGK=LOGK, BT=BT_BWD, SCALE=_round_f32(scale), num_warps=8)
        # 4. the value walk (the K17 walk: K11B's per-channel walk plus the
        #    dense lane's adjoint).
        dstate_seg = torch.empty((batch, heads, R, value_dim),
                                 dtype=torch.float32, device=device)
        _k17_bwd_value_walk_kernel[(rows, nblocks)](
            traj, payload, gates, dreads_c, decay32, live_pre.view(torch.int8),
            carry,
            nr_ch, w_ch, dstrength_ch,
            schedule.slot, schedule.op, schedule.erase.view(torch.int8),
            dpayload, ddense, ddecay_ch, dgates_ch, dstate_seg,
            segt, tokens, start, R,
            *payload.stride(), *gates.stride(), *dreads_c.stride(),
            *schedule.slot.stride(),
            H=heads, V=value_dim, VP=VP, RP=RP, LOGV=LOGV, BT=BT_BWD,
            BV=BV_BWD, GATED=1 if gated else 0, DENSE=1 if use_dense else 0,
            num_warps=4)
        # 5. the key walk (K11's stride-GENERALIZED walk, imported verbatim
        #    -- K10's own hardcodes the 2-lane layout; the key channel is
        #    untouched by gates and deposits).
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
    return (dpayload, ddense, dwrite_keys, dquery_keys, ddecay, dgates,
            dstate, dkeys)


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::k17_scan_fwd", mutates_args=())
def _k17_scan_fwd_op(
    payload: torch.Tensor, dense: torch.Tensor,
    write_keys: torch.Tensor, query_keys: torch.Tensor, decay: torch.Tensor,
    gates: torch.Tensor, state: torch.Tensor, live: torch.Tensor,
    keys: torch.Tensor, rank: int, segment_tokens: int, tau_match: float,
    tau_merge: float, tau_pop: float, tie_break: float, scale: float,
    gated: bool, use_dense: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """The whole (layer, window) dense-composed scan as ONE opaque op (the
    K4/K6/K10/K11/K11B/K13/K15 hosting pattern)."""
    _check_dense_shapes(payload, dense, write_keys, query_keys, decay, gates,
                        state, live, keys, int(rank), int(segment_tokens),
                        bool(gated), bool(use_dense))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k17_forward_launch(payload, dense, write_keys, query_keys, decay,
                              gates, state, live, keys, pol32, int(rank),
                              int(segment_tokens), bool(gated),
                              bool(use_dense), collect=False,
                              warps=_seq_warps(int(rank), payload.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"])


@_k17_scan_fwd_op.register_fake
def _(payload, dense, write_keys, query_keys, decay, gates, state, live,
      keys, rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break,
      scale, gated, use_dense):
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


def _k17_scan_setup(ctx, inputs, output):
    (payload, dense, write_keys, query_keys, decay, gates, state, live, keys,
     rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break, scale,
     gated, use_dense) = inputs
    ctx.save_for_backward(payload, dense, write_keys, query_keys, decay,
                          gates, state, live, keys, *output[4:])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)
    ctx.gated = bool(gated)
    ctx.use_dense = bool(use_dense)


@torch.library.custom_op("x45::k17_scan_bwd", mutates_args=())
def _k17_scan_bwd_op(
    payload: torch.Tensor, dense: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor, gates: torch.Tensor,
    dreads: torch.Tensor, slot: torch.Tensor, op: torch.Tensor,
    erase: torch.Tensor, live_after: torch.Tensor,
    payload_field: torch.Tensor, live_pre: torch.Tensor,
    ckpt_state: torch.Tensor, ckpt_live: torch.Tensor, ckpt_keys: torch.Tensor,
    rank: int, segment_tokens: int, scale: float, gated: bool,
    use_dense: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: the dense-composed
    segment loop (per-channel value replay -> key replay -> the additive
    prep -> the K17 value walk -> the key walk) runs inside, decisions as
    recorded constants (the standing doctrine)."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k17_backward_body(payload, dense, write_keys, query_keys, decay,
                              gates, dreads, schedule, live_pre, ckpt_state,
                              ckpt_live, ckpt_keys, int(rank),
                              int(segment_tokens), float(scale), bool(gated),
                              bool(use_dense))


@_k17_scan_bwd_op.register_fake
def _(payload, dense, write_keys, query_keys, decay, gates, dreads, slot, op,
      erase, live_after, payload_field, live_pre, ckpt_state, ckpt_live,
      ckpt_keys, rank, segment_tokens, scale, gated, use_dense):
    device = payload.device
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    return (torch.empty_like(payload),
            torch.empty_like(dense),
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


def _k17_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys):
    """The adjoint: dpayload, ddense, dwrite_keys, dquery_keys, ddecay (per
    channel), dgates (per channel), dstate, dkeys; the decision channel and
    the liveness carry no gradient by construction."""
    (payload, dense, write_keys, query_keys, decay, gates, state, live, keys,
     slot, op, erase, live_after, payload_field, live_pre, ckpt_state,
     ckpt_live, ckpt_keys) = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    (dpayload, ddense, dwrite_keys, dquery_keys, ddecay, dgates, dstate,
     dkeys) = _k17_scan_bwd_op(payload, dense, write_keys, query_keys, decay,
                               gates, dreads, slot, op, erase, live_after,
                               payload_field, live_pre, ckpt_state,
                               ckpt_live, ckpt_keys, ctx.rank,
                               ctx.segment_tokens, ctx.scale, ctx.gated,
                               ctx.use_dense)
    return (dpayload, ddense, dwrite_keys, dquery_keys, ddecay, dgates,
            dstate, None, dkeys, None, None, None, None, None, None, None,
            None, None, None)


torch.library.register_autograd("x45::k17_scan_fwd", _k17_scan_backward,
                                setup_context=_k17_scan_setup)


# --------------------------------------------------------------- the wrapper


def dense_scan(payload, dense, write_keys, query_keys, decay, gates, *,
               policy: RoutingPolicy, gated: bool = True, use_dense: bool = True,
               rank=None, rank_max=None,
               segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
               initial_state=None, initial_live=None, initial_keys=None,
               collect: bool = False) -> RoutedResult:
    """One whole (layer, window) dense-composed scan at PER-TOKEN granularity.

    The K15 wrapper's drop-in for the write-form machine: the composed scan's
    own refusals, the ADDITIVE read, the per-channel DECAY vector and write
    gate, plus the dense lane 2 (the best-match-live-row deposit).  The
    kernel path is ONE opaque op pair per scan; every refusal takes the
    EAGER ORACLE -- the same semantics, the parity fixed point -- never a
    silent change."""
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
                         "write-form registration; it is a chunk-dial "
                         "control")
    if policy.grow:
        raise ValueError("growth is refused by the write-form registration: "
                          "growth is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the write-form registration requires rank == "
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
            out = _k17_forward_launch(
                payload, dense, write_keys, query_keys, decay, gates, state,
                live, keys, pol32, rank, segment_tokens, gated, use_dense,
                collect=True,
                warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                            ScanStats(), schedule, out["weights"])
    if (payload.is_cuda and separated_available(rank, value_dim, key_dim)):
        reads, state_out, live_out, _keys, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live, _ckpt_keys = \
            _k17_scan_fwd_op(payload, dense, write_keys, query_keys, decay,
                             gates, state, live, keys, rank,
                             int(segment_tokens), policy.tau_match,
                             policy.tau_merge, policy.tau_pop,
                             policy.tie_break, policy.read_scale,
                             bool(gated), bool(use_dense))
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same semantics, the parity fixed point
    ref = dense_oracle_scan(payload, dense, write_keys, query_keys, decay,
                            gates, policy=policy, gated=gated,
                            use_dense=use_dense, rank=rank, rank_max=rank_max,
                            initial_state=state, initial_live=live,
                            initial_keys=keys, collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class DenseWriteComposedMixer(ComposedReadMixer):
    """The K17 mixer: the K15 composed mixer (the same projections, the same
    per-channel gate channels, the same committed ``channel_decay`` birth,
    the additive read) with the DENSE WRITE channel added -- the b_proj-delta
    form: ``dense = m_d*sigmoid(d_proj(x)) (.) payload``, deposited at the
    best-match live row inside the scan (lane 2).  At birth the mixer IS the
    composed machine bit for bit: ``dense == 0`` exactly (the committed
    zero-born gate conventions -- ``d_proj`` born at zero weight, ``m_d``
    born at exact zero), so the deposit is the IEEE identity and the scan's
    every channel is the composed machine's own."""

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
        super().__init__(width, heads, value_dim, rank, rank_max, key_dim,
                        policy, ctx=ctx, segment_tokens=segment_tokens)
        # the dense channel (the b_proj form): GDN2's own single bias-free
        # write-gate Linear shape, born at exact zero weight
        self.d_proj = nn.Linear(width, heads * value_dim, bias=False)
        self.dense_scalar = nn.Parameter(torch.zeros(heads))
        self.apply_dense_birth()

    def apply_dense_birth(self):
        """The dense channel's zero-born birth (the committed conventions):
        ``d_proj`` at exact zero weight and ``m_d`` at exact zero, so
        ``dense == 0`` at step 0 and the mixer is born as the composed
        machine bit for bit (the gradient reaches ``m_d`` at step 0 through
        sigmoid(0) = 0.5; the projection trains once the scalar moves --
        the two-step wake-up, receipted by the suite)."""
        with torch.no_grad():
            self.d_proj.weight.zero_()
            self.dense_scalar.zero_()

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
        # the dense channel (the b_proj-delta form): born at exact zero
        dense = self.dense_scalar.view(1, 1, -1, 1) \
            * torch.sigmoid(self.d_proj(x)).view(
                batch, tokens, self.heads, self.value_dim) * payload
        result = dense_scan(payload, dense, write_keys, query_keys, lam_vec,
                            gates, policy=self.policy, gated=gated,
                            use_dense=True, rank=self.rank,
                            rank_max=self.rank_max,
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


def convert_model_to_dense_write(model,
                                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``ComposedReadMixer`` for the dense-write one, sharing
    parameters (the K10/K11/K11B/K13/K15 conversion pattern: the
    ``nn.Linear`` modules the optimizer already saw are the very same
    objects), adding the dense channel (the b_proj-delta form) at the
    zero-born birth.  Every carried module is carried over UNTOUCHED, so at
    birth the converted model IS the K15 composed machine bit for bit and
    the discrete schedule is the banked winner's bit for bit at fixed
    inputs.  ``k15_composition.py`` is not edited; this builds on the K15
    conversion's output.  The parameter DELTA is the registration's (p3's
    own 8,657,040 + the dense channel's 3,538,992 at depth 24; stated,
    asserted by the probe, not hidden)."""
    for block in model.blocks:
        mixer = block.mixer
        if not isinstance(mixer, ComposedReadMixer):
            raise ValueError("convert_model_to_dense_write expects a "
                             "K15-composed model (run "
                             "convert_model_to_composition first); got "
                             f"{type(mixer).__name__}")
        sep = DenseWriteComposedMixer.__new__(DenseWriteComposedMixer)
        torch.nn.Module.__init__(sep)
        sep.__dict__.update(mixer.__dict__)
        sep._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        sep.segment_tokens = int(segment_tokens)
        sep.width = int(mixer.value.weight.shape[1])
        heads = int(mixer.heads)
        value_dim = int(mixer.value_dim)
        # the dense channel (the b_proj form), born at exact zero
        sep.d_proj = nn.Linear(sep.width, heads * value_dim, bias=False)
        sep.dense_scalar = nn.Parameter(torch.zeros(heads))
        sep.apply_dense_birth()
        block.mixer = sep
    return model


__all__ = ["DenseWriteComposedMixer", "convert_model_to_dense_write",
           "dense_parameter_delta", "dense_scan", "dense_oracle_scan",
           "_k17_forward_launch", "_k17_backward_body", "LANES",
           "separated_available", "DEFAULT_KEY_DIM", "DEFAULT_CONTEXT",
           "DEFAULT_SEGMENT_TOKENS", "ComposedReadMixer",
           "convert_model_to_composition", "composed_oracle_scan",
           "composed_scan", "composed_parameter_delta"]
