"""K21 -- THE SCAN COST PROFILE: the SCRATCH MICRO-BRACKET file.

Zero training.  This file is the task's named scratch space: every lever
below is a RE-IMPLEMENTATION or a LAUNCH-CONFIG probe of the committed
composed scan's forward kernel, lived in ONLY here -- the committed engine
files (``k15_composition.py`` and everything it imports) are imported
verbatim and NEVER edited; the main lane's receipt hash-binds them to prove
it.  Nothing in this file is imported by any committed engine or by the
training machines; it exists to answer one question with measurements:

    is the composed scan's 23.40 ms forward at the registered shape
    INHERENT to the slot semantics, or INCIDENTAL to a kernel engineered
    receipts-first?

THE RECEIPTS BAR (the lane's own discipline): a lever only COUNTS as a
speedup if it reprodu the committed scan's outputs BIT-IDENTICALLY
(``torch.equal`` on every output channel of the committed opaque op pair:
reads, state, live, keys, slot, op, erase, live_after, payload, live_pre,
ckpt_state, ckpt_live, ckpt_keys).  Anything else changes the machine and
is out of scope; every lever here is checked, and a lever that cannot
preserve bit-exactness is REPORTED as such, never shipped.

THE INSTRUMENTS (three kinds, all on captured REAL activations at the
registered shape -- batch 8 x 2048 tokens, width 576, heads 2, rank 128,
value_dim 128, key_dim 16, segment 64):

1. ``launch-config sweep`` -- the COMMITTED kernel verbatim (the committed
   ``_k15_forward_launch``, which already takes warps/stages), relaunched at
   every (num_warps, num_stages) bracket.  Launch metadata is not semantics;
   bit-identity is still verified per bracket because Triton's reduction
   lowering can in principle depend on the layout.

2. ``the scratch kernel`` -- a faithful copy of ``_k15_seq_fwd_kernel`` with
   constexpr-gated levers.  The faithful configuration (all flags zero) is
   bit-identity-checked against the committed op pair first; if the copy
   does not match, the instrument FAILS CLOSED and nothing it says counts.

   The levers (each a pure code motion or a same-pairings reduction form,
   each independently gated so each can be bit-identity-checked alone):

   * ``REDUCE_NR``   the value-norm fold ``_fold_axis2`` (``tl.sum``'s
     reduction-layout machinery) over the whole (rank, value) tile, every
     token, replaced by ``_split_fold_axis2`` -- the DEFINED-ORDER
     adjacent-pair tree the committed kernel already uses for the small
     (rank, key) tiles (the K10 gate-D note: same pairings, same single IEEE
     adds, bit-identical, no ``tl.sum`` layout machinery).  Each fold step
     here is a size-2 sum -- ONE IEEE add, order-free -- on the same
     pairings, so this lever is bit-identical by construction.
   * ``REDUCE_SUM``  the three axis-0 reductions (the read gather, the
     CORRECT-branch ``old_pre``, the write ``old_row``) replaced by k6's
     ``_fold_axis0`` -- the defined adjacent-pair tree along the rank axis
     via size-2 sums.  The committed form is a full-axis ``tl.sum`` whose
     reduction ORDER is implementation-defined; whether it shares the
     adjacent-pair pairings is EMPIRICAL -- the bit-identity check decides,
     and a mismatch is an honest "cannot preserve bit-exactness" result.
   * ``LAZY_NORMS``  the per-token value-norm/strength chain
     (``nr``/``strength``/``fw``/``sw``/``a_mix``/``b_mix``) MOVED INSIDE
     the scalar ``if merge:`` branch it alone feeds.  Dataflow-identical:
     the chain's values are pure functions of the pre-branch state, and
     nothing outside the merge branch consumes them; the K19 op-mix receipts
     say merges are <= 0.34% of tokens, so the committed kernel computes a
     two-pass 16K-element fold per token to serve a <1% event.  Bit-identity
     still verified empirically.
   * ``LAZY_OLD_ROW`` the write-side ``old_row`` hit-sum MOVED INSIDE the
     ``else`` of ``if erase0`` (it is consumed only there; ``erase0`` is
     true on CORRECT/MERGE/POP -- ~98% of tokens at the registered op mix).
     Dataflow-identical for the same reason; verified empirically.

3. ``ablation brackets`` -- NOT speedups and never claimed as such: the
   faithful copy with ONE pass deleted per bracket (constexpr ``ABL``),
   used only to attribute the committed kernel's measured milliseconds to
   its constituent passes.  Ablated variants change outputs by design;
   their TIMING DELTAS vs the faithful copy are the per-pass cost
   breakdown.  The ablations are chosen so the discrete op mix is UNCHANGED
   (only value-channel passes and I/O are ablated; the decision sim sweep
   is NOT ablated because zeroing it would change the op mix and corrupt
   the attribution).

Usage: this module is driven by ``k21_scan_profile.py`` (the main lane);
it never runs on its own.
"""
from __future__ import annotations

import time

import torch
import triton
import triton.language as tl

from .algebra import APPEND, CORRECT, MERGE, NONE, POP
from .k1_schedule_scan import _add_rn, _mul_rn, _sub_rn
from .k6_sequential_matrix import (_div_rn32, _fold_axis0, _fold_axis2,
                                   _fold_vec, _log2_up, _pow2, _round_f32,
                                   _seq_warps, _sqrt_rn32)
from .k10_separated_keys import _split_fold_axis2
from .k15_composition import LANES, _k15_scan_fwd_op
from .schedule import EPS

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))

# the ablation brackets (attribution instruments, never speedups)
ABL_NONE = 0            # the faithful copy
ABL_READ = 1            # no read sweep: no scores, no read gather
ABL_NORMS = 2           # no per-token value-norm fold (strength chain)
ABL_HITSUMS = 3         # no old_pre / old_row hit-sums
ABL_DECAY = 4           # no per-channel decay tile multiply
ABL_CKPT = 6            # no segment checkpoint stores
ABL_RECORD = 7          # no record-channel stores (schedule + live_pre)
ABLATIONS = (ABL_NONE, ABL_READ, ABL_NORMS, ABL_HITSUMS, ABL_DECAY,
             ABL_CKPT, ABL_RECORD)
ABLATION_NAMES = {
    ABL_NONE: "none (the faithful copy)",
    ABL_READ: "no read sweep (scores + gather; reads stored as zeros)",
    ABL_NORMS: "no per-token value-norm fold (nr/strength := 1)",
    ABL_HITSUMS: "no hit-sums (old_pre/old_row := 0)",
    ABL_DECAY: "no per-channel decay tile multiply",
    ABL_CKPT: "no segment checkpoint stores",
    ABL_RECORD: "no record-channel stores (slot/op/erase/live_after/"
                "payload lanes/live_pre)",
}
# the jit-visible constexpr twins (Triton requires globals referenced
# inside @jit code to be tl.constexpr instances)
_ABL_NONE = tl.constexpr(ABL_NONE)
_ABL_READ = tl.constexpr(ABL_READ)
_ABL_NORMS = tl.constexpr(ABL_NORMS)
_ABL_HITSUMS = tl.constexpr(ABL_HITSUMS)
_ABL_DECAY = tl.constexpr(ABL_DECAY)
_ABL_CKPT = tl.constexpr(ABL_CKPT)
_ABL_RECORD = tl.constexpr(ABL_RECORD)


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k21_scratch_fwd_kernel(
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
    ABL: tl.constexpr, REDUCE_NR: tl.constexpr, REDUCE_SUM: tl.constexpr,
    LAZY_NORMS: tl.constexpr, LAZY_OLD_ROW: tl.constexpr,
):
    """The faithful copy of ``_k15_seq_fwd_kernel`` (same arithmetic, same
    defined-order folds) with the K21 levers gated by constexpr.  At all
    flags zero this MUST reproduce the committed op pair bit for bit -- the
    lane's instrument check; the levers are then bracketed one at a time."""
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
        if ABL != _ABL_RECORD:
            tl.store(live_pre_base + t * H * R + ranks, live.to(tl.int8),
                     mask=keep_r)
        if ABL != _ABL_CKPT:
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
        # strengths -- K6 verbatim; K13: no longer part of the read weights).
        # LAZY_NORMS moves this whole chain inside the merge branch it
        # alone feeds (dataflow-identical; the check decides).
        if LAZY_NORMS == 0:
            if ABL == _ABL_NORMS:
                nr = tl.full((RP,), 1.0, tl.float32)
            else:
                if REDUCE_NR:
                    nr = tl.reshape(_sqrt_rn32(_split_fold_axis2(
                        _mul_rn(acc, acc), VP, LOGV)), (RP,))
                else:
                    nr = tl.reshape(_sqrt_rn32(
                        _fold_axis2(_mul_rn(acc, acc), VP, LOGV)), (RP,))
            strength = _mul_rn(nr, tl.where(live & keep_r, 1.0, 0.0))
        # ---- the ADDITIVE read weights on the KEY channel.
        nk = tl.reshape(_sqrt_rn32(_split_fold_axis2(_mul_rn(kacc, kacc), KP,
                                                    LOGK)), (RP,))
        nqk = tl.reshape(_sqrt_rn32(_fold_vec(_mul_rn(qk, qk), KP, LOGK)), (1,))
        uqk = _div_rn32(qk, tl.broadcast_to(tl.maximum(nqk, _EPS), (KP,)))
        live_r = live & keep_r
        live_any = tl.sum(live_r.to(tl.int32), 0) > 0
        if ABL == _ABL_READ:
            w = tl.zeros((RP,), tl.float32)
        else:
            scores = _div_rn32(
                _mul_rn(tl.reshape(_split_fold_axis2(_mul_rn(uqk[None, :],
                                                            kacc), KP, LOGK),
                                   (RP,)), SCALE),
                tl.maximum(nk, _EPS))
            # K13: the additive read -- the raw score, live-gated.
            w = tl.where(live_r, scores, 0.0)
        # the read: the K6 floor channel (a native rank-axis sum).
        if ABL == _ABL_READ:
            read_o = tl.zeros((VP,), tl.float32)
        elif REDUCE_SUM:
            read_o = tl.reshape(
                _fold_axis0(_mul_rn(w[:, None], acc), RP, LOGR), (VP,))
        else:
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
        # the gated lane 0 (K11B verbatim).
        if LAZY_NORMS == 0:
            fw = tl.maximum(tl.sum(tl.where(ranks == best, strength, 0.0), 0),
                            _EPS)
            sw = tl.maximum(
                tl.sum(tl.where(ranks == second, strength, 0.0), 0), _EPS)
            total = fw + sw
            a_mix = _div_rn32(fw, total)
            b_mix = _div_rn32(sw, total)
        carries = (op0 == _APPEND_CODE) | (op0 == _CORRECT_CODE)
        hit = keep_r & (slot0 == ranks)
        if merge:
            if LAZY_NORMS == 1:
                # the same chain, computed only when its sole consumer runs
                # (the values are pure functions of the same pre-branch
                # state -- bit-identical by dataflow).
                if ABL == _ABL_NORMS:
                    nr = tl.full((RP,), 1.0, tl.float32)
                else:
                    if REDUCE_NR:
                        nr = tl.reshape(_sqrt_rn32(_split_fold_axis2(
                            _mul_rn(acc, acc), VP, LOGV)), (RP,))
                    else:
                        nr = tl.reshape(_sqrt_rn32(
                            _fold_axis2(_mul_rn(acc, acc), VP, LOGV)), (RP,))
                strength = _mul_rn(nr, tl.where(live & keep_r, 1.0, 0.0))
                fw = tl.maximum(
                    tl.sum(tl.where(ranks == best, strength, 0.0), 0), _EPS)
                sw = tl.maximum(
                    tl.sum(tl.where(ranks == second, strength, 0.0), 0),
                    _EPS)
                total = fw + sw
                a_mix = _div_rn32(fw, total)
                b_mix = _div_rn32(sw, total)
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
                    if ABL == _ABL_HITSUMS:
                        old_pre = tl.zeros((VP,), tl.float32)
                    elif REDUCE_SUM:
                        old_pre = tl.reshape(_fold_axis0(
                            tl.where(hit[:, None], acc, 0.0), RP, LOGR),
                            (VP,))
                    else:
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
        if ABL != _ABL_DECAY:
            acc = _mul_rn(acc, tl.broadcast_to(lam[None, :], (RP, VP)))
        if LAZY_OLD_ROW == 0:
            if ABL == _ABL_HITSUMS:
                old_row = tl.zeros((VP,), tl.float32)
            elif REDUCE_SUM:
                old_row = tl.reshape(_fold_axis0(
                    tl.where(hit[:, None], acc, 0.0), RP, LOGR), (VP,))
            else:
                old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)

        # ---- record the decision channel (the 2-lane (b, T, h, L) layout).
        if ABL != _ABL_RECORD:
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
            if LAZY_OLD_ROW == 1:
                # the same sum, computed only on the not-erased path that
                # consumes it (acc is unchanged in between -- bit-identical
                # by dataflow).
                if ABL == _ABL_HITSUMS:
                    old_row = tl.zeros((VP,), tl.float32)
                elif REDUCE_SUM:
                    old_row = tl.reshape(_fold_axis0(
                        tl.where(hit[:, None], acc, 0.0), RP, LOGR), (VP,))
                else:
                    old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)
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


# ---------------------------------------------------------- the launch wrapper

OUTPUT_KEYS = ("reads", "state", "live", "keys", "slot", "op", "erase",
               "live_after", "payload", "live_pre", "ckpt_state",
               "ckpt_live", "ckpt_keys")


def scratch_forward_launch(payload, write_keys, query_keys, decay, gates,
                           state, live, keys, policy_f32, rank,
                           segment_tokens, gated, warps, stages, *,
                           abl=ABL_NONE, reduce_nr=0, reduce_sum=0,
                           lazy_norms=0, lazy_old_row=0):
    """The committed ``_k15_forward_launch``'s own allocation and launch
    pattern (verbatim), pointed at the scratch kernel with the lever flags."""
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
    weights = torch.empty((1,), dtype=torch.float32, device=device)
    tau_match, tau_merge, tau_pop, tie, scale = policy_f32
    _k21_scratch_fwd_kernel[(batch * heads,)](
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
        TIE=tie, SCALE=scale, COLLECT=0,
        GATED=1 if gated else 0,
        STAGES=stages,
        ABL=abl, REDUCE_NR=reduce_nr, REDUCE_SUM=reduce_sum,
        LAZY_NORMS=lazy_norms, LAZY_OLD_ROW=lazy_old_row,
        num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, keys=out_keys,
                slot=slot, op=op, erase=erase, live_after=live_after,
                payload=payload_field, live_pre=live_pre,
                ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                ckpt_keys=ckpt_keys)


# ------------------------------------------------------------- the instruments


def committed_reference(payload, write_keys, query_keys, decay, gates,
                       policy_f32, rank, segment_tokens, gated):
    """The committed opaque op pair's own outputs -- the bit-identity bar."""
    tau_match, tau_merge, tau_pop, tie, scale = policy_f32
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    state = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                        device=device)
    live = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
    keys = torch.zeros(batch, heads, rank, key_dim, dtype=torch.float32,
                       device=device)
    out = _k15_scan_fwd_op(payload, write_keys, query_keys, decay, gates,
                           state, live, keys, int(rank), int(segment_tokens),
                           tau_match, tau_merge, tau_pop, tie, scale,
                           bool(gated))
    return dict(zip(OUTPUT_KEYS, out))


def bit_identity(reference: dict, candidate: dict) -> dict:
    """Per-output-channel ``torch.equal`` -- the receipts bar.  A lever
    counts only if every channel is bit-identical."""
    fields = {}
    all_equal = True
    for name in OUTPUT_KEYS:
        ref, got = reference[name], candidate[name]
        equal = bool(torch.equal(ref, got))
        if ref.is_floating_point():
            diff = float((ref - got).abs().max()) if ref.numel() else 0.0
        else:
            diff = float((ref != got).sum()) if ref.numel() else 0.0
        fields[name] = dict(bit_identical=equal,
                            max_abs_diff_or_mismatch_count=diff)
        all_equal &= equal
    return dict(bit_identical=all_equal, channels=fields)


def time_launch(fn, warmup: int, iters: int) -> list[float]:
    """CUDA-synced per-iteration wall of a zero-arg launch closure (ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.monotonic()
        fn()
        torch.cuda.synchronize()
        samples.append((time.monotonic() - started) * 1000.0)
    return samples
