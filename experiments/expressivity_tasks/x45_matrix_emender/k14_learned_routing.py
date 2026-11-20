"""K14 -- THE LEARNED MANAGEMENT ROUTING ARM: the op routing itself made a
learned policy, on the K10 separated-key lineage (the operator-corrected
program: zero-learned-management was never a mandate -- the op-routing
decision channel is an open lever, and the C5 receipts name it our poorest
surface: 27.7K gate params + three hand-set thresholds vs GDN2's 27.4M
learned management parameters).

Why this file exists
--------------------
The op-set audit (`RESULT_OP_SET_AUDIT.md`) receipts the degeneration: the
threshold chain routes CORRECT to 97.9% of tokens in identity-dominated
rewrites, MERGE sits at ~0.3%, POP at 0.0% -- three hand-set thresholds
(``tau_match`` 0.45, ``tau_merge`` 0.85, ``tau_pop`` 0.10) deciding every
management verb.  This module replaces that FIXED chain with a LEARNED
ROUTER: per-token, input-dependent, a small module from the causal context
producing five per-op scores, DISCRETE argmax at the forward (the recorded
schedule -- decisions stay discrete, the scan stays exact given them),
trained through the K12 straight-through estimator pattern.  Registered in
``PREREGISTRATION_LEARNED_MANAGEMENT.md`` (a DRAFT until the operator signs
it off; the birth construction was supervisor-approved before this engine
was locked).

The semantics (the registration's 1, pre-stated; the supervisor's rulings
cited)
-------------------------------------------------------------------------
1. **The birth anchor (P5, supervisor-approved: branch-active margins + the
   reserved runner-up).**  The threshold chain is a LEXICOGRAPHIC TREE, and
   an argmax over per-op scores can only reproduce a tree if the scores
   ENCODE its branch structure (additive raw margins provably cannot: in the
   no-match region an additive CORRECT score ``d1 - d2`` is unbounded above
   while the chain says APPEND).  The birth score ``M_k`` of op ``k`` is
   therefore built from the chain's OWN branch conditions -- computed
   bit-identically to K10's selection (the same defined-order folds, the
   same ``_round_f32`` thresholds) -- and the chain's own margin deltas:

   ``d1 = best - tau_match``, ``d2 = second - tau_merge``,
   ``d3 = tau_pop - worst`` (single fp32 subs);

   * MERGE (active iff correct & second >= tau_merge): ``M = d1 + d2``
   * CORRECT (active iff correct & not merge):      ``M = d1``
   * APPEND (active iff ~correct & free slots):     ``M = -d1``
   * POP (active iff ~correct & ~appendable & full & worst <= tau_pop &
     any live):                                      ``M = d3``
   * NONE (the else branch):                        ``M = +0.0``
   * every INACTIVE op scores the reserved constant ``-1.0``.

   Exactly one branch is active per token (the chain's own conditions), and
   every active margin is provably ``>= 0`` (``d1 >= 0`` on the correct
   branch, ``-d1 > 0`` on APPEND, ``d3 >= 0`` on POP) while the reserved
   constant is ``< 0``, so the argmax is the chain's winner UNIQUELY on
   every input -- bit-exact birth with NO tie-break reliance (the K12
   ``+0.0`` discipline, one step stronger: the zero-born learned component
   is added with an exact IEEE add to margins that are never ``-0.0``, so
   nothing rides on the argmax's tie order at birth).  The winner's score
   IS its threshold delta -- "score margins born at the threshold deltas",
   literally.
2. **The learned router (P4, supervisor-approved: the per-head
   ``Linear(ctx*width -> 5)``, context-only input).**  The full per-op
   score is ``s_k = M_k + g_k`` where ``g`` is ONE per-head linear module
   per mixer over the DETACHED causal context (the same
   ``ctx*width = 2304``-dim window the key projections see), born at EXACT
   zeros with the K12 ``+0.0`` normalisation (``g = router(x_detached) +
   (+0.0)``; IEEE ``-0.0 + +0.0 = +0.0``) so at birth ``s = M`` bit for
   bit and the FROZEN-router control is the banked winner bit for bit
   FOREVER.  **+553,200 parameters at the registered shape** (24 mixers x
   2 heads x (5*2304 + 5) = 0.37% of the machine, ~2% of GDN2's 27.4M
   management mass) -- stated, asserted by the probe, not hidden.  Deeper
   modules and context+margin inputs are the named follow-ups.
3. **The feasibility masks (supervisor-approved semantics guards, NOT
   learned).**  APPEND is masked to ``-inf`` when no free slot exists and
   POP when no live row does: the chain never picks them there, so birth
   parity is untouched, and a trained router can never APPEND-write into a
   live row (P3) or POP an empty store.  Every other op is mechanically
   well-defined for any token (the kernel's per-op lane arithmetic handles
   dead-row targets exactly as K10's oracle already does).
4. **The recorded schedule doctrine INTACT.**  The router produces the op
   schedule; the scan stays exact given it (the Lean algebra is
   schedule-conditional, so every learned schedule is licensed).  The
   argmax is DISCRETE at the forward and recorded; the per-op scores are a
   RECORDED channel ``(b, T, h, 5)`` (P1) so the backward's estimator needs
   no replay.  The op-derived machinery per op (slot, lanes, blend,
   liveness) is K10's own, driven by the chosen op: CORRECT/MERGE target
   the best row, APPEND the first free slot, POP/NONE record the worst.
5. **The zero-gradient FINDING (the K12 wall, unchanged).**  Under the
   recorded-schedule doctrine the op choice is a backward constant, so the
   STRUCTURAL gradient to the router through the eager oracle is EXACTLY
   ZERO (asserted in the tests).  "The model can learn which op to use"
   REQUIRES an admitted estimator -- the same structural fact the placement
   lane surfaced, now on the op channel.
6. **The admitted estimator (K12-identical, supervisor-approved).**  The
   kernel path's backward is K10's own ``_k10_backward_body`` imported
   VERBATIM (the six value/key gradients UNCHANGED -- the recorded schedule
   fully determines them) plus ONE small torch reduction on the recorded
   scores: with ``p = softmax(s over the five ops)``, ``k*`` the recorded
   op and ``c_t = <dpayload_t, p_t>`` the EXISTING value walk's own
   write-content output (no new walk), the straight-through surrogate is

   ``dL/ds_k = sum_t c_t * p_{k*} * (delta_{kk*} - p_k)``,

   returned as ``d(router_scores)`` so autograd chains it through the
   router module; the score inputs are DETACHED, ONLY the router learns.
   **The estimator's bias, disclosed verbatim (the supervisor's rescope):
   ``c_t`` is zero on MERGE/POP/NONE tokens (the imported backward's
   ``dpayload`` vanishes on those lanes), so the estimator reinforces the
   chosen op where the write content mattered and gives NO direct signal to
   tokens the chain routed to MERGE/POP/NONE.  A null on the
   POP/MERGE-revival prediction is therefore UNINTERPRETABLE as a
   learned-management verdict for those ops -- it receipts only that the
   write-content surrogate does not indirectly flip content-carrying tokens
   into them; a read-path POP/MERGE estimator is the named follow-up.**

The arithmetic spec is K10's verbatim (defined-order folds, fp32
decision/read arithmetic, correctly-rounded IEEE ops, libdevice ``exp``,
``mul.rn``/``add.rn``/``sub.rn`` -- no FMA contraction), extended by the
router's five scalar subs/adds per token (single correctly-rounded ops, so
oracle and kernel are bit-equal by construction on the new channel).

The kernels
------------
``_k14_seq_fwd_kernel``   K10's persistent whole-window walk with the
                          selection chain replaced by the score/argmax
                          block: per token the K10 chain quantities
                          bit-identically (the birth anchor), the five
                          birth margins, the router's learned scores
                          (loaded per token), the feasibility masks, the
                          discrete argmax (first-max, op-code order), then
                          K10's own per-op machinery driven by the chosen
                          op.  Records K10's decision channel shapes PLUS
                          the per-op scores (always -- the backward's
                          recorded input) and the chain's own op (the
                          birth anchor's flip-rate instrument, collect
                          only).

The backward's SIX value/key gradients are K10's own
(``_k10_backward_body`` imported VERBATIM).  The estimator is a small torch
reduction on recorded tensors -- bit-exact by construction, gated by an
fp64 autograd reference of the surrogate objective.  The wiring is ONE
opaque ``torch.library`` op pair per (layer, window) scan
(``x45::k14_scan_fwd``/``x45::k14_scan_bwd``, the K4/K6/K10/K12 hosting
pattern) with the same refusals by name.

Doctrine, unchanged: no ``k10_*``/`k11_*``/`k12_*`` source is edited; K10's
spec helpers, its backward body, its selection quantities and its
mixer/conversion are IMPORTED.  The K11 lane (cards 0/1) and the K12
sources are untouched by this file.
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
from .k1_schedule_scan import _add_rn, _mul_rn, _sub_rn
from .k10_separated_keys import (
    DEFAULT_CONTEXT, DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS, MIMETIC_SEED,
    SeparatedSequentialMixer, _check_separated_shapes, _k10_backward_body,
    _split_fold_axis2, convert_model_to_separated_keys, mimetic_key_family,
    separated_available)
from .k6_sequential_matrix import (_div_rn32, _fold_axis2, _fold_vec, _fp32,
                                    _log2_up, _norm_defined, _pow2,
                                    _round_f32, _seq_warps, _sqrt_rn32,
                                    tree_fold)
from .key_separation_probe import causal_context
from .model import DECAY_FLOOR, DECAY_HORIZON_TOKENS
from .scan import RoutedResult, ScanStats
from .schedule import EPS, RoutingPolicy

__all__ = ["DEFAULT_KEY_DIM", "DEFAULT_CONTEXT", "DEFAULT_SEGMENT_TOKENS",
           "MIMETIC_SEED", "RoutedResult", "ScanStats", "MatrixSchedule",
           "RoutingPolicy", "EPS", "ROUTER_OPS", "RESERVED_SCORE",
           "RoutingSeparatedMixer", "convert_model_to_routing",
           "convert_model_to_separated_keys", "routing_available",
           "routing_oracle_scan", "routing_scan", "routing_parameter_delta",
           "separated_available", "_k14_forward_launch",
           "_k14_backward_body"]

# The five op scores, indexed BY OP CODE (NONE, APPEND, CORRECT, POP, MERGE
# = 0..4): the argmax's first-max tie-break is therefore the op-code order,
# and the scores tensor's last axis is (b, T, h, 5) in that order.
ROUTER_OPS = (NONE, APPEND, CORRECT, POP, MERGE)
# The reserved runner-up constant (supervisor-approved): every INACTIVE
# branch's birth score.  Any value < 0 preserves the birth exactness (every
# active margin is provably >= 0); this value sets the post-birth flip
# barrier (margin_winner + 1.0) and the STE softmax's temperature at birth.
RESERVED_SCORE = -1.0


def routing_available(rank: int, value_dim: int, key_dim: int,
                      value_limit: int = 16384,
                      key_limit: int = 8192) -> bool:
    """The persistent walk holds the whole store and its key table (K10's
    own gate); the router's five per-token scalars ride in registers."""
    return separated_available(rank, value_dim, key_dim, value_limit,
                               key_limit)


# ------------------------------------------------------------- the oracle


def _routing_branches(best_value, second_value, worst_value, free_count,
                      any_live, tau_match, tau_merge, tau_pop):
    """The threshold chain's OWN branch conditions, verbatim from K10's
    ``_key_selection`` (the birth anchor's single source of truth on the
    oracle side; the kernel inlines the same expressions).  Exactly one
    branch is active per token."""
    correct = best_value >= tau_match
    merge = correct & (second_value >= tau_merge)
    need = ~correct
    appendable = need & (free_count > 0)
    full = free_count <= 0
    pop = (~correct) & (~appendable) & full & (worst_value <= tau_pop) \
        & any_live
    none = ~(correct | appendable | pop)
    return dict(correct=correct, merge=merge, appendable=appendable, pop=pop,
                none=none)


def _birth_scores(branches, d1, d2, d3):
    """The branch-active margin encoding (the registration's 1.1): the ACTIVE
    branch's op scores its own threshold delta, every inactive op the
    reserved constant.  Returns a ``(..., 5)`` fp32 tensor in op-code order
    (NONE, APPEND, CORRECT, POP, MERGE)."""
    shape = d1.shape + (5,)
    scores = torch.full(shape, RESERVED_SCORE, dtype=torch.float32,
                        device=d1.device)
    scores[..., NONE] = torch.where(branches["none"], 0.0, RESERVED_SCORE)
    scores[..., APPEND] = torch.where(branches["appendable"], -d1,
                                      RESERVED_SCORE)
    scores[..., CORRECT] = torch.where(branches["correct"]
                                       & (~branches["merge"]), d1,
                                       RESERVED_SCORE)
    scores[..., POP] = torch.where(branches["pop"], d3, RESERVED_SCORE)
    scores[..., MERGE] = torch.where(branches["merge"], d1 + d2,
                                     RESERVED_SCORE)
    return scores


def _routing_decision(sim, live, router_scores, tau_match, tau_merge, tau_pop,
                      tie, rank):
    """One token's LEARNED routing on the key-channel similarity (the oracle
    side of the kernel's score/argmax block).  ``sim`` is the DETACHED
    routing similarity with dead rows at -2; ``router_scores`` is this
    token's ``g`` ``(b, h, 5)`` fp32 (already detached by the caller).
    Everything is fp32 and correctly-rounded, so the kernel is bit-equal by
    construction (the K10 parity discipline extended to the new channel)."""
    penalty = torch.arange(rank, dtype=torch.float32,
                           device=sim.device) * tie
    score = sim - penalty
    best_index = score.argmax(-1)
    best_value = sim.gather(-1, best_index.unsqueeze(-1)).squeeze(-1)
    second_score = score.scatter(-1, best_index.unsqueeze(-1), -3.0)
    second_index = second_score.argmax(-1)
    second_value = sim.gather(-1, second_index.unsqueeze(-1)).squeeze(-1)
    worst_index = score.argmin(-1)
    worst_value = sim.gather(-1, worst_index.unsqueeze(-1)).squeeze(-1)
    free = ~live
    free_count = free.sum(-1)
    any_live = live.any(-1)
    free_slot = free.to(torch.int64).argmax(-1)     # first free index
    branches = _routing_branches(best_value, second_value, worst_value,
                                 free_count, any_live, tau_match, tau_merge,
                                 tau_pop)
    d1 = best_value - tau_match                     # single fp32 sub
    d2 = second_value - tau_merge
    d3 = tau_pop - worst_value
    s = _birth_scores(branches, d1, d2, d3) + router_scores
    # the feasibility masks (semantics guards, not learned): APPEND needs a
    # free slot, POP a live row.  Birth-neutral: the chain never picks them
    # there, so the masks only define the post-deviation support.
    s[..., APPEND] = torch.where(free_count > 0, s[..., APPEND],
                                 torch.full_like(s[..., APPEND],
                                                 float("-inf")))
    s[..., POP] = torch.where(any_live, s[..., POP],
                              torch.full_like(s[..., POP], float("-inf")))
    op0 = s.argmax(-1)                              # first max wins
    merge = op0 == MERGE
    correct_family = (op0 == CORRECT) | merge
    is_append = op0 == APPEND
    slot0 = torch.where(correct_family, best_index,
                        torch.where(is_append, free_slot, worst_index))
    erase0 = (op0 == CORRECT) | merge | (op0 == POP)
    live0 = op0 != POP
    chain_op0 = torch.where(branches["correct"],
                            torch.where(branches["merge"],
                                        torch.full_like(op0, MERGE),
                                        torch.full_like(op0, CORRECT)),
                            torch.where(branches["appendable"],
                                        torch.full_like(op0, APPEND),
                                        torch.where(branches["pop"],
                                                    torch.full_like(op0, POP),
                                                    torch.full_like(op0,
                                                                    NONE))))
    return dict(best_index=best_index, second_index=second_index,
                worst_index=worst_index, op0=op0, slot0=slot0,
                erase0=erase0, live0=live0, chain_op0=chain_op0,
                scores=s, branches=branches,
                carries=is_append | (op0 == CORRECT))


def routing_oracle_scan(payload, write_keys, query_keys, decay, router_scores,
                        *, policy: RoutingPolicy, rank=None, rank_max=None,
                        initial_state=None, initial_live=None,
                        initial_keys=None, collect=True):
    """The learned-routing arm's eager per-token oracle: K10's oracle with
    the selection chain replaced by the score/argmax block.  The parity fixed
    point; differentiable through the value/key channels exactly as K10's,
    and HONEST about the router: the op choice is discrete, so the structural
    gradient to the router through this oracle is EXACTLY ZERO (the finding;
    the kernel path's backward carries the admitted straight-through
    estimator instead -- the registration's disclosure).
    """
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    if write_keys.shape != query_keys.shape:
        raise ValueError("write_keys and query_keys disagree")
    if write_keys.shape[:-1] != payload.shape[:-1]:
        raise ValueError("keys must share the payload's (batch, tokens, heads)")
    if router_scores.shape != (batch, tokens, heads, 5):
        raise ValueError(f"router_scores must be (batch, tokens, heads, 5) = "
                         f"{(batch, tokens, heads, 5)}; got "
                         f"{tuple(router_scores.shape)}")
    if router_scores.dtype != torch.float32 or not router_scores.is_contiguous():
        raise ValueError("router_scores must be a contiguous fp32 tensor")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the learned "
                          "routing registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the learned routing "
                          "registration: it is a chunk-boundary concept")
    if rank is None:
        rank = value_dim
    if rank_max is None:
        rank_max = int(rank)
    if int(rank) != int(rank_max):
        raise ValueError("the learned routing registration requires rank == "
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
    RP, VP, KP = _pow2(rank), _pow2(value_dim), _pow2(key_dim)
    tau_match, tau_merge, tau_pop = (_round_f32(policy.tau_match),
                                     _round_f32(policy.tau_merge),
                                     _round_f32(policy.tau_pop))
    tie = _round_f32(policy.tie_break)
    scale = _round_f32(policy.read_scale)
    lanes = 2 if policy.ops else 1
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
    scores_out = (torch.zeros(batch, tokens, heads, 5, dtype=torch.float32,
                              device=device) if collect else None)
    chain_op_out = (torch.zeros(batch, tokens, heads, dtype=torch.int64,
                                device=device) if collect else None)
    b_index = torch.arange(batch, device=device)[:, None]
    h_index = torch.arange(heads, device=device)[None, :]
    for t in range(tokens):
        if collect:
            live_pre_out[:, t] = live
        p_t, wk_t, qk_t = p32[:, t], wk32[:, t], qk32[:, t]
        lam_t = d32[:, t]                                     # (b, h)
        # ---- the read: K10 verbatim (certified weights on the key channel).
        nr = _norm_defined(state, -1, VP)                      # (b, h, rank)
        nk = _norm_defined(keys, -1, KP)                       # (b, h, rank)
        nqk = _norm_defined(qk_t, -1, KP)                       # (b, h)
        uqk = qk_t / nqk.clamp_min(EPS).unsqueeze(-1)
        scores = (scale * tree_fold(uqk.unsqueeze(2) * keys, -1, KP)) \
            / nk.clamp_min(EPS)
        live_f = live.to(compute_dtype)
        any_live = live.any(-1, keepdim=True)
        masked = scores.masked_fill(~live, float("-inf"))
        safe = torch.where(any_live, masked, torch.zeros_like(masked))
        pmax = safe.max(-1).values
        expo = torch.exp(safe - pmax.unsqueeze(-1))
        denom = tree_fold(expo * live_f, -1, RP)               # (b, h)
        any_live_flat = live.any(-1)
        denom_safe = torch.where(any_live_flat, denom, torch.ones_like(denom))
        probs = torch.where(any_live, expo / denom_safe.unsqueeze(-1),
                            torch.zeros_like(expo))
        w = probs * nr * live_f
        reads[:, t] = tree_fold(w.unsqueeze(-1) * state, -2, RP)
        if collect:
            weights_out[:, t] = w
        # ---- the decisions: the LEARNED ROUTER on the DETACHED key channel.
        # The chain quantities are K10's own (the birth anchor); the op is
        # the argmax over the five per-op scores (the registration's 1.1-1.3).
        keys_det = keys.detach()
        nkd = _norm_defined(keys_det, -1, KP)
        wk_det = wk_t.detach()
        uk = wk_det / _norm_defined(wk_det, -1, KP).clamp_min(EPS).unsqueeze(-1)
        sim = tree_fold(uk.unsqueeze(2) * keys_det, -1, KP) \
            / nkd.clamp_min(EPS)                               # (b, h, rank)
        sim = sim.masked_fill(~live, -2.0)
        sel = _routing_decision(sim, live, router_scores[:, t].detach(),
                                tau_match, tau_merge, tau_pop, tie, rank)
        if collect:
            scores_out[:, t] = sel["scores"]
            chain_op_out[:, t] = sel["chain_op0"]
        # ---- the strength and the payload lane: K10 verbatim.
        strength = _norm_defined(state.detach(), -1, VP) * live.to(compute_dtype)
        blend = None
        if bool(sel["op0"].eq(MERGE).any()):
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
        lane0 = torch.where(sel["carries"].unsqueeze(-1), p_t,
                            torch.zeros_like(p_t))
        if blend is not None:
            lane0 = torch.where(sel["op0"].eq(MERGE).unsqueeze(-1), blend,
                                lane0)
        # ---- the key lane: K10 verbatim.
        key_lane = torch.where(sel["op0"].eq(POP).unsqueeze(-1),
                               torch.zeros_like(wk_t), wk_t)
        fields["slot"][:, t, :, 0] = sel["slot0"]
        fields["slot"][:, t, :, 1] = sel["second_index"]
        fields["op"][:, t, :, 0] = sel["op0"]
        fields["op"][:, t, :, 1] = torch.where(sel["op0"].eq(MERGE),
                                               torch.full_like(sel["op0"], MERGE),
                                               torch.full_like(sel["op0"], NONE))
        fields["erase"][:, t, :, 0] = sel["erase0"]
        fields["erase"][:, t, :, 1] = sel["op0"].eq(MERGE)
        fields["live_after"][:, t, :, 0] = sel["live0"]
        fields["live_after"][:, t, :, 1] = False
        fields["payload"][:, t, :, 0, :] = lane0
        # ---- apply: DECAY (values only), then the lanes in the registered order.
        state = state * lam_t.unsqueeze(-1).unsqueeze(-1)
        rows_at = state[b_index.expand(batch, heads),
                        h_index.expand(batch, heads), sel["slot0"]]
        written = torch.where(sel["erase0"].unsqueeze(-1), lane0,
                              rows_at + lane0)
        state = state.index_put((b_index.expand(batch, heads),
                                h_index.expand(batch, heads),
                                sel["slot0"]), written)
        keys_at = keys[b_index.expand(batch, heads),
                       h_index.expand(batch, heads), sel["slot0"]]
        keys = keys.index_put(
            (b_index.expand(batch, heads), h_index.expand(batch, heads),
             sel["slot0"]),
            torch.where(sel["op0"].ne(NONE).unsqueeze(-1), key_lane, keys_at))
        live = live.index_put((b_index.expand(batch, heads),
                               h_index.expand(batch, heads), sel["slot0"]),
                              torch.where(sel["op0"].ne(NONE), sel["live0"],
                                          live.gather(-1,
                                                      sel["slot0"].unsqueeze(-1)
                                                      ).squeeze(-1)))
        if bool(sel["op0"].eq(MERGE).any()):
            # masked per (b,h): only a MERGING lane's source row dies (the
            # K12-fixed form; the K10 oracle's latent multi-lane bug is not
            # copied -- see the registration's pre-stated finding).
            keys_at2 = keys[b_index.expand(batch, heads),
                            h_index.expand(batch, heads), sel["second_index"]]
            rows2 = state[b_index.expand(batch, heads),
                          h_index.expand(batch, heads), sel["second_index"]]
            live2 = live.gather(-1, sel["second_index"].unsqueeze(-1)
                                ).squeeze(-1)
            state = state.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads), sel["second_index"]),
                torch.where(sel["op0"].eq(MERGE).unsqueeze(-1),
                            torch.zeros(batch, heads, value_dim,
                                        dtype=compute_dtype, device=device),
                            rows2))
            keys = keys.index_put(
                (b_index.expand(batch, heads), h_index.expand(batch, heads),
                 sel["second_index"]),
                torch.where(sel["op0"].eq(MERGE).unsqueeze(-1),
                            torch.zeros(batch, heads, key_dim,
                                        dtype=compute_dtype, device=device),
                            keys_at2))
            live = live.index_put(
                (b_index.expand(batch, heads),
                 h_index.expand(batch, heads), sel["second_index"]),
                torch.where(sel["op0"].eq(MERGE),
                            torch.zeros(batch, heads, dtype=torch.bool,
                                        device=live.device), live2))
    schedule = MatrixSchedule(**fields)
    return dict(reads=reads, state=state, live=live, keys=keys,
                schedule=schedule, weights=weights_out, live_pre=live_pre_out,
                scores=scores_out, chain_op=chain_op_out, rank=int(rank))


# ------------------------------------------------------------- the forward kernel

_EPS = tl.constexpr(EPS)
_BIG_INDEX = tl.constexpr(1 << 30)
_NONE_CODE = tl.constexpr(int(NONE))
_APPEND_CODE = tl.constexpr(int(APPEND))
_CORRECT_CODE = tl.constexpr(int(CORRECT))
_POP_CODE = tl.constexpr(int(POP))
_MERGE_CODE = tl.constexpr(int(MERGE))
_RESERVED = tl.constexpr(RESERVED_SCORE)


@triton.jit(do_not_specialize=['T', 'SEG'])
def _k14_seq_fwd_kernel(
    payload, write_keys, query_keys, decay, router_scores,
    state_in, live_in, keys_in,
    reads, state_out, live_out, keys_out,
    slot_out, op_out, erase_out, live_after_out, payload_out, live_pre,
    ckpt_state, ckpt_live, ckpt_keys, weights_out,
    op_scores_out, chain_op_out,
    T, R, ROWS, SEG, KD,
    pbs, pts, phs, pvs,
    wbs, wts, whs, wks,
    qbs, qts, qhs, qks,
    dbs, dts, dhs,
    rbs, rts, rhs,
    H: tl.constexpr, V: tl.constexpr, VP: tl.constexpr, KP: tl.constexpr,
    RP: tl.constexpr, L: tl.constexpr,
    LOGV: tl.constexpr, LOGR: tl.constexpr, LOGK: tl.constexpr,
    TAU_MATCH: tl.constexpr, TAU_MERGE: tl.constexpr, TAU_POP: tl.constexpr,
    TIE: tl.constexpr, SCALE: tl.constexpr, COLLECT: tl.constexpr,
    STAGES: tl.constexpr,
):
    """K10's whole-window walk with the selection chain replaced by the
    learned router: per token the K10 chain quantities bit-identically (the
    birth anchor), the five branch-active margin scores plus the router's
    learned per-op scores, the feasibility masks, the discrete argmax
    (first-max, op-code order), then K10's own per-op machinery driven by the
    chosen op.  The per-op scores are ALWAYS recorded (the backward's
    estimator input); the chain's own op is collect-only (the flip-rate
    instrument)."""
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

    acc = tl.load(state_in + row.to(tl.int64) * R * V
                  + ranks[:, None] * V + values[None, :], mask=keep, other=0.0)
    kacc = tl.load(keys_in + row.to(tl.int64) * R * KD
                   + ranks[:, None] * KD + kdims[None, :], mask=keep_r[:, None]
                   & keep_k[None, :], other=0.0)
    live = tl.load(live_in + row * R + ranks, mask=keep_r, other=0) != 0

    payload_row = payload + batch_index * pbs + head_index * phs
    wk_row = write_keys + batch_index * wbs + head_index * whs
    qk_row = query_keys + batch_index * qbs + head_index * qhs
    decay_row = decay + batch_index * dbs + head_index * dhs
    router_row = router_scores + batch_index.to(tl.int64) * rbs \
        + head_index * rhs
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
    scores_base = (op_scores_out + batch_index.to(tl.int64) * T * H * 5
                   + head_index * 5)
    chain_base = (chain_op_out + batch_index.to(tl.int64) * T * H
                  + head_index)

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
                     + kdims[None, :], kacc,
                     mask=keep_r[:, None] & keep_k[None, :])
        p = tl.load(payload_row + t * pts + values * pvs, mask=keep_v,
                    other=0.0).to(tl.float32)
        wk = tl.load(wk_row + t * wts + kdims * wks, mask=keep_k,
                     other=0.0).to(tl.float32)
        qk = tl.load(qk_row + t * qts + kdims * qks, mask=keep_k,
                     other=0.0).to(tl.float32)
        lam = tl.load(decay_row + t * dts).to(tl.float32)

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

        # ---- the decisions: K10's chain quantities bit-identically, then the
        # LEARNED ROUTER's per-op scores and argmax (the registration's
        # 1.1-1.3): the ACTIVE branch's op scores its own threshold-delta
        # margin, every inactive op the reserved -1.0, plus the router's
        # learned per-op score; the feasibility masks are semantics guards.
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
        free = keep_r & (~live)
        free_count = tl.sum(free.to(tl.int32), 0)
        free_found = tl.min(tl.where(free, ranks, _BIG_INDEX), 0)
        free_slot = tl.where(free_found < _BIG_INDEX, free_found, 0)
        # the chain's OWN branch conditions (the birth anchor; K10's
        # selection expressions verbatim).
        correct = best_value >= TAU_MATCH
        merge_c = correct & (second_value >= TAU_MERGE)
        need = ~correct
        appendable = need & (free_count > 0)
        full = free_count <= 0
        pop_c = (~correct) & (~appendable) & full & (worst_value <= TAU_POP) \
            & live_any
        none_c = ~(correct | appendable | pop_c)
        # the margin deltas (single correctly-rounded fp32 ops).
        d1 = _sub_rn(best_value, TAU_MATCH)
        d2 = _sub_rn(second_value, TAU_MERGE)
        d3 = _sub_rn(TAU_POP, worst_value)
        # the router's learned per-op scores (detached input, fp32).
        g_none = tl.load(router_row + t * rts + 0)
        g_append = tl.load(router_row + t * rts + 1)
        g_correct = tl.load(router_row + t * rts + 2)
        g_pop = tl.load(router_row + t * rts + 3)
        g_merge = tl.load(router_row + t * rts + 4)
        # the branch-active birth margins plus g (the +0.0-born router leaves
        # s = M bit for bit), then the feasibility masks.
        s0 = _add_rn(tl.where(none_c, 0.0, _RESERVED), g_none)
        s1 = _add_rn(tl.where(appendable, -d1, _RESERVED), g_append)
        s2 = _add_rn(tl.where(correct & (~merge_c), d1, _RESERVED), g_correct)
        s3 = _add_rn(tl.where(pop_c, d3, _RESERVED), g_pop)
        s4 = _add_rn(tl.where(merge_c, _add_rn(d1, d2), _RESERVED), g_merge)
        s1 = tl.where(free_count > 0, s1, float('-inf'))
        s3 = tl.where(live_any, s3, float('-inf'))
        smax = tl.maximum(tl.maximum(tl.maximum(tl.maximum(s0, s1), s2), s3),
                          s4)
        op0 = tl.where(s0 == smax, _NONE_CODE,
                       tl.where(s1 == smax, _APPEND_CODE,
                                tl.where(s2 == smax, _CORRECT_CODE,
                                         tl.where(s3 == smax, _POP_CODE,
                                                  _MERGE_CODE))))
        # the chain's own op (the birth anchor's flip-rate instrument).
        chain_op0 = tl.where(correct,
                             tl.where(merge_c, _MERGE_CODE, _CORRECT_CODE),
                             tl.where(appendable, _APPEND_CODE,
                                      tl.where(pop_c, _POP_CODE,
                                               _NONE_CODE)))
        tl.store(scores_base + t * H * 5 + 0, s0)
        tl.store(scores_base + t * H * 5 + 1, s1)
        tl.store(scores_base + t * H * 5 + 2, s2)
        tl.store(scores_base + t * H * 5 + 3, s3)
        tl.store(scores_base + t * H * 5 + 4, s4)
        if COLLECT:
            tl.store(chain_base + t * H, chain_op0.to(tl.int64))
        # the op-derived machinery (K10's own, driven by the chosen op).
        merge = op0 == _MERGE_CODE
        correct_family = (op0 == _CORRECT_CODE) | merge
        is_append = op0 == _APPEND_CODE
        slot0 = tl.where(correct_family, best,
                         tl.where(is_append, free_slot, worst))
        erase0 = (op0 == _CORRECT_CODE) | merge | (op0 == _POP_CODE)
        live0 = op0 != _POP_CODE
        # ---- the MERGE blend from the CURRENT value rows (K6 verbatim).
        fw = tl.maximum(tl.sum(tl.where(ranks == best, strength, 0.0), 0),
                        _EPS)
        sw = tl.maximum(tl.sum(tl.where(ranks == second, strength, 0.0), 0),
                        _EPS)
        total = fw + sw
        a_mix = _div_rn32(fw, total)
        b_mix = _div_rn32(sw, total)
        carries = is_append | (op0 == _CORRECT_CODE)
        if merge:
            coef = tl.where(ranks == best, a_mix,
                           tl.where(ranks == second, b_mix, 0.0))
            blend = tl.sum(_mul_rn(coef[:, None], acc), 0)
            lane0 = blend
        else:
            lane0 = tl.where(carries, p, 0.0)

        # ---- record the decision channel (the scan's (b, T, h, lanes) layout).
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

        # ---- apply: DECAY (values only), then the value lanes in order (K6
        # verbatim), then the KEY lane.
        acc = _mul_rn(acc, lam)
        hit = keep_r & (slot0 == ranks)
        if erase0:
            new_row = lane0
        else:
            old_row = tl.sum(tl.where(hit[:, None], acc, 0.0), 0)
            new_row = _add_rn(old_row, lane0)
        acc = tl.where(hit[:, None], new_row[None, :], acc)
        # NONE touches nothing: the liveness update applies only to an op
        # that writes (K10's own ``tl.where(hit, live0, live)`` relies on the
        # chain's guarantee that a NONE always targets a LIVE row on a FULL
        # store; the learned router can pick NONE with free slots, when the
        # recorded target may be dead -- the masked form is the oracle's
        # own semantics and bit-identical at birth).
        live = tl.where(hit & (op0 != _NONE_CODE), live0, live)
        hit2 = merge & keep_r & (second == ranks)
        acc = tl.where(hit2[:, None], 0.0, acc)
        live = tl.where(hit2, False, live)
        # the key lane: every op but NONE re-files the target row (POP clears
        # the address); MERGE zeroes the source row's key with its value.
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
             + kdims[None, :], kacc,
             mask=keep_r[:, None] & keep_k[None, :])


# ------------------------------------------------------------ the forward wrapper


def _check_routing_shapes(payload, write_keys, query_keys, decay,
                          router_scores, state, live, keys, rank,
                          segment_tokens):
    """Refuse anything the routing path cannot express, before any launch."""
    if not (payload.is_cuda and write_keys.is_cuda and query_keys.is_cuda
            and decay.is_cuda and router_scores.is_cuda):
        raise ValueError("the routing scan is a device kernel; inputs must "
                         "be on CUDA")
    batch, tokens, heads, value_dim = payload.shape
    if router_scores.shape != (batch, tokens, heads, 5):
        raise ValueError(f"router_scores {tuple(router_scores.shape)} is not "
                         f"(batch, tokens, heads, 5) = "
                         f"{(batch, tokens, heads, 5)}")
    if router_scores.dtype != torch.float32 \
            or not router_scores.is_contiguous():
        raise ValueError("router_scores must be a contiguous fp32 tensor")
    # K10's own checks carry the rest (state/keys/live/key shapes).
    _check_separated_shapes(payload, write_keys, query_keys, decay, state,
                            live, keys, rank, segment_tokens)


def _k14_forward_launch(payload, write_keys, query_keys, decay, router_scores,
                        state, live, keys, policy_f32, rank, segment_tokens,
                        collect, warps, stages):
    """The persistent walk: one launch, every output allocated contiguous."""
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    lanes = 2
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
    op_scores = torch.empty((batch, tokens, heads, 5), dtype=torch.float32,
                            device=device)
    chain_op = (torch.empty((batch, tokens, heads), dtype=torch.int64,
                            device=device) if collect
               else torch.empty((1,), dtype=torch.int64, device=device))
    tau_match, tau_merge, tau_pop, tie, scale = policy_f32
    handle = _k14_seq_fwd_kernel[(batch * heads,)](
        payload, write_keys, query_keys, decay, router_scores,
        state, live.view(torch.int8), keys,
        reads, out_state, out_live.view(torch.int8), out_keys,
        slot, op, erase.view(torch.int8), live_after.view(torch.int8),
        payload_field, live_pre.view(torch.int8),
        ckpt_state, ckpt_live.view(torch.int8), ckpt_keys, weights,
        op_scores, chain_op,
        tokens, rank, batch * heads, int(segment_tokens), key_dim,
        *payload.stride(), *write_keys.stride(), *query_keys.stride(),
        *decay.stride(),
        router_scores.stride(0), router_scores.stride(1),
        router_scores.stride(2),
        H=heads, V=value_dim, VP=_pow2(value_dim), KP=_pow2(key_dim),
        RP=_pow2(rank), L=lanes,
        LOGV=_log2_up(_pow2(value_dim)), LOGR=_log2_up(_pow2(rank)),
        LOGK=_log2_up(_pow2(key_dim)),
        TAU_MATCH=tau_match, TAU_MERGE=tau_merge, TAU_POP=tau_pop,
        TIE=tie, SCALE=scale, COLLECT=1 if collect else 0,
        STAGES=stages, num_warps=warps)
    return dict(reads=reads, state=out_state, live=out_live, keys=out_keys,
                slot=slot, op=op, erase=erase, live_after=live_after,
                payload=payload_field, live_pre=live_pre,
                ckpt_state=ckpt_state, ckpt_live=ckpt_live,
                ckpt_keys=ckpt_keys, weights=weights, op_scores=op_scores,
                chain_op=chain_op, kernel=handle)


# ---------------------------------------------- the estimator's backward reduction


def _router_estimator(payload, dpayload, op_scores, op0):
    """The admitted straight-through estimator (the registration's 1.6, the
    K12 pattern): with ``p = softmax(s over the five recorded op scores)``,
    ``k*`` the recorded op and ``c_t = <dpayload_t, p_t>`` the imported value
    walk's own write-content output (no new walk),

    ``dL/ds_k = c_t * p_{k*} * (delta_{kk*} - p_k)``

    returned as ``d(router_scores)`` so autograd chains it through the
    router module.  Everything is a reduction over RECORDED tensors -- no
    replay, bit-exact by construction; the inputs are detached by the
    doctrine, so only the router's parameters learn.  ``c_t`` is zero on
    MERGE/POP/NONE tokens (the imported backward's ``dpayload`` vanishes on
    those lanes): the disclosed bias, stated verbatim in the registration."""
    c = (dpayload.to(torch.float32) * payload.to(torch.float32)).sum(-1)
    n = c.reshape(-1)
    probs = torch.softmax(op_scores.reshape(-1, 5), dim=-1)
    kstar = op0.reshape(-1).to(torch.int64)
    rows = torch.arange(n.shape[0], device=op_scores.device)
    pk = probs[rows, kstar]
    onehot = torch.zeros_like(probs)
    onehot[rows, kstar] = 1.0
    ds = pk.unsqueeze(-1) * (onehot - probs)
    ds = n.unsqueeze(-1) * ds
    return ds.view(op_scores.shape), c


# ------------------------------------------------- the backward body


def _k14_backward_body(payload, write_keys, query_keys, decay, dreads,
                       schedule, live_pre, ckpt_state, ckpt_live, ckpt_keys,
                       op_scores, rank, segment_tokens, scale):
    """K10's six gradients VERBATIM (the recorded schedule fully determines
    them; ``_k10_backward_body`` is imported, never edited) PLUS the router's
    admitted straight-through estimator ``d(router_scores)`` -- the
    registration's disclosed surrogate, the only channel whose gradient is
    not the true adjoint."""
    six = _k10_backward_body(payload, write_keys, query_keys, decay, dreads,
                             schedule, live_pre, ckpt_state, ckpt_live,
                             ckpt_keys, int(rank), int(segment_tokens),
                             float(scale))
    dpayload, dwrite_keys, dquery_keys, ddecay, dstate, dkeys = six
    d_router, _coeff = _router_estimator(payload, dpayload, op_scores,
                                         schedule.op[..., 0])
    return (dpayload, dwrite_keys, dquery_keys, ddecay, d_router, dstate,
            dkeys)


# ------------------------------------------------------------ the op pair


@torch.library.custom_op("x45::k14_scan_fwd", mutates_args=())
def _k14_scan_fwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor,
    query_keys: torch.Tensor, decay: torch.Tensor,
    router_scores: torch.Tensor, state: torch.Tensor, live: torch.Tensor,
    keys: torch.Tensor, rank: int, segment_tokens: int, tau_match: float,
    tau_merge: float, tau_pop: float, tie_break: float, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """The whole (layer, window) routing scan as ONE opaque op: the
    persistent walk's every K10 channel as its outputs PLUS the recorded
    per-op scores (the backward's estimator input), so autograd itself keeps
    the backward's saves alive."""
    _check_routing_shapes(payload, write_keys, query_keys, decay,
                          router_scores, state, live, keys, int(rank),
                          int(segment_tokens))
    pol32 = (_round_f32(tau_match), _round_f32(tau_merge), _round_f32(tau_pop),
             _round_f32(tie_break), _round_f32(scale))
    out = _k14_forward_launch(payload, write_keys, query_keys, decay,
                              router_scores, state, live, keys, pol32,
                              int(rank), int(segment_tokens), collect=False,
                              warps=_seq_warps(int(rank), payload.shape[-1]),
                              stages=3)
    return (out["reads"], out["state"], out["live"], out["keys"], out["slot"],
            out["op"], out["erase"], out["live_after"], out["payload"],
            out["live_pre"], out["ckpt_state"], out["ckpt_live"],
            out["ckpt_keys"], out["op_scores"])


@_k14_scan_fwd_op.register_fake
def _(payload, write_keys, query_keys, decay, router_scores, state, live,
      keys, rank, segment_tokens, tau_match, tau_merge, tau_pop, tie_break,
      scale):
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    device = payload.device
    nseg = -(-tokens // int(segment_tokens))
    shape = (batch, tokens, heads, 2)
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
                        dtype=torch.float32, device=device),
            torch.empty((batch, tokens, heads, 5), dtype=torch.float32,
                        device=device))


def _k14_scan_setup(ctx, inputs, output):
    payload, write_keys, query_keys, decay, router_scores, state, live, \
        keys, rank, segment_tokens, tau_match, tau_merge, tau_pop, \
        tie_break, scale = inputs
    # the backward needs the inputs' values and the walk's own recorded
    # channels; the router's LEARNED scores are not needed as VALUES (the
    # estimator consumes the RECORDED per-op scores), only their gradient
    # is returned.
    ctx.save_for_backward(payload, write_keys, query_keys, decay, state,
                          live, keys, *output[4:])
    ctx.rank, ctx.segment_tokens = int(rank), int(segment_tokens)
    ctx.scale = float(scale)


@torch.library.custom_op("x45::k14_scan_bwd", mutates_args=())
def _k14_scan_bwd_op(
    payload: torch.Tensor, write_keys: torch.Tensor, query_keys: torch.Tensor,
    decay: torch.Tensor, keys: torch.Tensor, dreads: torch.Tensor,
    op_scores: torch.Tensor, slot: torch.Tensor, op: torch.Tensor,
    erase: torch.Tensor, live_after: torch.Tensor, payload_field: torch.Tensor,
    live_pre: torch.Tensor, ckpt_state: torch.Tensor, ckpt_live: torch.Tensor,
    ckpt_keys: torch.Tensor, rank: int, segment_tokens: int, scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor]:
    """The whole window's adjoint as its own opaque op: K10's segment loop
    (imported verbatim) plus the router's estimator; decisions as recorded
    constants (the standing doctrine), the chosen op among them."""
    schedule = MatrixSchedule(slot=slot, erase=erase, payload=payload_field,
                              op=op, live_after=live_after)
    return _k14_backward_body(payload, write_keys, query_keys, decay, dreads,
                              schedule, live_pre, ckpt_state, ckpt_live,
                              ckpt_keys, op_scores, int(rank),
                              int(segment_tokens), float(scale))


@_k14_scan_bwd_op.register_fake
def _(payload, write_keys, query_keys, decay, keys, dreads, op_scores, slot,
      op, erase, live_after, payload_field, live_pre, ckpt_state, ckpt_live,
      ckpt_keys, rank, segment_tokens, scale):
    device = payload.device
    batch, tokens, heads, value_dim = payload.shape
    key_dim = write_keys.shape[-1]
    return (torch.empty_like(payload), torch.empty_like(write_keys),
            torch.empty_like(query_keys),
            torch.empty((batch, tokens, heads), dtype=torch.float32,
                        device=device),
            torch.empty((batch, tokens, heads, 5), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, value_dim), dtype=torch.float32,
                        device=device),
            torch.empty((batch, heads, rank, key_dim), dtype=torch.float32,
                        device=device))


def _k14_scan_backward(ctx, dreads, dstate_out, dlive, dkeys_out, dslot, dop,
                        derase, dlive_after, dpayload_field, dlive_pre,
                        dckpt_state, dckpt_live, dckpt_keys, d_op_scores):
    """The adjoint: dpayload, dwrite_keys, dquery_keys, ddecay,
    d(router_scores) (the admitted estimator), dstate, dkeys; the decision
    channel, the liveness and the recorded scores carry no gradient by
    construction; the scalar policy arguments carry none by type."""
    payload, write_keys, query_keys, decay, state, live, keys, slot, op, \
        erase, live_after, payload_field, live_pre, ckpt_state, ckpt_live, \
        ckpt_keys, op_scores = ctx.saved_tensors
    batch, tokens, heads, value_dim = payload.shape
    if dreads is None:
        dreads = torch.zeros(batch, tokens, heads, value_dim,
                             dtype=torch.float32, device=payload.device)
    dpayload, dwrite_keys, dquery_keys, ddecay, d_router, dstate, dkeys = \
        _k14_scan_bwd_op(payload, write_keys, query_keys, decay, keys, dreads,
                         op_scores, slot, op, erase, live_after,
                         payload_field, live_pre, ckpt_state, ckpt_live,
                         ckpt_keys, ctx.rank, ctx.segment_tokens, ctx.scale)
    return (dpayload, dwrite_keys, dquery_keys, ddecay, d_router, dstate,
            None, dkeys, None, None, None, None, None, None, None)


torch.library.register_autograd("x45::k14_scan_fwd", _k14_scan_backward,
                                setup_context=_k14_scan_setup)


# --------------------------------------------------------------- the wrapper


def routing_scan(payload, write_keys, query_keys, decay, router_scores, *,
                 policy: RoutingPolicy, rank=None, rank_max=None,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                 initial_state=None, initial_live=None, initial_keys=None,
                 collect: bool = False) -> RoutedResult:
    """One whole (layer, window) separated-key scan at PER-TOKEN granularity
    with the LEARNED op routing.

    The K10 wrapper's drop-in with the routing channel: the five-op
    machinery evaluated against the CURRENT state and key table at every
    token, the OP chosen by the learned router's argmax over the five
    per-op scores (born at the threshold chain's own margins).  The kernel
    path is ONE opaque op pair per scan; every refusal takes the EAGER
    ORACLE -- the same forward semantics, the parity fixed point, whose
    router gradient is the honest structural zero (the finding; the
    estimator lives on the kernel path, and the probe asserts that path is
    the one taken):

    * ``policy.ops=False`` (the append-only control) and ``policy.grow`` are
      refused outright (chunk-dial concepts); a rank prefix is refused by name;
    * a store the persistent walk cannot hold (``routing_available``), or
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
    if router_scores.shape != (batch, tokens, heads, 5) \
            or router_scores.dtype != torch.float32 \
            or not router_scores.is_contiguous():
        raise ValueError(f"router_scores must be a contiguous fp32 "
                         f"(batch, tokens, heads, 5) = "
                         f"{(batch, tokens, heads, 5)} tensor; got "
                         f"{tuple(router_scores.shape)} "
                         f"{router_scores.dtype}")
    if not policy.ops:
        raise ValueError("the append-only control is refused by the learned "
                          "routing registration; it is a chunk-dial control")
    if policy.grow:
        raise ValueError("growth is refused by the learned routing "
                          "registration: it is a chunk-boundary concept")
    if rank != rank_max:
        raise ValueError("the learned routing registration requires rank == "
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
            out = _k14_forward_launch(
                payload, write_keys, query_keys, decay, router_scores, state,
                live, keys, pol32, rank, segment_tokens, collect=True,
                warps=_seq_warps(rank, value_dim), stages=3)
        schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                  payload=out["payload"], op=out["op"],
                                  live_after=out["live_after"])
        return RoutedResult(out["reads"], out["state"], out["live"], rank,
                            ScanStats(), schedule, out["weights"])
    if (payload.is_cuda and routing_available(rank, value_dim, key_dim)):
        reads, state_out, live_out, _keys, _slot, _op, _erase, _live_after, \
            _payload, _live_pre, _ckpt_state, _ckpt_live, _ckpt_keys, \
            _op_scores = _k14_scan_fwd_op(
                payload, write_keys, query_keys, decay, router_scores, state,
                live, keys, rank, int(segment_tokens), policy.tau_match,
                policy.tau_merge, policy.tau_pop, policy.tie_break,
                policy.read_scale)
        return RoutedResult(reads, state_out, live_out, rank, ScanStats(),
                            None, None)
    # the eager oracle: the same forward semantics, the parity fixed point
    # (its router gradient is the structural zero -- the finding)
    ref = routing_oracle_scan(payload, write_keys, query_keys, decay,
                              router_scores, policy=policy, rank=rank,
                              rank_max=rank_max, initial_state=state,
                              initial_live=live, initial_keys=keys,
                              collect=False)
    return RoutedResult(ref["reads"], ref["state"], ref["live"], rank,
                        ScanStats(), None, None)


# ------------------------------------------------------- the mixer and the model


class RoutingSeparatedMixer(nn.Module):
    """The K14 mixer: the K10 separated mixer VERBATIM (its projections, its
    DECAY gate, its mimetic key birth) plus ONE per-head router module -- a
    ``Linear(ctx*width, 5)`` per head over the DETACHED causal context, born
    at EXACT zeros with the ``+0.0`` normalisation (P5: at birth the mixer
    IS the banked K10 winner bit for bit -- the zero-born scores leave the
    branch-active margin encoding untouched, and the argmax reproduces the
    threshold chain exactly), trainable through the admitted straight-through
    estimator on the kernel path.  ``router_trainable=False`` pins the router
    at birth: the never-fires falsifier control."""

    def __init__(self, width: int, heads: int, value_dim: int, rank: int,
                 rank_max: int, key_dim: int, policy: RoutingPolicy,
                 ctx: int = DEFAULT_CONTEXT,
                 segment_tokens: int = DEFAULT_SEGMENT_TOKENS,
                 router_trainable: bool = True):
        super().__init__()
        self.heads, self.value_dim = heads, value_dim
        self.rank, self.rank_max = rank, rank_max
        self.policy = policy
        self.key_dim, self.ctx = int(key_dim), int(ctx)
        self.segment_tokens = int(segment_tokens)
        self.width = width
        self.value = nn.Linear(width, heads * value_dim, bias=False)
        self.key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.query_key = nn.Linear(ctx * width, heads * key_dim, bias=False)
        self.out = nn.Linear(heads * value_dim, width, bias=False)
        self.decay = nn.Linear(width, heads, bias=True)
        with torch.no_grad():
            lam0 = math.exp(-1.0 / DECAY_HORIZON_TOKENS)
            self.decay.bias.fill_(math.log(lam0 / (1.0 - lam0)))
        self.router = nn.Linear(ctx * width, heads * 5, bias=True)
        with torch.no_grad():
            self.router.weight.zero_()
            self.router.bias.zero_()
        self.router.weight.requires_grad_(bool(router_trainable))
        self.router.bias.requires_grad_(bool(router_trainable))
        self.apply_mimetic_init()

    def apply_mimetic_init(self, seed: int = MIMETIC_SEED):
        """K10's mimetic key birth, verbatim.  The router is NOT part of the
        mimetic family: it is born at exact zero whatever its arm (the
        ``+0.0`` discipline; the birth machine is the banked K10 winner)."""
        with torch.no_grad():
            family = mimetic_key_family(self.width, self.key_dim, self.heads,
                                        seed)
            for projection in (self.key, self.query_key):
                projection.weight.zero_()
            for head in range(self.heads):
                block = family[head].t()          # (key_dim, width)
                rows = slice(head * self.key_dim, (head + 1) * self.key_dim)
                cols = slice((self.ctx - 1) * self.width, self.ctx * self.width)
                self.key.weight[rows, cols] = block
                self.query_key.weight[rows, cols] = block

    def router_scores(self, x: torch.Tensor) -> torch.Tensor:
        """The learned per-op scores from the DETACHED causal context:
        fp32 OUTSIDE the autocast regime (the router's arithmetic is fp32
        decision-side math, like the scan's own), born at exact ``+0.0``
        (the ``+0.0`` add normalises the ``-0.0`` corner of a zero matmul).
        ``(b, T, width) -> (b, T, heads, 5)``."""
        batch, tokens, _ = x.shape
        context = causal_context(x, self.ctx).reshape(
            batch, tokens, self.ctx * self.width)
        with torch.autocast(device_type="cuda", enabled=False):
            g = self.router(context.detach().to(torch.float32)) + 0.0
        return g.view(batch, tokens, self.heads, 5)

    def mix(self, x: torch.Tensor):
        batch, tokens, _ = x.shape
        context = causal_context(x, self.ctx)          # (b, T, ctx, width)
        context = context.reshape(batch, tokens, self.ctx * self.width)
        payload = self.value(x).view(batch, tokens, self.heads, self.value_dim)
        write_keys = self.key(context).view(batch, tokens, self.heads,
                                            self.key_dim)
        query_keys = self.query_key(context).view(batch, tokens, self.heads,
                                                  self.key_dim)
        decay = torch.sigmoid(self.decay(x)).clamp(min=DECAY_FLOOR,
                                                    max=1.0 - 1e-6)
        result = routing_scan(payload, write_keys, query_keys, decay,
                              self.router_scores(x), policy=self.policy,
                              rank=self.rank, rank_max=self.rank_max,
                              segment_tokens=self.segment_tokens)
        return (self.out(result.reads.reshape(batch, tokens,
                                              self.heads * self.value_dim)),
                result)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.mix(x)
        return out


def convert_model_to_routing(model, router_trainable: bool = True,
                             segment_tokens: int = DEFAULT_SEGMENT_TOKENS):
    """Swap every ``SeparatedSequentialMixer`` for the routing one, sharing
    parameters (the K10/K12 conversion pattern: the ``nn.Linear`` modules the
    optimizer already saw are the very same objects), adding ONE per-head
    router per mixer -- born at exact zeros and trainable (the learned arm),
    or pinned (the falsifier control).  Every value-side and key-side module
    is carried over UNTOUCHED, so at birth the converted model's discrete
    schedule is the banked K10 winner's bit for bit.  The module's kaiming
    draw is bracketed by an RNG-state save/restore so the conversion cannot
    shift any downstream stream the banked row depends on.  ``k10_*`` source
    is not edited; this builds on its conversion's output.  The parameter
    DELTA is the registration's stated accounting (+553,200 at the
    registered shape), asserted by the probe, not hidden."""
    for block in model.blocks:
        mixer = block.mixer
        if not isinstance(mixer, SeparatedSequentialMixer):
            raise ValueError("convert_model_to_routing builds on "
                             "convert_model_to_separated_keys' output; got "
                             f"{type(mixer).__name__}")
        rm = RoutingSeparatedMixer.__new__(RoutingSeparatedMixer)
        torch.nn.Module.__init__(rm)
        rm.__dict__.update(mixer.__dict__)
        rm._modules = dict(mixer._modules)     # own copy; the old mixer keeps its
        rm.segment_tokens = int(segment_tokens)
        # the router module: born at exact zeros; the RNG-state guard keeps
        # the creation draw out of the banked row's stream (the probe's
        # step-1 bit-identity assert is the second guard).
        cpu_state = torch.random.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state()
                      if torch.cuda.is_available() else None)
        router = nn.Linear(rm.ctx * rm.width, rm.heads * 5, bias=True)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
        with torch.no_grad():
            router.weight.zero_()
            router.bias.zero_()
        router.weight.requires_grad_(bool(router_trainable))
        router.bias.requires_grad_(bool(router_trainable))
        rm.router = router
        block.mixer = rm
    return model


def routing_parameter_delta(depth: int, heads: int, ctx: int,
                            width: int) -> int:
    """The registration's parameter accounting, computed not hardcoded: one
    per-head router (``5 * ctx*width + 5`` parameters) per mixer."""
    return depth * heads * (5 * ctx * width + 5)


__all__ = ["DEFAULT_KEY_DIM", "DEFAULT_CONTEXT", "DEFAULT_SEGMENT_TOKENS",
           "MIMETIC_SEED", "RoutedResult", "ScanStats", "MatrixSchedule",
           "RoutingPolicy", "EPS", "APPEND", "CORRECT", "MERGE", "NONE",
           "POP", "ROUTER_OPS", "RESERVED_SCORE", "RoutingSeparatedMixer",
           "convert_model_to_routing", "convert_model_to_separated_keys",
           "routing_available", "routing_oracle_scan", "routing_scan",
           "routing_parameter_delta", "separated_available",
           "_k14_forward_launch", "_k14_backward_body", "_router_estimator"]
