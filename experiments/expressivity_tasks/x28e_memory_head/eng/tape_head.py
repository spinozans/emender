"""Eager fp32 slot-store tape head — X28E phase 1 (M1 engineering only, pre-M2).

Binding spec: ``experiments/expressivity_tasks/x28e_memory_head/REGISTRATION.md``
(the "routing rule (complete executable specification)" section). Pattern
reference: ``ndm/triton/mlp_mem_fused.py`` (sequential scan, sparse
checkpoints, reverse-replay BPTT) — PHASE 1 IS EAGER ONLY, no Triton kernels.

Per (layer, head) the recurrent state is a variable-length store of slots;
each slot is a (key, value) pair in fp32 cells plus architectural bookkeeping
(live/dead flag, age, usage) and a decaying strength scalar. All slot
arithmetic is fp32. In this eager implementation the store is a fixed-size
buffer of ``budget`` slots plus a live mask (one store per (batch element,
head); the batch element carries its own store).

Per-token operation dispatch (in order, per the routing rule):

  1. READ — a learned query scores every live slot (scaled dot product);
     the read output is the soft-matched value mixture
     ``sum_i softmax(score)_i * strength_i * value_i`` (pre-pop, pre-write).
  2. POP — a READ-PATH action: the read's best-matched live slot (QUERY
     argmax) has its strength reduced by the learned pop strength when that
     strength exceeds ``pop_threshold``. The pop target is recorded
     separately from the write target.
  3. WRITE-ENABLE — W-ON / W-OFF (traced NO-WRITE). Placeholder rule
     (TO-BE-BOUND-AT-FREEZE): the learned write-strength gate exceeds
     ``write_enable_threshold``.
  4. WRITE (W-ON only) — exactly one of, evaluated in order:
       R-EMPTY          no live slots                -> APPEND
       R-NOMATCH        best write-match score < match_threshold -> APPEND
       R-ZERO-RESIDUAL  match && squared value residual < eps_residual -> APPEND
       R-CONFLICT       match && residual >= eps_residual -> DELTA-RULE
                       CORRECTION on the matched slot
       R-CAP (FINAL OVERRIDE) any APPEND requested while the budget is full
                       -> MERGE (delta-rule onto the best write match; no
                       slot is ever allocated past B).
  5. DECAY/AGE — live-slot strength *= decay_rate, age += 1; slots whose
     strength falls to <= free_threshold are freed (dead). Pop, merge and
     decay never allocate; only APPEND allocates (reusing freed slots, so
     total allocation never exceeds B).

Zero learned management: routing is ARCHITECTURAL and fully detached from
gradients. Every routing decision (argmax indices, threshold comparisons,
case codes, liveness) is computed from detached quantities and recorded as
integral trace tensors. Gradient flows ONLY through keys, values, queries,
and the write/pop/append gate magnitudes (strengths/gates).

FLAGGED engineering interpretations (TO-BE-BOUND-AT-FREEZE at M2; reported,
never silently decided):

  * POP semantics — "the best-matching live slot is marked dead with that
    strength" is implemented as a strength decrement
    ``strength <- strength * (1 - pop_strength)``; a slot dies when its
    strength reaches the registered ``free_threshold`` (the same death rule
    as usage-decay). This keeps the pop magnitude learnable ("gradient
    touches ... strengths"); the binary-kill reading is the flagged
    alternative for the freeze.
  * Soft-strength read — the read mixture weights each live slot by its
    strength; this is what gives the pop/decay magnitudes a differentiable
    effect. Hard liveness (the live/dead flag) still governs every routing
    decision and dead slots are excluded from reads exactly.
  * WRITE-ENABLE rule — placeholder: write-strength gate > threshold (the
    registration freezes the actual rule and its constants at M2). The gate
    magnitude doubles as the delta-rule write strength.
  * APPEND gate — its input includes the detached write-error magnitude
    against the best-matching live slot (registration: "the gate's input
    includes the write-error magnitude"); the gate strength scales the
    appended value (append writes ``(key, gate * value)``).
  * "Budget full" = no free (non-live) slot. Dead slots are reusable free
    capacity, so total allocation never exceeds B.
  * Match scoring — scaled dot products (query vs slot key for reads and
    pops; write key vs slot key for write routing), scaled by 1/sqrt(K).
  * Decay order — applied at end of token; a slot appended at token t is
    decayed once on its birth token; the free check runs after pop and
    again after decay. Free-by-decay joins the trace (G10, supervisor
    resolution extending the P1-6 gate): the trace carries per-token int8
    strength-death codes (FREED_POP=1 at the post-pop check,
    FREED_DECAY=2 at the decay check); replay consumes them as inputs and
    never recomputes a death decision from (rounded) state.
  * Read is PRE-write per the routing rule ("the head FIRST performs a
    READ ... and THEN a WRITE-ENABLE decision"): a token's own write is
    never visible to that token's read output (flagged: the final token's
    write is gradient-dead unless a later read consumes it).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .routing_trace import (
    NO_SLOT, NO_WRITE, R_CAP, R_CONFLICT, R_EMPTY, R_NOMATCH, R_ZERO_RESIDUAL,
    FREED_POP, FREED_DECAY, RoutingTrace,
)

# Soft mask value for dead slots in score tensors (exact softmax zero via
# underflow, no NaN for all-dead rows).
NEG_MASK = -1.0e30


# ---------------------------------------------------------------------------
# Architectural constants (registered placeholders — TO-BE-BOUND-AT-FREEZE).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TapeHeadConfig:
    """Head dimensions and the registered architectural constants.

    Thresholds/decay constants are FIXED ARCHITECTURAL CONSTANTS bound in
    the engineering freeze (M2) with their values receipted; the defaults
    below are PLACEHOLDER VALUES documented as TO-BE-BOUND-AT-FREEZE. Decay
    defaults satisfy the T2 inertness calculation (see :func:`fade_horizon`).
    """

    key_dim: int = 64                 # K
    value_dim: int = 128              # V
    n_heads: int = 4                  # head multiplicity (bound at freeze, T1)
    budget: Optional[int] = None      # B; None => unbounded arm (T5 machine cap)
    unbounded_cap: int = 2 ** 17      # 131,072 slots per (batch element, head) (T5)

    # --- routing thresholds (placeholder defaults, TO-BE-BOUND-AT-FREEZE) ---
    match_threshold: float = 0.5       # R-NOMATCH boundary (write-match score)
    eps_residual: float = 1e-6        # R-ZERO-RESIDUAL boundary (squared residual)
    pop_threshold: float = 0.5         # POP fires above this read-path strength
    write_enable_threshold: float = 0.5  # placeholder W-ON/W-OFF rule constant
                                       # (G14 proposed freeze value; the phase-1
                                       # 0.25 default was superseded pre-M2)

    # --- usage-decay constants (registered; T2 inertness receipt at freeze) --
    free_threshold: float = 1e-3       # strength <= this => slot freed (dead)
    decay_rate: float = 0.999995       # per-token strength decay (see fade_horizon)
    usage_decay: float = 0.999         # per-token usage-counter decay (EMA)

    @property
    def slots(self) -> int:
        return self.budget if self.budget is not None else self.unbounded_cap


def fade_horizon(cfg: TapeHeadConfig) -> float:
    """T2 inertness calculation: tokens until a fresh slot (strength 1.0)
    decays to ``free_threshold`` under per-token multiplicative decay:

        horizon = ln(free_threshold) / ln(decay_rate)

    T2 obligation: horizon >= 10x the longest registered input. The longest
    registered input at phase-1 time is the 90,904-token numerics smoke, so
    the default constants must give horizon >= 909,040 tokens. (FLAG: the
    cap-stress smoke length is not yet registered; the freeze receipt must
    re-run this calculation against the actual longest registered input.)
    With the placeholder defaults (0.999995, 1e-3): horizon ~ 1.38e6 tokens.
    """
    return math.log(cfg.free_threshold) / math.log(cfg.decay_rate)


# ---------------------------------------------------------------------------
# State: the per-(batch element, head) slot store.
# ---------------------------------------------------------------------------
@dataclass
class TapeState:
    keys: torch.Tensor      # [B, NH, S, K] fp32, gradient-carrying
    values: torch.Tensor    # [B, NH, S, V] fp32, gradient-carrying
    strength: torch.Tensor  # [B, NH, S] fp32 in (0, 1], gradient-carrying
    live: torch.Tensor      # [B, NH, S] bool  (the live/dead flag)
    age: torch.Tensor       # [B, NH, S] int64 (tokens since allocation)
    usage: torch.Tensor     # [B, NH, S] fp32 accumulated read mass (bookkeeping)

    def live_counts(self) -> torch.Tensor:
        return self.live.sum(-1)  # [B, NH]


def init_state(batch_size: int, cfg: TapeHeadConfig, device, dtype=torch.float32) -> TapeState:
    S, K, V, NH = cfg.slots, cfg.key_dim, cfg.value_dim, cfg.n_heads
    return TapeState(
        keys=torch.zeros(batch_size, NH, S, K, device=device, dtype=dtype),
        values=torch.zeros(batch_size, NH, S, V, device=device, dtype=dtype),
        strength=torch.zeros(batch_size, NH, S, device=device, dtype=dtype),
        live=torch.zeros(batch_size, NH, S, device=device, dtype=torch.bool),
        age=torch.zeros(batch_size, NH, S, device=device, dtype=torch.int64),
        usage=torch.zeros(batch_size, NH, S, device=device, dtype=dtype),
    )


def _onehot(idx: torch.Tensor, size: int, fires: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """[B, NH] integral indices + [B, NH] bool fire mask -> [B, NH, S] float."""
    return F.one_hot(idx.clamp(min=0), num_classes=size).to(dtype) * fires.unsqueeze(-1).to(dtype)


# ---------------------------------------------------------------------------
# The eager sequential scan (the architectural reference; also the replay
# engine when a trace is supplied).
# ---------------------------------------------------------------------------
def tape_scan(k, q, v, write_strength, pop_strength, append_logit, state: TapeState,
              cfg: TapeHeadConfig, append_err_coef=None, trace: Optional[RoutingTrace] = None,
              collect_trace: bool = True):
    """Sequential per-token tape-head scan (fp32 slot arithmetic).

    Args (all per-head per-token, shapes ``[B, T, NH, *]``):
        k, q           : [B, T, NH, K]  write key / read query (learned).
        v              : [B, T, NH, V]  write value (learned).
        write_strength : [B, T, NH]     delta-rule write magnitude; placeholder
                         W-ON rule: ``> write_enable_threshold`` (W-OFF traced
                         NO-WRITE). Module feeds sigmoid; the scan accepts any
                         fp32 magnitude in [0, 1].
        pop_strength   : [B, T, NH]     read-path pop magnitude; POP fires when
                         ``> pop_threshold`` (architectural decision).
        append_logit   : [B, T, NH]     append-gate pre-activation; the gate is
                         ``sigmoid(logit + append_err_coef * write_error_mag)``
                         where the error magnitude is DETACHED (the gate input
                         includes the write-error magnitude; routing does not).
        state          : entry TapeState (fp32).
        trace          : optional RoutingTrace over the SAME token range as the
                         inputs — REPLAY MODE: routing decisions are INPUTS,
                         never recomputed from (rounded) state.
        collect_trace  : collect (or echo, in replay mode) the trace.

    Returns ``(read_out [B, T, NH, V], final TapeState, RoutingTrace | None)``.

    Routing is computed from detached quantities only; the differentiable
    paths are the read mixture, the delta-rule correction, the appended
    content, and the strength updates.
    """
    B, T, NH = k.shape[0], k.shape[1], k.shape[2]
    K, V, S = k.shape[-1], v.shape[-1], state.live.shape[-1]
    assert K == cfg.key_dim and V == cfg.value_dim and NH == cfg.n_heads
    assert S == cfg.slots and S >= 1
    k = k.float(); q = q.float(); v = v.float()
    write_strength = write_strength.float()
    pop_strength = pop_strength.float()
    append_logit = append_logit.float()
    if append_err_coef is None:
        append_err_coef = torch.zeros(NH, device=k.device)
    else:
        append_err_coef = append_err_coef.float()
    scale = 1.0 / math.sqrt(K)
    dev = k.device

    keys, values, strength, live = state.keys, state.values, state.strength, state.live
    age, usage = state.age, state.usage

    tr_case: list = []
    tr_matched: list = []
    tr_pop_fired: list = []
    tr_pop_target: list = []
    tr_alloc_target: list = []
    tr_w_on: list = []
    tr_freed: list = []
    read_outs: list = []

    for t in range(T):
        k_t, q_t, v_t = k[:, t], q[:, t], v[:, t]          # [B, NH, *]
        beta = write_strength[:, t]                         # [B, NH]
        pop = pop_strength[:, t]
        a_logit = append_logit[:, t]

        # ---------------- READ (content-addressed, pre-pop/pre-write) --------
        q_scores = torch.einsum('bhsk,bhk->bhs', keys, q_t) * scale   # attached
        q_masked = q_scores.masked_fill(~live, NEG_MASK)
        q_probs = torch.softmax(q_masked, dim=-1)
        # Soft-strength weighting (FLAGGED interpretation): dead slots are
        # excluded exactly via the live mask; live slots are weighted by their
        # (differentiable) strength.
        read_weights = q_probs * strength * live
        read_out = torch.einsum('bhs,bhsv->bhv', read_weights, values)
        read_outs.append(read_out)
        usage = usage * cfg.usage_decay + read_weights.detach()

        # ---------------- POP (read-path; architectural decision) -----------
        has_live = live.any(-1)                                     # [B, NH]
        if trace is None:
            q_det = q_scores.detach().masked_fill(~live, NEG_MASK)
            read_best = q_det.argmax(-1)                            # [B, NH]
            pop_fired = (pop.detach() > cfg.pop_threshold) & has_live
        else:
            pop_fired = trace.pop_fired[:, t]
            read_best = trace.pop_target[:, t].clamp(min=0)
        pop_target = torch.where(pop_fired, read_best, NO_SLOT)
        onehot_pop = _onehot(read_best, S, pop_fired, strength.dtype)
        # "marked dead with that strength" (FLAGGED: strength decrement; the
        # slot dies at free_threshold). Attached to the pop magnitude only.
        strength = strength * (1.0 - onehot_pop * pop.unsqueeze(-1))
        if trace is not None and trace.freed is not None:
            # G10: the strength-death decision is a replay INPUT, never
            # recomputed from (possibly bf16-rounded) replayed strengths.
            freed = trace.freed[:, t] == FREED_POP                  # [B, NH, S]
        else:
            freed = live & (strength.detach() <= cfg.free_threshold)
            freed_code_t = freed.to(torch.int8) * FREED_POP
        live = live & ~freed
        has_live = live.any(-1)                                     # post-pop

        # ---------------- WRITE-ENABLE + five-case routing -------------------
        if trace is None:
            beta_det = beta.detach()
            w_on = beta_det > cfg.write_enable_threshold
            k_scores = torch.einsum('bhsk,bhk->bhs', keys, k_t) * scale
            k_det = k_scores.detach().masked_fill(~live, NEG_MASK)
            write_best = k_det.argmax(-1)                           # [B, NH]
            best_score = k_scores.detach().gather(-1, write_best.unsqueeze(-1)).squeeze(-1)
            matched_ok = has_live & w_on
            nomatch = matched_ok & (best_score < cfg.match_threshold)
            case = torch.full((B, NH), NO_WRITE, dtype=torch.int64, device=dev)
            case = torch.where(w_on & ~has_live, R_EMPTY, case)
            case = torch.where(nomatch, R_NOMATCH, case)
            corr_target = write_best  # R-CONFLICT/R-CAP correction target (recomputed below)
        else:
            # REPLAY MODE: routing decisions are INPUTS, never recomputed.
            case = trace.case[:, t]
            w_on = trace.w_on[:, t]
            alloc_target_t = trace.alloc_target[:, t]
            matched_t = trace.matched[:, t]
            append_fires = alloc_target_t >= 0
            corr_target = matched_t.clamp(min=0)
        # shared: squared value residual against the write-match target
        gather_idx = corr_target.view(B, NH, 1, 1).expand(B, NH, 1, V)
        v_best = values.gather(2, gather_idx).squeeze(2)             # attached
        residual = ((v_t - v_best) ** 2).sum(-1)
        residual_d = residual.detach()
        if trace is None:
            matched_ok_m = matched_ok & ~nomatch
            zero_res = matched_ok_m & (residual_d < cfg.eps_residual)
            conflict = matched_ok_m & ~zero_res
            case = torch.where(zero_res, R_ZERO_RESIDUAL, case)
            case = torch.where(conflict, R_CONFLICT, case)
            append_req = (case == R_EMPTY) | (case == R_NOMATCH) | (case == R_ZERO_RESIDUAL)
            free_exists = (~live).any(-1)
            case = torch.where(append_req & ~free_exists, R_CAP, case)
            append_fires = append_req & free_exists
            alloc_idx = (~live).to(torch.long).argmax(-1)          # first free slot
            alloc_target_t = torch.where(append_fires, alloc_idx, NO_SLOT)
            matched_t = torch.where(has_live, write_best, NO_SLOT)

        # append-gate strength: input includes the DETACHED write-error magnitude
        err_mag = torch.where(has_live, residual_d.clamp_min(0.0).sqrt(),
                              torch.zeros_like(residual_d))
        append_gate = torch.sigmoid(a_logit + append_err_coef * err_mag)

        # ---------------- APPEND (the only allocating op) --------------------
        onehot_alloc = _onehot(alloc_target_t, S, append_fires, values.dtype)
        if bool(append_fires.any()):
            keys = (keys * (1.0 - onehot_alloc.unsqueeze(-1))
                    + onehot_alloc.unsqueeze(-1) * k_t.unsqueeze(-2))
            new_values = (values * (1.0 - onehot_alloc.unsqueeze(-1))
                          + onehot_alloc.unsqueeze(-1)
                          * (append_gate.unsqueeze(-1).unsqueeze(-1) * v_t.unsqueeze(-2)))
        else:
            new_values = values
        strength = strength * (1.0 - onehot_alloc) + onehot_alloc * 1.0
        age = age * (1 - onehot_alloc.to(torch.int64))
        usage = usage * (1.0 - onehot_alloc)
        # PHASE-2 PARITY FIX (flagged in the phase-2 report): the phase-1 line
        # ``live = live | append_fires.unsqueeze(-1)`` broadcast [B,NH,1] over
        # [B,NH,S], marking EVERY slot live whenever an append fired. It was
        # invisible to the phase-1 tests (fresh stores: revived slots carry
        # strength 0 and die again at the end-of-token free check) but corrupts
        # any store with dead-but-strong slots. The append revives exactly the
        # allocated slot (the one-hot is already masked by ``append_fires``):
        live = live | onehot_alloc.to(torch.bool)

        # ---------------- DELTA-RULE CORRECTION / MERGE (R-CONFLICT, R-CAP) --
        corr_fires = (case == R_CONFLICT) | (case == R_CAP)
        if bool(corr_fires.any()):
            onehot_corr = _onehot(corr_target, S, corr_fires, values.dtype)
            v_corrected = v_best + beta.unsqueeze(-1) * (v_t - v_best)
            values = (new_values * (1.0 - onehot_corr.unsqueeze(-1))
                      + onehot_corr.unsqueeze(-1) * v_corrected.unsqueeze(-2))
        else:
            values = new_values

        # ---------------- DECAY / AGE / FREE (end of token) ------------------
        strength = torch.where(live, strength * cfg.decay_rate, strength)
        age = age + live.to(torch.int64)
        if trace is not None and trace.freed is not None:
            freed2 = trace.freed[:, t] == FREED_DECAY                # G10: replay input
        else:
            freed2 = live & (strength.detach() <= cfg.free_threshold)
            freed_code_t = freed_code_t + freed2.to(torch.int8) * FREED_DECAY
        live = live & ~freed2

        # ---------------- trace record ---------------------------------------
        if collect_trace:
            tr_case.append(case)
            tr_matched.append(matched_t if trace is None else trace.matched[:, t])
            tr_pop_fired.append(pop_fired)
            tr_pop_target.append(pop_target)
            tr_alloc_target.append(alloc_target_t)
            tr_w_on.append(w_on)
            if trace is None:
                tr_freed.append(freed_code_t)                       # G10 codes
            else:
                tr_freed.append(trace.freed[:, t] if trace.freed is not None
                                else torch.zeros(B, NH, S, dtype=torch.int8,
                                                 device=dev))

    if collect_trace:
        trace_out = RoutingTrace(
            case=torch.stack(tr_case, dim=1),
            matched=torch.stack(tr_matched, dim=1),
            pop_fired=torch.stack(tr_pop_fired, dim=1),
            pop_target=torch.stack(tr_pop_target, dim=1),
            alloc_target=torch.stack(tr_alloc_target, dim=1),
            w_on=torch.stack(tr_w_on, dim=1),
            freed=torch.stack(tr_freed, dim=1),   # int8 [B,T,NH,S] (G10)
        )
    else:
        trace_out = None

    read_out = torch.stack(read_outs, dim=1)  # [B, T, NH, V]
    final_state = TapeState(keys=keys, values=values, strength=strength,
                            live=live, age=age, usage=usage)
    return read_out, final_state, trace_out


# ---------------------------------------------------------------------------
# The eager PyTorch module (learned projections; zero learned management).
# ---------------------------------------------------------------------------
class TapeHead(nn.Module):
    """Eager slot-store tape head: learned key/query/value projections plus
    learned write/pop/append gate strengths. Routing, thresholds, decay and
    merge selection are architectural (constants + detached decisions).

    Gradient touches ONLY the projections' keys, values, queries and the
    write/pop/append strengths/gates (registration: "Zero learned
    management"). ``d_model`` and head multiplicity are TO-BE-BOUND-AT-
    FREEZE (flagged: the registration does not bind the input width).
    """

    def __init__(self, d_model: int, cfg: TapeHeadConfig):
        super().__init__()
        self.cfg = cfg
        NH, K, V = cfg.n_heads, cfg.key_dim, cfg.value_dim
        self.W_q = nn.Linear(d_model, NH * K, bias=True)      # read queries
        self.W_k = nn.Linear(d_model, NH * K, bias=True)      # write keys
        self.W_v = nn.Linear(d_model, NH * V, bias=True)      # write values
        self.W_write = nn.Linear(d_model, NH, bias=True)      # write-strength gate (placeholder W-ON rule)
        self.W_pop = nn.Linear(d_model, NH, bias=True)        # pop-strength gate
        self.W_append = nn.Linear(d_model, NH, bias=True)      # append-gate pre-activation
        # Learned scalar per head: how strongly the (detached) write-error
        # magnitude enters the append gate ("the gate's input includes the
        # write-error magnitude against the best-matching live slot").
        self.append_err_coef = nn.Parameter(torch.zeros(NH))

    def forward(self, x: torch.Tensor, state: Optional[TapeState] = None,
                trace_override: Optional[RoutingTrace] = None,
                collect_trace: bool = True):
        """x: [B, T, d_model]. Returns (read_out [B,T,NH,V], state, trace)."""
        B, T, D = x.shape
        cfg = self.cfg
        NH, K, V = cfg.n_heads, cfg.key_dim, cfg.value_dim
        assert D == self.W_q.in_features
        k = self.W_k(x).view(B, T, NH, K)
        q = self.W_q(x).view(B, T, NH, K)
        v = self.W_v(x).view(B, T, NH, V)
        write_strength = torch.sigmoid(self.W_write(x))       # [B, T, NH]
        pop_strength = torch.sigmoid(self.W_pop(x))           # [B, T, NH]
        append_logit = self.W_append(x)                      # [B, T, NH]
        if state is None:
            state = init_state(B, cfg, x.device, torch.float32)
        return tape_scan(k, q, v, write_strength, pop_strength, append_logit,
                         state, cfg, append_err_coef=self.append_err_coef,
                         trace=trace_override, collect_trace=collect_trace)


def head_parameter_count(d_model: int, cfg: TapeHeadConfig) -> int:
    """Exact parameter count of :class:`TapeHead` (freeze receipt arithmetic).

    = d_model*NH*(2K + V)  + NH*(2K + V)      (W_q, W_k, W_v weights + biases)
    + 3*(d_model*NH + NH)                     (W_write, W_pop, W_append)
    + NH                                      (append_err_coef)
    """
    NH, K, V = cfg.n_heads, cfg.key_dim, cfg.value_dim
    return (d_model * NH * (2 * K + V) + NH * (2 * K + V)
            + 3 * (d_model * NH + NH) + NH)


def layer_parameter_count(d_model: int, cfg: TapeHeadConfig) -> int:
    """Exact parameter count of the full layer (oracle.TapeLayer: head + W_o)."""
    return head_parameter_count(d_model, cfg) + cfg.n_heads * cfg.value_dim * d_model + d_model


__all__ = [
    "TapeHeadConfig", "TapeState", "init_state", "tape_scan", "TapeHead",
    "fade_horizon", "head_parameter_count", "layer_parameter_count",
    "NEG_MASK",
]
