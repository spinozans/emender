# K14 — the learned management routing arm: the op routing itself made a learned policy, born at the banked thresholds

Stage 14 of the X45 kernel lane, and the operator-corrected program's answer
to the C5 receipts' poorest surface (27.7K gate params + three hand-set
thresholds vs GDN2's 27.4M learned management parameters): **zero-learned-
management was never a mandate — the op-routing decision channel is an open
lever**.  The threshold routing (cos-sim vs `tau_match` -> CORRECT;
second-best vs `tau_merge` -> MERGE; strength vs `tau_pop` -> POP; else
APPEND) is replaced by a LEARNED ROUTER — per-token, input-dependent, a
small per-head module from the causal context producing five per-op scores,
DISCRETE argmax at the forward (the recorded schedule: decisions stay
discrete, the scan stays exact given them), trained through the K12
straight-through estimator pattern.  The registration is the committed draft
`PREREGISTRATION_LEARNED_MANAGEMENT.md` (operator sign-off pending); the
BIRTH CONSTRUCTION was escalated and supervisor-approved before the engine
was locked.  This document is the semantics and the GATE receipts;
`RESULT_ROUTING.md` is the probe.

**Headline, as narrowly as the evidence supports.**

* **The birth bar holds bit-exactly, with NO tie-break reliance (P5, one
  step stronger than K12's).**  The supervisor-approved construction —
  **branch-active margins + the reserved runner-up** — encodes the chain's
  own branch conditions (exactly one active per token) and scores the
  ACTIVE op's own threshold delta (MERGE `d1+d2`, CORRECT `d1`, APPEND
  `−d1`, POP `d3`, NONE `+0.0`; `d1 = best−tau_match` etc.), every INACTIVE
  op the reserved constant −1.0.  Every active margin is provably ≥ 0 > −1.0,
  so **the birth argmax is UNIQUE on every input** — the chain reproduced
  bit-exactly with no tie-order dependence at all (the gate receipt's
  uniqueness lane: min margin 1.0, the runner-up at the reserved constant on
  100% of tokens).  A chain is a LEXICOGRAPHIC TREE and additive raw margins
  provably cannot reproduce one (in the no-match region an additive CORRECT
  score `d1−d2` is unbounded above while the chain says APPEND) — the scores
  must encode the branch structure, and this form does.
* **Birth parity: 0 flips at the registered shape** (17M+ lanes), the k14
  launch at the zero-born router vs K10's OWN launch on every K10 channel —
  and parity holds at a FLIPPED router too (0 flips on every channel
  including the recorded per-op scores and the chain's own op channel),
  while a magnitude-0.9 random router moves **41.3% of ops off the chain**:
  the learned regime is exercised, not just reached.
* **THE FROZEN-ROUTER FALSIFIER IS BIT-IDENTICAL END TO END — full-trajectory
  identity, no reduction-order caveat needed.**  All 180 constant-phase
  training rows bit-identical to the banked winner's own
  `training.jsonl` (max |Δ| 0.0 at every step), step-1 bit-identical, and
  the landing post-cooldown EXACTLY the re-scored baseline's own
  **1.6832871096474784** (EMA 1.8054 too) — stronger than the K11B §2.4-class
  bar (step-1 + landing parity with a disclosed caveat), exactly as the
  construction predicts: the forward is the chain bit for bit and the six
  gradients are K10's own imported body on an identical recorded schedule.
* **THE ZERO-GRADIENT FINDING, on the op channel** (the K12 wall): under
  the recorded-schedule doctrine the op choice is a backward constant, so
  the STRUCTURAL gradient to the router through the eager oracle is EXACTLY
  ZERO (`oracle_drouter_is_none: true`) — "the model can learn which op to
  use" REQUIRES an admitted estimator, the same structural fact the
  placement lane surfaced.
* **The admitted estimator (K12-identical, supervisor-approved, with the
  rescope verbatim):** the kernel path's backward is K10's own
  `_k10_backward_body` imported VERBATIM (the six gradients BIT-EQUAL to
  K10's own body on the same recorded channels) plus ONE torch reduction on
  the RECORDED per-op scores (no replay): `dL/ds_k = Σ_t c_t · p_{k*} ·
  (δ_{kk*} − p_k)` with `p = softmax(s)` and `c_t = <dpayload_t, p_t>` the
  value walk's own write-content output.  Gated against the fp64 autograd
  of its own defined surrogate objective (7.0e-6 on the gate fixture), the
  surrogate objective itself fp64-gradchecked, nonzero at the zero-born
  router at step 0.  **The disclosed bias, verbatim: `c_t` is zero on
  MERGE/POP/NONE tokens, so the estimator gives NO direct signal to tokens
  the chain routed to MERGE/POP/NONE — a POP/MERGE-revival null is
  UNINTERPRETABLE as a learned-management verdict for those ops (the
  read-path POP/MERGE estimator is the named follow-up).**

## 1. What was built

| file | role |
| --- | --- |
| `PREREGISTRATION_LEARNED_MANAGEMENT.md` | the DRAFT registration (operator sign-off pending): the approved construction, the arms, M-M1..M-M4 with the refutation clauses, the verified baseline anchor, the falsifier bar |
| `k14_learned_routing.py` | the eager per-token oracle (`routing_oracle_scan`), the persistent forward kernel with the score/argmax block (`_k14_seq_fwd_kernel`), the estimator (`_router_estimator`), the `x45::k14_scan_fwd/bwd` op pair, the wrapper (`routing_scan`), the mixer and the conversions (`RoutingSeparatedMixer`, `convert_model_to_routing`) |
| `test_k14_routing.py` | 18 tests: gates A/B (birth + flipped router parity, birth parity vs K10's launch, birth-uniqueness, feasibility masks, the crafted all-five-ops regime, the frozen-router training falsifier, the mixer-level autocast birth parity, ragged/bf16/determinism, the heterogeneous-lane MERGE pin), the finding, the estimator (end-to-end through the custom op + fp64 autograd), the six-gradients receipts, refusals, the conversion accounting (+553,200) |
| `k14_gate_receipt.py` + `receipts/k14_gates.json` | the registered-shape receipts: gates A/B/D, the birth-parity lane, the uniqueness lane, the flip-rate lane, the finding, the estimator receipt |
| `k14_probe_routing.py` + `run_k14_routing_leased.sh` | the matched-exposure probe (seeds 3401/3402, corrected scoring) with the op-mix/flip-rate instruments, and its detached leased driver (cards 5/6) |

No `k10_*`/`k11_*`/`k12_*` source, PREREGISTRATION, `lattice.py` or
`matched_arms.py` was edited; K10's spec helpers, its backward body, its
selection quantities and its mixer/conversion are IMPORTED.  The K11 lane
(cards 0/1), the K12 sources, the K13 lane (cards 0/1) and the X46 sweep
(cards 2/3) are untouched; this lane owns cards 5/6.

### 1.1 The semantics (the registration's 0, pre-stated)

1. **The birth anchor (P5).**  K10's chain quantities computed
   bit-identically (the same defined-order folds, the same `_round_f32`
   thresholds); the chain's own branch conditions (exactly one active per
   token) decide which op scores its own threshold-delta margin — every
   inactive op the reserved −1.0 — and the argmax reproduces the banked
   winner's threshold routing EXACTLY, uniquely, on every input.
2. **The learned router (P4, the simplest form).**  `s_k = M_k + g_k` with
   `g` ONE per-head `Linear(ctx*width -> 5)` per mixer over the DETACHED
   causal context, born at EXACT zeros with the K12 `+0.0` normalisation
   (+553,200 parameters at the registered shape, ~2% of GDN2's management
   mass — stated, asserted, not hidden).  Deeper modules and
   context+margin inputs are the named follow-ups.
3. **The feasibility masks (semantics guards, NOT learned).**  APPEND
   masked at no-free-slot, POP at no-live-rows — birth-neutral (the chain
   never picks them there), the P3 guard post-deviation.
4. **The recorded schedule doctrine INTACT.**  The router produces
   schedules; the scan stays exact given them; the Lean algebra is
   schedule-conditional so every learned schedule is licensed by the same
   theorems.  The per-op scores are a RECORDED channel (P1); the chain's
   own op is a collect channel (the flip-rate instrument).
5. **The estimator + the rescope.**  As quoted above, verbatim from the
   registration.

### 1.2 One engineering note for the record

K10's kernel applies `live0 = (op != POP)` to the target row even for NONE
ops — inert under the chain (a NONE always targets a live row on a full
store) but WRONG under a learned router that picks NONE with free slots,
when the recorded target may be dead: the K14 kernel masks the liveness
update by `op != NONE` (the oracle's own semantics), bit-identical at birth
(receipted: 0 flips).  Found by this lane's parity tests; no `k10_*` file
touched.

## 2. Gates A, B and D (the receipts: `receipts/k14_gates.json`)

**Gate A — parity at the registered shape** (batch 8, 2048 tokens, 2 heads,
rank 128, value_dim 128, key_dim 16, 64-token segments, bf16 inputs): every
channel — the schedule fields, the payload lane, the read weights, the
pre-decision liveness, the exit state/liveness/key table AND the recorded
per-op scores + the chain's own op — bit-equal against the eager oracle at
the BIRTH router and at a magnitude-0.9 FLIPPED router: **0 flips** in both
lanes; the read at the fp32-vs-fp64 floor (inside the 8× noise factor).
The **birth-parity lane**: the k14 launch at `g = +0.0` vs K10's OWN launch
— 0 flips on every K10 channel, the chain's op recorded token-for-token.
The **uniqueness lane**: the birth argmax margin ≥ 1.0 on every token (the
reserved-gap separation), the runner-up at the reserved constant on 100% of
tokens.  The **flip-rate lane**: 41.3% of ops moved by the magnitude-0.9
random router.  The suite adds: the crafted all-five-ops regime (the birth
histogram covers NONE/APPEND/CORRECT/POP/MERGE with parity), the
feasibility-mask corners (a magnitude-10 APPEND score cannot APPEND into a
full store; a magnitude-10 POP score cannot POP an empty store; the masked
scores record `-inf`), the frozen-router 3-step training falsifier, the
mixer-level birth parity under bf16 autocast (the K11B §0.5 lesson), ragged
segments, determinism, and the refusals by name.

**Gate B — the gradients.**  The six K10 gradients through the k14
backward (K10's own body, imported verbatim) sit within `NOISE_FACTOR` (8×)
of the fp32-vs-fp64 floors on the flipped-router fixture, and are BIT-EQUAL
to K10's own body's outputs on the same recorded channels (the receipt
that the estimator changes nothing else).  The router's gradient is NOT the
true adjoint — it is the admitted estimator, gated by the fp64 autograd of
its own DEFINED surrogate objective (max abs diff **7.0e-6**), nonzero at
the zero-born router at step 0, with the zero-gradient finding asserted on
the same fixture.  The fp64 gradcheck validates the six K10 channels
through the routing oracle at birth (the flipped regime is not gradcheckable
by construction — its argmax margins are arbitrary and the flip boundary is
the recorded-schedule doctrine's own wall; its gradients are gated by the
floors test instead), and the surrogate objective itself is fp64-gradchecked
smooth in the recorded scores.

**Gate D — the walls.**  Forward **19.0 ms vs K10's own 19.0 ms** per
(layer, window) scan at the registered shape — **the routing channel is
wall-clock free**; the backward is K10's body plus the estimator's torch
reduction on the recorded scores (15.0 ms per scan, no replay kernel at
all); 255 registers / 4 spills (K10's own 255/2, K11B's 255/6 — at the
register ceiling like its siblings, receipted).

## 3. The probe

`RESULT_ROUTING.md` (the six arms at seeds 3401/3402 under the corrected
scoring; the seed-3401 baseline reused from the K12 lane's own re-scored
receipt after the in-code config+protocol identity assert — the verified
post-cooldown anchor **1.6832871096474784**, the 1.6892 row being K12's
forward-degenerate fixed-w bracket, not the baseline).

## 4. Recommended next step

The operator reads `PREREGISTRATION_LEARNED_MANAGEMENT.md` and
`RESULT_ROUTING.md` and rules on the follow-ups the results name: the
read-path / trajectory-sensed estimator (the write-content surrogate's
verdict at probe scale), the context+margin module, or the tempered
surrogate — none run without a ruling.
