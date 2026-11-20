# Pre-registration draft — LEARNED MANAGEMENT ROUTING (K14): the op routing itself made a learned policy, born at the banked thresholds

**Date:** 2026-11-24 · **Status: DRAFT, operator sign-off pending.** This is a
NEW registration (it changes WHICH op a token's write takes — the
op-routing decision channel, the C5 receipts' poorest surface: **27.7K gate
params + three hand-set thresholds vs GDN2's 27.4M learned management
parameters**), written and committed BEFORE any probe measurement is
claimed.  The operator's correction is the registration's frame:
**zero-learned-management was never a mandate** — the op-routing decision
channel is an open lever.  The birth construction was escalated and
SUPERVISOR-APPROVED before this engine was locked (the five rulings cited
below); probe-scale engineering checks (parity gates, gradchecks, the
estimator receipts, memory, timing) are pre-registration evidence and are
banked (`receipts/k14_gates.json`, the 18-test suite green, the K10 suite
green with the module loaded); **no board-scale (2700 s) claim is run before
the operator signs this document off.**  No `k10_*`/`k11_*`/`k12_*` source,
PREREGISTRATION or `matched_arms.py` is edited — K10's spec helpers, its
backward body, its selection quantities and its mixer/conversion are
IMPORTED; the K11 lane (cards 0/1), the K12 sources, the K13 lane and the
X46 sweep are untouched.  This lane owns cards 5/6.

## 0. The construction (supervisor-approved before the engine was locked)

The threshold chain (best cos `>= tau_match` → CORRECT; `+ second >=
tau_merge` → MERGE; else free slot → APPEND; else full & worst `<= tau_pop`
→ POP; else NONE) is a **LEXICOGRAPHIC TREE**, and an argmax over per-op
scores can only reproduce a tree if the scores ENCODE its branch structure
(additive raw margins provably cannot: in the no-match region an additive
CORRECT score `d1 - d2` is unbounded above while the chain says APPEND).
The approved birth form — **branch-active margins + the reserved runner-up**:

1. **The birth anchor (P5).**  Per token, K10's own chain quantities
   computed bit-identically (the same defined-order folds, the same
   `_round_f32` thresholds): `d1 = best − tau_match`, `d2 = second −
   tau_merge`, `d3 = tau_pop − worst`.  The chain's OWN branch conditions
   (exactly one active per token) decide which op scores its own
   threshold-delta margin — MERGE: `d1+d2`, CORRECT: `d1`, APPEND: `−d1`,
   POP: `d3`, NONE: `+0.0` — and **every inactive op scores the reserved
   constant −1.0**.  Every active margin is provably `>= 0` and the
   reserved constant is `< 0`, so the argmax is the chain's winner UNIQUELY
   on every input: **bit-exact birth with NO tie-break reliance** (the
   `receipts/k14_gates.json` birth-uniqueness lane: min margin 1.0, the
   runner-up at the reserved constant on 100% of tokens).  The winner's
   score IS its threshold delta — "score margins born at the threshold
   deltas", literally.
2. **The learned router (P4, the simplest form).**  The full per-op score
   is `s_k = M_k + g_k` where `g` is ONE per-head `Linear(ctx*width ->
   5)` per mixer over the DETACHED causal context (the same 2304-dim
   window the key projections see), born at EXACT zeros with the K12
   `+0.0` normalisation (IEEE `−0.0 + +0.0 = +0.0`), so at birth `s = M`
   bit for bit and the FROZEN-router control is the banked winner bit for
   bit FOREVER.  **+553,200 parameters** at the registered shape (24 mixers
   × 2 heads × (5·2304 + 5)) = **0.37% of the machine — ~2% of GDN2's
   27.4M management mass**: routing becomes learned at probe scale.  Deeper
   modules and context+margin inputs (`d1,d2,d3` as module features) are
   the NAMED FOLLOW-UPS, not this registration.
3. **The feasibility masks (semantics guards, NOT learned — the
   supervisor's own ruling).**  APPEND is masked to `−inf` when no free
   slot exists and POP when no live row does.  The chain never picks them
   there, so birth parity is untouched; the masks keep a trained router
   from ever APPEND-writing into a live row (P3) or POPping an empty
   store.  Every other op is mechanically well-defined for any token.
4. **The recorded schedule doctrine INTACT (the Lean licence).**  The
   router produces the op schedule; the scan stays exact given it.  The
   Lean algebra is SCHEDULE-CONDITIONAL — `EmenderMemory.token_conditional_scan`,
   `Event.combined_chunked_scan`, `MemoryOperator` scan-composition — so
   **every learned schedule is licensed by the same theorems**; the
   per-token scan remains bit-exact against the eager oracle (gate A: 0
   flips at the registered shape at the birth router AND at a flipped
   router that moves 41.3% of ops).  The argmax is DISCRETE at the forward
   and recorded; the per-op scores are a RECORDED channel `(b, T, h, 5)`
   (P1) so the backward's estimator needs no replay.  The op-derived
   machinery per op (slot, lanes, blend, liveness) is K10's own, driven by
   the chosen op; P2 holds — the router chooses the op, never the
   thresholds' arithmetic, the payload, the key values or the liveness
   bookkeeping.
5. **The zero-gradient FINDING (the K12 wall, unchanged).**  Under the
   recorded-schedule doctrine the op choice is a backward constant, so the
   STRUCTURAL gradient to the router through the eager oracle is EXACTLY
   ZERO (asserted in the tests, `oracle_drouter_is_none: true` in the gate
   receipt).  "The model can learn which op to use" REQUIRES an admitted
   estimator — the same structural fact the placement lane surfaced, now
   on the op channel.
6. **The admitted estimator (K12-identical, supervisor-approved).**  The
   kernel path's backward is K10's own `_k10_backward_body` imported
   VERBATIM (the six value/key gradients BIT-EQUAL to K10's own body on
   the same recorded channels — the gate receipt) plus ONE small torch
   reduction on the recorded scores: with `p = softmax(s over the five
   ops)`, `k*` the recorded op and `c_t = <dpayload_t, p_t>` the EXISTING
   value walk's own write-content output (no new walk),

   `dL/ds_k = Σ_t c_t · p_{k*} · (δ_{kk*} − p_k)`,

   returned as `d(router_scores)` so autograd chains it through the router
   module; the score inputs are DETACHED, ONLY the router learns.  The
   estimator equals the fp64 autograd of its own DEFINED surrogate
   objective to 7.0e-6 on the gate fixture, is nonzero at the zero-born
   router at step 0 (P5's step-0 receipt), and the surrogate objective is
   itself fp64-gradchecked (smooth in the recorded scores).

   **The estimator's bias, disclosed VERBATIM (the supervisor's rescope):
   `c_t` is zero on MERGE/POP/NONE tokens (the imported backward's
   `dpayload` vanishes on those lanes), so the estimator reinforces the
   chosen op where the write content mattered and gives NO direct signal
   to tokens the chain routed to MERGE/POP/NONE.  A null on the
   POP/MERGE-revival prediction (M-M2) is therefore UNINTERPRETABLE as a
   learned-management verdict for those ops — it receipts only that the
   write-content surrogate does not indirectly flip content-carrying tokens
   into them; a read-path POP/MERGE estimator (the row's read-landscape
   sensitivity) is the named follow-up.**

## 1. The machines and the arms

Every arm is the banked K10-keys probe machine VERBATIM
(`matched_arms.build_matrix("matrix_ops", 128, seed, k5b=False)` + the
parameter-neutral separated-keys conversion, the audited 149,981,808), the
routing arms adding the K14 module (+553,200, asserted in-code).  The
recorded-schedule doctrine makes the discrete op sequence a generated
object; the frozen-router falsifier is the banked winner itself (below).

| arm | machine | router |
|---|---|---|
| `baseline` (the FROZEN-THRESHOLDS control) | the banked winner (thresholds) | none (the K10 machine) |
| `routing_learned` | + the K14 module | zero-born, trainable via the estimator |
| `routing_frozen` (the FROZEN-ROUTER falsifier) | + the K14 module | pinned at exact zeros, `requires_grad=False` |

**The baseline anchor, verified before locking (the supervisor's required
correction, applied).**  The seed-3401 baseline row is the K12 lane's own
re-scored baseline (`receipts/k12_baseline_seed3401.json`):
constant-180 eval **1.841756020273481** (asserted bit-identical to the
banked run's own `training.jsonl` rows and eval) and post-cooldown
**1.6832871096474784** — NOT the 1.6892253501074654 row, which is K12's
forward-degenerate fixed-`w` bracket arm (`placement_fixed0p25/0p5/1p0`, all
three bit-identical).  **The lineage choice, stated:** the comparison runs
on the **ops-diagnosis lineage** (the banked AdamW nomination optimizer,
where the audit's degeneration receipts — CORRECT 0.97939 identity-dominated,
MERGE 0.00313, POP 0.00000 — and K12's full mechanism telemetry live).
BOTH GDN2 references are quoted as CONTEXT, not as compared rows: the
same-machine gap (GDN2 post-cooldown **1.3522** at its own protocol) and
the best-vs-best structural residual (**~0.08**: our speedrun-era 1.4338 vs
GDN2's 1.3522, `RESULT_PROTOCOL_CORRECTION.md`), which is the "~0.07" the
M-M4 read measures against.  Reuse of the K12 seed-3401 baseline receipt is
made ONLY after asserting config+protocol identity (the same
`build_matrix` call, key_dim 16, ctx 4, segment 64, seed 3401, exposure
20,022,866, the same optimizer dict, the same corrected scoring) — asserted
in-code by the probe; the seed-3402 baseline runs FRESH.

**The falsifier bar (the K11B §2.4 class, stated exactly).**  The
frozen-router control's step-1 train loss MUST be BIT-IDENTICAL to the
banked same-seed row's (the 0-flip-at-birth bar at probe scale), and its
landing MUST be at parity with the banked same-seed row.  **Full-trajectory
bit-identity is EXPECTED here** (stronger than the general §2.4 bar): at a
zero-born frozen router the forward is the banked winner's own chain bit
for bit (gate A: 0 flips vs K10's own launch) and the six gradients are
K10's own body verbatim on an identical recorded schedule, so the only
allowed divergence source is none.  The per-step loss deltas against the
banked row are measured and receipted HONESTLY either way — if
reduction-order growth appears despite the identical schedule, it is
disclosed like K11B §2.4's own caveat, never silently weakened.

## 2. The probe protocol (the lane's matched-exposure discipline, the
corrected scoring instrument)

Every arm runs the banked K10-keys probe protocol VERBATIM — the same
build order, the registered flagship stream, the same data order, the same
AdamW (the banked row's own optimizer config, fused), the same warmup, grad
clip and honest clock — to the matched-exposure landing (20,022,866
targets, 180 constant-LR steps), then the CORRECTED SCORING
(`RESULT_PROTOCOL_CORRECTION.md`, the standing discipline):

* **PRIMARY** — the post-COOLDOWN eval: a 24-step linear cooldown
  (13.3% of 180; factor `(25−k)/24`, never zero) continues the same
  stream, then the document-disjoint calibration eval (7 windows / 14,336
  targets).
* **SECONDARY (ranking-only)** — the EMA-0.99 iterate at step 180.
* **Banked comparability** — the constant-LR step-180 eval, retained; the
  seed-3401 baseline reproduces the banked **1.841756020273481**
  (asserted in-code); every LEARNED/frozen arm's step-1 train loss must
  equal the banked row's step-1 loss bit-identically (the 0-flip-at-birth
  bar at probe scale, asserted in-code).

Seeds **3401 and 3402, both registered upfront** (the operator's task names
both): 3401 primary, 3402 the seed-matched twin for every arm (baseline,
learned, frozen) — six arms total, one leased card each, cards 5/6.

Mechanism telemetry on every arm (the K10 probe's own instruments — op
rates, per-token occupancy, read entropy, identity share, the
KEY-COLLAPSE watch — imported) PLUS this registration's own:

* **THE OP MIX IS THE HEADLINE INSTRUMENT** (the audit's degeneration
  diagnosis is what a learned router should move): per-arm per-token lane-0
  op rates, the TOTAL-VARIATION distance of the learned arm's op mix vs
  the seed-matched baseline's, and the **flip-rate instrument** — the
  fraction of tokens whose learned op differs from the chain's own op
  recorded at the same pre-token state (the chain's op is a collect channel
  of the K14 kernel; it is well-defined per token even after the
  trajectories diverge);
* the trained router telemetry: per-layer router weight norms, the score
  margins, and the per-op learned contributions `g`;
* the read-mask sweep (`side_by_side.k10_read_mask_sweep`, the C4
  diagnostic) on every arm.

## 3. PRE-REGISTERED MECHANISM PREDICTIONS (committed before the probe)

The anchors are the BANKED K10-keys probe row (seed 3401,
`receipts/k10_probe.json`, re-measured under the same instruments by the
K12 baseline): eval NLL **1.841756020273481** (post-cooldown
**1.6832871096474784**), occupancy **24.03** live slots, read entropy
**1.6541** nats (≈ **5.23** effective rows), CORRECT **0.97939**, APPEND
**0.01748**, MERGE **0.00313**, POP **0.00000** per token, identity share
**0.1032**; and the C5 gap references quoted as context in §1.

* **M-M1 (THE OP MIX IS THE HEADLINE INSTRUMENT — the learned op mix
  shifts materially from the threshold mix).**  Met if the learned arm's
  op mix TV-distance vs the seed-matched baseline's own is **>= 0.02**,
  OR the flip-rate instrument reads **>= 1% of tokens**.  Unchanged if TV
  <= 0.005 and flip-rate <= 0.1%.  **Refuted if the router converges back
  to threshold behaviour — which would itself receipt that the thresholds
  were already optimal** (the refutation is a first-class result either
  way).
* **M-M2 (POP/MERGE usage rises where the audit found them idle:
  POP 0.00000, MERGE 0.00313).**  Met if POP rises to **>= 0.001/token**
  or MERGE to **>= 0.01/token** (≈3x banked).  **Read with the §0.6 rescope
  verbatim: a null here is UNINTERPRETABLE as a learned-management verdict
  for those ops — it receipts only that the write-content surrogate does
  not indirectly flip content-carrying tokens into them; the read-path
  POP/MERGE estimator is the named follow-up.**
* **M-M3 (identity-CORRECT share falls).**  The audit's diagnosis is
  CORRECT as identity-dominated byte rewriting (identity share 0.1032);
  a router that moves management should thin it.  Met if the learned
  arm's identity share falls **>= 0.02 absolute** below the seed-matched
  baseline's own re-measured value; unchanged within ±0.005; refuted if
  it rises.
* **M-M4 (THE NLL QUESTION — no directional prediction).**  If the
  learned router's post-cooldown NLL closes **>= 0.02 nats** of the ~0.07
  best-vs-best structural residual vs the seed-matched baseline under the
  same protocol, **learned management is a real lever and the leaner
  registration gains the router**.  If NLL parity with a MOVED op mix:
  the ops are **threshold-insensitive bookkeeping at probe scale** — a
  first-class receipt either way.  If parity with an UNMOVED mix: **the
  thresholds were optimal** (also first-class).  All NLL deltas are read
  against the lane's standing 0.075-nat endpoint-variance band: anything
  inside it is directional, never banked.
* **Refutation clause.**  If the mechanism numbers do not move while NLL
  does (or vice versa), the dissociation is reported honestly.  A null is
  a result.  The M-M2 rescope (§0.6) binds the POP/MERGE reading verbatim.

## 4. Claim set (registered, pending sign-off)

* **S1 (quality).**  The arms train at the registered shape under the
  corrected protocol; the seed-3401 baseline reproduces the banked row's
  constant-180 eval bit-identically; every new arm's step-1 loss equals
  the banked row's bit-identically; M-M4 evaluated with the receipts, the
  three-way read and the endpoint band stated above.
* **S2 (mechanism).**  M-M1/M-M2/M-M3 evaluated against the seed-matched
  anchors, each reported met/not-met/refuted; the M-M2 rescope applied
  verbatim; the falsifier bar (§1) receipted per-step.
* **S3 (algebra).**  The gates: birth parity bit-equality (0 flips vs
  K10's own launch), parity at the birth and the flipped router, the
  birth-argmax uniqueness lane, the frozen-router falsifier, the
  zero-gradient finding, the estimator-vs-fp64-autograd receipt, the
  six-gradients-bit-equal receipt, the fp64 gradchecks (the six channels
  at birth; the surrogate objective through the router path), and the
  K10 suite green with the module loaded — all in
  `receipts/k14_gates.json` and the tests, BEFORE any probe claim.
* **S4 (cost).**  The honest walls/tok/s per arm against the baseline's
  own (gate D: the routing channel is WALL-CLOCK FREE at the registered
  shape — forward 19.0 ms vs K10's own 19.0 ms, 255 registers / 4 spills;
  the backward is K10's body plus the estimator's torch reduction, 15.0
  ms per scan).

## 5. Honest scope and caveats

* The +553,200 router parameters are 0.37% of the machine (the banked
  winner has none) — stated; the comparison is parameter-matched to within
  0.4%, and the C5 contrast (2% of GDN2's management mass) is the honest
  frame, not a parity claim.
* The estimator is a SURROGATE, not the true adjoint (§0.5/§0.6); the
  learned arm's router trajectory is read with that bias disclosed, and
  the frozen-router control is the estimator-free falsifier.
* The context-only module input (the supervisor-approved P4-simplest
  form) cannot see the margins directly; the margins enter only through
  `M`'s additive encoding.  The context+`d1,d2,d3` module is the named
  follow-up if M-M1 lands "unchanged".
* The birth margins include the reserved −1.0 as a registered constant:
  it sets the post-birth flip barrier (margin_winner + 1.0, ~1.2–1.5
  typical at probe) and the STE softmax's temperature at birth; any
  value < 0 preserves birth exactness, and the value is disclosed, not
  tuned.
* Single protocol, probe scale, seeds 3401 and 3402; the 20M matched
  exposure; the 0.075-nat endpoint-variance band applies to every NLL
  endpoint here; the board-scale (2700 s) run is gated on the operator's
  sign-off of this document (and the K10/K11/K12 drafts it inherits).
* One engineering note for the record (found by this lane's parity tests,
  in THIS module's kernel only; no `k10_*` file touched): K10's kernel
  applies `live0 = (op != POP)` to the target row even for NONE ops —
  inert under the chain (a NONE always targets a live row on a full
  store) but WRONG under a learned router that picks NONE with free
  slots, when the recorded target may be dead.  The K14 kernel masks the
  liveness update by `op != NONE` (the oracle's own semantics),
  bit-identical at birth (receipted by the birth-parity lane: 0 flips).

## 6. Files

`k14_learned_routing.py` (the oracle, the forward kernel, the estimator,
the op pair, the wrapper, the mixer, the conversion), `test_k14_routing.py`
(18 tests: gates A/B, the finding, the estimator, the falsifier, the
feasibility masks, the birth-uniqueness lane, the heterogeneous-lane pin,
refusals, determinism, the conversion accounting), `k14_gate_receipt.py` +
`receipts/k14_gates.json` (the registered-shape gates), `k14_probe_routing.py`
+ `run_k14_routing_leased.sh` (the probe and its detached leased driver,
cards 5/6), `receipts/k14_*.json` (the arm receipts + the assemble
summary), `K14_ROUTING.md`.  No `k10_*`/`k11_*`/`k12_*` source is edited.
