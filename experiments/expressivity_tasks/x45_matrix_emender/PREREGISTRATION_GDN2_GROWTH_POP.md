# Pre-registration (DRAFT — operator sign-off pending) — GDN2 with growth + POP: head-recycling, the exact-capacity op transplanted onto the dense machine

**Date:** 2026-11-22 · **Status:** draft. The POP semantics below are
PROPOSED (head-recycling) and are ESCALATED before the arms lock; nothing in
this draft is settled until that reply is recorded here. The saturation
regime the arms run in is ruled separately (recorded in §7).

**Authorisation:** the operator's new arm — GDN2 GROWTH+POP. The audit's C2
finding is the claim this experiment finally tests ON GDN2's SIDE:
[`RESULT_OP_SET_AUDIT.md`](RESULT_OP_SET_AUDIT.md) §1.3 — **POP is the only
exact-capacity operation in either machine's vocabulary** ("to free a stored
direction exactly, a machine must zero a row and mark it reusable
(`pop_death_iff`, `freeMask_iff`); GDN2's two erasure channels cannot: the
rank-1 erase acts only along the current key, and the decay `exp(g) < 1`
strictly, so the state is emptied only asymptotically, never at a step"), and
GDN2 has no analog. POP is idle in every winning arm of OUR machine (0 in
K8/K10/side-by-side; 0.031 in the banked K6 probe) — the controller program's
ownership-expiry verb, unexplored as a trained mechanism. This experiment
gives GDN2 the missing verb and runs the pair that isolates it.

**Companion registrations (imported, never edited):**
[`PREREGISTRATION_GDN2_GROWTH.md`](PREREGISTRATION_GDN2_GROWTH.md) — the
growth axis (heads), the state-saturation trigger family, the birth protocol,
the S-gate receipts; this draft builds directly on its committed machinery
(`gdn2_growth_smoke.py`, imported verbatim). The corrected scoring:
[`RESULT_PROTOCOL_CORRECTION.md`](RESULT_PROTOCOL_CORRECTION.md).

**Rules inherited verbatim:** GDN2's model code is READ-ONLY — the POP
mechanism is state surgery at kernel boundaries in the committed growth-smoke
pattern (a wrapper/registrar; no `k10_*/k11_*/k12_*` edits, no edits to the
existing PREREGISTRATIONs, `lattice.py`, or the C3 lane's files). The K8
ruling governs the trigger: **write-pressure statistics do not gate** (the
K8 idle-store receipt — our registered write-reuse trigger fired at rate
~1.0 on an *idle* store); the gating statistic measures the STATE itself.
Impossibility is escalated, never patched.

---

## 1. The semantics under test (proposed, escalated in §8)

**HEAD-RECYCLING.** At the fixed 64-token statistic boundaries (the kernel's
own `chunk_size = 64`, the registered granularity), for each (block, head):
when the per-head saturation statistic crosses the POP threshold, zero that
head's state — `S_h → 0` exactly — making the head *virgin again*. Nothing
else changes:

* **State surgery only.** No parameter is created, removed, or re-shaped;
  the head's keys, gates and readout stay exactly what they were (trained,
  live, in the optimizer with untouched AdamW state). The shape
  `[B, H, d_k, d_v]` is unchanged — kernel-legal by construction (the same
  per-head isolation the growth smoke receipted: the chunk kernels never mix
  heads).
* **Exactness is the point.** After the surgery the popped head's state is
  **exactly zero** — the one erasure neither GDN2's rank-1 erase (current key
  only, partial) nor its decay (`exp(g) < 1` strictly) can ever perform
  (audit §1.3). This is the transplant of our `S[j] = 0; free the slot` onto
  the dense machine: the *slot* is the head's whole state; the *free* is its
  return to virgin superposition capacity.
* **The honest asymmetry vs our POP.** Our POP frees an addressable slot
  (liveness bookkeeping, allocator, reusable address). GDN2 has no
  addresses: the freed capacity is the head's key space as a whole —
  recycling, not slot management. The audit's exactness claim (never at a
  step, vs our never-at-a-step lack) transfers; the allocator claim does not
  and is not claimed.

**The wake-up/reuse analysis (does a recycled head re-enter training
cleanly? — the design question this experiment must answer before the arms).
The answer is yes, and MORE cleanly than the growth birth, with one honest
cost:**

* *Re-entry is immediate.* A recycled head's parameters were never touched:
  its `k_proj` rows are live, so it writes into its virgin state **within
  the first post-pop chunk** (the growth registration §3's own analysis: a
  live-key head with zero state fills it by its own writes; S-2's receipt of
  3.37 Frobenius mass). Its readout path was never silenced (no `o_proj`
  zeroing — that was the GROWTH birth's trick): gradients flow to its
  parameters at the **first post-pop step** (PS-3's live-re-entry receipt) —
  no one-step wake-up is needed. The head re-enters training as a fully
  trained head with an empty state.
* *The gradient flow is the exact-capacity backward.* The pop zeroes the
  carried state at a chunk boundary; the kernel's backward returns `dh0`, so
  the carry is differentiable (the growth smoke's S-3 carry-grad receipt:
  relative diff 0.0), and the mask's backward zeroes the popped head's
  gradient flow from post-pop reads — **the popped content is gone and
  receives no gradient**, exactly our POP's semantics (`pop_death_iff`).
  The smoke receipts this exactly (PS-3: popped head's pre-pop state grad
  0.0, other heads flow normally).
* *The honest cost: a real output perturbation, by design.* Unlike the
  growth event (output-invariant by the `o_proj` zeroing), a pop REMOVES the
  head's contribution from the readout until its state refills — the
  capacity-freeing act is observable at the event (PS-1's live-tau row
  receipts the perturbation). If the popped content was load-bearing, the
  loss pays for the free locally. That is the registered POP trade on our
  machine too, and it is the mechanism under test: saturation says the
  head's key space is exhausted; recycling bets that a virgin state written
  by the CURRENT context beats a saturated state written by stale contexts.
* *Reuse statistics are first-class outputs.* Every fire is receipted
  (boundary, head, the per-head smoothed PR that fired it, the saturation
  before/after) — the arms report how many pops, which heads, and the
  saturation trajectory (Deliverable 2's registered reporting).

## 2. The trigger (saturation-based, per the session's ruling)

**The gating statistic — the per-head participation ratio**, the same
statistic family the supervisor decided for growth (§5 of the growth
registration), at per-head granularity instead of mean-over-heads. Per
(block, head, boundary): `PR(S_h) = (Σσ)² / Σσ² ∈ [1, d_k]` of the boundary
state `S_h ∈ ℝ^{128×256}`; **batch-mean** per head; **smoothed over the last
4 boundaries** (reset on fire; reset at window start); fires at

```
smoothed batch-mean PR_h ≥ τ_pop
```

**Why saturation and not a write-pressure statistic:** the K8 receipt
(write-reuse fired at ~1.0 on an idle store) — the same ruling that gated
growth. The delta-self-interference co-diagnostic ι of the growth
registration stays NON-GATING and is reported alongside on every arm.

**The threshold:** the registered candidate is `τ_pop = 0.75 · d_k = 96`,
the per-head analog of `τ_sat`. No per-head PR receipts exist; the disclosed
calibration protocol is the growth registration's own (§5), run by THIS lane
on the same saturation-regime stream so the receipts cross-check the C3
lane's: the FIXED arm's per-boundary per-head PR is receipted in-regime; if
96 is unreachable in-regime, the calibrated threshold is fixed and committed
BEFORE the arms run; otherwise 96 stands. Per the supervisor's regime
ruling (§7), the calibration scans BOTH horizons (context 8192 and 16384)
and picks the SHORTEST horizon at which `τ_sat = 96` is reachable for the
growth statistic; if neither reaches, the receipt is escalated (no arm runs
on an invented horizon choice).

**Interaction with growth (the composed arm).** Both triggers evaluate at
the same 64-token boundaries from the same probe trajectory. Precedence:
**growth first** (a model-wide event at the registered mean-over-(block,
head, batch) statistic ≥ τ_sat, resetting all per-head bookkeeping at the
new head count — the newborn heads are virgin, PR ≈ 0, and cannot fire);
then per-head pops (an individually saturated head recycles without growing
the machine). The layering is the design: growth relieves the LAYER's
capacity pressure; POP relieves ONE HEAD's. The birth receipt (PS-1)
guarantees the composed arm with pop disabled is bit-equal to GDN2+growth.

## 3. Birth: pop disabled ⇒ bit-equal to GDN2+growth

The pop machinery is compiled into the same walk as growth (one code path,
`segment_walk`): with `τ_pop` unreachable the trigger is still evaluated at
every boundary and decides nothing, and the armed walk is **bit-identical**
to the growth-only walk (PS-1's receipt, across a mid-window growth event).
The control arm of the pair (§6) is therefore the registered GDN2+growth
mechanism verbatim, run fresh in this session and regime; the POP arm
differs from it by exactly the fires of §2. The wrapper parity gate (PS-0)
receipts the stronger claim the arms depend on: the segmented real pass
with NO events reproduces the core's own training forward exactly (the
growth smoke's S-2 bit-identity: the segmentation is the kernel's own chunk
decomposition).

## 4. THIS STAGE's scope: the feasibility smoke (no quality claim)

Block-level, on the committed growth-smoke pattern (one registered-shape
GDN2 block, 2 heads; `gdn2_growth_pop_smoke.py`, importing
`gdn2_growth_smoke.py` verbatim), receipts to
`receipts/gdn2_growth_pop_smoke.json`:

* **PS-0 wrapper parity** — the never-fires segmented real pass vs the
  core's own training forward, fp32 and bf16 autocast.
* **PS-1 inertness at birth** — pop disabled (unreachable τ, trigger
  evaluated) bit-identical to the growth-only walk across a mid-window
  growth event; the same machinery at a reachable τ DOES fire.
* **PS-2 carry legality across a pop** — the popped head's boundary state
  EXACTLY zero; per-head isolation (never-popped heads bit-identical to
  the un-popped walk); the output untouched before the first pop boundary;
  refill mass within one chunk; pops composed with a growth event (incl. a
  pop on a born-new head).
* **PS-3 gradient legality** — the masked carry's leaf grads bit-equal to
  the explicit construction; the popped head's pre-pop state receives
  EXACTLY zero gradient from post-pop segments; training through forced
  pops: finite, the recycled head's parameters live at the first post-pop
  step, the parameter set (and with it the optimizer state) untouched.
* **PS-4 long-horizon kernel tolerance** — forward+backward at the
  regime's contexts T ∈ {8192, 16384} at H ∈ {2, 4, 8} plus growth+pop
  walks at the long shapes: the fresh kernel-shape receipts the regime
  ruling requires, walls and peaks receipted. Any kernel impossibility at
  the long shapes is escalated, not patched.
* **PS-5 statistic anchors** — the per-head PR (τ_pop's statistic) and the
  mean-over-heads PR (τ_sat's statistic) from the real kernel's own
  trajectory at init at T=8192, plus the SVD wall (the calibration's
  overhead receipt).

**Failure handling:** any PS-gate that cannot be met with GDN2's code
untouched is an impossibility receipt and an escalation, not a patch.

## 5. Falsifiable predictions (pre-stated, both directions live)

No directional prediction on the headline — both outcomes are results (the
session's standing rule):

* **P-P1 (the binding receipt, inherited).** In the saturation regime the
  fixed-state bound's shadow binds the control (the growth registration's
  P-G1): saturation statistics climb with horizon/association load.
* **P-P2 (the op thesis).** If exact-capacity recycling pays on the dense
  machine: GDN2+growth+POP beats GDN2+growth by a margin outside the
  endpoint-variance band (0.075) under the corrected primary score, with
  fires receipted in-regime (saturation before/after the pops measured and
  reported).
* **P-P3 (the honest controller-only verdict).** If POP adds nothing even
  where its trigger binds — the arms tie, or the POP arm loses — then the
  C2 exact-capacity claim is controller-only ON GDN2's ARCHITECTURE TOO:
  the verb is legal, cheap, and worthless for byte-LM quality at this
  scale, the first positive-or-negative receipt the op has ever had on
  either machine's own lanes. A tie does NOT refute the formal claim (the
  fixed-state exit exists and is exact — that is a theorem); it receipts
  the claim's quality irrelevance in this regime, exactly as K8 receipted
  growth's tie (2.0186 / 2.0240 / 2.0278).
* **P-P4 (never-fires degeneration).** If τ_pop never fires in-regime
  after calibration, the POP arm degenerates to the control — the
  informative null, reported as such (the K8 tie branch).

## 6. The arms (LOCKED ONLY AFTER the §8 escalation reply)

Same-session pair, matched exposure, the saturation regime per the
supervisor's ruling (§7): **`gdn2_growth`** (the control: the registered
GDN2 arm + the growth registration §3 verbatim, run fresh in this session
and regime) and **`gdn2_growth_pop`** (the same + §1's head-recycling).
Matched exposure = the standing probe rule (20,000,000 targets,
train-split, the honest clock), constant-LR phase then the corrected
24-step linear cooldown; **the post-cooldown calibration eval is the
PRIMARY score** (constant-LR step-180 eval retained for comparability;
EMA-0.99 secondary, ranking-only) per RESULT_PROTOCOL_CORRECTION.md.
Parameter deltas disclosed (POP adds none; growth's are the registration's
own). The POP arm's recycling statistics are reported as first-class
receipts: fires per step, per (block, head), the smoothed PR at each fire,
saturation before/after, refill trajectories. Single seed (3401, the
standing seed), stated in every receipt. Impossibility, calibration
anomalies, and material cross-check divergence from the C3 lane's receipts
are escalated, never silently reconciled.

## 7. The saturation regime (the supervisor's ruling, 2026-11-22)

Option (B): the **long-horizon variant of the registered Pile stream** —
same documents, longer windows via `lattice.Stream.windows(context=...)`
with NO `lattice.py` edit; document-bounded windows only (never pack across
boundaries, truncate at the boundary, document-disjoint calibration/test
splits). **Horizon scan:** this lane's own τ_sat calibration runs at BOTH
context 8192 and 16384 and picks the SHORTEST horizon at which
`τ_sat = 96` is reachable; if neither reaches, escalate with the receipt —
no arm runs on an invented horizon choice. Budget = the standing probe
rule: 20M-target matched exposure, 180 constant-LR steps + the 24-step
cooldown primary. Fresh kernel-shape receipts at the longer T (PS-4). The
C3 lane (cards 4/5) runs the four-arm version under the identical ruling;
this lane's calibration receipts cross-check theirs, and material
divergence is escalated.

## 8. The escalation (recorded before the arms lock)

The POP semantics choice is escalated with this draft and the smoke
receipts: **head-recycling (proposed)** vs the alternatives the analysis
surfaces, with the refusals stated:

* **(a) Singular-direction zeroing** ("partial pop": zero only the top
  singular directions of `S_h`) — REFUSED: not an exact-capacity act in the
  audit's sense (the whole-slot zero is), requires an in-wrapper SVD at
  every boundary, and is not the registered POP semantics on our machine
  either.
* **(b) Decay boosting for saturated heads** ("soft pop": raise that
  head's effective decay) — REFUSED by the audit's own finding: decay
  `exp(g) < 1` never reaches zero at a step; the exactness claim (the
  entire point of C2) is forfeited; also parameter/gate surgery per head
  at runtime, not state surgery.
* **(c) Token-granularity or mid-chunk pops** — REFUSED: the registered
  K8 discipline is boundary-only discrete ops, and mid-chunk surgery would
  need segmentation inside the kernel's own chunk (a GDN2 code edit,
  forbidden).
* **(d) Head-recycling at 64-token boundaries on the per-head saturation
  statistic (PROPOSED)** — exact, kernel-legal by per-head isolation,
  state-only, live re-entry, the honest analog of our POP on the dense
  machine.

The supervisor's reply is recorded here verbatim when it lands; the arms
(§6) lock only under it.

## 9. Commands (this stage)

```bash
eval "$(GPU_LEASE_VISIBLE=6 scripts/gpu_lease.sh acquire 1 --wait)"
OMP_NUM_THREADS=4 NDM_PIN_TRITON_AUTOTUNE=0 /home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_pop_smoke \
    --out experiments/expressivity_tasks/x45_matrix_emender/receipts/gdn2_growth_pop_smoke.json
```

## 10. SMOKE RECEIPTS (appended post-draft; the draft above is unchanged)

**Run:** 2026-11-22, card 6 (broker lease `GPU_LEASE_VISIBLE=6 ... acquire 1`),
`/home/erikg/emender/.venv/bin/python`, fp32 with tf32 disabled,
`NDM_PIN_TRITON_AUTOTUNE=0` (the growth smoke's receipted environment
knob). Receipt: `receipts/gdn2_growth_pop_smoke.json`; driver:
`gdn2_growth_pop_smoke.py` (the committed growth smoke imported verbatim).
**All six PS-gates MET.** One registered-shape GDN2 block, 2 heads —
block-level feasibility, no quality claim.

* **PS-0 wrapper parity — MET.** The never-fires segmented real pass vs
  the core's own training forward: fp32 max diff **1.98e-9** (output scale
  3.0e-3; ~6.5e-7 relative — the same autotune/reduction-order noise the
  growth smoke's S-1 receipted; not bit-identical, receipted as measured);
  **bf16 autocast: 0.0** (the bf16 rounding erases the fp32 noise). The
  wrapper the arms install reproduces GDN2's training forward to fp noise
  with no events.
* **PS-1 inertness at birth — MET.** Pop disabled (τ_pop unreachable, the
  trigger still evaluated at every boundary) is **bit-identical**
  (diff 0.0) to the growth-only walk across a mid-window growth event —
  the control arm IS GDN2+growth by construction. The same machinery at a
  reachable τ fires at **all 31 boundaries** (τ_pop = 1.0; every fire
  receipted with its smoothed statistic) and perturbs the output by
  3.4e-3 — the capacity-freeing act is observable, by design.
* **PS-2 pop carry legality — MET.** Forced pops at chunks 8/20/28 (head 1
  pre-growth; head 3, a born-new grown head; head 0, a shared head
  post-growth): the popped heads' states at the surgery boundaries are
  **exactly 0.0** (the exact-capacity receipt); the never-popped head's
  final state is **bit-identical** to the un-popped walk (per-head
  isolation); the output is **bit-identical (0.0) before the first pop
  boundary**; every popped head refills **within one chunk** (Frobenius
  mass 1.58–2.00 — live keys write into the virgin state immediately).
* **PS-3 gradient legality — MET.** The pop-masked carry's leaf gradients
  (q/k/v/b/w) are **bit-equal (diff 0.0)** to the explicit
  concatenate-a-zero-slice construction; the popped head's pre-pop state
  receives **exactly 0.0** gradient from post-pop segments while the other
  head's carry gradient flows normally (0.85e-3) — the exact-capacity
  backward, our `pop_death_iff` semantics on the dense machine. Training
  through forced pops (4 steps, pops every window): losses finite and
  descending (0.9996 → 0.9987), grads finite, the recycled head's k_proj
  and o_proj receive **nonzero gradient at the first post-pop step**
  (2.1e-6 / 1.3e-6 — live re-entry, no one-step wake-up needed), and the
  parameter set is **unchanged** across every pop (a pop is state-only
  surgery; the optimizer state is untouched by construction).
* **PS-4 long-horizon kernel tolerance — MET.** Whole-window forward+
  backward at the regime's contexts, all finite: T=8192 walls
  0.045 / 0.052 / 0.090 s (peaks 0.97 / 1.57 / 3.05 GB) at H = 2 / 4 / 8;
  T=16384 walls 15.9 / 16.1 / 54.0 s (peaks 1.91 / 3.09 / 6.04 GB) — the
  T=16384 walls are the **first-call Triton autotune sweep at the new
  shape** under `NDM_PIN_TRITON_AUTOTUNE=0` (the receipted environment
  knob, a one-time per-shape cost, not a kernel fault); the growth+pop
  segment walks (the arms' real path, 64-token chunks) run the full
  T=8192 / T=16384 windows with pops at every boundary in **4.7 / 9.5 s**,
  outputs and states finite. **No kernel impossibility at any long-horizon
  shape.**
* **PS-5 statistic anchors — RECEIPTED.** At init, random inputs, T=8192
  (127 boundaries): the per-head PR (the POP trigger statistic)
  head 0 ≈ **46**, head 1 ≈ **67** (smoothed max; raw max 46.6 / 67.7) —
  per-head PR is strongly heterogeneous at init, the first evidence the
  per-head granularity is a real axis (a head can saturate while the
  layer mean does not); the mean-over-heads growth statistic max
  **56.9** — both below the registered 96 at init on random inputs, as
  intended; the in-regime calibration on the C3 stream (§7) is the
  threshold's authority. The SVD wall for the full per-boundary PR
  trajectory: **2.7 s** per walk at T=8192 (batch 2, one block) — the
  calibration's overhead receipt.

**Verdict of the feasibility stage:** head-recycling is kernel-legal on
GDN2 by construction (state-only surgery at the kernel's own 64-token
boundaries; shapes unchanged; per-head isolation receipts the
never-popped heads bit-identical), exact (the only exact zero either
machine's erasure vocabulary can produce), differentiable (the
exact-capacity backward receipted: popped content receives zero
gradient), and live (a recycled head re-enters training at the first
post-pop step, no wake-up protocol needed). The arms (§6) lock only
after the §8 escalation reply.

