# RESULT_PLACEMENT.md — the learned placement policy measured: the honest null, with the machinery receipted as fully functional and the operator's prior direction refuted

**Date:** 2026-11-23 · **Status: complete — all six arms landed. THE HEADLINE: free-slot placement is decorative on natural text at probe scale. Every arm's NLL delta sits inside the 0.075-nat endpoint-variance band and below the pre-stated 0.01 win bar; the operator's prior direction ("write where the evicted content lived" — locality) was forced and HURTS (+0.0059); the supervisor-approved completing control shows the anti-local sign ALSO hurts (+0.0047); the learned arm is the only arm ahead (+0.0062, directional-only, unattributable to either sign at fixed magnitude) and its own optimizer left |w| ≤ 0.03 — the model's own verdict that the lever is not worth pulling, exactly as the registration's F1 rarity analysis predicted. The three-way read lands on (iii), with the zero-gradient finding and the estimator machinery as this stage's durable receipts.** ·
**Authorisation:** the operator's K12 task (the named lever: the op-set audit's ALLOCATION gap), with the supervisor's pre-build rulings on the three escalated points — (1) the admitted estimator (option (a), straight-through softmax, detached score inputs, only the per-head w learns; option (c) rejected); (2) the searched-placement control (the fixed-w bracket), later corrected in-flight: the ruled positive bracket proved FORWARD-DEGENERATE (argmax(w·sim) is invariant under positive scaling — three arms that are provably one machine, their bit-identical landings a determinism receipt), and the completing NEGATIVE control was approved; (3) the birth tie-break and residue form confirmed as specified. The registration is the DRAFT `PREREGISTRATION_PLACEMENT.md`, committed before any probe measurement; its predictions are evaluated below verbatim, unmet bands included.

## 0. The ladder, stated first

All six arms at the exact probe protocol verbatim (the 20,000,000-target matched-exposure rule → the 20,022,866 landing, 180 constant-LR steps, the banked K10-keys 1.8418-lineage config, AdamW, seed 3401, the registered flagship stream, the document-disjoint calibration eval at 7 windows / 14,336 targets, the honest clock), then the corrected scoring (24-step linear cooldown PRIMARY, EMA-0.99 secondary, constant-180 retained for banked comparability).

| arm | constant-180 (banked-comparable) | **post-cooldown (PRIMARY)** | EMA-180 (secondary) | Δ vs baseline (PRIMARY) | fire rate (of APPENDs) |
|---|---:|---:|---:|---:|---:|
| **baseline** (the banked K10-keys config re-scored) | **1.841756020273481** (= the banked row, BIT-IDENTICAL) | **1.68329** | 1.80540 | — | — |
| **placement_learned** (w zero-born, the estimator) | 1.81986 | **1.67710** | 1.80354 | **+0.0062** (directional-only) | 12.5 % (925/7376) |
| placement_fixed+0.25 | 1.85084 | 1.68923 | 1.81280 | −0.0059 (worse) | 10.9 % |
| placement_fixed+0.5 | 1.85084 | 1.68923 | 1.81280 | −0.0059 (identical: degenerate) | 10.9 % |
| placement_fixed+1.0 | 1.85084 | 1.68923 | 1.81280 | −0.0059 (identical: degenerate) | 10.9 % |
| placement_fixed−0.25 (the completing control) | 1.83242 | 1.68798 | 1.80520 | −0.0047 (worse) | 12.6 % |

Three protocol receipts carry the table:

* **The baseline reproduces the banked row BIT-IDENTICALLY end-to-end**: its constant-180 eval equals the banked **1.841756020273481** to the last digit, and all 180 constant-phase step losses are bit-identical to the banked run's own `training.jsonl` (asserted in-code) — the strongest protocol-equivalence receipt any lane has banked, now under the corrected instrument too.
* **The learned arm's step-1 loss is bit-identical to the banked row's** (the 0-flip-at-birth bar at probe scale); its step-2 loss diverges exactly as designed — the estimator moved the zero-born `w` off `+0.0` after the first step.
* **The three positive fixed arms are bit-identical to EACH OTHER** — the forward-degeneracy receipt (below).

## 1. The three-way read, landed

The pre-stated three-way read (the supervisor's ruling), evaluated against the receipts:

> **(iii) THE HONEST NULL: free-slot placement is decorative on natural text.**
> Every arm's delta sits inside the 0.075-nat endpoint-variance band and below
> the pre-stated 0.01 win bar; BOTH fixed signs hurt slightly (+w −0.0059
> worse, −w −0.0047 worse) while the learned arm's +0.0062 is
> directional-only and unattributable to either sign at fixed magnitude. The
> estimator MACHINERY functioned end-to-end — the zero-born w received
> gradient at step 0 (P5's step-0 bar), the per-head signs trained (29 of 48
> heads materially negative), placement fired on 11–13 % of APPENDs — but
> the optimizer left |w| ≤ 0.03: its own verdict that the lever is not worth
> pulling, exactly as the registration's F1 rarity analysis predicted
> (placement can differ from first-free only in multi-free-slot windows,
> which the banked op mix — POP 0.00000, MERGE 0.00313 per token — makes
> ~0.2 % of tokens). C5's write/read form owns the residual.

The operator's own framing, stated honestly: the prior direction — "write
where the evicted content lived" — was tested at full strength (the fixed
+w arm receipts the policy doing exactly what it is told: chosen-slot residue
similarity +0.036 vs the free-slot mean +0.000) and it **hurts**. Given the
choice, the model wrote AWAY from the dead residue's neighbourhood (29/48
heads negative) and then barely used the lever at all. The question "can the
model know where to write?" gets its answer: **it can — the machinery
receipt says so — and having the choice, it declines to use it.** That is
the more interesting answer, and it is a result, not a failure: it closes
the op-set audit's allocation question at probe scale.

## 2. The pre-registered predictions, evaluated (all against matched anchors)

* **P-1 (NLL) — direction MET, band MET, win bar NOT met.** The learned arm
  improves on the baseline (+0.0062 post-cooldown, inside the pre-stated
  [0.0, 0.03] band), but no arm clears the pre-stated ≥ 0.01 win bar → **no
  second seed**, receipted. Stated honestly: the constant-180 delta is
  larger (−0.0219) but the corrected score shrinks it 3.5× — the placement
  effect behaves like a mild training-dynamics effect, not converged
  quality.
* **P-2 (address locality forms) — NOT met.** Δ(Pearson r between live-pair
  slot distance and key cosine) vs the baseline's own −0.008: learned
  −0.003, fixed+w −0.013, fixed−w −0.018 — all short of the −0.02 bar. The
  placement policy itself does what it is told (the chosen-slot residue
  similarities: +0.036 at +w, −0.027 at −w — the sign structure works), but
  the live-pair geometry barely moves: the store's address space is not
  reorganising.
* **P-3 (effective read rows rise) — NOT met.** e^H: baseline 4.78, learned
  4.72, fixed+w 4.85, fixed−w 4.70 — the ±0.5-row band is nowhere near met;
  the arms move it ±0.07 in BOTH directions.
* **P-4 (decorative stores fall) — NOT met.** Read-mask layers with ΔNLL ≥
  0.02: baseline 7, learned 9, fixed+w 7, fixed−w 9 of 24 — short of the ≥
  10 bar. (The corrected-scoring baseline itself measures 7 — matching the
  side-by-side's AdamW-era anchor of 7 load-bearing layers.)
* **P-5 (MERGE may revive) — NOT met, honestly.** MERGE stays at
  0.0029–0.0030 per token everywhere (the banked row's own 0.00313) —
  near-address slots do NOT coalesce; there is **no positive per-token
  MERGE receipt**, and the K6 falsification stands.
* **The refutation clause does not need to fire**: the mechanism numbers did
  not move and the NLL did not materially improve — a coherent null, not a
  dissociation.

## 3. The durable receipts of this stage

1. **THE ZERO-GRADIENT FINDING** (the supervisor's own ruling, the stage's
   structural receipt): under P2+P3 the placement effect is purely discrete,
   so the structural gradient to `w` is EXACTLY ZERO (asserted on the
   oracle; `receipts/k12_gates.json` receipts it) — "the model can learn
   where to write" REQUIRES an admitted estimator, and that requirement is
   a structural fact of this architecture (Paper 1's zero-learned-
   management wall, surfaced from the placement side).
2. **The estimator machinery works**: the supervisor-approved straight-
   through surrogate (score inputs detached, only the +48 per-head `w`
   learns) reached the zero-born weight at step 0, trained per-head signs,
   and is receipted bit-precisely against its DEFINED torch reference (max
   diff 0.0) — while the six K10 gradients pass through unchanged (bit-
   equal to K10's own backward body). The birth bar (0 flips vs K10's own
   launch, 17.4 M lanes) and the frozen-w falsifier (bit-equal through
   training) hold.
3. **The forward-distinct placement family**: `{w = 0, w > 0, w < 0}` — the
   positive fixed-w bracket is scale-invariant (a defect of the ruling,
   corrected in-flight; the three arms' bit-identical landings double as
   the run-to-run determinism receipt). Any future placement work starts
   from this corrected family structure.
4. **The K10 eager oracle's latent multi-lane MERGE bug**, found by this
   lane's fixtures, fixed in this module's copy only, unreachable by K10's
   own gate fixtures and inert in production (no banked result affected);
   K11's oracle flagged as carrying the same pattern.
5. **The honest cost**: the placement channel costs ~5 % forward (20 vs 19
   ms/scan at the registered shape) and nothing at the score that matters.

## 4. Residuals, stated

* Single seed per arm (3401); the pre-stated win bar was not met, so the
  second-seed rule correctly did not fire — the NLL deltas here are
  directional reads inside the 0.075-nat endpoint band, never bankable
  claims.
* Probe scale, one protocol (20 M matched exposure + the 24-step cooldown
  extension); the 2700 s board and C14+ scale remain untested, as every
  prior lane stated for its own numbers. A null at probe scale does not
  preclude a board-scale effect — but nothing here earns the board run.
* The placement decision window is structurally rare (~0.2 % of tokens);
  the probe measures that rare window, and a true but tiny effect (≤ 0.01
  nats) would be invisible at this seed count — the honest scope of the
  null is "no effect ≥ the win bar", not "no effect whatsoever".
* The estimator's disclosed bias (write-content sensitivity, not trajectory
  sensitivity) bounds the learned arm's interpretation: the optimizer's
  verdict "leave |w| tiny" is evidence about the SURROGATE's landscape;
  option (c) (attached score inputs) remains the named follow-up if the
  operator wants the stronger test — but nothing in this result motivates
  it.
* The address-locality instrument's first implementation had a corrcoef
  axis bug (caught pre-banking by a synthetic sanity check; one learned-arm
  landing discarded and re-run bit-identically). An honest bonus
  measurement survives it: the FIRST-FREE rule itself already carries
  TEMPORAL address locality at init (mean r = −0.146 on the untrained
  machine) — the baseline's own trained r is the only honest P-2 anchor.

## Provenance and receipts

No `k10_*`/`k11_*` source, PREREGISTRATION, `lattice.py` or GDN2 model code
was edited; K10's backward body, its selection chain, its mixer/conversion,
the K10 probe's instruments, `protocol_correction._train_corrected` (the
corrected-scoring loop) and `side_by_side.k10_read_mask_sweep` are IMPORTED
verbatim. The K11 lane (cards 0/1) untouched. Receipts:
`receipts/k12_gates.json` (gates A/B/D, the birth-parity and fire lanes,
the finding, the estimator), `receipts/k12_<label>.json` per arm (each with
the full config/bindings, the three scorings, the mechanism diagnostics, the
placement receipts, the address-locality instrument, the read-mask sweep and
the banked-equivalence receipts), `receipts/k12_placement_summary.json`
(the assemble + the prediction evaluations), the smoke receipt, and the
per-arm run dirs under `k12_runs/` with live training logs and consoles.
Session lambda01, cards 2/3, RTX 6000 Ada, torch 2.9.1+cu128. Committed per
artifact as each landed.

## Commands

```bash
# one arm, detached on a leased card from this lane's pool (cards 2/3)
setsid bash -c 'OMP_NUM_THREADS=4 \
    experiments/expressivity_tasks/x45_matrix_emender/run_k12_placement_leased.sh \
    baseline_seed3401 \
    > …/k12_runs/baseline_seed3401.console.txt 2>&1' &

# the assemble: the corrected ladder + the prediction evaluations + the three-way read
OMP_NUM_THREADS=4 /home/erikg/emender/.venv/bin/python -m \
    experiments.expressivity_tasks.x45_matrix_emender.k12_probe_placement assemble

# the lane's tests (18) + the K10 suite green with the module loaded (19)
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k12_placement.py -q
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
    experiments/expressivity_tasks/x45_matrix_emender/test_k10_separated_keys.py -q
```
