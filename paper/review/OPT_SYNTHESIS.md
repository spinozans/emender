# OPT_SYNTHESIS — cross-probe aggregation, composed regime R*, and the 1.3B CMA-over-controls protocol

**Task:** `opt-synth` (OPT_SPEC.md §6) · **Role:** Programmer · **Date:** 2026-06-10
· small scale (dim 256 / 32 heads / N32 / depth 4 / mlp 2.0), fp32, schedule-free
AdamW, shared §3 battery + JCC metric (§1.3), **FUSED Triton path only** (Erik's
mandate; `typed-gdn2` `use_triton_e97=True`, loud-guard active — see §6).

This document aggregates the four OPTIMIZATION-NOT-ARCHITECTURE lever probes
(`opt-headlr`, `opt-initspread`, `opt-norm`, `opt-minimal`), states which levers
move the local optimum and by how much, assembles the single best optimized regime
**R\*** for the GDN+nonlin mixture, re-runs R\* once at small scale to confirm the
composed JCC (§6.4), and finalizes the **CMA-ES-over-controls protocol** for the
1.3B validation (§4.2/§6.5). It is the gate `opt-1p3b` consumes.

---

## 0. TL;DR

- **The cross-probe verdict is the convergent-loss null, extended from architecture
  to optimization — with two real but *small* training-regime levers on top.**
  Against a properly **LR-tuned** GDN-2 incumbent, the optimized mixture's edge is
  carried overwhelmingly by the **substrate** (the mixture simply *has* counting
  heads GDN-2 lacks), not by the training levers. Decomposed to pure lever effect
  (vs the substrate-default B₂), **every lever is at or below the pre-registered
  floor Δ\* = 0.03**: per-head-type LR **+0.021**, placement-knob-LR **+0.015**,
  norm/gate/decay-init **+0.008** (NULL), and the minimal-core ablation removes only
  the MLP (capability-neutral). No single lever clears Δ\* on its own.
- **Two levers are nonetheless worth carrying** because their value is *reliability
  and the worst corner*, not headline JCC: (1) **per-head-type LR at compute-class
  5×** (opt-headlr `c5`) — converges all 3 seeds and holds 4/4 corners where the
  default is seed-fragile, decisive on length-extrapolation (T=512 c5 0.998 vs B
  0.853); (2) **placement** (opt-initspread) — the *house* allocation (≥50 %
  `gdn2_recall`) is **necessary**: naive uniform/skew/center spread **collapses the
  track corner** (−0.31 JCC). Placement is not a "win" so much as a *constraint that
  must not be violated*.
- **Composed regime R\* (after the mandatory §6.4 re-run): house placement (50 %
  `gdn2_recall` neg-eig + 25 % counting `e97_delta`/`refit-del` + 25 % `nonlin`,
  n_heads=32, MLP kept) + `head_lr_compute_mult=5` (`c5`).** The initially-composed
  candidate also stacked `decay_init=slow`, but the small-scale re-run showed it
  **regresses recall** in combination with the LR lever (rstar 0.930 vs c5 0.969) —
  so it is **dropped** (additivity is not assumed; the re-run caught the negative
  interaction). The knob-LR=20 companion was never stackable (mutually exclusive
  with the per-head-type LR groups in the trainer; both drive the compute side).
  At identical harness the surviving lever's pure lift is **+0.006** (sub-Δ\*) —
  its value is reliability + worst-corner + length-extrapolation, not headline JCC.
- **1.3B protocol (§5):** CMA-ES-optimize **both** controls at ~1.3B —
  **CMA-best GDN-2** (`--model gdn2`) and **CMA-best m2rnn** (`--model m2rnn`; a CMA
  1.3B config already exists at `hf_v03_fix_staging/m2rnn-cma-1.3b`, dim 1920 /
  depth 21) — so the §1.4 bar uses `B = CMA-best GDN-2`, best-vs-best, no hobbled
  baselines.

---

## 1. Unified leaderboard (OPT_SPEC §6.2)

All four probes' `JCC_ROWS.jsonl` were re-scored against a **single reconciled
ceiling set** (§1.1) so they are directly comparable. Headline `JCC = min_c r_c`
over the scored corners {recall, counting, step-growth, track}, seed-averaged.
Aggregator: `experiments/expressivity_tasks/aggregate_opt_synth.py`; full table
(42 arms) in `results_opt_synth/UNIFIED_LEADERBOARD.txt`.

### 1.1 Reconciled frozen ceilings S_c (§6.1)

The four probes ran their control/specialist arms independently and each wrote a
slightly seed-divergent `opt_ceilings.json` (a known coordination point the spec
assigns to synth). Reconciliation rule, faithful to §1.3 ("the best accuracy any
single specialist reaches on corner c"): **S_c = max over the four probes'
specialist ceilings** — the *hardest* (most conservative) denominator. Because both
`R` and `B` divide by the **same** S_c, **ΔJCC is ceiling-invariant**, so the
per-lever verdicts below are robust to this choice (each probe independently
verified this on its own arms).

| corner | reconciled S_c | from |
|---|---|---|
| recall | 0.9607 | opt-minimal (`min_no_mlp`) |
| counting | 0.8962 | opt-norm (`spec_refit`) |
| step_growth | 0.9987 | opt-initspread (`B_gdn2`@1e-3) |
| track | 0.9999 | opt-minimal |

Written to `opt_ceilings.json` under key `opt_synth_reconciled` (alongside each
probe's own block, preserved). Counting is the binding corner for every GDN-2/house
arm — the headline JCC ≈ the counting ratio throughout.

### 1.2 Top of the unified board (against reconciled S_c)

| rank | probe:arm | JCC | SE | recall | count | step | track | held | note |
|---|---|---|---|---|---|---|---|---|---|
| 1 | norm:decay_slow3 | 0.973 | .000 | 0.997 | 0.974 | 0.974 | 1.000 | 4.0 | 2 seeds |
| 2 | norm:decay_slow | 0.969 | .006 | 1.000 | 0.996 | 0.969 | 1.000 | 4.0 | 2 seeds |
| 3 | norm:combined_best | 0.965 | .008 | 1.000 | 0.981 | 0.977 | 0.989 | 3.7 | 3 seeds |
| 4 | **headlr:headlr_c5** | **0.964** | .008 | 0.987 | 0.965 | 0.976 | 1.000 | 3.7 | **3 seeds, all converge** |
| 5 | norm:B2_default | 0.964 | .001 | 0.964 | 0.991 | 0.980 | 1.000 | 4.0 | substrate default |
| 6 | norm:lam_1.3 | 0.962 | .008 | 0.969 | 0.967 | 0.980 | 1.000 | 3.5 | |
| 7 | **initspread:house_klr20** | **0.952** | .006 | 0.974 | 0.956 | 1.000 | 1.000 | 3.3 | placement winner |
| 8 | headlr:headlr_c2 | 0.951 | .011 | 0.984 | 0.951 | 0.980 | 0.999 | 3.7 | |
| … | initspread:house_klr1 (=B₂) | 0.937 | .006 | 0.974 | 0.937 | 1.000 | 0.999 | 3.3 | |
| … | minimal:min_full (=B₂) | 0.930 | .008 | 0.986 | 0.930 | 0.979 | 0.999 | 3.0 | |
| … | **GDN-2 incumbent B** (mean) | **0.901** | — | 0.99 | ~0.90 | ~0.98 | 1.00 | 3.0 | bottleneck = counting |
| bottom | initspread:uniform4/skew/center | 0.63–0.65 | — | ~0.95 | ~0.91 | 1.00 | **0.65** | 1.7–2 | **track collapse** |
| bottom | minimal:min_no_negeig | 0.291 | .017 | 0.92 | 0.91 | 0.97 | **0.29** | 1.0 | neg-eig removed |

**Reading.** Absolute JCC compresses under the hardest reconciled ceiling (e.g.
headlr_c5 0.994→0.964 because the reconciled counting/recall denominators are
higher than headlr's per-length ones); the *ranking and the ΔJCC gaps are what
matter*. The substrate-default arms (`B2_default`/`min_full`/`house_klr1`) cluster
at 0.93–0.96; the LR/placement levers nudge a few points up; the GDN-2 incumbent
sits at 0.90 bottlenecked on counting; the *wrong* placements crash to 0.63–0.65.

### 1.3 Harness-consistency check (§6.2) — PASS

`B` (GDN-2) appears in all four probes; its JCC must agree within seed noise or a
harness has drifted:

| probe | B arm | B JCC |
|---|---|---|
| opt-headlr | gdn2-default (5e-4) | 0.894 |
| opt-initspread | B_gdn2 (1e-3) | 0.886 |
| opt-norm | B_gdn_lr1e3 (1 seed) | 0.924 |
| opt-minimal | B_lr0p001 | 0.901 |

Mean **0.901**, spread **0.038**. The spread is fully explained and *not* a drift
bug: (i) the counting corner (the binding corner) averages a slightly different
witness set per probe — initspread uses `modular_counter`+`dyck` only, the others
add `anbncn_viability` (easier) → raises counting; (ii) norm's 1e-3 point is a
single seed; (iii) probes report B at different "best" LRs (5e-4 vs 1e-3). All
ranking is by **within-probe ΔJCC**, which is invariant to these. Harness is
consistent.

---

## 2. Per-lever contributions and verdicts (OPT_SPEC §6.3)

The honest per-lever measure is the **pure contribution vs the substrate-default
B₂** (`JCC(R) − JCC(B₂)`); `JCC(R) − JCC(B)` mixes in the substrate (the
counting-head advantage GDN-2 structurally lacks — an architecture effect, out of
scope for the optimization levers).

| lever (probe) | best arm | vs incumbent B | **pure (vs B₂)** | clears Δ\*=0.03? | verdict |
|---|---|---|---|---|---|
| **per-head-type LR** (opt-headlr) | `c5` (compute 5×) | +0.076 | **+0.021** | no | **GO vs B**, lever sub-Δ\*; value = reliability + extrapolation |
| **placement** (opt-initspread) | `house_klr20` | +0.066 | **+0.015** (knob-LR part) | no (knob-LR) / **yes** (placement = +0.31) | **GO vs B**; *placement is a hard constraint*, knob-LR companion sub-Δ\* |
| **norm/gate/decay-init** (opt-norm) | `decay_slow` | +0.057…+0.070 | **+0.008…+0.021** | no | **NULL** (CMA defaults near-optimal; only retention→1 direction positive) |
| **minimal core** (opt-minimal) | `min_no_mlp` | — | — (ablation) | — | core = house mixture; only MLP removable (capability), KEEP for bpb |

### 2.1 What each lever actually buys

- **Per-head-type LR (GO vs B; pure +0.021, sub-Δ\*).** Driving the compute-class
  heads at **5×** lets counting/step-growth converge without disturbing
  recall/track; **10–20× backfires** (recall/track collapse, SE≈0.20). The
  falsifier (drive recall 10×) is inferior, confirming the direction is *drive
  compute*. The pure headline contribution is below the floor, but its real value
  is **reliability** (c5 converges all 3 seeds, holds 4/4 corners; B₂/B are
  2/3-seed fragile) and **length-extrapolation** (T=512: c5 0.998 vs B 0.853,
  +0.145). Carry it.

- **Placement (GO vs B; the lever is a *constraint*, not a +Δ\* gain).** The
  knob-LR companion (klr1→klr20) adds only +0.015. The *placement itself* is worth
  **≈ +0.31 JCC** — but as a **floor, not a ceiling**: the house allocation (≥50 %
  `gdn2_recall`, neg-eig) is **necessary** because naive uniform/skew/center
  "spread everything" placements **collapse the track corner** (s5 ≈ 0.63–0.65 vs
  1.00, converged, all 3 seeds — track capability scales with the gdn-neg recall
  allocation; the dedicated `e97_track` heads do *not* substitute). The synthesis
  lesson: *use the house placement and do not spread the recall mass thin.*
  `recallheavy` (62.5 % recall) also wins → the operating window is wide, not a
  knife-edge.

- **Norm/gate/decay-init (NULL).** No init knob moves JCC beyond Δ\* once the
  substrate is fixed. The CMA-tuned cell defaults (`lam_max=1.585`,
  `beta_max=2.747`, gate default, head-norm on) are already near-optimal. The one
  positive axis is **decay_init=slow (retention→1)**: directionally correct
  (recall needs to hold the binding; the mod-k counter needs to hold its residue),
  but +0.008–0.021, sub-Δ\*, and **saturating** (δ=3 gives no headroom over δ=2).
  `decay_fast` (retention→0) is strongly harmful (−0.16). Carry decay_slow as a
  free, harmless, directionally-correct default; carry nothing else.

- **Minimal core.** Necessity ranking: **neg-eig (+0.690) ≫ short-conv (+0.344) ≫
  gate (+0.055) > nonlin-in-time (+0.031 ≈ Δ\*) > MLP (+0.024, removable for
  capability)**. Every *recurrent* component is load-bearing; the post-mixer MLP is
  the only droppable piece, and only for the capability battery — it is known to
  matter for LM held-out bpb, so R\* **keeps the MLP** for the 1.3B runs. The
  practical synthesis output: the levers should optimize the *training* of this
  cell, not shrink the function class.

### 2.2 The crux the four probes agree on

All four independently reached the same structural conclusion: **the win over the
GDN-2 incumbent is the substrate (the counting heads), not the training lever.**
opt-minimal makes this sharpest — a properly **LR-tuned** GDN-2 (5e-4→1e-3 fixes
its step-growth/counting weakness, which is itself an LR artifact, not an
architectural ceiling) **ties** the capability-complete mixture (+0.030 = exactly
Δ\*) at **2.7× fewer parameters**. This is the standing convergent-loss null,
now extended to the optimization line: *at convergent loss and matched compute, a
well-trained GDN-2 is hard to beat on joint capability coverage.* The two surviving
levers (compute-5× LR, house placement) buy **reliability and the worst corner**,
not a headline win — which is exactly why the 1.3B test against **CMA-best** GDN-2
(not a default baseline) is the real verdict.

---

## 3. The composed regime R* and its small-scale re-run (OPT_SPEC §6.4)

### 3.1 Composition candidate (and why knob-LR=20 is dropped)

> This is the *candidate* composition that goes into the §6.4 re-run. The re-run
> (§3.3) revised it: `decay_init=slow` regressed and was dropped, leaving the
> **final R\* in §3.4**. The reasoning below for the placement/LR-lever choices
> stands; only the decay axis changed.

R\* composes the per-lever winners onto the opt-minimal core:

```
substrate / placement : house mixture — 50% gdn2_recall (gdn_allow_neg_eigval=1)
                        + 25% counting (e97_delta @ scale / refit-del @ fp32)
                        + 25% nonlin,  n_heads=32   (= the minimal load-bearing core)
per-head-type LR      : --head_lr_recall_mult 1.0 --head_lr_compute_mult 5.0   (opt-headlr c5)
decay init            : --decay_init slow                                       (opt-norm)
MLP                   : KEPT (mlp_ratio 2.0)   — load-bearing for LM bpb at scale
neg-eig / gate /
short-conv            : ON  (all load-bearing per opt-minimal)
```

**Why not also `--knob_lr_mult 20` (initspread klr20)?** Two reasons, both
decisive: (1) the trainer's param-grouping makes the per-head-type LR groups and
the legacy knob-LR group **mutually exclusive** (`train_hybrid.py:489–532`: with
`head_lr_*_mult` set, `knob_lr_mult` is ignored with a WARN) — they cannot both
apply; (2) both levers **drive the compute side**, and opt-headlr showed
compute-side over-drive (10–20×) **collapses recall/track**. opt-initspread's
*actual* GO lever was **placement** (already in the house mixture), with knob-LR a
sub-Δ\* +0.015 companion. The principled composition therefore uses the stronger,
more reliable per-head-type LR (`c5`, pure +0.021) as the single LR lever and does
not stack a second compute-side boost. (An `klr20` arm is re-run alongside R\* as
the control that confirms this choice.)

### 3.2 R* re-run — arms (identical harness, removes cross-probe noise)

`experiments/expressivity_tasks/run_opt_synth_rstar.py`, 3 seeds, full scored
battery, `modular_counter`/`modular_quadratic` @16000 steps (still-climbing at 8000
per all four probes), the rest @8000 / controls @5000, fp32, fused path:

| arm | lever on the house mixture |
|---|---|
| `b2_house` | none (= B₂ anchor) |
| `c5` | `head_lr_compute_mult=5` (opt-headlr winner) |
| `klr20` | `knob_lr_mult=20` (opt-initspread knob-LR) |
| **`rstar`** | `head_lr_compute_mult=5` + `decay_init=slow` (the composed candidate) |

Scored against the §1.1 reconciled ceilings by
`aggregate_opt_synth.py --rstar`. **Decision:** R\* holds iff
`JCC(rstar) ≥ max(JCC(c5), JCC(klr20))` within seed noise; else fall back to the
best single lever and note the interaction (§6.4).

### 3.3 R* re-run — results (96 runs, 0 fails, ~42 min, fp32 fused path)

Scored against the §1.1 reconciled ceilings, seed-averaged JCC = mean over seeds of
the per-seed worst-corner ratio (`aggregate_opt_synth.py --rstar`;
`results_opt_synth/RSTAR_TABLE.txt`). All arms converged on the accuracy-plateau
gate (≥23/24 hard runs).

| arm | JCC | SE | recall | counting | step | track | conv | vs B₂ |
|---|---|---|---|---|---|---|---|---|
| `b2_house` (= B₂ anchor) | 0.963 | .001 | 0.963 | 0.992 | 0.980 | 1.000 | 24/24 | — |
| **`c5`** (compute 5×) | **0.969** | .005 | 0.970 | 0.998 | 0.977 | 1.000 | 23/24 | **+0.006** |
| `klr20` (knob-LR 20×) | 0.966 | .004 | 0.966 | 0.981 | 0.980 | 1.000 | 23/24 | +0.003 |
| `rstar` (c5 + decay_slow) | 0.930 | .005 | **0.930** | 0.990 | 0.972 | 0.998 | 24/24 | **−0.033** |

**Verdict: the composition REGRESSES — fall back to the best single lever (c5).**
`JCC(rstar)=0.930 < max(JCC(c5), JCC(klr20))=0.969`. The interaction is **negative
and specific to recall**: adding `decay_init=slow` (retention→1) on top of the 5×
compute-class LR drops the **recall** corner (0.970 → 0.930) while leaving
counting/step/track intact. opt-norm forwarded decay_slow as "free + harmless", but
*in composition with the per-head-type LR lever it is NOT harmless* — pushing
retention→1 while the compute heads train 5× aggressively interferes with the recall
head's binding precision. This is precisely the levers-interact / additivity-not-
assumed failure mode §6.4 exists to catch; the re-run earned its cost.

**Two further findings from the identical-harness re-run:**
1. **The pure lever lifts are sub-Δ\* and tiny** — c5 +0.006, klr20 +0.003 vs the
   B₂ substrate default (Δ\*=0.03). With cross-probe witness-set/ceiling noise
   removed, the per-head-type LR and knob-LR levers contribute *almost nothing* to
   headline JCC at convergent loss. This **confirms the convergent-loss null at the
   composition level**, independent of the per-probe estimates.
2. **c5 ≥ klr20 at identical harness** (0.969 vs 0.966), with c5's edge on counting
   (0.998 vs 0.981) — justifying the §3.1 choice of the per-head-type LR over the
   knob-LR as the single LR lever (they were never stackable anyway).

### 3.4 Final R* carried to 1.3B (post-re-run)

> **R\* = house placement (50 % `gdn2_recall` neg-eig + 25 % counting + 25 % `nonlin`,
> n_heads=32, MLP kept) + `head_lr_compute_mult=5` (`c5`).** `decay_init` is left at
> default (the slow variant regressed recall in composition). This is the best
> single-lever regime at identical harness (JCC 0.969); its value over a tuned GDN-2
> is **reliability + worst-corner + length-extrapolation**, not a headline JCC gain
> (the pure lever lift is +0.006, sub-Δ\*).

`--decay_init slow` is therefore **removed** from the §4.4 `layer_kwargs` for the
1.3B run. (Should a 1.3B ablation want to re-test it without the LR lever, it is a
one-flag arm; but the small-scale evidence is that it interacts badly with the
carried LR lever.)

---

## 4. The 1.3B CMA-ES-over-controls protocol (OPT_SPEC §4.2 / §6.5 — Erik's directive)

At 1.3B we do **not** compare against default-trained controls. Both controls are
**CMA-ES-optimized at the 1.3B budget for this application**, so each sits at *its
own* optimum (best-vs-best, no suboptimal-geometry artifacts). The §1.4 decision
rule at 1.3B uses **`B = CMA-best GDN-2`** — the strongest possible incumbent —
making any GO verdict maximally honest.

### 4.1 Machinery (reuse, no new search code)

`scripts/cmaes_search_v2.py` — two-phase **LHS exploration → CMA-ES refinement**,
population 16, σ=0.35, `min_generations=6` (consecutive=3, threshold=0.005). It
already supports both control model types as first-class `--model` values with
calibrated param calculators (`calc_gdn2_params`, `calc_m2rnn_params` in
`scripts/calc_dim.py`) and search spaces (verified present in `SEARCH_SPACES`).
Fitness = held-out validation loss on the committed pretraining slices (minimize),
the same objective the 1.3B CMA leaderboard used.

### 4.2 Search A — CMA-best GDN-2 (the incumbent B at its optimum)

```bash
python scripts/cmaes_search_v2.py \
    --model gdn2 \
    --params 1.3B \
    --train_minutes 30 \
    --gpus 0,1,2,3,4,5,6,7 \
    --phase both --lhs_samples 48 \
    --output cma_runs/opt1p3b_gdn2
```
(flags verified: `--params` takes a string size e.g. `1.3B`, default `480M`;
`--output`, `--phase both`, `--lhs_samples`, `--train_minutes`, `--gpus` all in
`cmaes_search_v2.py` argparse. `--param_tolerance` optional.)

Search space (6D, from `SEARCH_SPACES['gdn2']`): `dim ∈ [1024,4096] (×128)`,
`expansion ∈ [1,3]`, `depth ∈ [10,50]`, `n_heads ∈ [8,64]`, `lr ∈ [1e-4,3e-3]
(log)`, `batch_size ∈ [1,128] (log, memory-clamped)`. `--gdn_allow_neg_eigval 1`
fixed ON (the recall+track substrate; the optimization line's GDN-2 incumbent
always has neg-eig). Param calculator `calc_gdn2_params` enforces the ~1.3B target
within tolerance. **Note (opt-minimal):** GDN-2's counting/step-growth weakness is
an LR artifact — the CMA `lr` axis will find the un-hobbled point, so this B is the
hardest honest incumbent.

### 4.3 Search B — CMA-best m2rnn (the raw-write power-separation foil)

A CMA m2rnn 1.3B config **already exists**:
`hf_v03_fix_staging/m2rnn-cma-1.3b/config.json` (level `m2rnn`, **dim 1920,
depth 21**, expansion 1.0, gate sigmoid, `linear_state=false`, checkpoint_loss
2.6277 @ step 1.467M). **Reuse it directly** as the CMA-best m2rnn control — no new
search needed unless `opt-1p3b` wants a fresh seed:

```bash
# Reuse the existing CMA-best m2rnn geometry (preferred — already converged):
#   dim=1920 depth=21 expansion=1.0 level=m2rnn linear_state=0 gate=sigmoid
# Or re-run CMA from scratch at the 1.3B target if a fresh search is wanted:
python scripts/cmaes_search_v2.py \
    --model m2rnn \
    --params 1.3B \
    --train_minutes 30 --gpus 0,1,2,3,4,5,6,7 \
    --phase both --lhs_samples 48 \
    --output cma_runs/opt1p3b_m2rnn
```

m2rnn search space (`_E88_SEARCH_SPACE`): `dim, n_heads ∈ [32,2000], n_state ∈
{16,32,48,64}, depth ∈ [10,50], lr, batch_size`. m2rnn is the family where the
**state-nonlinearity knob is load-bearing in the *opposite* direction**
(`M2RNN_LINEAR_ABLATION.md`: linear ≫ tanh) — the honest "different write rule
entirely" foil. If the optimized GDN+nonlin R\* cannot separate from CMA-best
m2rnn on coverage, that bounds the claim.

### 4.4 R* at 1.3B — geometry and wiring (not CMA-searched; it is the candidate)

R\* is **not** CMA-searched (it is the arm under test, carried from the small-scale
synthesis). It is param-matched to ~1.3B by scaling the §2.3 cell to the 1.3B dims
and wired as the `typed-gdn2-lm` LadderLM level (verified present,
`ladder_lm.py:565`), with the R\* knobs passed through `layer_kwargs=` (the
documented candidate-knob path, `ladder_lm.py:1092/1136`):

```python
layer_kwargs = {
    'head_type_logits': HOUSE_LOGITS,   # 50/25/25 gdn2_recall/counting/nonlin at the scaled n_heads
    'gdn_allow_neg_eigval': 1,
    'refit_has_mom': 0,                 # if the counting slice uses refit-del; at bf16 scale e97_delta is also available
    # NOTE: decay_init is left at DEFAULT — the §3.3 re-run showed decay_init='slow'
    # regresses recall in composition with the LR lever (dropped from R*).
}
# per-head-type LR (head_lr_compute_mult=5) is an OPTIMIZER-side param-grouping,
# applied in the 1.3B trainer exactly as train_hybrid.py:489-538 (recall-class
# 'gdn' sub-block at 1x, compute-class unified/refit/nonlin sub-blocks at 5x).
```

To match the controls' geometry per size, R\*'s `dim/depth/n_heads` are set to the
**CMA-best GDN-2** geometry from Search A (R\* and its incumbent share geometry so
the only difference is the head mixture + training regime), and reported alongside
the m2rnn geometry. This realizes "**m2rnn AND gdn-2 geometry per size**" — each
control at its own CMA optimum, R\* at the GDN-2 geometry it is meant to improve on.

### 4.5 Two measurements at 1.3B (both required)

1. **Capability coverage at scale** — run the §3 battery on the **1.3B-shaped
   cell** (scale the head config to 1.3B dims, train on the probes), compute JCC at
   scale; the §1.4 bar must hold (`B = CMA-best GDN-2`). Exactly as
   `E97_WITHIN_LAYER_SYNTHESIS.md` Q1 / the 0.48B scale-pilot did.
2. **Held-out BPB** — the LM run on the committed slices (`COMMA_PILE_BPB` /
   `heldout_multislice`), **averaged (schedule-free) weights**, **token-matched AND
   wall-clock-noted** (the within-layer wall-loss turned on wall-clock — report both;
   `e97-lm-1p3b` showed token-win can flip to wall-loss).

### 4.6 1.3B verdict rule

**GO** iff R\* clears §1.4 at 1.3B on JCC (worst-corner, beyond noise, vs
`B = CMA-best GDN-2`) **AND** does not regress held-out BPB vs `B`. Otherwise
**NULL** — the convergent-loss null extended from architecture to optimization, a
clean honest negative that closes the line (consistent with the §0 standing nulls).
Given the small-scale finding that every pure lever is sub-Δ\* and a tuned GDN-2
ties the mixture, **the pre-registered expectation at 1.3B is NULL**; the levers'
defensible upside is reliability/worst-corner and length-extrapolation, which the
scale battery (eval at long T) is best positioned to detect.

---

## 5. Hand-off to `opt-1p3b`

1. **R\* config** (§3.4, post-re-run) — `typed-gdn2-lm`, house placement,
   `head_lr_compute_mult=5`, **decay_init=default** (decay_slow dropped: regressed
   recall in composition), MLP kept, neg-eig/gate/short-conv on; geometry =
   CMA-best GDN-2's.
2. **Controls** — Search A (CMA-best GDN-2, §4.2) as `B`; Search B / the existing
   `m2rnn-cma-1.3b` (§4.3) as the m2rnn foil.
3. **Reconciled ceilings** — `opt_ceilings.json` key `opt_synth_reconciled` (§1.1)
   for the scale-battery JCC; recompute scale ceilings from the 1.3B specialist arms
   if available (the small-scale S_c are a sanity anchor, not the scale denominator).
4. **Decision rule** — §4.6; `B = CMA-best GDN-2`, Δ\* = max(0.03, 2·SE_seed), GO
   needs worst-corner JCC beyond noise **and** no BPB regression, both token- and
   wall-clock-accounted.

### 5.1 RESOLVED — 1.3B verdict (`opt-1p3b`, 2026-06-10): **NULL**

`opt-1p3b` ran the validation (`paper/review/OPT_1P3B_RESULTS.md`,
`experiments/opt_1p3b/`): R\* (typed mixture + `head_lr_compute_mult=5`, fp32 FUSED) vs
**CMA-best GDN-2** (`gdn-1.3b`, 1.35B) and **CMA-best m2rnn** (`m2rnn-cma-1.3b`, 1.31B),
matched wall-clock on real Comma-Pile, fused kernels. **Both §4.6 gates fail:**
held-out **BPB R\* 2.220 > GDN-2 1.765** (also token-matched 6.083 > 6.011), worst-corner
**JCC R\* 0.095 < GDN-2 0.274**. The pre-registered expectation holds: **optimization does
not move the optimum at scale** — the convergent-loss null extends from architecture to
the optimization line, best-vs-best. R\* is *less* sample-efficient at 1.3B (the 5×
compute-class LR aces step_growth but starves recall/track — the §3.3 interaction, severe
at scale).

---

## 6. Provenance / FUSED-path compliance / artifacts

- **FUSED-ONLY (Erik's mandate):** every R\* re-run trains the FUSED Triton path.
  `typed-gdn2` defaults `use_triton_e97=True` (`typed_head_mixture.py:226`) with the
  loud guard (`:540–555`) that **raises** rather than silently falling back to the
  eager T-scan. The fp32 substrate uses `gdn2_recall` (FLA fused) + `refit-del`
  (fused sequential Triton fwd+bwd) + `nonlin` (UnifiedCell) — the exact fused
  configuration the three prior fp32 probes validated. The smoke confirmed head-type
  LR groups active (recall-class 52 params @1×, compute-class 88 @5×) and no eager
  WARN. No `use_triton*=False` / `fused=False` / `cplx_force_sequential` anywhere.
- **Aggregator:** `experiments/expressivity_tasks/aggregate_opt_synth.py`
  (reconciles ceilings, unified leaderboard, harness-consistency check, `--rstar`
  scoring). **R\* runner:** `run_opt_synth_rstar.py`.
- **Data:** `results_opt_synth/` (R\* run JSONs + logs), `UNIFIED_LEADERBOARD.txt`,
  reconciled block in `opt_ceilings.json`.
- **Inputs:** the four probes' `results_opt_{headlr,initspread,norm,minimal}/JCC_ROWS.jsonl`
  + `paper/review/OPT_{HEADLR,INITSPREAD,NORM,MINIMAL}_RESULTS.md`.
