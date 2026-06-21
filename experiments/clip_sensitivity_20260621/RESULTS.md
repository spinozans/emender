# clip-sensitivity-control — grad-clip ON-vs-OFF A/B (emender E97 vs gdn2-mlp)

**Task:** `clip-sensitivity-control` (the §6 control proposed in
`docs/GRAD_CLIP_ACCOUNTING.md`). **Date:** 2026-06-21. **Method:** 12 REAL
fused-kernel training runs (3 seeds × 4 cells) on the Pile, single-GPU, bf16,
matched-token. No fabrication; every number below is parsed from a committed
`train.log` / `result.json`.

## Why this run exists

The forensic audit (`GRAD_CLIP_ACCOUNTING.md`) found **no committed run with
clipping disabled anywhere in the tree** (§4c, CLAIM #6) and that the
search-time cross-arm grad asymmetry was **unquantified** (§5, CLAIM #20,
flagged UNSUPPORTED). It argued the confound was low-risk and *conservative* for
the emender NO-GO verdict, but conceded this was **argued, not measured**. This
task converts that argument into a measured result.

## Design (clip is the ONLY variable)

2×2 grid, the two lb-compare MLP arms at their CMA-found 1.3B geometries:

| Arm | Cell | Geometry | tuned LR |
|---|---|---|---|
| **emender-mlp** | E97 split-edit (`e88_raw_write=1`) + SwiGLU MLP | dim1792 nh216 ns32 dep11 mlp2.262 | 1.0072e-3 |
| **gdn2-mlp** | GDN-2 mixer + SwiGLU MLP | dim2176 nh30 dep12 mlp3.259 | 4.7431e-4 |

× `{--grad_clip 1.0, --grad_clip 0}` × seeds `{42, 43, 44}` = **12 runs**.

- **Byte-identical construction** to lb-compare: built through
  `scripts/cmaes_search_v2.build_train_command` from the same `run_bpb.py`
  MODELS entries (re-verified against each arm's committed `args.json`).
- **Matched tokens** (per task): the wall-clock budget (`--train_minutes`) is
  **stripped** and replaced by a fixed `--steps 850`. `should_continue()`
  (`train.py:1425`) then terminates on step-count, so every cell sees exactly
  850 optimizer steps. Both arms are `batch_size=4`, `chunk_size=2048` ⇒
  **8192 tokens/step ⇒ equal steps == equal tokens** within *and* across arms
  (≈6.96 M tokens/cell). A non-finite **skip** does not advance `step`, so the
  step budget guarantees matched *updated* tokens even under instability.
- **Tuned LR held fixed**, **seed fixed** per cell ⇒ within an arm, clip-on and
  clip-off share data order + init, so **grad_clip is the only difference**.
- **bf16 + fused per NON-NEGOTIABLE #1**: emender → repo Triton split-edit
  kernel (`--use_triton 1`, `[fused-guard] … NO eager fallback` asserted);
  gdn2 → external FLA chunked GDN-2 fused kernel (`[fused-guard] … NO eager
  fallback`). **All 12/12 logs carry the guard line; 0 eager fallbacks.**
- Held-out: the **same fixed disjoint Pile-tail slice** lb-compare used
  (`heldout_p50k_2048.pt`, 64×2048 = 131 072 tokens, `bytes_per_token=3.8781`,
  rebuilt byte-identically, SEED=7777). BPB on schedule-free **non-averaged**
  (primary; the basis consistent with the CMA search loss — lb-compare
  correction #2/§methodological-finding) **and averaged** (secondary, a known
  short-budget artifact) weights.

## Raw per-cell results (12 runs, all rc=0, all 850/850 steps)

| seed | arm | clip | BPB nonavg | BPB avg | skips | non-finite stop | grad mean | grad max | %steps grad>1 | fused-guard |
|---:|---|---|---:|---:|---:|---|---:|---:|---:|:--:|
| 42 | emender | on  | 2.1006 | 2.1830 | 0 | no | 1.87 | 16.88 | 88.2% | ✅ E97 |
| 42 | emender | off | 2.0255 | 2.0922 | 0 | no | 2.16 | 36.78 | 92.9% | ✅ E97 |
| 42 | gdn2    | on  | 2.0265 | 2.1370 | 0 | no | 2.11 | 15.56 | 100%  | ✅ FLA |
| 42 | gdn2    | off | 2.0278 | 2.1569 | 0 | no | 2.34 |  8.29 | 100%  | ✅ FLA |
| 43 | emender | on  | 2.0237 | 2.0957 | 0 | no | 1.53 |  7.28 | 84.7% | ✅ E97 |
| 43 | emender | off | 2.0327 | 2.0925 | 0 | no | 1.72 |  6.58 | 92.9% | ✅ E97 |
| 43 | gdn2    | on  | 2.0226 | 2.1440 | 0 | no | 2.03 | 11.94 | 100%  | ✅ FLA |
| 43 | gdn2    | off | 2.0402 | 2.1695 | 0 | no | 2.25 |  7.04 | 100%  | ✅ FLA |
| 44 | emender | on  | 2.0395 | 2.1037 | 0 | no | 1.55 |  4.19 | 89.4% | ✅ E97 |
| 44 | emender | off | 2.0365 | 2.1083 | 0 | no | 1.66 |  4.69 | 96.5% | ✅ E97 |
| 44 | gdn2    | on  | 2.0236 | 2.1422 | 0 | no | 2.32 | 19.12 | 100%  | ✅ FLA |
| 44 | gdn2    | off | 2.0425 | 2.1651 | 0 | no | 2.40 |  9.73 | 100%  | ✅ FLA |

## (i) Non-finite / skip rate under clip-off

**Zero.** Across all 12 runs (including all 6 clip-off runs): **0** non-finite
grad **skips** (`train.py:1555`), **0** non-finite-loss **stops**
(`train.py:1496`), all 850/850 steps completed. Un-clipped pre-clip grad norms
stayed finite and modest at the tuned operating points (max **36.78** —
seed-42 emender clip-off — the single largest of all 12 runs; all other
clip-off maxima ≤ 9.73). **Clipping engages on 85–100 % of steps** (it is very
much "active"), yet **removing it diverges neither arm at this budget.**

> Scope: this is the lb-compare budget (850 steps ≈ 7 M tokens, schedule-free
> cold-start). It does **not** reach the multi-day-racer regime where the one
> bf16 inf occurred near step ~322 k (`GRAD_CLIP_ACCOUNTING.md` §4b). That inf
> is caught by **skip-on-nonfinite** (`5555b9d`), *not* clipping, and is
> out-of-horizon here. "No divergence in 850 steps" ≠ "never diverges."

## (ii) ΔBPB(clip-off − clip-on), per arm (paired; same seed/data/init)

| basis | emender ΔBPB | gdn2 ΔBPB |
|---|---|---|
| **non-avg (primary)** | **−0.023 ± 0.046** (−0.075, +0.009, −0.003) | **+0.013 ± 0.010** (+0.001, +0.018, +0.019) |
| avg (artifact) | −0.030 ± 0.053 (−0.091, −0.003, +0.005) | +0.023 ± 0.003 (+0.020, +0.026, +0.023) |

- **emender:** clip on/off makes **no consistent difference** — the mean
  straddles zero and is dominated entirely by the **seed-42 outlier** (−0.075,
  which did not replicate: +0.009, −0.003 at seeds 43/44).
- **gdn2:** clip-off is **consistently a hair worse** (all 3 seeds positive,
  tight ±0.010), i.e. clipping slightly *helps* gdn2 — the opposite arm from
  the one §5 predicted clipping would flatter.

## (iii) Does the emender−gdn2 BPB gap change between clip-on and clip-off?

`gap = BPB(emender) − BPB(gdn2)`; `Δgap = gap_clipoff − gap_clipon`.

| basis | gap clip-on | gap clip-off | **Δgap (mean ± std, n=3)** | per-seed Δgap |
|---|---:|---:|---:|---|
| **non-avg (primary)** | +0.030 ± 0.039 | −0.005 ± 0.003 | **−0.036 ± 0.036** | −0.076, −0.009, −0.022 |
| avg (artifact) | −0.014 ± 0.052 | −0.066 ± 0.010 | −0.053 ± 0.051 | −0.111, −0.029, −0.018 |

**Sign:** all 3 seeds give a (small) negative Δgap — clip-off nudges the gap
*toward* emender. **Magnitude:** the mean |Δgap| is **0.036 (non-avg) / 0.053
(avg)**, both **inside** the lb-compare BPB noise band (0.01–0.09,
`lb-compare-verdict`). The nudge comes from clip-off mildly *worsening gdn2*
(ii), not from helping emender (which is neutral).

## Decision rule → VERDICT

> Rule (`GRAD_CLIP_ACCOUNTING.md` §6): if `|Δgap| <` the lb-compare noise floor
> (~0.01–0.09) → **verdict robust, close the question**; a **stable** clip-off
> narrowing toward emender **beyond** that floor → **re-open** the emender verdict.

**VERDICT: ROBUST — the emender NO-GO is not overturned by grad clipping.**
On the primary (non-avg) basis `|Δgap| = 0.036 < 0.09`; on the secondary (avg)
basis `|Δgap| = 0.053 < 0.09`. The narrowing is **stable in sign but within the
noise floor in magnitude**, so the "narrows beyond floor ⇒ re-open" branch is
**not** triggered. clip-off is **stable** (0/12 divergence) for both arms.

**Why the multi-seed pass mattered (calibration):** the **single seed-42** run
alone gave `Δgap = −0.076` (non-avg) / **−0.111 (avg, which *exceeds* the 0.09
floor and would have nominally tripped "re-open")**, driven by an
**un-replicated** −0.075 emender clip-off improvement. Seeds 43/44 collapse both
to within-noise. A single-seed verdict here would have been a **false re-open**.
Independent confirmation of the ~0.08 noise floor: this run's fresh clip-on gdn2
non-avg BPB (2.0265, seed 42) differs from lb-compare's committed gdn2-mlp
(2.1013) by 0.075 at the *same* config/seed/slice — i.e. single-run non-avg BPB
noise at this budget is itself ~0.07–0.09, exactly the band the rule cites.

## Reconciliation with `GRAD_CLIP_ACCOUNTING.md` §5

- §5's **conclusion** ("verdict robust to the clip confound; low flip risk") is
  **CONFIRMED** — measured `|Δgap|` stays within noise on both bases, 3 seeds.
- §5's **conjectured mechanism/direction** ("clip=1.0 flatters the spikier
  emender arm; turning clip off would make emender *worse*") is **empirically
  corrected**: at the measured budget clip-off is **neutral for emender**
  (ΔBPB −0.023, straddles 0) and **mildly hurts gdn2** (+0.013, all seeds), so
  the residual gap shift is *toward* emender, not away — though within noise.
  The §5 worst-case ("clip-off makes emender diverge, the inf is emender's E97
  path") **did not occur** in 6/6 emender+clip-off runs at this budget.
- Closes the data gaps: §4c ("no clip-off run exists") and §20 / CLAIM #20
  (search-time asymmetry UNSUPPORTED) now have a measured clip-off A/B at the
  tuned operating points.

## Caveats / scope (honest)

1. **Short budget.** 850 steps ≈ 7 M tokens, cold-start schedule-free — the
   lb-compare budget, not convergence and not the multi-day racer. The clip
   on/off BPB effect at convergence / longer horizon is **not** measured here;
   the seed-42 transient (clip-off speeding emender's early descent) is exactly
   the kind of convergence-*rate* effect prior emender work shows vanishing at
   convergence (`emender-real-1p3b`).
2. **Tuned operating points only.** Each arm runs at its CMA-tuned LR (tuned
   *under* clip). Extreme cold-start spikes like the CMAES `grad 8064` (§3a)
   were *exploration* candidates (bad HPs); at the tuned config un-clipped grads
   stay ≤ ~37 over 850 steps.
3. **3 seeds.** Enough to expose the seed-42 outlier and bound |Δgap| under the
   floor, not enough for a tight CI. A longer-budget, more-seed confirm is the
   natural follow-up if a tighter bound is ever needed.

## Reproduce

```bash
python3 experiments/lb_compare_20260613/build_heldout_tensor.py        # fixed slice
bash    experiments/clip_sensitivity_20260621/orchestrate.sh           # seed 42 (4 cells)
bash    experiments/clip_sensitivity_20260621/orchestrate_seeds.sh     # seeds 43,44 (8 cells)
python3 experiments/clip_sensitivity_20260621/multiseed_aggregate.py   # gap analysis + verdict
```

Artifacts: `run_clip_ab.py` (driver), `orchestrate*.sh`, `aggregate_clip.py`,
`multiseed_aggregate.py`, `clip_ab_results*.json`,
`clip_ab_multiseed_analysis.json`, `runs/*/train.log` + `result.json`.
