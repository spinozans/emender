# Control CMA Verification — gdn2 (fla-gdn) & m2rnn 1.3B

**Task:** verify-control-cma · **Date:** 2026-06-11 · measured facts only (no framing).

VERDICT: **PARTIALLY VERIFIED, with two real asymmetries flagged.** Both control CMA
searches exist and were located + inspected (they are NOT missing). BUT the geometries
actually deployed as the `cma_gdn2` / `cma_m2rnn` controls are the **long-racer
pile-convergence anchors, not the CMA-search winners** — and for fla-gdn the two diverge
sharply. The fla-gdn search also never cleanly terminated.

---

## Where the records live

Durable artifacts are in `~/emender`, NOT `~/ndm` (per
`docs/HANDOFF_CMAES_GDN2_MLP_20260605.md` and auto-memory `e97-raw-1p3b-consolidated`).

CMA-search root: `/home/erikg/emender/experiments/local/cmaes_redo_1300m_20260529/`
- fla-gdn: `fla-gdn/fla-gdn_20260531_131116/`  (`cmaes_state.pkl`, `generations.jsonl`, 110 `eval_*` dirs)
- m2rnn:   `m2rnn/m2rnn_20260531_131116/`      (`cmaes_state_final.pkl`, `generations.jsonl`, `results.json`, 107 `eval_*` dirs)
- controller logs: `logs/fla-gdn.log`, `logs/m2rnn.log`

Anchor (deployed-geometry) definitions: `anchors_missing_20260531.json`,
`anchors_gdn2_primary.json`, `anchors_corrected.json` in the same root.

The 1.3B control CHECKPOINTS actually shipped (`hf_v03_fix_staging/gdn-1.3b`,
`hf_v03_fix_staging/m2rnn-cma-1.3b`) were trained to FULL convergence from scratch in
`/tmp/pile_convergence_3arch` (fla-gdn, 1.998M steps -> loss 2.6148 / BPB 0.976965) and
`/tmp/pile_convergence_m2rnn` (m2rnn, 1.467M steps -> loss 2.6277 / BPB 0.979845). Those
are training runs, not searches — they hold no `cmaes_state.pkl`.

---

## Measured CMA-search budget & convergence

| Control | popsize | generations | total evals | clean stop? | best AvgLoss | converged? |
|---|---|---|---|---|---|---|
| **fla-gdn** | 8 | ~13 (gen 0-12+, log reaches "Generation 13") | **110** (eval_0-108) | **NO** — no "SEARCH COMPLETE", no `_final` pkl; log ends mid-gen-13 at eval 108 | **6.0854** | plateaued (best 6.0854 held from gen 9) but **not cleanly terminated / possibly cut short** |
| **m2rnn** | 8 | **16** (gen 0-15) | **107** | **YES** — "SEARCH COMPLETE", 17.11 h, `cmaes_state_final.pkl` | **6.1161** | **converged** — flat from gen 2 (6.1198 -> 6.1161 over 14 gens = -0.004) |

Per-candidate budget: short proxy training run (~15 min/candidate, same harness class as
the e99 typed-gdn2 search whose median was 14.85 train-min/candidate).

### Best-fitness-per-generation (best_loss_so_far)

```
fla-gdn:  6.2628 6.2628 6.1188 6.1188 6.1188 6.1188 6.1188 6.1092 6.0991 6.0854 6.0854 6.0854  (gen0..11; sigma 0.290->0.124)
m2rnn:    6.3754 6.1865 6.1198 6.1198 6.1198 6.1198 6.1198 6.1198 6.1198 6.1198 6.1161 6.1161 6.1161 6.1161 6.1161 6.1161  (gen0..15; sigma 0.305->0.160)
```
Both curves flatten well before their last generation -> both reached a fitness plateau at
this proxy budget. fla-gdn improved later (down to 6.0854 at gen 9) than m2rnn (frozen at
6.116 from gen 10); the fla-gdn controller log shows the patience counter ("No improvement
N/2") repeatedly resetting, and the run was snapshotted still inside gen 13 without the
terminal banner.

---

## Deployed control geometry vs CMA-search winner — the key asymmetry

| Control | CMA-search BEST (measured) | DEPLOYED control (lm_runner.py / staging config) | match? |
|---|---|---|---|
| fla-gdn | dim **3712**, exp 2, depth **10**, n_heads **35**, lr 7.31e-4, bs 3 | dim **2688**, exp 2, depth **21**, n_heads **44**, ns 64, lr 2.871e-3, bs 4 | **NO — depth 10 vs 21, dim 3712 vs 2688.** Deployed = long-racer anchor, not the CMA winner |
| m2rnn | dim **2048**, n_heads **342**, n_state **16**, depth **21**, lr 5.50e-4, bs 5 | dim **1920**, n_heads **370**, n_state **16**, depth **21**, exp 1.0, lr 6.02e-4, bs 5 | **YES (~)** — same depth 21 / ns 16; dim & heads are param-match adjustments of the same point |

Source of the deployed geometries (verbatim, `anchors_missing_20260531.json`):
- fla-gdn: *"Corrected p50k_base FLA-GDN anchor from current **long racer geometry**, depth adjusted from 21 to 20 after tokenizer-aware accounting."* -> dim 2688 / exp 2 / depth 20 / h44 / lr 0.002871. (Shipped config.json shows depth 21.)
- m2rnn: *"Corrected p50k_base M2RNN anchor from current XMA **long racer geometry**."* -> dim 1920 / h370 / ns16 / depth 21 / lr 6.02e-4.

=> The `cma_*` labels in `experiments/emender_real_1p3b/lm_runner.py` are **accurate for
m2rnn but a misnomer for gdn2**: the deployed fla-gdn control is the hand-set long-racer
geometry (depth 21), which the CMA search did NOT select (it preferred a shallow-wide
depth-10/dim-3712 point). This is precisely the "convergence never re-verified" gap the
task suspected — confirmed for fla-gdn.

---

## Budget comparison vs the Emender CMA (emender-1p3b-cma)

Reference: the substrate's own search `experiments/e99_1p3b_cma` (level `typed-gdn2-lm`,
the Emender backbone) ran **popsize 8, 12 generations, 96 evals,
stop="completed_all_generations"**, median 14.85 train-min/candidate
(`artifacts/stop_reason.json`, `generations.jsonl`).

| Search | popsize | gens | evals | clean stop |
|---|---|---|---|---|
| fla-gdn control | 8 | ~13 | 110 | no |
| m2rnn control | 8 | 16 | 107 | yes |
| typed-gdn2 (Emender substrate, e99) | 8 | 12 | 96 | yes |
| **Emender CMA (emender-1p3b-cma, pending)** | must = 8 | target 12-16 | target ~96-110 | — |

**Asymmetries flagged:**
1. **Between the two controls:** m2rnn got 16 generations, fla-gdn ~13 (total evals nearly
   equal: 107 vs 110). Minor in eval count; popsize identical (8). Tolerable but not exact.
2. **fla-gdn search incompleteness:** fla-gdn never emitted "SEARCH COMPLETE" and has no
   `_final` state pickle — it plateaued but may have been cut short. m2rnn terminated cleanly.
3. **MAJOR (geometry provenance):** the deployed gdn2 control is NOT its CMA winner (depth
   21 vs CMA-preferred depth 10). So even a perfectly budget-matched Emender CMA is being
   compared against a control whose geometry came from a *separate* selection path
   (long-racer hand-tuning + full Pile convergence), not the symmetric proxy CMA. For
   strict fairness the Emender's CMA-best should be compared either (a) against the
   *CMA-winner* geometries of the controls under the same train-to-convergence treatment,
   or (b) all three trained under the identical matched-token/matched-wall budget (which is
   what `emender_real_1p3b` actually did — see its VERDICT.md — making the geometry-search
   asymmetry less load-bearing for THAT comparison, but it still means "cma_gdn2" is the
   long-racer geometry, not a search output).

**Fairness recommendation for emender-1p3b-cma:** use popsize 8 and 12-16 generations
(~96-110 evals) to sit inside the control envelope. Report its CMA-best geometry/mixture
and held-out BPB. Note in the comparison that the controls' deployed geometries are
long-racer anchors (fla-gdn) / search-consistent anchors (m2rnn), trained to full Pile
convergence — so any head-to-head must equalize the *training* budget, not just the
*search* budget.

---

## Validation checklist
- [x] gdn2 (fla-gdn): gens ~13 / 110 evals / popsize 8; plateaued, **not cleanly terminated** — from `cmaes_redo_1300m_20260529/fla-gdn/fla-gdn_20260531_131116/` + `logs/fla-gdn.log`.
- [x] m2rnn: 16 gens / 107 evals / popsize 8; **converged** (SEARCH COMPLETE, 17.11 h) — from `m2rnn/m2rnn_20260531_131116/results.json` + `logs/m2rnn.log`.
- [x] Budget compared vs Emender/typed-gdn2 CMA (popsize 8, 12 gen, 96 evals); asymmetries flagged (gens 13 vs 16 vs 12; fla-gdn incompleteness; deployed != CMA-winner for gdn2).
- [x] Findings committed.
