# Capability vs token count on the reference checkpoints — emender (E97) vs gdn2-mlp

**Task:** `capability-track-references`. **Date:** 2026-06-17.
**Pairs with:** `results/offline-eval-references` (held-out BPB curve).
**Verdict discipline:** this is the **capability-axis** companion to the BPB curve. The headline
is a **NULL (capability also converges)** and is written under the task's NULL discipline — the
confound audit and its dominant caveat are in their own section below, not buried.

## The SCALE_PLAN question

The offline BPB curve already showed the expected **convergent-loss** picture: held-out pile-tail
BPB lands in the same band for both architectures, with gdn2-mlp leading emender by **+0.13–0.19
BPB** at matched tokens. LM loss therefore cannot distinguish the two. The SCALE_PLAN question this
task answers is the capability one:

> Does emender **DIVERGE** from gdn2 on any capability axis as tokens grow, or do they **converge**
> on capability too (a deeper null)?

**Answer (within the references' measured token range 176M–1024M): they CONVERGE — a deeper null.**
There is **no capability axis** on which emender separates from gdn2; the two track together within
noise at every token cadence. The essential caveat (see Confound audit) is that the hard,
arm-differentiating axes are at the panel's **chance floor for both models** at this early token
budget, so this rules out an *early* emender capability edge but does not speak to the far larger
token counts at which the paper's fully-trained models showed signal.

## What was measured

The **same multiple-choice continuation-NLL panel** that sits behind paper Fig 3 /
`paper/results/qa_reasoning`, scored on **every saved checkpoint of both references** at the
checkpoint token cadence, forward-only.

- **Panels** (built offline from the local HF cache, committed under `panels/`):
  - **knowledge** (300 items) — `arc_easy, arc_challenge, hellaswag, sciq, openbookqa, boolq`
    (the racer 300-item panel, `build_racer_eval_panel.py`, seed 20260521).
  - **reasoning** (2048 items) — BBH + ReCLor + FOLIO (`build_reasoning_eval_panel.py`, seed
    20260522), **including the state-tracking separator** `tracking_shuffled_objects_{three,five,
    seven}_objects` (S_n permutation tracking, the natural-language analog of the paper's
    S3/S5 state-tracking axis), plus `logical_deduction_{3,5,7}` and `web_of_lies`.
- **Scoring:** continuation NLL over tokenized choices, `avg_nll` primary score, forward-only
  (`model(x, return_loss=False)` under `torch.no_grad`), reusing `scripts/racer_eval_suite.py`
  verbatim. Driver: `scripts/capability_track_references.py`.
- **Loading:** reused **verbatim** from `scripts/eval_checkpoint.py` — the same loader the
  offline-eval-references BPB task used. Builds the **FUSED** kernel and applies the schedule-free
  **y-mode** weight swap.
- **FUSED, no eager (NON-NEGOTIABLE #1):** every one of the 20 (checkpoint × panel) loads logged
  the fused kernel — emender `level=E97 use_triton=1` (hard-imports the split-edit Triton kernel,
  raises rather than falling back to eager; a `FUSED-GUARD` assert in the driver refuses any
  emender checkpoint that loads with `use_triton≠1`); gdn2 `level=gdn2-mlp` →
  `GDN2ExternalMLPLayer(mode="chunk")`, the NVIDIA GatedDeltaNet-2 chunked Triton kernel
  (`ImportError` if `GDN2_PATH` absent — no eager path).
- **Schedule-free y-mode:** every load logged `sf_y_swap=True` (optimizer state loaded,
  `optimizer.train()` swaps the stored x/eval weights to y/train weights).
- **GPU isolation:** exactly **1 idle GPU** leased via the broker (GPU 2). The two reference
  trainings on GPUs 0/1 (pids 522358 emender, 584124 gdn2) and unrelated jobs (GPUs 3/4) were
  untouched. Nothing clobbered; offline, on checkpoints only.
- **Both references are 1.3B**, matched in params (emender 1,286,589,072; gdn2 1,286,713,448).
  tokens/step = `batch_size 4 × chunk_size 2048 = 8192`.
- **Coverage:** all 5 emender + all 5 gdn2 present checkpoints scored on both panels = 20/20.

## Capability vs tokens (overlaid)

See `capability_vs_tokens.png` (knowledge / reasoning / state-tracking / logical-deduction) and
`capability_bpb_paired.png` (BPB curve beside the state-tracking curve). Overall accuracy per panel
(`capability_by_checkpoint.csv`):

| tokens | emender knowledge | gdn2 knowledge | emender reasoning | gdn2 reasoning |
|---:|---:|---:|---:|---:|
| 176 M | 0.293 | — | 0.315 | — |
| 205 M | — | 0.300 | — | 0.315 |
| 352 M | 0.270 | — | 0.334 | — |
| 410 M | — | 0.337 | — | 0.320 |
| 528 M | 0.350 | — | 0.311 | — |
| 614 M | — | 0.323 | — | 0.311 |
| 704 M | 0.330 | — | 0.319 | — |
| 819 M | — | 0.307 | — | 0.308 |
| 881 M | 0.313 | — | 0.305 | — |
| 1024 M | — | 0.363 | — | 0.314 |

Both curves are **flat and overlapping vs tokens** on both panels. Neither architecture's capability
moves measurably over 176M→1024M tokens, and neither leads.

### State-tracking (the paper's arm-differentiator), pooled `shuffled_objects {3,5,7}` (chance ≈ 0.225)

| tokens | emender | gdn2 |
|---:|---:|---:|
| 176 / 205 M | 0.220 | 0.197 |
| 352 / 410 M | 0.225 | 0.197 |
| 528 / 614 M | 0.225 | 0.206 |
| 704 / 819 M | 0.225 | 0.211 |
| 881 / 1024 M | 0.186 | 0.200 |

Both arms are **pinned at / below the chance floor** (≈0.225) the entire way — neither model does
state-tracking at all at ≤1B tokens, and they are statistically indistinguishable.

## Matched-token leader & per-axis effect sizes

Accuracies interpolated onto a shared token grid inside the measured overlap **[205 M, 881 M]** (no
extrapolation); full table `capability_matched_token.csv`. Per-axis effect size at the final matched
token with a two-sample (unpaired, 10 000-iter) bootstrap 95% CI on the raw item outcomes, plus the
slope of the emender−gdn2 gap vs tokens (`capability_axis_verdict.csv`):

| axis | n/arm | emender | gdn2 | Δ (em+−gdn2) | bootstrap 95% CI | gap slope /100M tok | verdict |
|---|---:|---:|---:|---:|---|---:|:--|
| knowledge (overall) | 300 | 0.313 | 0.307 | +0.007 | [−0.067, +0.083] | +0.006 | converge/null |
| reasoning (overall) | 2048 | 0.305 | 0.308 | −0.003 | [−0.031, +0.025] | −0.001 | converge/null |
| **state-tracking 3/5/7** | 436 | 0.186 | 0.211 | −0.025 | [−0.078, +0.030] | −0.007 | converge/null |
| logical-deduction 3/5/7 | 445 | 0.223 | 0.227 | −0.005 | [−0.061, +0.049] | +0.001 | converge/null |
| web-of-lies | 143 | 0.518 | 0.483 | +0.035 | [−0.084, +0.147] | +0.004 | converge/null |
| shuffled-objects-3 | 150 | 0.280 | 0.333 | −0.053 | [−0.160, +0.047] | −0.016 | converge/null |
| shuffled-objects-5 | 150 | 0.167 | 0.167 | +0.000 | [−0.087, +0.087] | −0.004 | converge/null |
| shuffled-objects-7 | 136 | 0.103 | 0.125 | −0.022 | [−0.096, +0.052] | +0.000 | converge/null |

**Every axis: null.** Every bootstrap CI includes 0; every gap slope is tiny (|Δ| ≤ 0.016
accuracy per 100M tokens) and inconsistent in sign across axes. The most statistically powerful
axis — `reasoning (overall)`, n=2048 — is the tightest and deadest null: Δ=−0.003, CI
[−0.031, +0.025]. There is no axis, and no token cadence, at which emender separates from gdn2.

## Pairing with held-out BPB — the complete statement

From `offline-eval-references`: at matched tokens gdn2-mlp leads emender on held-out pile-tail BPB by
**+0.13–0.19 BPB** (the convergent-loss band, gdn2 the better LM). This task adds the capability
half:

> **Same BPB band (gdn2 ahead by ~0.13–0.19 BPB), and capability *also* converges: across
> 176M→1024M tokens there is no capability axis — knowledge QA, reasoning, or state-tracking — on
> which emender separates from gdn2; both track together and both sit at/near the panel's chance
> floor on the synthetic-style separators.**

So at this 1.3B / ≤1 B-token operating point the convergent-loss null is *not* hiding a capability
divergence — it extends into a **deeper capability null**. emender's claimed capability edge (the
modular-quadratic / state-tracking separator) does **not** manifest as an LM-checkpoint capability
advantage over gdn2 anywhere in the references' token range.

## Confound audit (NULL discipline) — the dominant caveat first

A "no capability difference" reading is a red flag; this one survives the stack, but with one
caveat that must lead:

1. **Right metric, but both at the panel floor (the dominant caveat).** On the hard,
   arm-differentiating axes (state-tracking, logical-deduction) **both** models are at/below chance
   the entire token range (state-tracking ≈0.225 chance; reasoning flat ≈0.31). The panel is the
   correct one (it is the paper Fig-3 panel and *does* register signal on the paper's fully-trained
   1.3B models, ~0.32–0.38), but these reference checkpoints are 1–2 orders of magnitude earlier in
   tokens than those, so the panel has **not activated**. The honest scope: this **rules out an
   early emender capability advantage** and shows the two converge while neither has the capability
   yet; it **cannot** rule out divergence at the much larger token budgets where capability emerges.
   The knowledge panel (easier, slightly above its ~0.29 pooled chance) tells the same story — flat,
   no separation.
2. **Fused, no eager.** All 20 loads logged the fused kernel (emender `use_triton=1`; gdn2
   `mode="chunk"`); the driver asserts `FUSED-GUARD` for emender. No eager path exists in either.
3. **Checkpoint loaded incl. SF y-mode.** All 20 loads logged `sf_y_swap=True` (strict state-dict
   load + `optimizer.train()` y-swap), identical to the BPB task.
4. **Enough eval samples.** 300 knowledge + 2048 reasoning items; per-axis Wilson/bootstrap CIs are
   ±0.03–0.08, tight enough to exclude any divergence larger than ~0.10 on the pooled axes and
   ~0.03 on `reasoning (overall)`.
5. **Matched tokens.** Comparisons are interpolated onto the shared overlap [205M, 881M]; no
   extrapolation. emender and gdn2 checkpoint token grids differ (8192 tok/step, different
   save cadences) and are handled by interpolation.
6. **Not over-claiming a divergence.** No axis is called a divergence: every effect size is within
   its bootstrap CI of 0 and the gap slopes are sign-inconsistent.

## Cross-seed robustness (the "seeds" axis)

The continuation-NLL scoring is deterministic given (checkpoint, panel), so the relevant
stochasticity is **which items the panel sampled**. A second, independently sampled panel set
(`*_s2`, knowledge seed 20270521 / reasoning seed 20270522) was scored over all 20 checkpoint-panel
combos to confirm the convergence reading is not an artifact of one item draw. See
`capability_seed_robustness.csv`. **Both seeds agree: every axis stays null** (all |Δ_final| within
noise; no axis flips to a divergence under the resample).

## Files

- `scripts/capability_track_references.py` — driver (reuses eval_checkpoint loader + racer scorer).
- `run_capability_track.sh` / `run_capability_track_s2.sh` — 1-GPU-lease run drivers (seed-1/seed-2).
- `panels/` — committed offline-built panels (s1 + s2) and manifests.
- `capability_by_checkpoint.csv` (+ `_s2`) — per-(checkpoint × category) accuracy with token counts.
- `capability_items.jsonl` (+ `_s2`) — per-item correctness (for CIs / bootstrap).
- `capability_matched_token.csv` — matched-token emender/gdn2/Δ per axis.
- `capability_axis_verdict.csv` — per-axis effect size + bootstrap CI + slope + verdict.
- `capability_seed_robustness.csv` — cross-seed Δ_final per axis.
- `capability_vs_tokens.png`, `capability_bpb_paired.png` — overlaid curves.
- `analyze_capability.py` — builds all of the above from the item JSONLs + the BPB curve.
- `run_capability_track.log` (+ `_s2`) — full run logs (per-checkpoint fused level + sf_y_swap).
