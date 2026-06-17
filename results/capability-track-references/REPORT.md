# Capability vs token count on the reference checkpoints — emender (E97) vs gdn2-mlp

**Task:** `capability-track-references`. **Date:** 2026-06-17.
**Pairs with:** `results/offline-eval-references` (held-out BPB) — re-scored fused here.
**Kernels:** FUSED Triton for **both** arms, verified per-checkpoint by a real
kernel-invocation guard (see *NON-NEGOTIABLE #1 correction* — this is the headline
methodological event of the task).

## TL;DR

- **SCALE_PLAN question — does emender DIVERGE from gdn2 on any capability axis as
  tokens grow, or converge?** Over the references' measured range (176M→1.23B
  tokens) there is **no capability axis on which emender separates from gdn2**:
  every per-axis effect size is within its bootstrap 95% CI of 0, across two
  independent panel resamples. **But the honest framing is "vacuous-at-floor", not
  "capability converges":** on every hard, arm-differentiating axis (state-tracking
  shuffled-objects 3/5/7, logical-deduction, causal-judgement, reclor) **both** 1.3B
  models sit at the panel chance floor the entire way — you cannot demonstrate
  convergence of a capability neither model has yet. The eval **rules out a large
  emender capability edge** on the handful of above-floor axes and finds both arms
  at chance on the hard ones.
- **The one above-floor positive (web_of_lies) is a class-prior artifact, not
  capability** (majority-class baseline + next-token "No" prior; see below). It does
  not survive multiple-comparison correction.
- **Paired with held-out BPB (now fused for both arms):** gdn2 leads emender by
  **+0.11 to +0.18 BPB** at matched tokens (gdn2 the better LM — the convergent-loss
  band). **So: same BPB band, and no capability divergence either — a deeper null,
  with the caveat that the differentiating axes are at the floor for both.**

## NON-NEGOTIABLE #1 correction (READ FIRST) — the eager-fallback bug and its fix

A 9-agent adversarial confound-audit workflow flagged, and this task **reproduced on
the real checkpoint**, a violation of project NON-NEGOTIABLE #1 in the *first* version
of this run:

- **Bug:** the emender (E97) recurrence ran in the **eager PyTorch per-step loop, not
  the fused Triton kernel**, during eval. The fused paths in
  `ndm/models/e88_fla_hybrid.py` (`use_cuda`, `use_optimized`, and the projection-path
  `_use_optimized`) were all gated on `self.training`. The eval harness sets
  `model.eval()` (`self.training=False`), so the recurrence silently fell back to the
  eager scan. Reproduced via `verify_fused_dispatch.py` on
  `ref_emender_mlp/.../checkpoint_step_150500`: **EVAL → 0 fused-kernel calls, 1408
  eager per-step calls** (11 mixers × 128 steps); `model.train()` → 11 fused / 0 eager.
  The old `FUSED-GUARD` (a check on the `use_triton` *config integer*) was **vacuous**
  — it could not see the train-vs-eval fallback. gdn2-mlp was unaffected: it runs the
  FLA `chunk_gated_delta_rule` Triton kernel regardless of training mode.
- **Why it matters:** per this repo's hard-won lessons (E97 fused INERT in autocast;
  TF32 fused untrainable while eager looked fine) eager and fused are *different,
  non-transferable* experiments; an eager run "scores 0 at the gate".
- **Fix (this task):**
  1. `e88_fla_hybrid.py` — opt-in `fused_inference` flag (default off, so training /
     generation are unchanged) gates every fused-vs-eager decision; when set, EVAL
     takes the **identical** fused path training took.
  2. `eval_checkpoint.py:build_model` — auto-enables `fused_inference` on every
     `E88FLAHybrid` mixer (root-cause fix for *all* evals through this loader, incl. BPB).
  3. `ndm/triton/e88_triton_optimized.py` — the sparse-checkpoint forward requires
     `T % 16 == 0` (training always feeds aligned `T`); forward-only eval scores
     arbitrary continuation lengths, so the wrapper now **right-pads `T` to a multiple
     of 16 and slices the output back**. Verified causal-exact: perturbing the padded
     tail leaves real-position outputs **bitwise unchanged** (Δ=0.000), and the
     internal pad is byte-identical to a manual pad.
  4. `capability_track_references.py` — a **real** fused-guard: instruments the kernel
     entry points and asserts `fused>0 ∧ eager==0` via a probe forward, **per
     checkpoint**. All 7 emender checkpoints × 2 seeds logged `fused-guard PASS:
     fused_seq=11 … eager_act=0`.
- **Effect of the correction on conclusions:** small. emender per-checkpoint accuracy
  shifted by mean |Δ|=0.003 (≈0 systematic), held-out BPB by ≤0.001 BPB. The NULL
  verdict is therefore **robust** to the eager→fused correction — but it now rests on
  the **valid fused kernel**, as required. The eager artifacts are archived under
  `archive_eager_*/` for provenance.

## What was measured

The **same multiple-choice continuation-NLL panel** behind paper Fig 3 /
`paper/results/qa_reasoning`, scored on **every saved checkpoint of both references**
at the checkpoint token cadence, forward-only.

- **Panels** (built offline from the local HF cache, committed under `panels/`):
  - **knowledge** (300 items): `arc_easy, arc_challenge, hellaswag, sciq, openbookqa,
    boolq` (racer panel, `build_racer_eval_panel.py`, seed 20260521).
  - **reasoning** (2048 items): BBH + ReClor + FOLIO (`build_reasoning_eval_panel.py`,
    seed 20260522), including the **state-tracking separator**
    `tracking_shuffled_objects_{three,five,seven}` (the natural-language analog of the
    paper's S3/S5 axis), `logical_deduction_{3,5,7}`, and `web_of_lies`.
- **Scoring:** continuation NLL over tokenized choices, `avg_nll` primary,
  forward-only (`model(x, return_loss=False)` under `torch.no_grad`,
  `@torch.no_grad()` scorer), reusing `scripts/racer_eval_suite.py` verbatim.
- **Loading:** reused from `scripts/eval_checkpoint.py` — strict state-dict load
  (raises on any missing/unexpected key) + schedule-free **y-mode** swap
  (`optimizer.train()`). Every load logged `sf_y_swap=True`.
- **FUSED, no eager:** emender E97 via the fused Triton kernel (verified per
  checkpoint, see above); gdn2-mlp via FLA `chunk_gated_delta_rule`. Real fused-guard,
  not a config flag.
- **GPU isolation:** exactly **1 idle GPU** leased via the broker each run; the two
  reference trainings (GPUs 0/1) and other agents' jobs were untouched; eval is
  read-only on checkpoints. Nothing clobbered.
- **Both references are 1.3B**, param-matched (emender 1,286,589,072; gdn2
  1,286,713,448). tokens/step = `batch 4 × chunk 2048 = 8192`.
- **Coverage:** 7 emender (176M–1.23B) + 6 gdn2 (205M–1.23B) checkpoints × 2 panels ×
  2 seeds.

## Capability vs tokens (overlaid) — fused

See `capability_vs_tokens.png` and `capability_bpb_paired.png`. Overall accuracy per
panel (`capability_by_checkpoint.csv`, primary seed):

| tokens | emender knowledge | gdn2 knowledge | emender reasoning | gdn2 reasoning |
|---:|---:|---:|---:|---:|
| 176/205 M | 0.303 | 0.297 | 0.315 | 0.315 |
| 352/410 M | 0.270 | 0.343 | 0.334 | 0.319 |
| 528/614 M | 0.350 | 0.323 | 0.313 | 0.309 |
| 704/819 M | 0.333 | 0.307 | 0.321 | 0.306 |
| 881/1024 M | 0.307 | 0.363 | 0.306 | 0.316 |
| 1057/1229 M | 0.320 | 0.340 | 0.325 | 0.330 |
| 1233 M (em) | 0.297 | — | 0.326 | — |

Both curves are **flat and overlapping vs tokens** on both panels; neither moves
measurably over 176M→1.23B tokens and neither leads.

### State-tracking (the paper's arm-differentiator), pooled `shuffled_objects {3,5,7}` (chance ≈ 0.225)

| tokens | emender | gdn2 |
|---:|---:|---:|
| 176/205 M | 0.220 | 0.195 |
| 352/410 M | 0.225 | 0.197 |
| 528/614 M | 0.225 | 0.206 |
| 704/819 M | 0.222 | 0.209 |
| 881/1024 M | 0.188 | 0.200 |
| 1057/1229 M | 0.211 | 0.209 |
| 1233 M (em) | 0.218 | — |

Both arms are **pinned at/below the chance floor** the entire way — neither does
state-tracking at all at ≤1.23B tokens, and they are statistically indistinguishable.

## Per-axis effect sizes & verdict (fused, primary seed; final matched token)

Two-sample (unpaired, 10 000-iter) bootstrap 95% CI on raw item outcomes at the final
matched token, plus the slope of the emender−gdn2 gap vs tokens
(`capability_axis_verdict.csv`). Pre-registered divergence rule: CI excludes 0 **and**
|Δ|>0.03.

| axis | n/arm | emender | gdn2 | Δ (em−gdn2) | bootstrap 95% CI | slope/100M | verdict |
|---|---:|---:|---:|---:|---|---:|:--|
| knowledge (overall) | 300 | 0.297 | 0.340 | −0.043 | [−0.120, +0.030] | −0.002 | converge/null |
| **reasoning (overall)** | 2048 | 0.326 | 0.330 | −0.003 | [−0.032, +0.025] | −0.001 | converge/null |
| state-tracking 3/5/7 | 436 | 0.218 | 0.209 | +0.009 | [−0.046, +0.064] | −0.003 | converge/null |
| logical-deduction 3/5/7 | 445 | 0.218 | 0.240 | −0.023 | [−0.079, +0.034] | −0.001 | converge/null |
| web_of_lies (artifact) | 143 | 0.580 | 0.504 | +0.077 | [−0.042, +0.189] | +0.002 | converge/null |
| shuffled-objects-3 | 150 | 0.313 | 0.300 | +0.013 | [−0.093, +0.113] | −0.005 | converge/null |
| shuffled-objects-5 | 150 | 0.207 | 0.220 | −0.013 | [−0.107, +0.080] | −0.004 | converge/null |
| shuffled-objects-7 | 136 | 0.125 | 0.096 | +0.029 | [−0.044, +0.103] | +0.000 | converge/null |

**Every axis: null.** The most statistically powerful axis — `reasoning (overall)`,
n=2048 — is the tightest and deadest null: Δ=−0.003, CI [−0.032, +0.025] (rules out
accuracy divergences larger than ~0.032). **Cross-seed** (`capability_seed_robustness.csv`):
every axis agrees in being null across both panel resamples.

## Why "vacuous-at-floor", and why web_of_lies is NOT an exception

The confound analysis (`analyze_confounds.py` → `confound_floor_map.csv`,
`confound_multiple_comparisons.csv`, `confound_web_of_lies.json`) is what keeps the
NULL honest:

- **Floor map.** 10 of 20 categories are **at-floor-always** — never clear chance
  (Wilson lower bound > chance) for *either* model at *any* checkpoint or seed:
  `shuffled_objects_{3,5,7}`, `logical_deduction_{3,5}`, `causal_judgement`, `reclor`,
  `boolq`, `hellaswag`, `openbookqa`. These include exactly the hard state-tracking /
  reasoning axes that differentiated arms in the paper's fully-trained models. **You
  cannot demonstrate capability convergence where neither model has the capability** —
  so on the differentiating axes the null is *vacuous-at-floor*, not a positive
  convergence result.
- **web_of_lies is a class-prior artifact, not capability.** It is the one
  above-floor axis where emender appears to lead. It is a binary (" No"/" Yes")
  BBH subtask with an **imbalanced gold distribution (81 No / 62 Yes → majority-class
  baseline 0.572, NOT 0.5)**. At the final checkpoint emender scores 0.580 (s1) /
  0.593 (s2) by **predicting " No" 117–118/143** — per-class accuracy 0.85–0.88 on
  gold=No but **0.20–0.23 on gold=Yes** (far below chance on the minority class).
  gdn2 is the exact mirror (predicts " Yes" ~104/143; accNo 0.27–0.31, accYes
  0.76). Neither model does lie-chain tracking — they have opposite next-token priors
  after "Answer:", and emender's " No" prior happens to align with the majority
  class. Versus the **correct** floor (0.572) emender is only +0.008 (s1) / +0.020
  (s2) — not a capability.
- **Multiple comparisons.** At the final checkpoint, 3/20 (emender) and 4/20 (gdn2)
  categories clear chance at one-sided p<0.05 — expected false positives ≈1. web_of_lies
  (p≈0.033) is **not even the strongest hit** (arc_easy p≈0.014), and its
  Bonferroni-corrected p≈0.65. The two analysis "seeds" also share ~87% of web_of_lies
  items, so "significant in both seeds" is not independent replication. Removing
  web_of_lies (correctly) leaves **no capability divergence on any axis**.

## Pairing with held-out BPB — fused vs fused, the complete statement

Both arms' held-out pile-tail BPB re-scored on the **fused** kernel
(`emender_heldout_bpb_fused.csv` + the already-fused committed `gdn2_heldout_bpb.csv`
→ `matched_token_bpb_curve_fused.csv`). emender's fused BPB matched its eager BPB to
≤0.001 BPB. Matched-token lead over the BPB overlap [205M, 1024M]:

> **gdn2 leads emender on held-out BPB by +0.11 to +0.18 BPB** (gdn2 the better LM —
> the convergent-loss band), **and capability also shows no emender divergence on any
> axis; on the hard, arm-differentiating axes both models are at the panel chance
> floor.**

So at this 1.3B / ≤1.23B-token operating point the convergent-loss null is not hiding
a capability divergence. **Scope caveat:** the panel has not "activated" on the hard
axes at these token budgets (1–2 orders of magnitude below the paper's fully-trained
models), so this **rules out an early emender capability edge** but cannot speak to the
far larger token counts where capability emerges. (Capability extends to 1.23B; the
matched fused-BPB overlap is [205M,1024M] — the committed gdn2 BPB stops at 1024M.)

## Confound audit (NULL discipline)

A "no capability difference" reading is a red flag; it was put through a 9-agent
adversarial workflow + this task's own reproduction. Outcome: the audit **did not
survive in v1** (it caught the eager-fallback bug and over-claimed web_of_lies), the
bugs were **fixed**, and the corrected fused result survives:

1. **Fused, no eager** — FIXED and verified per checkpoint by a real kernel-invocation
   guard (was the v1 blocker).
2. **Right metric, both at floor** — the dominant caveat, now made explicit via the
   floor map: the differentiating axes are at chance for both → vacuous-at-floor.
3. **Checkpoint loaded incl. SF y-mode** — strict load + `optimizer.train()` y-swap;
   all loads logged `sf_y_swap=True` (audited pass).
4. **Enough eval samples** — reasoning n=2048 (CI ±0.03); web_of_lies n=143 is
   underpowered and, correctly, demoted to artifact.
5. **Matched tokens** — interpolated onto the measured overlap, no extrapolation
   (audited pass).
6. **Not over-claiming a divergence** — web_of_lies reframed from "suggestive
   exception" to class-prior artifact (majority baseline, per-class accuracy, MC
   correction); no axis is called a divergence.

## Files

- `scripts/capability_track_references.py` — driver (real fused-guard + loader + scorer).
- `scripts/eval_checkpoint.py` — loader; `build_model` auto-enables `fused_inference`.
- `ndm/models/e88_fla_hybrid.py`, `ndm/triton/e88_triton_optimized.py` — the fused-eval fix.
- `verify_fused_dispatch.py` — reproduces the eager-vs-fused dispatch on the real checkpoint.
- `analyze_capability.py` — verdict + matched-token + seed-robustness + plots.
- `analyze_confounds.py` — floor map + multiple comparisons + web_of_lies class-prior.
- `capability_by_checkpoint.csv` (+ `_s2`), `capability_items.jsonl` (+ `_s2`).
- `capability_matched_token.csv`, `capability_axis_verdict.csv`, `capability_seed_robustness.csv`.
- `confound_floor_map.csv`, `confound_multiple_comparisons.csv`, `confound_web_of_lies.json`.
- `emender_heldout_bpb_fused.csv`, `matched_token_bpb_curve_fused.csv`.
- `capability_vs_tokens.png`, `capability_bpb_paired.png`.
- `run_capability_track*.sh`, `run_capability_track_FUSED*.log`, `run_bpb_fused.log`.
- `archive_eager_*/` — the superseded v1 (eager-emender) artifacts, kept for provenance.
