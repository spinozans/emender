# QA / Reasoning Progression — Internal Reference

## Overview

This document collects the quiz/QA eval results for the four 1.27B-parameter racer
models (E88, FLA-GDN, Mamba2, M2RNN) run periodically during training on The Pile
with 2k context. The "quiz" is a multiple-choice continuation-NLL harness: the model
scores each answer choice by computing the negative log-likelihood of continuing the
prompt with that choice, then picks the lowest-NLL continuation as its prediction.

## The Eval System

### scripts/racer_eval_suite.py

The main batched eval runner. Scores multiple-choice items by continuation NLL.
Accepts JSON/JSONL probe files (or the built-in 40-item probe from
`knowledge_continuation_probe.py`) and supports two scoring modes:
- `stateful-prefix`: reuses recurrent hidden state after the shared prompt prefix
  (used for E88 and M2RNN, which have simple tensor states)
- `full-sequence`: groups by input length to avoid padding-sensitive score drift
  (used for FLA-GDN and Mamba2)

Primary metric reported: `avg_nll` (NLL normalized by number of choice tokens).
Also computes `nll` (raw) and `pmi_avg` (PMI = conditional NLL minus neutral-prefix NLL).

### scripts/build_racer_eval_panel.py

Builds a JSONL probe file by sampling from 6 HuggingFace datasets:
- `arc_challenge` (ARC Challenge, science questions, 4 choices)
- `arc_easy` (ARC Easy, 4 choices)
- `hellaswag` (commonsense completion, 4 choices)
- `sciq` (science QA, 4 choices)
- `openbookqa` (elementary science, 4 choices)
- `boolq` (yes/no reading comprehension, 2 choices)

Default: 50 items per task → 300 total items. This is the "fact" or "knowledge" panel.

### scripts/build_reasoning_eval_panel.py

Builds a JSONL probe file from reasoning-focused datasets:
- `reclor` (logical reasoning, 4 choices)
- `folio` (formal logic / first-order inference, 3 choices: True/False/Uncertain)
- 12 BIG-Bench Hard (BBH) subtasks:
  - boolean_expressions, causal_judgement, date_understanding, disambiguation_qa,
    formal_fallacies, logical_deduction_{3,5,7}_objects,
    tracking_shuffled_objects_{3,5,7}_objects, web_of_lies

Default: up to 160 items per task, capped at 2048 total.

### scripts/generate_racer_samples.py

Generates freeform text samples from checkpoint. Used for qualitative inspection.
Also provides shared utilities (checkpoint loader, model builder, ScheduleFree weight
swap) imported by the other eval scripts.

### scripts/knowledge_continuation_probe.py

Contains a hard-coded built-in 40-item probe covering 6 categories:
- `facts` (8 items): geography, basic science, authorship
- `concepts` (8 items): biology, physics, word meanings
- `code` (8 items): Python, SQL, HTML, JS, git, C idioms
- `math` (6 items): arithmetic, derivatives, Pythagorean theorem, primes
- `format` (6 items): Stack Overflow, LaTeX, paper abstract, patent style
- `coherence` (4 items): cause-effect, pronoun resolution, discourse

This is the "quiz" that was run first (at earlier training steps). Random baseline
is 25% (4-choice items throughout), except format items also have 4 choices.

### scripts/run_periodic_racer_evals.py

Orchestration script: takes snapshots every N minutes during live training runs,
hardlinks newest checkpoint per model into stable paths, then invokes
`racer_eval_suite.py`. Default: 4 snapshots, 360-minute intervals.

Configured for 4 racers at fixed /tmp checkpoint directories:
- E88: `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/...`
- FLA-GDN: `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/...`
- Mamba2: `/tmp/pile_convergence_3arch/ctx2k/mamba2_resume_ckpt/...`
- M2RNN: `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/...`

Output root: `~/racer_eval_runs/ctx2k_periodic`

---

## Eval Log Locations

Output lives under `~/racer_eval_runs/` (not `~/elman/`):

```
~/racer_eval_runs/
  ctx2k_panel_20260521_initial/          # First 300-item panel run (2026-05-21)
    snapshot_20260521_222601/
  ctx2k_panel_periodic_20260521/         # Periodic runs, 6h intervals (2026-05-22)
    periodic.log
    snapshot_20260522_043436/
    snapshot_20260522_103913/
    snapshot_20260522_164536/
  ctx2k_fact_reasoning_20260522/         # Fact + reasoning panels in one shot
    snapshot_20260522_223345/
      fact_e88_fla/                      # Fact panel for E88 + FLA-GDN
      fact_mamba_m2/                     # Fact panel for Mamba2 + M2RNN
      reasoning_e88_fla/                 # Reasoning panel for E88 + FLA-GDN
      reasoning_mamba_m2/                # Reasoning panel for Mamba2 + M2RNN
  ctx2k_panel_2k_20260522/               # Full-size (~2k item) fact panel
    e88_fla/snapshot_20260522_215134/
    mamba_m2/snapshot_20260522_215134/
```

Earlier knowledge probe logs in:
```
/tmp/racer_generations_20260518/
  knowledge_probe_systematic_{E88,FLA-GDN,M2RNN,Mamba2}.json
  knowledge_probe_avg_{e88,fla,m2rnn,mamba2}.json
  knowledge_probe.json
  knowledge_probe_mamba2.json
```

---

## Built-in Probe (40-item): Progression

Scored with `avg_nll`. Random baseline: 25%.

| Model   | Step  | Accuracy | Code  | Coherence | Concepts | Facts | Format | Math  |
|---------|-------|----------|-------|-----------|----------|-------|--------|-------|
| E88     | 705k  | 0.725    | 0.875 | 1.000     | 0.750    | 0.750 | 0.667  | 0.333 |
| FLA-GDN | 945k  | 0.750    | 0.875 | 1.000     | 0.750    | 0.500 | 0.833  | 0.667 |
| M2RNN   | 615k  | 0.675    | 0.875 | 1.000     | 0.625    | 0.625 | 0.667  | 0.333 |
| Mamba2  | 1266k | 0.750    | 0.875 | 1.000     | 0.750    | 0.750 | 0.667  | 0.500 |

All models score 100% on coherence (pronoun resolution, cause-effect). All score ≥87.5% on
code idioms. Math is the hardest category: 33–67%.

---

## Racer Panel (300-item, 6 NLP tasks): Progression

Scored with `avg_nll`. Per-task random baselines: 25% (4-choice), 50% (BoolQ 2-choice).
Combined panel random baseline ≈ 29% (5 × 25% + 1 × 50%, equal 50-item slices).

### E88 (NDM model)

| Step  | Loss   | Accuracy | ARC-C | ARC-E | BoolQ | HellaSwag | OpenBookQA | SciQ  |
|-------|--------|----------|-------|-------|-------|-----------|------------|-------|
| 891k  | 2.6305 | 0.353    | 0.260 | 0.500 | 0.360 | 0.300     | 0.180      | 0.520 |
| 909k  | 2.6566 | 0.343    | 0.200 | 0.520 | 0.380 | 0.260     | 0.200      | 0.500 |
| 924k  | 2.6541 | 0.353    | 0.260 | 0.520 | 0.380 | 0.300     | 0.180      | 0.480 |
| 942k  | 2.7485 | 0.367    | 0.320 | 0.520 | 0.420 | 0.260     | 0.180      | 0.500 |

Range covered: steps 891k–942k (51k steps, ~6 hours between snapshots).

### FLA-GDN

| Step  | Loss   | Accuracy | ARC-C | ARC-E | BoolQ | HellaSwag | OpenBookQA | SciQ  |
|-------|--------|----------|-------|-------|-------|-----------|------------|-------|
| 1188k | 2.7644 | 0.383    | 0.300 | 0.540 | 0.600 | 0.280     | 0.180      | 0.400 |
| 1209k | 2.7054 | 0.377    | 0.200 | 0.560 | 0.600 | 0.260     | 0.240      | 0.400 |
| 1230k | 2.7414 | 0.377    | 0.280 | 0.560 | 0.580 | 0.240     | 0.180      | 0.420 |
| 1251k | 2.6597 | 0.380    | 0.260 | 0.480 | 0.600 | 0.280     | 0.220      | 0.440 |

### Mamba2

| Step  | Loss   | Accuracy | ARC-C | ARC-E | BoolQ | HellaSwag | OpenBookQA | SciQ  |
|-------|--------|----------|-------|-------|-------|-----------|------------|-------|
| 1605k | 2.8549 | 0.353    | 0.220 | 0.440 | 0.500 | 0.260     | 0.220      | 0.480 |
| 1635k | 2.7762 | 0.360    | 0.200 | 0.560 | 0.420 | 0.280     | 0.220      | 0.480 |
| 1665k | 2.7353 | 0.377    | 0.260 | 0.540 | 0.480 | 0.260     | 0.180      | 0.540 |
| 1695k | 2.5768 | 0.360    | 0.220 | 0.500 | 0.460 | 0.280     | 0.220      | 0.480 |

### M2RNN

| Step  | Loss   | Accuracy | ARC-C | ARC-E | BoolQ | HellaSwag | OpenBookQA | SciQ  |
|-------|--------|----------|-------|-------|-------|-----------|------------|-------|
| 810k  | 2.6616 | 0.377    | 0.280 | 0.520 | 0.560 | 0.300     | 0.180      | 0.420 |
| 828k  | 2.8516 | 0.363    | 0.320 | 0.500 | 0.540 | 0.220     | 0.180      | 0.420 |
| 846k  | 2.7462 | 0.383    | 0.320 | 0.500 | 0.620 | 0.280     | 0.140      | 0.440 |
| 861k  | 2.7276 | 0.367    | 0.300 | 0.480 | 0.560 | 0.280     | 0.220      | 0.360 |

---

## Reasoning Panel (BBH + ReCLor + FOLIO): Latest Snapshot Only

Scored with `avg_nll`. Run at one snapshot (2026-05-22 ~22:33 UTC).

Random baselines by task:
- BBH boolean_expressions, causal_judgement, web_of_lies: 50% (2 choices)
- BBH date_understanding, disambiguation_qa, formal_fallacies: varies (~25-33%)
- BBH logical_deduction_*, tracking_shuffled_objects_*: varies by object count
- ReCLor: 25% (4 choices)
- FOLIO: 33% (3 choices: True/False/Uncertain)

| Model   | Step  | Overall Acc | Bool.Exp | Causal | DateU | DisambQA | FormalFal | LogDed3 | LogDed5 | LogDed7 | Track3 | Track5 | Track7 | WebLies | FOLIO | ReCLor |
|---------|-------|-------------|----------|--------|-------|----------|-----------|---------|---------|---------|--------|--------|--------|---------|-------|--------|
| E88     | 957k  | 0.319       | 0.593    | 0.486  | 0.303 | 0.423    | 0.424     | 0.349   | 0.213   | 0.114   | 0.267  | 0.160  | 0.110  | 0.427   | 0.367 | 0.234  |
| FLA-GDN | 1272k | 0.350       | 0.593    | 0.507  | 0.324 | 0.415    | 0.570     | 0.308   | 0.213   | 0.161   | 0.300  | 0.187  | 0.125  | 0.601   | 0.360 | 0.228  |
| M2RNN   | 879k  | 0.336       | 0.593    | 0.507  | 0.324 | 0.359    | 0.596     | 0.349   | 0.227   | 0.128   | 0.300  | 0.173  | 0.132  | 0.434   | 0.360 | 0.214  |
| Mamba2  | 1722k | 0.324       | 0.579    | 0.521  | 0.248 | 0.387    | 0.437     | 0.390   | 0.180   | 0.148   | 0.300  | 0.220  | 0.096  | 0.510   | 0.333 | 0.186  |

Key observations:
- `boolean_expressions` (True/False): 57–59%. Random = 50%. Marginal gain (~7–9pp).
- `web_of_lies` (2-choice logical tracing): 43–60%. FLA-GDN strongest (60%).
- `formal_fallacies`: FLA-GDN 57%, M2RNN 60% — substantially above random (33%).
- `logical_deduction_7_objects`: 10–16%. Random ≈ 14%. Near-chance.
- `tracking_shuffled_objects_7`: 10–13%. Near-chance (random ≈ 14%).
- `reclor` (logical reading comprehension): 19–23%. Near-chance (random = 25%).
- `folio` (first-order logic): 33–37%. Close to random (33%).

Structural observations:
- All models struggle on the "multi-step object tracking" tasks (tracking_shuffled_objects_*,
  logical_deduction_7_objects) — performance approaches random as object count grows.
- E88 and M2RNN's stateful prefix caching is unused for the reasoning panel (batches differ).
- FLA-GDN has a notable lead on formal_fallacies (0.570) and web_of_lies (0.601).

---

## Fact Panel (2k items): Latest Snapshot

Scored with `avg_nll`. Same 6 tasks as the 300-item panel but with larger samples (~300+
items per task). Single snapshot.

| Model   | Step  | Loss   | Overall | ARC-C | ARC-E | BoolQ | HellaSwag | OpenBookQA | SciQ  |
|---------|-------|--------|---------|-------|-------|-------|-----------|------------|-------|
| E88     | 957k  | 2.6113 | 0.347   | 0.241 | 0.351 | 0.417 | 0.329     | 0.229      | 0.500 |
| FLA-GDN | 1269k | 2.6615 | 0.368   | 0.258 | 0.366 | 0.586 | 0.311     | 0.231      | 0.440 |
| Mamba2  | 1719k | 2.6527 | 0.361   | 0.268 | 0.354 | 0.537 | 0.289     | 0.237      | 0.466 |
| M2RNN   | 876k  | 2.7040 | 0.368   | 0.261 | 0.340 | 0.563 | 0.314     | 0.271      | 0.443 |

---

## Summary: Score Trajectory for E88 (NDM) — Full Range

The only model with enough data points to show progression:

| Step  | Panel          | N    | Accuracy | Notes                      |
|-------|----------------|------|----------|----------------------------|
| 705k  | Built-in probe | 40   | 0.725    | Early run, 6-category probe |
| 891k  | Racer panel    | 300  | 0.353    | Different eval, harder tasks |
| 909k  | Racer panel    | 300  | 0.343    |                            |
| 924k  | Racer panel    | 300  | 0.353    |                            |
| 942k  | Racer panel    | 300  | 0.367    |                            |
| 957k  | Fact panel     | ~1.4k| 0.345    |                            |
| 957k  | Reasoning panel| ~2k  | 0.319    | BBH/ReCLor/FOLIO           |

The jump from 0.725 (40-item probe) to 0.353 (300-item racer panel) reflects panel
difficulty, not regression: the built-in probe uses very easy items (all models near
ceiling on code/coherence), while the racer panel uses standard NLP benchmarks where
1.27B models are expected to score 30–45%.

---

## Caveats

1. **Step counts are not comparable across models.** FLA-GDN ran ~1.27M steps while
   E88 ran ~957k steps at the time of the latest snapshot. Step counts reflect tokens
   seen differently due to batch sizes.

2. **Panel sampling is seeded but not shuffled the same way each run.** The 300-item
   panel uses a fixed seed (20260521) for deterministic item selection, but two
   different run directories use the same panel and their items should be identical.
   The 2k panel uses a different selection.

3. **Score noise.** With 50 items per task in the 300-item panel, accuracy values have
   standard error ≈ √(p(1-p)/50) ≈ 6–7 pp. Differences < 3 pp should not be over-interpreted.

4. **No temporal step progression for FLA-GDN / Mamba2 / M2RNN on reasoning panel.**
   Only one reasoning panel snapshot exists. Progression over training is documented
   only for the 300-item fact panel.
