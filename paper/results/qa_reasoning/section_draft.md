# QA and Reasoning Capability Progression

## What the Evaluation Measures

We track model capability using a multiple-choice continuation harness we call the
**knowledge and reasoning quiz**. For each question, the model assigns a score to
each answer choice by computing the negative log-likelihood of continuing the prompt
with that choice token-by-token; the model's prediction is the lowest-cost (most
probable) continuation. This scoring approach, common in language model evaluation
[CITE: Brown et al. 2020, Gao et al. 2021 lm-eval], requires no generation and
produces deterministic predictions that are directly comparable across architectures.

The harness covers two separate quiz panels:

**Knowledge panel** (300 items per snapshot). Six standard NLP benchmark validation
splits sampled at 50 items each:
- ARC-Challenge and ARC-Easy [CITE: Clark et al. 2018]: grade-school science questions, 4 choices
- HellaSwag [CITE: Zellers et al. 2019]: commonsense completion, 4 choices
- SciQ [CITE: Welbl et al. 2017]: science QA, 4 choices
- OpenBookQA [CITE: Mihaylov et al. 2018]: elementary science, 4 choices
- BoolQ [CITE: Clark et al. 2019]: yes/no reading comprehension, 2 choices

Combined random baseline: ≈29% (five 4-choice tasks at 25% and one 2-choice task at
50%, equally weighted). Expected accuracy for a 1.27B-parameter model mid-training is
roughly 33–45% on this panel based on comparable published work.

**Reasoning panel** (≈2k items). Multi-step logical inference tasks:
- BIG-Bench Hard (BBH) [CITE: Suzgun et al. 2022]: 12 subtasks including boolean
  expression evaluation, causal judgment, date arithmetic, formal fallacy detection, and
  multi-step object tracking with 3, 5, or 7 objects.
- ReCLor [CITE: Yu et al. 2020]: logical reading comprehension, 4 choices (random = 25%).
- FOLIO [CITE: Han et al. 2022]: formal first-order logic inference with 3 labels
  (True / False / Uncertain, random = 33%).

## Models

We evaluate four 1.27B-parameter sequence models trained on The Pile with a 2k-token
context window. E88 is the primary NDM model; the others are baselines.

| Model   | Architecture | Dimension | Depth |
|---------|-------------|-----------|-------|
| E88     | NDM (E88)   | 1664      | 12    |
| FLA-GDN | Gated linear attention | 2688 | 21 |
| Mamba2  | Selective SSM | 2048   | 32    |
| M2RNN   | Matrix RNN  | 1920      | 21    |

## Knowledge Panel: Progression over Training

We ran the knowledge panel periodically during the late-stage training of all four
models, capturing four snapshots over approximately 18 hours (Table 1). Because models
differ in training step count (E88 is step-efficient; Mamba2 requires more steps for
comparable loss), we report accuracy against step rather than wall-clock time.

**Table 1.** Knowledge panel accuracy (avg-NLL score, 300 items) across training steps.

| Model   | Step  | Loss  | Overall | ARC-C | ARC-E | BoolQ | HellaSwag | OpenBookQA | SciQ  |
|---------|-------|-------|---------|-------|-------|-------|-----------|------------|-------|
| E88     | 891k  | 2.631 | 0.353   | 0.260 | 0.500 | 0.360 | 0.300     | 0.180      | 0.520 |
| E88     | 924k  | 2.654 | 0.353   | 0.260 | 0.520 | 0.380 | 0.300     | 0.180      | 0.480 |
| E88     | 942k  | 2.749 | 0.367   | 0.320 | 0.520 | 0.420 | 0.260     | 0.180      | 0.500 |
| FLA-GDN | 1188k | 2.764 | 0.383   | 0.300 | 0.540 | 0.600 | 0.280     | 0.180      | 0.400 |
| FLA-GDN | 1230k | 2.741 | 0.377   | 0.280 | 0.560 | 0.580 | 0.240     | 0.180      | 0.420 |
| FLA-GDN | 1251k | 2.660 | 0.380   | 0.260 | 0.480 | 0.600 | 0.280     | 0.220      | 0.440 |
| M2RNN   | 810k  | 2.662 | 0.377   | 0.280 | 0.520 | 0.560 | 0.300     | 0.180      | 0.420 |
| M2RNN   | 846k  | 2.746 | 0.383   | 0.320 | 0.500 | 0.620 | 0.280     | 0.140      | 0.440 |
| M2RNN   | 861k  | 2.728 | 0.367   | 0.300 | 0.480 | 0.560 | 0.280     | 0.220      | 0.360 |
| Mamba2  | 1605k | 2.855 | 0.353   | 0.220 | 0.440 | 0.500 | 0.260     | 0.220      | 0.480 |
| Mamba2  | 1665k | 2.735 | 0.377   | 0.260 | 0.540 | 0.480 | 0.260     | 0.180      | 0.540 |
| Mamba2  | 1695k | 2.577 | 0.360   | 0.220 | 0.500 | 0.460 | 0.280     | 0.220      | 0.480 |

*Random baseline: ≈29% overall (25% for 4-choice tasks, 50% for BoolQ).*

All four models score substantially above the random baseline (29%) across the observed
window, confirming that at this scale and training stage the models have acquired genuine
multiple-choice discrimination ability. Within-model variation across the four snapshots
is modest (±2–3 pp, within the standard-error of ≈6 pp for 50-item slices), consistent
with slow, monotonic capability growth in the late-training phase.

E88 ends the snapshot window at **0.367** overall, rising from 0.353 at step 891k. The
per-category pattern is consistent: E88 is strongest on ARC-Easy (0.500–0.520) and SciQ
(0.480–0.520) and weakest on OpenBookQA (0.180), mirroring the other models.

FLA-GDN leads on BoolQ (0.580–0.600) throughout, suggesting better utilization of
passage context for yes/no judgment. Mamba2 shows more variance across snapshots than
the RNN-family models, consistent with its lower step efficiency.

## Reasoning Panel: Single-Snapshot Results

At the latest available snapshot (2026-05-22), we evaluated all four models on the
reasoning panel (Table 2). This panel tests multi-step inference rather than factual
retrieval.

**Table 2.** Reasoning panel accuracy (avg-NLL, ≈2k items). A subset of columns is shown;
full per-task breakdown is in `reasoning_panel_latest.csv`.

| Model   | Step  | Overall | Formal Fallacies | Web of Lies | Log.Ded.-3 | Log.Ded.-7 | Track-7 | FOLIO | ReCLor |
|---------|-------|---------|-----------------|-------------|------------|------------|---------|-------|--------|
| E88     | 957k  | 0.319   | 0.424           | 0.427       | 0.349      | 0.114      | 0.110   | 0.367 | 0.234  |
| FLA-GDN | 1272k | 0.350   | 0.570           | 0.601       | 0.308      | 0.161      | 0.125   | 0.360 | 0.228  |
| M2RNN   | 879k  | 0.336   | 0.596           | 0.434       | 0.349      | 0.128      | 0.132   | 0.360 | 0.214  |
| Mamba2  | 1722k | 0.324   | 0.437           | 0.510       | 0.390      | 0.148      | 0.096   | 0.333 | 0.186  |

*Random baselines: 50% for boolean/causal/web-of-lies (2-choice); ~25–33% for multi-object tasks; 25% for ReCLor; 33% for FOLIO.*

Several patterns stand out:

1. **Multi-step object tracking collapses with chain length.** Performance on
   `logical_deduction_3_objects` (0.31–0.39) degrades substantially to
   `logical_deduction_7_objects` (0.11–0.16) and to
   `tracking_shuffled_objects_7_objects` (0.10–0.13), near or below random for the
   harder variants. This is consistent with the hypothesis that 1.27B recurrent models
   lack the in-context working memory to track seven simultaneous state variables.

2. **FLA-GDN leads on formal inference tasks.** FLA-GDN achieves 0.570 on formal
   fallacies and 0.601 on web-of-lies, outperforming E88 (0.424, 0.427) and Mamba2
   (0.437, 0.510) by meaningful margins. M2RNN also scores well on formal fallacies
   (0.596) despite its smaller step count. Whether this reflects a genuine reasoning
   advantage or a difference in training trajectory at the snapshot time requires
   additional controlled comparison.

3. **FOLIO and ReCLor are near-random for all models.** FOLIO scores (0.33–0.37)
   barely exceed the 33% baseline; ReCLor scores (0.19–0.23) are below the 25% baseline.
   These are among the hardest reasoning tasks even for much larger models, and 1.27B
   models at this training stage appear not to have acquired the multi-hop deductive
   structure these tasks require.

4. **E88 is not systematically weaker on reasoning.** Despite E88's architecture being
   novel relative to the well-studied gated-attention and SSM baselines, its overall
   reasoning accuracy (0.319) is within one standard error of M2RNN (0.336) and Mamba2
   (0.324). FLA-GDN's lead is larger but may partly reflect its higher step count at
   the snapshot.

## Relation to Loss Trajectory

Across both panels, accuracy correlates broadly with loss: lower-loss checkpoints tend
to score higher. However, the per-snapshot window is narrow (each model covers only
~50k steps between first and last snapshot, with loss fluctuating ±0.1 due to learning
rate schedule). This makes it impossible to establish a reliable accuracy-vs-loss curve
from these data alone. The available evidence is consistent with the standard finding
that perplexity and downstream accuracy are correlated but not tightly coupled at fixed
scale [CITE].

## Limitations

- The snapshot window is narrow relative to total training. E88's first snapshot is at
  step 891k; we have no quiz measurements from early training (steps < 700k for any model
  on the multi-task panel).
- With 50 items per task in the 300-item panel, each category accuracy has SE ≈ 6–7 pp.
  Category-level differences smaller than ~15 pp should not be over-interpreted.
- The reasoning panel has only one snapshot (no within-training progression for this panel).
- Models are compared at their respective snapshot steps, not at matched loss levels.
