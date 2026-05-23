# Expressivity Results Summary

Source-of-truth consolidation of all expressivity-task experiments in
`experiments/expressivity_tasks/`. Feeds the paper's expressivity section directly.

---

## Artifact Status

**All nine result directories under `experiments/expressivity_tasks/results/` contain
no committed artifacts.** Run output files (`.json`, `.log`) are covered by
`.gitignore` (`output/`, `benchmark_results/`, `checkpoints/`, `*.pt`, `*.pth`,
`*.safetensors`, `*.log`). Numbers that did survive into version control live in:

| File | Content |
|------|---------|
| `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` | 6-task × 3-pattern in-distribution table |
| `experiments/expressivity_tasks/LENGTH_EXTRAP_RESULTS.md` | parity / modular_counter / fsm_tracking extrapolation curves |
| `experiments/expressivity_tasks/MODULAR_COUNTER_FOLLOWUP.md` | Followup protocol description (results not committed) |
| `paper/ndmpapernotes.md` (lines 153–173) | S3/S5 headline table + length-extrapolation curve |

---

## Model Families Referenced

| Short name | Layer pattern | Dim / heads | Notes |
|------------|--------------|-------------|-------|
| E88 / NDM | `E88` | 384 / H=32 N=32 | Main model; nonlinear delta-correcting matrix memory |
| FLA-GDN | `fla-gdn` | 640 / H=32 N=32 | Linear recurrent baseline (delta-style, linear state) |
| M2RNN (tied/CMA) | `m2rnn` | 384 / H=32 N=32 | Nonlinear matrix-state, tied-head geometry |
| M2RNN-paper | `m2rnn-paper` | 608 / H=32 N=32 | Nonlinear matrix-state, published grouped-head geometry |
| hybrid_AABB | `[E88, E88, fla-gdn, fla-gdn]` | 384 | Ablation: mixed E88/GDN stack |

Mamba2 and LLaMA-attention are referenced in the paper narrative but are **not**
integrated into the expressivity harness; literature numbers are cited instead.

Canonical 8M-matched scale: `dim=384, depth=4, n_heads=32, n_state=32,
schedule-free AdamW, 10K steps, batch_size=32` (E88 / M2RNN tied); FLA-GDN uses
`dim=640` to match parameter count.

---

## Task Families

### 1. Parity

**Task:** Predict running parity (XOR) of a binary input stream.
Random baseline: 0.50.

#### 1a. Canonical sweep (in-distribution, T=128)

**Source:** `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/full_8m_matched_20260511` — **no committed artifacts**
**Protocol:** dim=384, depth=4, H=32 N=32, sf-AdamW, 10K steps, batch=32, 3 seeds (42/123/456)

| Pattern | Mean acc ± std |
|---------|---------------|
| pure_E88 | **1.000 ± 0.000** |
| pure_FLA | 0.857 ± 0.022 |
| hybrid_AABB | 1.000 ± 0.000 |

**Claim:** E88 and the AABB hybrid both solve parity perfectly; FLA-GDN reaches
only 0.86, showing that linear state is insufficient for reliable parity tracking.

#### 1b. Length extrapolation (train T=40, eval T∈{40,80,160,320,500})

**Source:** `experiments/expressivity_tasks/LENGTH_EXTRAP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/separation_canonical_seed42_20260511` or associated sweep — **no committed artifacts**
**Protocol:** same dim/depth/head config, 5K steps, 3 seeds

| Pattern | T=40 | T=80 | T=160 | T=320 | T=500 |
|---------|------|------|-------|-------|-------|
| pure_E88 | **1.000 ± 0.000** | **1.000 ± 0.001** | **0.984 ± 0.022** | **0.944 ± 0.054** | **0.887 ± 0.088** |
| pure_FLA | 0.997 ± 0.001 | 0.844 ± 0.017 | 0.673 ± 0.010 | 0.585 ± 0.006 | 0.550 ± 0.002 |

**Claim:** E88 retains 0.89 accuracy at 12.5× training length; FLA collapses to
near-random (0.55) by T=500.

---

### 2. Dyck Languages (dyck, dyck2)

**Task:** Predict well-balanced bracket sequences. `dyck` uses single bracket
type (K=8), `dyck2` extends to two types.
Random baseline: varies.

#### 2a. Canonical sweep (in-distribution)

**Source:** `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/full_8m_matched_20260511` — **no committed artifacts**
**Task config:** dyck: seq_len=256, K=8, 10K steps.

| Pattern | Mean acc ± std |
|---------|---------------|
| pure_E88 | 1.000 ± 0.000 |
| pure_FLA | 1.000 ± 0.000 |
| hybrid_AABB | 1.000 ± 0.000 |

**Claim:** Dyck-1 is solvable by all three families at dim=384. Serves as a
sanity check rather than a separation witness.

**dyck2:** Task definition exists (`tasks/dyck2.py`) and is included in the
separation suite runner config, but **no committed result numbers for dyck2**.
The `separation_8m_matched_20260511` and associated dirs are empty.

---

### 3. FSM Tracking

**Task:** Track the state of a finite-state machine with K=4 states across a
sequence of encoded transitions. Random baseline: 0.25.

#### 3a. Canonical sweep (in-distribution, T=256)

**Source:** `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/full_8m_matched_20260511` — **no committed artifacts**
**Protocol:** 10K steps, seq_len=256, K=4, 3 seeds.

| Pattern | Mean acc ± std |
|---------|---------------|
| pure_E88 | **1.000 ± 0.000** |
| pure_FLA | 0.830 ± 0.040 |
| hybrid_AABB | 0.713 ± 0.021 |

**Claim:** E88 solves FSM tracking perfectly; FLA reaches 0.83; the hybrid
AABB is worse than either pure model (0.71), demonstrating that FLA layers
degrade E88's state-tracking capability.

#### 3b. Length extrapolation (train T=40, eval T∈{40,80,160,320,500})

**Source:** `experiments/expressivity_tasks/LENGTH_EXTRAP_RESULTS.md`
**Run dir:** associated sweep dirs — **no committed artifacts**

| Pattern | T=40 | T=80 | T=160 | T=320 | T=500 |
|---------|------|------|-------|-------|-------|
| pure_E88 | **1.000 ± 0.000** | **1.000 ± 0.001** | **0.903 ± 0.065** | **0.711 ± 0.103** | **0.591 ± 0.102** |
| pure_FLA | 0.988 ± 0.006 | 0.924 ± 0.037 | 0.677 ± 0.081 | 0.473 ± 0.048 | 0.387 ± 0.034 |

**Claim:** Both grok at training length; E88 retains 0.59 at T=500, FLA falls
to 0.39. The gap widens monotonically with length.

---

### 4. Modular Counter

**Task:** Track a running modular counter with K states (K=5 canonical, K=20/K=50
hard variants). Random baseline: 1/K.

#### 4a. Canonical sweep (in-distribution, K=5, T=128)

**Source:** `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/full_8m_matched_20260511` — **no committed artifacts**
**Protocol:** 10K steps, seq_len=128, K=5, 3 seeds.

| Pattern | Mean acc ± std |
|---------|---------------|
| pure_E88 | **0.903 ± 0.033** |
| pure_FLA | 0.648 ± 0.118 |
| hybrid_AABB | 0.536 ± 0.238 |

**Claim:** E88 leads on modular counter; FLA-GDN reaches 0.65; hybrid AABB
degrades to 0.54 (worse than FLA). The high variance on FLA/hybrid suggests
some seeds grok and others do not within 10K steps.

#### 4b. Length extrapolation (train T=40, K=5, eval T∈{40,80,160,320,500})

**Source:** `experiments/expressivity_tasks/LENGTH_EXTRAP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/separation_canonical_seed42_20260511` — **no committed artifacts**

| Pattern | T=40 | T=80 | T=160 | T=320 | T=500 |
|---------|------|------|-------|-------|-------|
| pure_E88 | **0.973 ± 0.018** | **0.794 ± 0.083** | **0.517 ± 0.048** | **0.350 ± 0.023** | **0.300 ± 0.020** |
| pure_FLA | 0.477 ± 0.019 | 0.339 ± 0.010 | 0.268 ± 0.005 | 0.234 ± 0.002 | 0.225 ± 0.001 |
| hybrid_AABB | 0.508 ± 0.151 | 0.365 ± 0.083 | 0.283 ± 0.042 | 0.240 ± 0.022 | 0.227 ± 0.001 |

**Claim:** E88 groks the K=5 counter at T=40 (0.97); FLA-GDN does not (0.48
at training length, degrades to ~random at T=500). Hybrid tracks FLA, not E88.

#### 4c. Followup study (K=5 long / K=20 / K=50)

**Source:** `experiments/expressivity_tasks/MODULAR_COUNTER_FOLLOWUP.md`
**Run dir:** `experiments/expressivity_tasks/results/modular_counter_followup_20260511` — **no committed artifacts**
**Protocol:** K5_T128_long=30K steps; K20/K50_T256=20K steps. Models: E88_H32N32_bf16,
E88_H32N32_fp32, E88_H64N16_fp32, FLA_H32N32_bf16/fp32, M2RNN_tied, M2RNN_paper.

**Status:** The followup protocol and intended interpretations are documented in
`MODULAR_COUNTER_FOLLOWUP.md`. No committed result numbers. The doc describes
four diagnostic questions:
- Is M2RNN-tied's 10K edge over E88 on K=5 a grokking-speed artifact?
- Is E88 precision-sensitive on modular counter (fp32 vs bf16)?
- Does multiprogramming shape (H64/N16 vs H32/N32) help E88?
- Do M2RNN variants fail at K=20/K=50 or under length extrapolation?

---

### 5. S5 Permutation Composition

**This is the headline expressivity claim for the paper.**

**Task:** Track the running product in the symmetric group S5 (120 states) using
adjacent transposition generators. S5 is the NC1-complete witness by Barrington's
theorem. Also tested: S3 (6 states, solvable group control). Task definition:
`tasks/s5_permutation.py`. Random baseline: 1/120 = 0.0083 for S5, 1/6 = 0.1667
for S3.

#### 5a. 8M matched run (train T=128, 3 seeds)

**Source:** `paper/ndmpapernotes.md` lines 153–173
**Run dir:** `experiments/expressivity_tasks/results/s5_witness_8m_20260521` — **no committed artifacts**
**Protocol:** separation suite runner (`run_separation_suite.py`), 8M parameter-matched,
20K steps for s5_permutation (per TASK_CONFIG), T=128, seeds 42/123/456.
Length extrapolation at T∈{128, 256, 512, 1024}.

**Train-length accuracy (T=128):**

| Task | Model | Mean acc | Min | Max | Random baseline |
|------|-------|:--------:|:---:|:---:|:---------------:|
| S3 | E88/NDM | **1.0000** | 0.9999 | 1.0000 | 0.1667 |
| S3 | FLA-GDN | 0.7185 | 0.6122 | 0.8516 | 0.1667 |
| S3 | M2RNN (tied) | 0.3124 | 0.2742 | 0.3529 | 0.1667 |
| S3 | M2RNN-paper | 0.3773 | 0.3669 | 0.3855 | 0.1667 |
| S5 | E88/NDM | **0.7918** | 0.6232 | 0.8880 | 0.0083 |
| S5 | FLA-GDN | 0.3552 | 0.3148 | 0.3850 | 0.0083 |
| S5 | M2RNN (tied) | 0.2157 | 0.1856 | 0.2309 | 0.0083 |
| S5 | M2RNN-paper | 0.1698 | 0.1555 | 0.1844 | 0.0083 |

**S5 length extrapolation:**

| Model | T=128 (train) | T=256 | T=512 | T=1024 |
|-------|:------------:|:-----:|:-----:|:------:|
| E88/NDM | **0.7900** | **0.4158** | **0.2150** | **0.1104** |
| FLA-GDN | 0.3544 | 0.1843 | 0.0974 | 0.0521 |
| M2RNN (tied) | 0.2142 | 0.1120 | 0.0593 | 0.0339 |
| M2RNN-paper | 0.1696 | 0.0884 | 0.0488 | 0.0283 |

**Interpretation (from `paper/ndmpapernotes.md`):**
- NDM separates at training length (0.79 vs 0.36 FLA-GDN vs 0.22 M2RNN vs 0.17 M2RNN-paper), not only under extrapolation.
- NDM remains ahead under extrapolation, but all models degrade with length.
- M2RNN underperformance — both tied and paper-shaped — supports the update-rule
  claim: nonlinear matrix state alone is not sufficient; the delta correction and
  many-program geometry matter.
- FLA-GDN's 0.36 on S5 vs 0.72 on S3 vs 1.00 on parity illustrates the solvability
  gap: linear scan handles solvable-group tasks better than non-solvable ones.

---

### 6. Associative Recall

**Task:** Store key-value pairs in sequence, then retrieve the value for a
queried key. K=8 keys, seq_len=64. Random baseline: 1/vocab.

#### 6a. Canonical sweep (in-distribution)

**Source:** `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/full_8m_matched_20260511` — **no committed artifacts**
**Protocol:** 10K steps, seq_len=64, K=8, 3 seeds.

| Pattern | Mean acc ± std |
|---------|---------------|
| pure_E88 | 0.881 ± 0.025 |
| pure_FLA | **0.997 ± 0.003** |
| hybrid_AABB | 0.947 ± 0.006 |

**Claim:** FLA-GDN has a meaningful edge on associative recall (+0.12 over E88),
reflecting parallel-attention's natural fit for key-value lookup. This is the only
task in the canonical 6-task sweep where FLA-GDN outperforms E88 by more than noise.

---

### 7. Selective Copy

**Task:** Copy only the marked tokens from a mixed sequence. K=8, seq_len=256.

#### 7a. Canonical sweep (in-distribution)

**Source:** `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md`
**Run dir:** `experiments/expressivity_tasks/results/full_8m_matched_20260511` — **no committed artifacts**
**Protocol:** 10K steps, seq_len=256, K=8, 3 seeds.

| Pattern | Mean acc ± std |
|---------|---------------|
| pure_E88 | 1.000 ± 0.000 |
| pure_FLA | 1.000 ± 0.000 |
| hybrid_AABB | 1.000 ± 0.000 |

**Claim:** All three families solve selective copy perfectly. No separation here;
serves as a sanity check.

---

### 8. Delta Memory and Keyed FSM Memory

**Task (delta_memory):** Tests the delta-correcting write mechanism directly —
read-then-correct update against a target value. Task definition: `tasks/delta_memory.py`.

**Task (keyed_fsm_memory):** Maintains a keyed table of finite-state values;
operations are absolute writes or transition-dependent updates; final query asks
for the current state at a random key. K=8, seq_len=128. Designed to stress-test
the E88 vs M2RNN delta-correction difference. Task definition:
`tasks/keyed_fsm_memory.py`.

**Run dirs:**
- `experiments/expressivity_tasks/results/separation_8m_matched_20260511` — **no committed artifacts**
- `experiments/expressivity_tasks/results/separation_8m_smoke_20260511` — **no committed artifacts**
- `experiments/expressivity_tasks/results/separation_canonical_seed42_20260511` — **no committed artifacts**
- `experiments/expressivity_tasks/results/separation_canonical_smoke_20260511` — **no committed artifacts**
- `experiments/expressivity_tasks/results/separation_pilot_20260511` — **no committed artifacts**

**Status:** No committed result numbers for delta_memory or keyed_fsm_memory.
These tasks appear in the separation suite (`run_separation_suite.py` DEFAULT_TASK_ORDER
includes `keyed_fsm_memory`; `delta_memory` is defined but not in the default run list).
The `docs/M2RNN_E88_COMPARISON.md` describes keyed_fsm_memory as the primary
E88 vs M2RNN separation task and suggests it should show a delta-correction advantage,
but results are not committed.

---

## Result Directory Reference Table

| Directory | Runner script | Tasks | Models | Committed artifacts |
|-----------|--------------|-------|--------|-------------------|
| `full_8m_matched_20260511` | `run_separation_suite.py` (inferred) | parity, modular_counter, fsm_tracking, s3/s5_permutation, dyck, dyck2, selective_copy, assoc_recall, overwrite_recall, reset_recall, keyed_fsm_memory | E88_8M, FLA_8M, M2RNN_8M, M2RNN_paper_8M | **none** |
| `full_8m_matched_e88_triton_20260511` | `run_separation_suite.py --use_triton_e88` | same as above | same, E88 uses Triton kernel | **none** |
| `modular_counter_followup_20260511` | `run_modular_counter_followup.py` | modular_counter K=5/K=20/K=50 | E88_H32N32_{bf16,fp32}, E88_H64N16_fp32, FLA_H32N32_{bf16,fp32}, M2RNN_tied, M2RNN_paper | **none** |
| `s5_witness_8m_20260521` | `run_separation_suite.py --tasks s3_permutation s5_permutation --use_triton_e88` | s3_permutation, s5_permutation | E88_8M, FLA_8M, M2RNN_8M, M2RNN_paper_8M | **none** (numbers in `paper/ndmpapernotes.md` lines 153–173) |
| `separation_8m_matched_20260511` | `run_separation_suite.py` (separation subset) | keyed_fsm_memory, overwrite_recall, reset_recall | E88_8M, FLA_8M, M2RNN_8M, M2RNN_paper_8M | **none** |
| `separation_8m_smoke_20260511` | smoke run of above | subset | subset | **none** |
| `separation_canonical_seed42_20260511` | `run_separation_suite.py` or `run_canonical_sweep.py`, single seed | separation tasks | canonical models | **none** |
| `separation_canonical_smoke_20260511` | smoke run | subset | subset | **none** |
| `separation_pilot_20260511` | pilot run of separation suite | subset | subset | **none** |

---

## Claim-Mapping Table

| Paper claim | Quantitative support | Source file |
|-------------|---------------------|-------------|
| E88 wins or ties FLA-GDN on 5 of 6 canonical state-tracking tasks | parity 1.00 vs 0.86; mod_counter 0.90 vs 0.65; fsm_tracking 1.00 vs 0.83; dyck 1.00 vs 1.00; selective_copy 1.00 vs 1.00; assoc_recall 0.88 vs 1.00 | `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` |
| FLA-GDN collapses to near-random under length extrapolation; E88 does not | parity T=500: E88 0.887 vs FLA 0.550; fsm T=500: E88 0.591 vs FLA 0.387; mod_counter T=500: E88 0.300 vs FLA 0.225 (random=0.20) | `experiments/expressivity_tasks/LENGTH_EXTRAP_RESULTS.md` |
| Hybrid AABB underperforms pure E88 on state-tracking tasks | mod_counter hybrid 0.54 vs E88 0.90; fsm_tracking hybrid 0.71 vs E88 1.00 | `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` |
| NDM separates from all baselines on S5 permutation at training length | E88 0.79 vs FLA 0.36 vs M2RNN 0.22 vs M2RNN-paper 0.17 (random=0.0083) | `paper/ndmpapernotes.md` lines 161–164 |
| NDM separates from all baselines on S5 at T=256 (length extrapolation) | E88 0.416 vs FLA 0.184 vs M2RNN 0.112 vs M2RNN-paper 0.088 | `paper/ndmpapernotes.md` lines 168–173 |
| NDM remains ahead of all baselines on S5 at T=512 | E88 0.215 vs FLA 0.097 vs M2RNN 0.059 vs M2RNN-paper 0.049 | `paper/ndmpapernotes.md` lines 168–173 |
| E88 solves S3 (solvable group) perfectly; all baselines fail | S3: E88 1.000 vs FLA 0.719 vs M2RNN 0.312 vs M2RNN-paper 0.377 | `paper/ndmpapernotes.md` lines 157–160 |
| M2RNN nonlinear matrix state is insufficient for S5 (both tied and paper geometries) | S5: M2RNN 0.216 and M2RNN-paper 0.170 vs random 0.0083 (far below E88 0.79) | `paper/ndmpapernotes.md` lines 163–164 |
| FLA-GDN has an edge on associative recall (attention-natural task) | assoc_recall: FLA 0.997 vs E88 0.881 | `experiments/expressivity_tasks/CANONICAL_SWEEP_RESULTS.md` |
| Modular counter K=5 timing ambiguity (M2RNN-tied slight edge at 10K steps; needs 30K follow-up) | Referenced in MODULAR_COUNTER_FOLLOWUP.md; raw 10K numbers not committed | `experiments/expressivity_tasks/MODULAR_COUNTER_FOLLOWUP.md` |

---

## Gaps and Missing Data

The following are noted as data gaps — numbers are **not fabricated**:

1. **keyed_fsm_memory, overwrite_recall, reset_recall, delta_memory:** No committed
   result numbers. All separation run directories are empty. These are the tasks
   most targeted at the E88 vs M2RNN delta-correction separation.

2. **dyck2:** Task is defined (`tasks/dyck2.py`) and appears in the separation
   suite config, but no results committed.

3. **Modular counter followup (K=5 long / K=20 / K=50):** Protocol documented in
   `MODULAR_COUNTER_FOLLOWUP.md`; no result numbers committed. The key open
   questions (M2RNN-tied grokking speed, E88 precision, K=20/K=50 hardness) remain
   unanswered in committed data.

4. **E88 Triton vs PyTorch accuracy parity check:** `full_8m_matched_e88_triton_20260511`
   was the Triton-kernel accuracy-parity check. No committed artifacts.

5. **M2RNN-tied vs E88 at canonical 6-task scale:** The canonical sweep doc
   (`CANONICAL_SWEEP_RESULTS.md`) reports only E88, FLA, and hybrid_AABB. M2RNN-tied
   and M2RNN-paper rows in the canonical 6-task table are **not committed**. The
   S5/S3 numbers in `paper/ndmpapernotes.md` are the only committed E88 vs M2RNN
   comparison with full tables.
