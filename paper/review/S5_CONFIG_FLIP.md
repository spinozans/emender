# S5 CONFIG-FLIP 2×2: disentangling the linear-state KNOB from the CMA-found CONFIG

**Task:** `s5-config-flip` · **Date:** 2026-06-04 · Diagnostic only (no `paper/main.typ` edit).

## Question

The symmetric eval (`s5sym-eval`) found that **e88-linear** (`linear_state=1`) slightly
beats **e88-tanh** (`linear_state=0`) on S5. But each arm won via its *own*
300-step CMA-ES search, so the two arms differ in **two** things at once:

1. the **KNOB** — the state nonlinearity (`linear_state` 1 vs 0), and
2. the **CONFIG** — the CMA-found geometry/lr (config L: dim256/depth5/**H38**/N32,
   lr 0.00266 vs config T: dim256/depth5/**H39**/N32, lr 0.00295).

This task isolates the two by completing a 2×2 cross-flip. Two cells already
existed (cited from the eval, **not** rerun); this task ran the other two
(the off-diagonal flips), changing **only** `linear_state` relative to the
source config — no re-tuning.

|              | tanh (`linear_state=0`)         | linear (`linear_state=1`)          |
|--------------|----------------------------------|-------------------------------------|
| **config T** | **A** = `e88-tanh`  *(eval, cited)* | **D** = `e88-cfgT-linear` *(flip, new)* |
| **config L** | **C** = `e88-cfgL-tanh` *(flip, new)* | **B** = `e88-linear` *(eval, cited)* |

A and B are the original CMA winners; the diagonal A↔B is the comparison the
eval made. C and D are the flips that hold the geometry/lr fixed and toggle
only the knob.

## Recipe (identical across all four cells)

3 seeds {42,123,456}; schedule-free AdamW; batch 32; seq_len 128; expansion 1.0;
K 5. **S5** (`s5_permutation`): train T=128, 20 000 steps. **S3** control
(`s3_permutation`): train T=128, 10 000 steps. Eval grid T ∈ {128,256,512,1024}
(8 eval batches/length), end of training. A/B come verbatim from
`eval/summary.json`; C/D from this task's raw per-seed JSONs in `eval/`. The
flips reuse the source winner's geometry/lr **verbatim** and override **only**
`linear_state` (`use_gate=1` in all four — unchanged).

## Exact parameter counts (all four cells)

`linear_state` does **not** change the parameter count — it only swaps the
state recurrence's elementwise `tanh` for identity. Confirmed empirically: C
(config L + tanh) reports the same count as B (config L + linear); D (config T +
linear) the same as A (config T + tanh).

| cell | arm              | config | knob   | n_heads | **params**    |
|------|------------------|--------|--------|---------|---------------|
| A    | e88-tanh         | T      | tanh   | 39      | **8,069,766** (~8.07M) |
| D    | e88-cfgT-linear  | T      | linear | 39      | **8,069,766** (~8.07M) |
| C    | e88-cfgL-tanh    | L      | tanh   | 38      | **7,863,676** (~7.86M) |
| B    | e88-linear       | L      | linear | 38      | **7,863,676** (~7.86M) |

Matches the task's expectation (config T ≈ 8.07M H39, config L ≈ 7.86M H38).

## The 2×2 on S5@T128 (in-distribution; seed-mean ± SD, n=3)

|              | tanh (ls=0)        | linear (ls=1)      |
|--------------|--------------------|--------------------|
| **config T** | A: 0.9888 ± 0.0111 | D: **0.9998 ± 0.0002** |
| **config L** | C: 0.9967 ± 0.0042 | B: 0.9997 ± 0.0005 |

All four are at/near ceiling at T128 (≥0.989). The original A-vs-B gap
(+0.0109) is essentially fully recovered by flipping the knob on config T
(A→D = +0.0110), with the config making little difference at T128.

## S5 length-extrapolation grid (seed-mean ± SD, n=3)

| cell | arm              | cfg | knob   | T128            | T256            | T512            | T1024           |
|------|------------------|-----|--------|-----------------|-----------------|-----------------|-----------------|
| A    | e88-tanh         | T   | tanh   | 0.9888 ± 0.0111 | 0.6296 ± 0.0372 | 0.3216 ± 0.0176 | 0.1678 ± 0.0107 |
| B    | e88-linear       | L   | linear | 0.9997 ± 0.0005 | 0.7515 ± 0.1253 | 0.3909 ± 0.0779 | 0.2002 ± 0.0334 |
| C    | e88-cfgL-tanh    | L   | tanh   | 0.9967 ± 0.0042 | 0.6909 ± 0.0935 | 0.3551 ± 0.0504 | 0.1847 ± 0.0270 |
| **D**| **e88-cfgT-linear** | **T** | **linear** | **0.9998 ± 0.0002** | **0.8436 ± 0.0324** | **0.4384 ± 0.0192** | **0.2255 ± 0.0079** |

The dynamic range lives in the extrapolation regime (T≥256). There, **D
(config T + linear) is the strongest cell at every length** — and it is the one
cell the symmetric eval never tested.

## S3 control length-extrapolation grid (seed-mean ± SD, n=3)

| cell | arm              | cfg | knob   | T128            | T256            | T512            | T1024           |
|------|------------------|-----|--------|-----------------|-----------------|-----------------|-----------------|
| A    | e88-tanh         | T   | tanh   | 1.0000 ± 0.0000 | 0.9976 ± 0.0011 | 0.8929 ± 0.0378 | 0.6152 ± 0.0343 |
| B    | e88-linear       | L   | linear | 1.0000 ± 0.0000 | 0.9919 ± 0.0070 | 0.8646 ± 0.0957 | 0.6480 ± 0.1502 |
| C    | e88-cfgL-tanh    | L   | tanh   | 1.0000 ± 0.0000 | 0.9924 ± 0.0055 | 0.8045 ± 0.1018 | 0.5846 ± 0.1269 |
| D    | e88-cfgT-linear  | T   | linear | 0.9992 ± 0.0012 | 0.9584 ± 0.0675 | 0.8374 ± 0.1972 | 0.6716 ± 0.2398 |

The solvable S3 control confirms all four configurations train and solve the
task (T128 = 1.00 for all but D, which is 0.999). At S3 extrapolation the four
cells are statistically indistinguishable (overlapping SDs); the S5 differences
are therefore not an artifact of one config being broken. D carries large S3
seed variance at T512/1024 (driven by seed 42 underperforming on the control —
see per-seed below), so its S3 ordering is not meaningful; the diagnostic signal
is S5.

## Disentanglement verdict

Deltas at **S5@T128** (the metric the CMA optimized), plus the extrapolation
trend (Δ at T256, the first held-out length, in parentheses):

- **KNOB effect — is linear > tanh at FIXED config?** **Yes, robustly.**
  - At config L: B − C = +0.0030 @T128  (+0.0606 @T256)
  - At config T: D − A = +0.0110 @T128  (+0.2140 @T256)
  - Linear beats tanh in **all 8** config×length comparisons (2 configs × 4
    lengths). The effect is consistent and grows with extrapolation length,
    largest on config T.

- **CONFIG effect — is config L > config T at FIXED knob?** **No clean main
  effect; small and sign-flipping (an interaction).**
  - At tanh:   C − A = +0.0079 @T128  (+0.0613 @T256) — config L slightly better.
  - At linear: B − D = −0.0001 @T128  (−0.0921 @T256) — config **T** better, and
    increasingly so at longer T (−0.047 @T512, −0.025 @T1024).
  - i.e. config L helps under tanh but **hurts** under linear. There is no
    config that is uniformly better; the CONFIG axis interacts with the knob and
    is dominated by the knob in magnitude at the extrapolation lengths.

- **Conclusion.** The **e88-linear win is driven by the `linear_state` KNOB, not
  by the CMA-found config.** Holding geometry/lr fixed and flipping only the
  knob reproduces (and at T≥256 amplifies) the linear advantage on both
  configs. The geometry that CMA picked for the *linear* arm (config L) is in
  fact **mildly suboptimal for the linear state**: config T + linear (cell D)
  beats config L + linear (cell B, the published e88-linear) at **every** T≥256
  — +0.092 @T256, +0.048 @T512, +0.025 @T1024 — and with lower seed variance.

- **Rough-CMA caveat (implicated).** Because the win is knob-driven and config T
  + linear out-extrapolates the linear arm's own CMA config, the 300-step CMA
  search did **not** explore deeply or fairly between the two arms. Each arm got
  an independent ~300-step search with `linear_state` fixed; the linear arm's
  search settled on H38/lr 0.00266, but the H39/lr 0.00295 geometry found by the
  tanh arm's search transfers to the linear state and does *better* on held-out
  lengths. The truncated, per-arm search undersampled the geometry space, so the
  published config ranking under-credits the linear knob's true ceiling. A fair
  comparison would re-search (or at least cross-evaluate) geometries within each
  knob setting — exactly what cell D shows was missed.

## Per-seed spread (n=3; high-variance arms flagged)

The eval flagged high seed variance for some arms (notably B = e88-linear at
T256, SD 0.125). Per-seed S5 accuracy for the two new flip cells:

**C = e88-cfgL-tanh (config L + tanh):**
| T    | seed42 | seed123 | seed456 | SD     |
|------|--------|---------|---------|--------|
| 256  | 0.7945 | 0.6129  | 0.6654  | 0.0935 |
| 512  | 0.4100 | 0.3107  | 0.3448  | 0.0504 |
| 1024 | 0.2150 | 0.1633  | 0.1759  | 0.0270 |

**D = e88-cfgT-linear (config T + linear):**
| T    | seed42 | seed123 | seed456 | SD     |
|------|--------|---------|---------|--------|
| 256  | 0.8391 | 0.8138  | 0.8780  | 0.0324 |
| 512  | 0.4368 | 0.4200  | 0.4583  | 0.0192 |
| 1024 | 0.2277 | 0.2168  | 0.2320  | 0.0079 |

D is not only the highest-mean S5 cell at extrapolation but also the **tightest
across seeds** (e.g. T256 SD 0.032 vs B's 0.125) — the config-T geometry makes
the linear state both better and more stable on S5 than its own CMA config (B).
(On the S3 control, by contrast, D's seed 42 underperforms — S3 T512 seed42 =
0.612 vs 0.978/0.923 — inflating D's S3 SD; this does not affect the S5 signal.)

## Provenance / reproduce

- Flip driver: `scripts/eval_s5_config_flip.py` (REAL `train_hybrid.py`; no mocks;
  GPU safety gate <2GB; idempotent). Aggregator: `scripts/aggregate_s5_config_flip.py`.
- GPU gate: all 8 GPUs idle (2 MiB each) at launch → used GPUs [0–7] round-robin.
  Logged to `eval/logs/flip_gpus_used.txt`. 12/12 runs OK, 0 failed.
- Raw per-seed JSONs: `eval/e88-cfgL-tanh_{S5,S3}_seed{42,123,456}.json` and
  `eval/e88-cfgT-linear_{S5,S3}_seed{42,123,456}.json` (alongside the cited
  `e88-tanh_*`/`e88-linear_*`). Combined table: `eval/config_flip_summary.json`.
- A, B cited from `eval/summary.json` (s5sym-eval); C, D from this task.
