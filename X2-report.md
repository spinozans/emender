# X2 — released-pair invertible-map probes

## Status and preregistered instrument

Status: **complete**. Exact suffix-only floor throughout: **0.026874%** (1/3721).
No absent result is imputed. All accuracies below are percentages, full-vocabulary argmax exact joint class `61*x+y`; both coordinates must match. Invalid classes fail both coordinates.

The live supervisor amended the reviewed scratch dim1024/depth2 plan: use the released 1.2866B E97 and matched 1.3B GDN2-MLP loaders/geometry with Schedule-Free train/y recovery. Scratch probes are NOT run (available only for a future separately approved mechanism study). Recipe: full fine-tune bf16 AdamW (0.9,0.95), lr5e-5, warmup300, wd0.01, clip0.5, batch32. E97/GDN2 seeds0/1/2; identity seed0; one-step diagnostics all three seed0. Frozen paired full-input-deduplicated 128 train/512 val/512 test per depth. Production drivers128/512/1024 plus two init tokens.

Stopping is symmetric: validation every250 completed updates, first running-best gain <0.001 accuracy over2000 updates stops training, hard cap60000. Best validation accuracy, then lower CE, then earlier update selects ALL arms. Eval0 anchors the first window. Selection identities/hashes are frozen before test/extrapolation evaluation. Full-state running, coordinates, terminal, post-suffix7 burn-in counts/losses are retained in JSON. No targets ever enter model inputs.

Implementation fact: the existing fused E97 kernel accepts unaligned lengths only in forward-only mode. The first calibration attempt failed at update1 for both E97 variants; GDN2 was explicitly stopped to restart consistently. `attempt1_alignment/` retains all failures, traces and partial logs. Supervisor approved identical constant-zero right padding to multiple16 on every arm and split (3->16,130->144). Pad logits are sliced away before loss/scoring; pads occur strictly after real causal prefixes, so they cannot contaminate real positions or leak targets. Permanent fixed-weight forward/loss and training masked-pad gradient tests pass. Only the E97 carried-state tanh is replaced by identity; all parameter keys/values/geometry remain identical at initialization.

## Per-seed selected results

| Run | Best val | Selected update | Stop update/reason | Test1x | Test4x | Test8x | Suffix floor |
|---|---:|---:|---|---:|---:|---:|---:|
| headline_e97_seed0 | 0.027466% | 1500 | 2000/plateau | 0.030518% | 0.022125% | 0.021362% | 0.026874% |
| headline_e97_seed1 | 0.022888% | 1000 | 2000/plateau | 0.018311% | 0.017548% | 0.019073% | 0.026874% |
| headline_e97_seed2 | 0.022888% | 0 | 2000/plateau | 0.032043% | 0.027466% | 0.032234% | 0.026874% |
| headline_gdn2_seed0 | 0.036621% | 1250 | 2000/plateau | 0.022888% | 0.023270% | 0.020790% | 0.026874% |
| headline_gdn2_seed1 | 0.032043% | 1000 | 2000/plateau | 0.018311% | 0.020218% | 0.020409% | 0.026874% |
| headline_gdn2_seed2 | 0.025940% | 2000 | 2000/plateau | 0.022888% | 0.017548% | 0.020027% | 0.026874% |
| headline_e97_identity_seed0 | 0.027466% | 0 | 2000/plateau | 0.025940% | 0.022125% | 0.026512% | 0.026874% |
| calibration_e97_seed0 | 0.195312% | 750 | 2750/plateau | 0.195312% | not applicable | not applicable | 0.026874% |
| calibration_gdn2_seed0 | 0.976562% | 3000 | 3750/plateau | 0.390625% | not applicable | not applicable | 0.026874% |
| calibration_e97_identity_seed0 | 0.195312% | 3500 | 3750/plateau | 0.000000% | not applicable | not applicable | 0.026874% |

## All-seed aggregate (mean ± sample SD; SEM in JSON)

- e97: best val 0.024414% ± 0.002643%; test_1x 0.026957% ± 0.007527%, test_4x 0.022380% ± 0.004964%, test_8x 0.024223% ± 0.007031%; floor 0.026874%.
- gdn2: best val 0.031535% ± 0.005359%; test_1x 0.021362% ± 0.002643%, test_4x 0.020345% ± 0.002863%, test_8x 0.020409% ± 0.000381%; floor 0.026874%.

Per-seed paired E97−GDN2 differences (accuracy fractions) and all counts are in `results_summary.json`. A best seed is never substituted for the three-seed mean. Seed uncertainty is not a token-binomial interval.

## Coordinates, arithmetic calibration, and cost

| Run | Test1x x | Test1x y | Test1x terminal | Parameters | Updates | Real training tokens | Tokens/s training | Wall seconds | Suffix floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| headline_e97_seed0 | 1.152039% | 1.081848% | 0.000000% | 1286589072 | 2000 | 8320000 | 7474.6 | 1289.0 | 0.026874% |
| headline_e97_seed1 | 1.145935% | 1.060486% | 0.000000% | 1286589072 | 2000 | 8320000 | 8253.4 | 1173.8 | 0.026874% |
| headline_e97_seed2 | 1.699829% | 1.689148% | 0.000000% | 1286589072 | 2000 | 8320000 | 7538.7 | 1279.4 | 0.026874% |
| headline_gdn2_seed0 | 1.316833% | 1.203918% | 0.000000% | 1286713448 | 2000 | 8320000 | 7603.0 | 1270.7 | 0.026874% |
| headline_gdn2_seed1 | 1.440430% | 1.318359% | 0.000000% | 1286713448 | 2000 | 8320000 | 7829.5 | 1233.4 | 0.026874% |
| headline_gdn2_seed2 | 1.301575% | 1.129150% | 0.000000% | 1286713448 | 2000 | 8320000 | 7460.0 | 1290.7 | 0.026874% |
| headline_e97_identity_seed0 | 1.635742% | 1.676941% | 0.000000% | 1286589072 | 2000 | 8320000 | 7735.1 | 1247.0 | 0.026874% |
| calibration_e97_seed0 | 9.179688% | 0.781250% | 0.195312% | 1286589072 | 2750 | 264000 | 747.2 | 451.7 | 0.026874% |
| calibration_gdn2_seed0 | 24.804688% | 0.976562% | 0.390625% | 1286713448 | 3750 | 360000 | 576.6 | 749.0 | 0.026874% |
| calibration_e97_identity_seed0 | 4.492188% | 1.757812% | 0.000000% | 1286589072 | 3750 | 360000 | 749.8 | 606.1 | 0.026874% |

- e97: 11 matrix-state layers, 2,433,024 recurrent matrix elements per sequence (short-conv caches excluded); per-layer shapes/flags in config.
- gdn2: 12 matrix-state layers, 5,898,240 recurrent matrix elements per sequence (short-conv caches excluded); per-layer shapes/flags in config.
- e97_identity: 11 matrix-state layers, 2,433,024 recurrent matrix elements per sequence (short-conv caches excluded); per-layer shapes/flags in config.

Real training tokens include the two initialization positions but exclude right-padding overhead. Physical padded tokens/sequence and matrix-state element counts per layer are in each config. Released heads are retained (full vocab), not replaced by the scratch plan's 3721-class heads. Released GDN2 configuration governs even where it differs from the old scratch control's eigenvalue setting; actual flags are recorded in headline configs.

## Matched-step panels

Selected-checkpoint panels above are not equal-step comparisons. Below: every retained mandatory fixed step and the union of actual headline stopping steps. Arms are NEVER trained past their plateau to fill a later panel. Any fixed panel beyond an arm's stop (4000/8000/12000/16000/18269 or the10K grid) is not reached, not zero accuracy.

| Run | Update | Test1x | Test4x | Test8x | Suffix floor |
|---|---:|---:|---:|---:|---:|
| headline_e97_seed0 | 1000 | 0.027466% | 0.024033% | 0.022888% | 0.026874% |
| headline_e97_seed0 | 2000 | 0.019836% | 0.016785% | 0.022316% | 0.026874% |
| headline_e97_seed1 | 1000 | 0.018311% | 0.017548% | 0.019073% | 0.026874% |
| headline_e97_seed1 | 2000 | 0.021362% | 0.019455% | 0.018311% | 0.026874% |
| headline_e97_seed2 | 1000 | 0.025940% | 0.019073% | 0.017738% | 0.026874% |
| headline_e97_seed2 | 2000 | 0.024414% | 0.019836% | 0.019646% | 0.026874% |
| headline_gdn2_seed0 | 1000 | 0.021362% | 0.023270% | 0.019646% | 0.026874% |
| headline_gdn2_seed0 | 2000 | 0.016785% | 0.022507% | 0.018120% | 0.026874% |
| headline_gdn2_seed1 | 1000 | 0.018311% | 0.020218% | 0.020409% | 0.026874% |
| headline_gdn2_seed1 | 2000 | 0.016785% | 0.021362% | 0.019264% | 0.026874% |
| headline_gdn2_seed2 | 1000 | 0.024414% | 0.020599% | 0.021172% | 0.026874% |
| headline_gdn2_seed2 | 2000 | 0.022888% | 0.017548% | 0.020027% | 0.026874% |
| headline_e97_identity_seed0 | 1000 | 0.021362% | 0.012207% | 0.019455% | 0.026874% |
| headline_e97_identity_seed0 | 2000 | 0.021362% | 0.014877% | 0.023651% | 0.026874% |

## Suffix calibration

Analytic suffix0/1/7 cap = 0.026874% once at least two drivers are omitted (width7 starts at driver9). The CPU verification rerun and `suffix_calibration.json` retain fresh 2,000,000-independent-trajectory measurements per prime, known initialization, seeds, exact counts, six-sigma checks, and the empirical old parity-transfer control. These are not learned-model upper bounds.

## Honest interpretation

No practical learned-computation separation is demonstrated: headline stop updates are [2000] (the first possible plateau is2000), with full-state scores near or below the suffix reference rather than successful map computation. At p=61, selected-checkpoint test1x means are E97 0.026957% and GDN2 0.021362%, against the 0.026874% suffix reference. One-step held-out full-state scores are E97/GDN2/identity 0.195312%, 0.390625%, 0.000000%. These arithmetic controls and the extrapolation panels must constrain any interpretation: failure to learn the one-step arithmetic blocks a depth-specific conclusion. This frozen128-example fine-tune tests learning efficiency of this released pair under one fixed recipe, not universal representability, impossibility, or CMA causality. The single-seed identity contrast is exploratory and not a multi-seed causal result; initialization and optimization confounds remain. Update0 selections, not learned checkpoints: ['headline_e97_seed2', 'headline_e97_identity_seed0']. Every finite-state map still admits a sufficiently large one-hot linear-state representation; these failures are not an expressivity impossibility result.

## Disclosed mid-run conditional protocol amendment

Registered **2026-10-16T13:31:24.243791+00:00** after headline validation through750/1000 updates had already been observed. This was registered before any headline stop/final-test evaluation, **not pre-launch**. `contingency_amendment.json` preserves exact per-arm observation counters. No stopping, selection, recipe, metric, or data changes followed.

Trigger iff BOTH architectures' three-seed mean best-val full-state accuracy <= **0.127%** (operator's binding rounded floor+0.1pp threshold). Trigger state: **True**. If triggered, released-pair p13/p31 calibrations disambiguating state-space size from task-family learning failure are a **separately authorized follow-up task**, NOT launched here. Fresh CPU13/31/61 arithmetic/data/suffix checks plus completed p61 one-step diagnostics across all three architectures satisfy current calibration scope by explicit supervisor ruling.

## Receipts and reproducibility

- `contract.json`: committed/pushed preregistration and explicit live amendments; configs bind seeds, optimizer, source checkpoint/args and runner hashes.
- Per run: `frozen_pools.npz`, data hashes, `training.jsonl`, `validation.json`, selected/frozen checkpoints and SHA256 ledger, best-test and matched-test counts, full summary. Binary model files are local ignored artifacts, NOT in git; hashes/absolute paths reconstruct and identify them.
- `receipts/`: exact commands, GPU UUID/index snapshots, per-arm lease/console/exit records; local lambda01, no Slurm and no WorksGood service. `attempt1_alignment/` is preserved, never mislabeled as complete.
- CPU protocol/task/facade suite:71 passed (including54 protocol/task tests); fresh suffix verification passes. Final commands and clean-index proof accompany acceptance. DiLoCo resilience is unchanged: requirement matrix IDs not applicable.
- `receipts/final_audit.json`: all10 exit codes zero; selected and matched checkpoint binaries rehashed; observation counts, paired identity data, best/matched agreement, and selection-before-test chronology verified. All GPU leases released and all eight GPUs idle.
- Independent acceptance review remains required. Three-seed intervals are limited; fixed small training data and a single recipe cannot establish general task learnability. Any incomplete/failure arm is explicit above.

## Validation commands

```text
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m pytest -q tests/test_invertible_stage2.py tests/test_invertible_quadratic.py tests/test_e97_facade.py  # 71 passed
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.verify_invertible_quadratic --output experiments/expressivity_tasks/results_invertible_stage2/suffix_calibration.json  # all checks passed
bash experiments/expressivity_tasks/launch_invertible_stage2.sh calibration  # three completed diagnostics, exit0 after documented restart
bash experiments/expressivity_tasks/launch_invertible_stage2.sh headline  # seven completed headline runs, exit0
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.aggregate_invertible_stage2  # complete, receipt audit passed
CUDA_VISIBLE_DEVICES='' /usr/bin/python3 -m experiments.expressivity_tasks.results_invertible_stage2.receipts.audit_command  # ten-arm artifact audit passed
bash -n experiments/expressivity_tasks/launch_invertible_stage2.sh; compileall; cmp AGENTS.md CLAUDE.md; git diff --check  # passed
```
