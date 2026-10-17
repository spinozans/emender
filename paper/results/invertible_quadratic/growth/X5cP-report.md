# X5cP — frozen-T4 positional evaluation

## Finding P1: composition-depth reading CONFIRMED, with exact-count qualifications

Using the **pre-existing best-validation checkpoints**, high-accuracy full-state tracking is confined to driver1 for all E97 seeds, drivers1–3 for GDN2 seeds0/1, and drivers1–2 for GDN2 seed2. Here “solves” means at least the registered 90% accuracy level, not perfection. E97 seed2 gets **511/512**, not512, at driver1; GDN2 seed1 gets **508/512**, not512, at driver3. The reading is therefore confirmed as an empirical high-accuracy-depth description, **not** an exact binary step-function or an architecture capacity/mechanism claim.

Later positions are not uniformly zero: E97 seed0 has joint counts10/512 at driver2 and2/512 at driver3, with **338/512 x-correct** at driver2; GDN2 seed2 has4/512 joint and105/512 x-correct at driver3. These partial-coordinate results must not be erased by the depth shorthand. E97 seeds1/2 each have **1/512 terminal correct at best**, versus **0/512 at final**. No model solves driver4; isolated correct predictions do not establish terminal tracking. The all-zero-terminal statement applies **only to recorded final evaluations**.

## Frozen protocol and provenance

Root `C = experiments/expressivity_tasks/results_invertible_stage5c_growth_8m`.
Each run's evidence is `C/receipts/NAME/positional_receipts.json`. The evaluator is `experiments/expressivity_tasks/eval_invertible_stage5c_positions.py`.

- Exactly **one evaluation pass per best checkpoint**, six sequential checkpoints on one exclusively leased GPU, 16 batches of32 sequences each:3072 evaluated sequences and12288 scored states total. No training, re-selection, extra checkpoint search, or final-checkpoint forward pass.
- **T4 means four drivers**, not a Tesla T4 GPU. Actual device: lambda01 GPU0, NVIDIA RTX6000 Ada; Python3.12.3, Torch2.9.1+cu128, CUDA12.8. CPU-only audits and tests did not re-evaluate models.
- Same frozen validation NPZ and all three T4 array hashes verified against each run's original `data_receipt.json`; paired arms have identical NPZ and T4-array hashes for each seed. No data regenerated. Each sequence is six tokens: two initial-coordinate prefix tokens (unscored), then drivers1–4 at zero-based token indices2–5.
- Strict model-state load of saved **ScheduleFree eval x** weights; no optimizer transformation. `model.eval()`, `torch.no_grad()`, deterministic algorithms enabled, cuDNN benchmarking off, TF32 matmul off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; fixed seed, batch32, bf16 autocast/fp32 weights, identical causal right-padding to16 and slicing as training evaluation. No stochastic sampling.
- Joint argmax decoded as `x=token//61`, `y=token%61`; invalid predictions excluded from coordinate counts. Loss is full-state cross-entropy in nats, not separate coordinate losses. Out-of-range counts are all zero (task head vocab3721).
- Existing selection convention is maximum **aggregate** validation joint accuracy, then minimum loss on ties. The selected checkpoints were read from `result.json:stages[0].best`; **not** selected anew by position or by lowest loss. Original best/final validation rows, checkpoint metadata and payload SHA-256s were verified. Receipts embed both recorded metric sets and source hashes, including the exact executed evaluator. Runtime `source_commit` is the pre-pass HEAD `7f252b4b`; the new evaluator's content hash, not a claim that it was already in that commit, binds its source.

## Cross-run positional table

**Every row has count512.** J = joint correct; X/Y = coordinates of that same joint argmax; OOR = out-of-range. Loss values rounded for display; JSON retains full precision.

| Run | Driver | J /512 | X /512 | Y /512 | Mean loss (nats) | OOR |
|---|---:|---:|---:|---:|---:|---:|
| e97_seed0 | 1 | 512 | 512 | 512 | 0.000328074328 | 0 |
| e97_seed0 | 2 | 10 | 338 | 13 | 5.9019564 | 0 |
| e97_seed0 | 3 | 2 | 7 | 9 | 8.22510715 | 0 |
| e97_seed0 | 4 | 0 | 9 | 5 | 8.22432199 | 0 |
| e97_seed1 | 1 | 512 | 512 | 512 | 0.00175897031 | 0 |
| e97_seed1 | 2 | 0 | 8 | 9 | 8.22041816 | 0 |
| e97_seed1 | 3 | 0 | 8 | 10 | 8.22267422 | 0 |
| e97_seed1 | 4 | 1 | 5 | 6 | 8.22081955 | 0 |
| e97_seed2 | 1 | 511 | 511 | 511 | 0.0172856473 | 0 |
| e97_seed2 | 2 | 1 | 21 | 6 | 8.21725543 | 0 |
| e97_seed2 | 3 | 0 | 7 | 11 | 8.22081682 | 0 |
| e97_seed2 | 4 | 1 | 11 | 11 | 8.22174557 | 0 |
| gdn2_seed0 | 1 | 512 | 512 | 512 | 3.5696348e-06 | 0 |
| gdn2_seed0 | 2 | 512 | 512 | 512 | 1.0325842e-05 | 0 |
| gdn2_seed0 | 3 | 512 | 512 | 512 | 0.0124918098 | 0 |
| gdn2_seed0 | 4 | 0 | 9 | 6 | 8.22362626 | 0 |
| gdn2_seed1 | 1 | 512 | 512 | 512 | 8.20676268e-06 | 0 |
| gdn2_seed1 | 2 | 512 | 512 | 512 | 3.10297739e-05 | 0 |
| gdn2_seed1 | 3 | 508 | 512 | 508 | 0.0464451221 | 0 |
| gdn2_seed1 | 4 | 0 | 8 | 13 | 8.20892946 | 0 |
| gdn2_seed2 | 1 | 512 | 512 | 512 | 0.000705920311 | 0 |
| gdn2_seed2 | 2 | 512 | 512 | 512 | 0.00190872537 | 0 |
| gdn2_seed2 | 3 | 4 | 105 | 6 | 6.90898457 | 0 |
| gdn2_seed2 | 4 | 0 | 7 | 9 | 8.22628739 | 0 |

## Aggregate reconciliation and best-versus-final convention

Position sums equal **every aggregate integer count** from the recorded best evaluations for all six runs; canonical aggregate loss also matches **exactly (delta0.0)**. Driver4 equals each best terminal count. The position-mean versus canonical reduction discrepancy is only floating-point summation order (less than1e-6 nats). No mismatches were patched.

Below, each `best → final` pair reports the newly reproduced best aggregate versus the **recorded, not re-forwarded**, final aggregate. Denominators: joint/X/Y2048; terminal512. Different checkpoints need not yield equal counts: all differences are explicitly retained in each receipt's `best_minus_final_counts` and `best_minus_final_loss`. GDN2 seeds0/1 selected their final checkpoints, hence identical metrics. Other runs differ. Original final metrics were checked against the last validation row and the terminal receipt's result hash.

| Run | Best → final update | Joint best → final | X best → final | Y best → final | Terminal best → final | Loss best → final |
|---|---:|---:|---:|---:|---:|---:|
| e97_seed0 | 132000 → 190000 | 524 → 515 | 866 → 627 | 539 → 536 | 0 → 0 | 5.587928444 → 5.641993910 |
| e97_seed1 | 102500 → 104000 | 513 → 512 | 533 → 537 | 537 → 543 | 1 → 0 | 6.166417599 → 6.166159302 |
| e97_seed2 | 102000 → 104000 | 513 → 511 | 550 → 550 | 539 → 539 | 1 → 0 | 6.169275820 → 6.168220133 |
| gdn2_seed0 | 324000 → 324000 | 1536 → 1536 | 1545 → 1545 | 1542 → 1542 | 0 → 0 | 2.059032992 → 2.059032992 |
| gdn2_seed1 | 372000 → 372000 | 1532 → 1532 | 1544 → 1544 | 1545 → 1545 | 0 → 0 | 2.063853428 → 2.063853428 |
| gdn2_seed2 | 170000 → 204000 | 1028 → 1025 | 1136 → 1100 | 1039 → 1030 | 0 → 0 | 3.784471691 → 3.624307066 |

The composition-depth claim above uses **best-validation checkpoints only**. Final aggregates alone cannot prove which nonterminal positions were solved, so no final per-position claim is inferred. In particular, aggregate25%/50%/75% is not itself the positional evidence; the new table is.

## Checkpoint SHA-256 receipts

Both best and final payload hashes were recomputed, even where no final forward was requested. Payloads remain local/Git-ignored; receipts and evaluator are committed.

| Run | Best SHA-256 | Final SHA-256 |
|---|---|---|
| e97_seed0 | `04b9f5092230d3d54600771e784d5fb763d3b9dfe28f6029eaaea95556b0cdd3` | `2c635610da8f5c84512702256d4175bb35a8d61e80d6d762a4ff44693cf39a73` |
| e97_seed1 | `36036cb91df847adda4eb6e86d4e516fea3ced21811aca97216a0f4b96f71deb` | `17646e1d97b9a48ebca2a1ebb51a8b287c4d97935dd0a11853df123dbedaf05f` |
| e97_seed2 | `cb72cdc9317012ecec7e0f501edf3cc4c82b80f7bbfb6f6bc47e07d6260f58c0` | `ac3a261b3d12642958c2df22ca17ca15d66e82361f413ab5b5e954169ee47531` |
| gdn2_seed0 | `71c16a8dd906224ec2bc88656370136e91a076de9c2150ccc8fd6491d013afc1` | `71c16a8dd906224ec2bc88656370136e91a076de9c2150ccc8fd6491d013afc1` |
| gdn2_seed1 | `23a6550cb4c5254bafc3a91544ae41c2b7849353dbeb968abce9f256f4262c72` | `23a6550cb4c5254bafc3a91544ae41c2b7849353dbeb968abce9f256f4262c72` |
| gdn2_seed2 | `e0aa61284a8a3da0864dc7679e06e93cd872760d1c3e16f6a222195e7be6f4ac` | `9e7f0151fef9fa055506c88e12ea34a5a33770d07092353f488eb1ad5772ae62` |

Frozen-validation NPZ SHA-256 (same for E97 and GDN2 at each seed; individual T4 input/target/mask hashes are in every positional receipt):

| Seed | SHA-256 |
|---|---|
| 0 | `c7a31d5a206fc3bd144509ef1004a50bf13bb7b15e4c8f2fa89986557e6180b4` |
| 1 | `5e051705e045d7fff0acbfa1a953ef8ba4e7160ecc8d63b9175c4b8fa03f3c40` |
| 2 | `7d0ca54560fc9bee3ee03524ce7e7b09bea651b88e7b0910b8ff048027a80705` |

## Validation / reproducibility

Single GPU invocation (already completed; **do not rerun just to audit receipts**):

```bash
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
.venv/bin/python -m experiments.expressivity_tasks.eval_invertible_stage5c_positions
```

CPU validation:

```bash
.venv/bin/python -m pytest -q   tests/test_eval_invertible_stage5c_positions.py   tests/test_invertible_stage3_8m.py   tests/test_invertible_stage5_curriculum_8m.py   tests/test_invertible_stage5b_growth_8m.py   tests/test_replay_stagnation_rule.py   tests/test_invertible_quadratic.py
```

**105 passed** (four new positional/accounting/receipt tests,101 existing). GPU output: `C/receipts/positional/eval_console.txt`; final test output: `tests.txt`; independent CPU hash/terminal audit: `audit.txt`, all in that same positional evidence directory. Initial CPU suite had104 passed/1 failed because the new audit test expected a relative evaluator-source key, whereas Python `__file__` supplied an absolute key; fixed the test to accept either source spelling and reran CPU tests only. `tests_initial.txt` retains that failure; no GPU rerun, receipt edits, or evaluation discrepancies occurred. `git diff --check` and Python compile checks passed.

Limitations: this is a post-training validation diagnostic, not an independent test pool or a proof of exact computational depth. Loss tie-breaking and best-versus-final differences remain explicit; no superiority/capacity bound or suffix-only mechanism is claimed. No resilient-training behavior or Slurm queue binding is involved. Existing unrelated untracked X5b/X5c runtime artifacts were left alone. Independent reviewer acceptance remains supervisor-owned.
