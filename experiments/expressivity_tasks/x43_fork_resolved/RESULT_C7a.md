# X43 C7a — joint residual training fails the dense comparison

The preregistered improvement prediction is **refuted for both executed heads**. Every position remained in the differentiable mixture loss; neither hard position masking nor C6 support/OOD normalization was used. Joint training does not preserve useful anchors automatically: transformer neural weight collapses almost to zero on covered positions. This is a failed fixed-wall-clock recipe, not evidence that residual learning cannot work.

## Identical inference comparison

All losses are nats/byte on exactly 55,770 final targets (32,415 forks, 23,355 forced). Each frozen neural head uses the same C3 feature-gate calibration procedure on the separate 55,769 calibration targets, train-prefix count mixture, nonempty-support forced walk, and empty-support abstention. No test targets enter fitting. Dense rows reuse historical C1/C3 predictions; masked rows reuse C6 aware predictions, which also include conditioning and OOD normalization: they are a compound control, not a pure masking ablation. Boosted heads alone were trained in this session.

| Head | Training recipe | Fused total NLL | Fused fork NLL |
|---|---|---:|---:|
| emender | dense | 1.673012 | 2.424886 |
| emender | masked | 6.694321 | 10.433005 |
| emender | boosted | 3.718402 | 4.902387 |
| transformer | dense | 1.746996 | 2.558332 |
| transformer | masked | 7.312338 | 11.496303 |
| transformer | boosted | 14.647364 | 19.057854 |

The above common calibration protocol isolates the effect of the trained neural head as registered. The actual jointly trained gate is **not calibrated**; its separate deployment result uses the same probability mixture, walk and abstention rules:

| Boosted head | Joint-gate total | Joint-gate fork | Raw neural total | Raw neural fork |
|---|---:|---:|---:|---:|
| emender | 3.860648 | 5.147120 | 6.133178 | 6.267019 |
| transformer | 18.651658 | 25.947240 | 28.333415 | 28.675635 |

Both joint and common-calibration deployments lose to dense controls. Every 9,049 empty-support case and 17,066 nonempty-support zero-target fork remains scored; these are not floored or dropped. C6's index target probabilities are float32 exact-count ratios; the same arrays are used for all six common rows. Historical dense totals reproduce C3 within 1e-6.

## Gate telemetry

Values below are mean **neural** weights (joint g versus 1 minus C3 index weight) on final positions before the forced-walk override. Calibration gates were fitted on covered forks only; forced-position telemetry is extrapolation. Empty support is overridden to neural weight 1 in both protocols. Interval count is occurrence/support size; branch count is distinct supported bytes. Full strata, including boosted recalibrated gates and fork denominators, are in receipts/c7a/results.json.

| Head | Dimension | Bin [lo, hi) | N | Joint neural weight | C3 dense neural weight |
|---|---|---|---:|---:|---:|
| emender | interval_count | [0, 0.5) | 9049 | 1 | 1 |
| emender | interval_count | [0.5, 1.5) | 19396 | 0.421889 | 0.926972 |
| emender | interval_count | [1.5, 4.5) | 14357 | 0.419636 | 0.856115 |
| emender | interval_count | [4.5, 16.5) | 8707 | 0.415844 | 0.697316 |
| emender | interval_count | [16.5, 64.5) | 3265 | 0.412105 | 0.496887 |
| emender | interval_count | [64.5, 1e+10) | 996 | 0.40851 | 0.284004 |
| emender | branch_count | [0, 1) | 9049 | 1 | 1 |
| emender | branch_count | [1, 2) | 28651 | 0.422295 | 0.919807 |
| emender | branch_count | [2, 4) | 11708 | 0.417337 | 0.780635 |
| emender | branch_count | [4, 8) | 4602 | 0.410418 | 0.518374 |
| emender | branch_count | [8, 16) | 1420 | 0.402717 | 0.243068 |
| emender | branch_count | [16, 257) | 340 | 0.396619 | 0.0698241 |
| emender | context_length | [1, 2) | 5 | 0.405462 | 0.0105825 |
| emender | context_length | [2, 4) | 866 | 0.425687 | 0.389595 |
| emender | context_length | [4, 8) | 23719 | 0.462811 | 0.771386 |
| emender | context_length | [8, 16) | 29209 | 0.537014 | 0.914655 |
| emender | context_length | [16, 32) | 1913 | 0.804028 | 0.987156 |
| emender | context_length | [32, 128) | 58 | 1 | 1 |
| transformer | interval_count | [0, 0.5) | 9049 | 1 | 1 |
| transformer | interval_count | [0.5, 1.5) | 19396 | 2.09846e-06 | 0.97109 |
| transformer | interval_count | [1.5, 4.5) | 14357 | 1.73268e-06 | 0.942176 |
| transformer | interval_count | [4.5, 16.5) | 8707 | 1.34182e-06 | 0.856513 |
| transformer | interval_count | [16.5, 64.5) | 3265 | 1.13218e-06 | 0.680758 |
| transformer | interval_count | [64.5, 1e+10) | 996 | 9.05393e-07 | 0.430674 |
| transformer | branch_count | [0, 1) | 9049 | 1 | 1 |
| transformer | branch_count | [1, 2) | 28651 | 2.37529e-06 | 0.964913 |
| transformer | branch_count | [2, 4) | 11708 | 1.04114e-06 | 0.903865 |
| transformer | branch_count | [4, 8) | 4602 | 3.24699e-07 | 0.750146 |
| transformer | branch_count | [8, 16) | 1420 | 7.92889e-08 | 0.506757 |
| transformer | branch_count | [16, 257) | 340 | 2.48177e-08 | 0.210071 |
| transformer | context_length | [1, 2) | 5 | 3.05542e-07 | 0.0226658 |
| transformer | context_length | [2, 4) | 866 | 0.0103967 | 0.538831 |
| transformer | context_length | [4, 8) | 23719 | 0.0724337 | 0.880129 |
| transformer | context_length | [8, 16) | 29209 | 0.205006 | 0.96599 |
| transformer | context_length | [16, 32) | 1913 | 0.667015 | 0.99542 |
| transformer | context_length | [32, 128) | 58 | 1 | 1 |

Transformer's one-branch mean neural weight is about 2.4e-6, versus C3 dense's .965; on 16+ branches it is about 2.5e-8 versus .210. Emender stays around .40–.42 on covered positions but is much worse than its dense head. Gate behavior is a receipt, not an isolated causal ablation.

## Execution and validation

Registered before any training in commit a07a8eea (full hash in git); executable training/scoring source d2d590bf. The preregistration hash is in PROGRAM.md and training.json. GPU broker leased physical card 2, RTX 6000 Ada; no card 7 work, no Slurm/Frontier, no WorksGood service or subagents. Interpreter /home/erikg/emender/.venv/bin/python only. No checkpoints/.pt files produced or committed.

| Head | Configured trainable parameters | Steps | Tokens | Training seconds | Inference seconds |
|---|---:|---:|---:|---:|---:|
| emender | 2067144 | 229 | 468992 | 180.179879 | 1.326321 |
| transformer | 2067192 | 26380 | 54026240 | 180.003063 | 0.175010 |

Same seed 4301, batch 16, context 128, AdamW lr .0003 / decay .01, clipping 1, BF16 autocast with float32 normalized probabilities, 180-second loops including kernel and synchronization costs; imports/index/initialization/evaluation excluded. The gate adds four parameters to C6's matched backbones; unused conditioner/OOD allocations remain but get no gradients. All 29 C6 execution dependency hashes still match. Emender completed only 229 steps, compared with C6 dense's 5,668; its pronounced throughput loss is unexplained, so this is not a step-matched mechanism comparison. Transformer completed 26,380 steps versus C6 dense's 28,657, yet failed more severely. Historical dense initialization/precision and active-parameter counts are not exact matched reruns; controls were reused as directed.

Six CPU tests pass (gradient attenuation at forced anchors, nonzero fork/gate gradients, empty-support abstention, normalized mixture, and fail-closed ledger checks). validate_c7a.py independently replays 70 loss/denominator checks, prediction/index/source/dependency hashes, gate freeze, GPU/budget accounting and C3 dense reproduction; green. validate_program.py green after outcome binding. Initial pytest collection failed on a relative import; fixed. An existing mutation test assumed C6 was still registered and failed; fixed to select by status. Those failures are retained in manual_notes.json. Training runner exit 0; no nonfinite loss failures.

## Mechanism and residual risks

The premise that a dense mixture necessarily retains load-bearing anchors was a hypothesis and fails in this recipe. With fixed g, the neural log-loss gradient is multiplied by g*p_model(target)/p_fused(target); exact index coverage can attenuate **fork** gradients as well as forced gradients. The confidence-dependent gate adds another gradient path, so this factor alone is not the complete derivative. Self-containing training counts supply zero zero-target examples, whereas final test contains 17,066 zero-target forks. A gate can minimize training likelihood by trusting the index almost everywhere, leaving the neural head weak exactly where test needs it. Transformer gate collapse and low training mixture loss alongside severe held-out neural loss are consistent with that mechanism; no causal ingredient ablation was run.

Selectors and labels remain corpus-wide/transductive. Exact training counts exclude held-out occurrences, but context selection is not causal. One seed/corpus, no significance/generalization claim. GDN2 was not executed; Emender and transformer satisfy the prioritized scope. No fixed-step replay or throughput diagnosis, no prefix-only/leave-one-out exposure, no gate constraint/regularizer or independent dense auxiliary objective was tested. Do not scale this unconstrained recipe as a working solution. C6 was a compound failure, so it cannot isolate the hard-mask cause. No resilient DiLoCo behavior was changed or exercised; its requirement IDs are not applicable.
