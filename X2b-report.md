# X2b — fresh procedural training, frozen evaluation

## Status and preregistered instrument

Status: **complete**. Exact suffix-only floor throughout: **0.026874%** (1/3721).
No absent result is imputed. All accuracies below are percentages, full-vocabulary argmax exact joint class `61*x+y`; both coordinates must match. Invalid classes fail both coordinates.

The live supervisor amended the reviewed scratch dim1024/depth2 plan: use the released 1.2866B E97 and matched 1.3B GDN2-MLP loaders/geometry with Schedule-Free train/y recovery. Scratch probes are NOT run (available only for a future separately approved mechanism study). Recipe: full fine-tune bf16 AdamW (0.9,0.95), lr5e-5, warmup300, wd0.01, clip0.5, batch32. E97/GDN2 seeds0/1/2; identity seed0; one-step diagnostics all three seed0. Fresh procedural training batch32; unchanged X2 frozen512 val/512 test per depth. Archived128 training pool is not sampled. Production drivers128/512/1024 plus two init tokens.

Stopping is symmetric: validation every250 completed updates, trailing train-loss improvement = (mean loss over the first 250 updates of the 2,000-step window minus mean loss over the final 250 updates) / first-window mean; plateau iff that ratio < 1% AND best-val gain < 0.001 (0.1pp) at step >= 10,000, hard cap60000. Best validation accuracy, then lower CE, then earlier update selects ALL arms. Eval0 anchors the first window. Selection identities/hashes are frozen before test/extrapolation evaluation. Full-state running, coordinates, terminal, post-suffix7 burn-in counts/losses are retained in JSON. No targets ever enter model inputs.

Implementation fact: the existing fused E97 kernel accepts unaligned lengths only in forward-only mode. In the preserved X2 run, the first calibration attempt failed at update1 for both E97 variants; GDN2 was explicitly stopped to restart consistently. X2's archived `attempt1_alignment/` retains all failures, traces and partial logs. Supervisor approved identical constant-zero right padding to multiple16 on every arm and split (3->16,130->144). Pad logits are sliced away before loss/scoring; pads occur strictly after real causal prefixes, so they cannot contaminate real positions or leak targets. Permanent fixed-weight forward/loss and training masked-pad gradient tests pass. Only the E97 carried-state tanh is replaced by identity; all parameter keys/values/geometry remain identical at initialization.

## Per-seed selected results

| Run | Best val | Selected update | Stop update/reason | Test1x | Test4x | Test8x | Suffix floor |
|---|---:|---:|---|---:|---:|---:|---:|
| headline_e97_seed0 | 0.285339% | 10000 | 10000/plateau | 0.292969% | 0.087357% | 0.061226% | 0.026874% |
| headline_e97_seed1 | 0.312805% | 10000 | 10000/plateau | 0.277710% | 0.083160% | 0.055313% | 0.026874% |
| headline_e97_seed2 | 0.294495% | 9500 | 10000/plateau | 0.289917% | 0.086594% | 0.059128% | 0.026874% |
| headline_gdn2_seed0 | 1.664734% | 21000 | 21500/plateau | 1.647949% | 0.421906% | 0.230408% | 0.026874% |
| headline_gdn2_seed1 | 1.762390% | 23000 | 23750/plateau | 1.788330% | 0.456619% | 0.244904% | 0.026874% |
| headline_gdn2_seed2 | 1.803589% | 22500 | 23500/plateau | 1.751709% | 0.473022% | 0.250816% | 0.026874% |
| headline_e97_identity_seed0 | 0.273132% | 10000 | 10000/plateau | 0.270081% | 0.088501% | 0.061226% | 0.026874% |
| calibration_e97_seed0 | 88.281250% | 24250 | 28000/plateau | 85.937500% | not applicable | not applicable | 0.026874% |
| calibration_gdn2_seed0 | 94.726562% | 18000 | 20250/plateau | 90.234375% | not applicable | not applicable | 0.026874% |
| calibration_e97_identity_seed0 | 85.742188% | 24750 | 27000/plateau | 85.937500% | not applicable | not applicable | 0.026874% |

## All-seed aggregate (mean ± sample SD; SEM in JSON)

- e97: best val 0.297546% ± 0.013985%; test_1x 0.286865% ± 0.008074%, test_4x 0.085704% ± 0.002235%, test_8x 0.058556% ± 0.002998%; floor 0.026874%.
- gdn2: best val 1.743571% ± 0.071315%; test_1x 1.729329% ± 0.072817%, test_4x 0.450516% ± 0.026099%, test_8x 0.242043% ± 0.010501%; floor 0.026874%.

Per-seed paired E97−GDN2 differences (accuracy fractions) and all counts are in `results_summary.json`. A best seed is never substituted for the three-seed mean. Seed uncertainty is not a token-binomial interval.

## Coordinates, arithmetic calibration, and cost

| Run | Test1x x | Test1x y | Test1x terminal | Parameters | Updates | Real training tokens | Tokens/s training | Wall seconds | Suffix floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| headline_e97_seed0 | 2.510071% | 1.947021% | 0.000000% | 1286589072 | 10000 | 41600000 | 7605.7 | 5979.6 | 0.026874% |
| headline_e97_seed1 | 2.394104% | 1.847839% | 0.000000% | 1286589072 | 10000 | 41600000 | 8312.8 | 5493.9 | 0.026874% |
| headline_e97_seed2 | 2.508545% | 1.911926% | 0.000000% | 1286589072 | 10000 | 41600000 | 7807.3 | 5831.0 | 0.026874% |
| headline_gdn2_seed0 | 3.648376% | 3.231812% | 0.000000% | 1286713448 | 21500 | 89440000 | 7339.8 | 13248.3 | 0.026874% |
| headline_gdn2_seed1 | 3.575134% | 3.396606% | 0.000000% | 1286713448 | 23750 | 98800000 | 7479.0 | 14364.6 | 0.026874% |
| headline_gdn2_seed2 | 3.700256% | 3.350830% | 0.000000% | 1286713448 | 23500 | 97760000 | 7795.8 | 13651.3 | 0.026874% |
| headline_e97_identity_seed0 | 2.511597% | 1.921082% | 0.000000% | 1286589072 | 10000 | 41600000 | 7750.3 | 5878.6 | 0.026874% |
| calibration_e97_seed0 | 100.000000% | 85.937500% | 85.937500% | 1286589072 | 28000 | 2688000 | 751.7 | 4504.9 | 0.026874% |
| calibration_gdn2_seed0 | 100.000000% | 90.234375% | 90.234375% | 1286713448 | 20250 | 1944000 | 582.0 | 4011.1 | 0.026874% |
| calibration_e97_identity_seed0 | 100.000000% | 85.937500% | 85.937500% | 1286589072 | 27000 | 2592000 | 752.1 | 4336.9 | 0.026874% |

- e97: 11 matrix-state layers, 2,433,024 recurrent matrix elements per sequence (short-conv caches excluded); per-layer shapes/flags in config.
- gdn2: 12 matrix-state layers, 5,898,240 recurrent matrix elements per sequence (short-conv caches excluded); per-layer shapes/flags in config.
- e97_identity: 11 matrix-state layers, 2,433,024 recurrent matrix elements per sequence (short-conv caches excluded); per-layer shapes/flags in config.

Real training tokens include the two initialization positions but exclude right-padding overhead. Physical padded tokens/sequence and matrix-state element counts per layer are in each config. Released heads are retained (full vocab), not replaced by the scratch plan's 3721-class heads. Released GDN2 configuration governs even where it differs from the old scratch control's eigenvalue setting; actual flags are recorded in headline configs.

## Matched-step panels

Selected-checkpoint panels above are not equal-step comparisons. Below: every retained mandatory fixed step and the union of actual headline stopping steps. Arms are NEVER trained past their plateau to fill a later panel. Any fixed panel beyond an arm's stop (4000/8000/12000/16000/18269 or the10K grid) is not reached, not zero accuracy.

| Run | Update | Test1x | Test4x | Test8x | Suffix floor |
|---|---:|---:|---:|---:|---:|
| headline_e97_seed0 | 1000 | 0.047302% | 0.026703% | 0.029564% | 0.026874% |
| headline_e97_seed0 | 4000 | 0.051880% | 0.029373% | 0.030136% | 0.026874% |
| headline_e97_seed0 | 8000 | 0.205994% | 0.067520% | 0.051117% | 0.026874% |
| headline_e97_seed0 | 10000 | 0.292969% | 0.087357% | 0.061226% | 0.026874% |
| headline_e97_seed0 | 21500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed0 | 23500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed0 | 23750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed1 | 1000 | 0.038147% | 0.031662% | 0.027657% | 0.026874% |
| headline_e97_seed1 | 4000 | 0.050354% | 0.028992% | 0.027275% | 0.026874% |
| headline_e97_seed1 | 8000 | 0.224304% | 0.068283% | 0.049210% | 0.026874% |
| headline_e97_seed1 | 10000 | 0.277710% | 0.083160% | 0.055313% | 0.026874% |
| headline_e97_seed1 | 21500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed1 | 23500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed1 | 23750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed2 | 1000 | 0.036621% | 0.026703% | 0.028610% | 0.026874% |
| headline_e97_seed2 | 4000 | 0.050354% | 0.029373% | 0.028992% | 0.026874% |
| headline_e97_seed2 | 8000 | 0.230408% | 0.071335% | 0.048256% | 0.026874% |
| headline_e97_seed2 | 10000 | 0.283813% | 0.091553% | 0.062561% | 0.026874% |
| headline_e97_seed2 | 21500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed2 | 23500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_seed2 | 23750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_gdn2_seed0 | 1000 | 0.033569% | 0.025558% | 0.024986% | 0.026874% |
| headline_gdn2_seed0 | 4000 | 0.038147% | 0.030899% | 0.027466% | 0.026874% |
| headline_gdn2_seed0 | 8000 | 0.268555% | 0.093079% | 0.063324% | 0.026874% |
| headline_gdn2_seed0 | 10000 | 0.532532% | 0.157928% | 0.086594% | 0.026874% |
| headline_gdn2_seed0 | 12000 | 0.755310% | 0.196457% | 0.115776% | 0.026874% |
| headline_gdn2_seed0 | 16000 | 1.155090% | 0.313187% | 0.166702% | 0.026874% |
| headline_gdn2_seed0 | 18269 | 1.377869% | 0.374985% | 0.199890% | 0.026874% |
| headline_gdn2_seed0 | 20000 | 1.533508% | 0.405121% | 0.216866% | 0.026874% |
| headline_gdn2_seed0 | 21500 | 1.651001% | 0.434875% | 0.223732% | 0.026874% |
| headline_gdn2_seed0 | 23500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_gdn2_seed0 | 23750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_gdn2_seed1 | 1000 | 0.050354% | 0.030518% | 0.025558% | 0.026874% |
| headline_gdn2_seed1 | 4000 | 0.036621% | 0.027466% | 0.029373% | 0.026874% |
| headline_gdn2_seed1 | 8000 | 0.352478% | 0.106430% | 0.061798% | 0.026874% |
| headline_gdn2_seed1 | 10000 | 0.569153% | 0.153732% | 0.093651% | 0.026874% |
| headline_gdn2_seed1 | 12000 | 0.758362% | 0.198746% | 0.121689% | 0.026874% |
| headline_gdn2_seed1 | 16000 | 1.152039% | 0.293350% | 0.171852% | 0.026874% |
| headline_gdn2_seed1 | 18269 | 1.347351% | 0.370789% | 0.191307% | 0.026874% |
| headline_gdn2_seed1 | 20000 | 1.548767% | 0.408554% | 0.218773% | 0.026874% |
| headline_gdn2_seed1 | 21500 | unavailable: checkpoint pruned before stop step became required | unavailable: checkpoint pruned before stop step became required | unavailable: checkpoint pruned before stop step became required | 0.026874% |
| headline_gdn2_seed1 | 23500 | 1.824951% | 0.453186% | 0.249100% | 0.026874% |
| headline_gdn2_seed1 | 23750 | 1.777649% | 0.458145% | 0.242996% | 0.026874% |
| headline_gdn2_seed2 | 1000 | 0.044250% | 0.027847% | 0.024796% | 0.026874% |
| headline_gdn2_seed2 | 4000 | 0.059509% | 0.028610% | 0.027275% | 0.026874% |
| headline_gdn2_seed2 | 8000 | 0.379944% | 0.117493% | 0.076103% | 0.026874% |
| headline_gdn2_seed2 | 10000 | 0.607300% | 0.167465% | 0.096130% | 0.026874% |
| headline_gdn2_seed2 | 12000 | 0.842285% | 0.219727% | 0.125885% | 0.026874% |
| headline_gdn2_seed2 | 16000 | 1.226807% | 0.334549% | 0.185204% | 0.026874% |
| headline_gdn2_seed2 | 18269 | 1.448059% | 0.385666% | 0.204468% | 0.026874% |
| headline_gdn2_seed2 | 20000 | 1.585388% | 0.422668% | 0.223541% | 0.026874% |
| headline_gdn2_seed2 | 21500 | 1.634216% | 0.437927% | 0.236511% | 0.026874% |
| headline_gdn2_seed2 | 23500 | 1.745605% | 0.454330% | 0.240707% | 0.026874% |
| headline_gdn2_seed2 | 23750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_identity_seed0 | 1000 | 0.045776% | 0.025558% | 0.029755% | 0.026874% |
| headline_e97_identity_seed0 | 4000 | 0.050354% | 0.028992% | 0.030327% | 0.026874% |
| headline_e97_identity_seed0 | 8000 | 0.221252% | 0.078583% | 0.049210% | 0.026874% |
| headline_e97_identity_seed0 | 10000 | 0.270081% | 0.088501% | 0.061226% | 0.026874% |
| headline_e97_identity_seed0 | 21500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_identity_seed0 | 23500 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |
| headline_e97_identity_seed0 | 23750 | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | unavailable: arm already stopped; no extra training | 0.026874% |

## Suffix calibration

Analytic suffix0/1/7 cap = 0.026874% once at least two drivers are omitted (width7 starts at driver9). The CPU verification rerun and `suffix_calibration.json` retain fresh 2,000,000-independent-trajectory measurements per prime, known initialization, seeds, exact counts, six-sigma checks, and the empirical old parity-transfer control. These are not learned-model upper bounds.

## Honest interpretation

Fresh data rescues arithmetic learning: one-step test E97/GDN2/identity = 85.937500%, 90.234375%, 85.937500%. The long-task ordering favors GDN2, not E97: selected test1x means 1.729329% versus 0.286865%; at matched10000 updates, GDN2/E97 means are 0.569661%/0.284831%. All compare with the 0.026874% suffix reference. Both architectures remain low in absolute full-state accuracy and degrade with depth; this does not establish robust invertible-map computation. Unequal stopping/selection steps are explicit, and the single-seed identity contrast is exploratory. These released-pair fixed-recipe results do not prove universal expressivity, impossibility, or CMA causality.

## Amendment #2 registration

```json
{
  "amendment": 2,
  "registered_at": "2026-10-16T13:56:49.456820+00:00",
  "base_commit": "ca79f05b195f7af9ba7a110140254eff044ea902",
  "authority": "taskX2b-freshdata.md and live supervisor window-mean/capacity rulings before launch",
  "training": "task.generate_batch(32, drivers+2, training_rng) fresh procedural per batch; seed0/1/2; archived train pool not used",
  "evaluation": "X2 frozen validation and test pools unchanged, same seeds and hashes",
  "plateau": "trailing train-loss improvement = (mean loss over the first 250 updates of the 2,000-step window minus mean loss over the final 250 updates) / first-window mean; plateau iff that ratio < 1% AND best-val gain < 0.001 (0.1pp) at step >= 10,000",
  "capacity": "If free capacity <400GiB at a checkpoint check, retain best-val chain, every10K grid, latest250 rolling, own stop, legacy fixed {1000,4000,8000,12000,16000,18269}, and all known headline stop steps; scan published frozen selections each prune. Each prune records removed steps and retained hashes. Any cross-arm panel cell whose checkpoint was already pruned before its step became required is reported as unavailable/pruned - never imputed, never re-created by interpolation, never silently dropped. Stop steps broadcast via selection_frozen.json; running arms scan each250 check.",
  "evidence_basis": {
    "X2_report_sha256": "548651e16a2d05a886b90a7fac0d14e4562e9fa4171dd6e46b17601c13bd7923",
    "X2_results_summary_sha256": "eb6a7289186edbc0218941f9952ac2ebece12b68573e5c844e2bcd5e47fb60a2",
    "observed_train_loss": "13.2->0.03 nats, frozen128 pool memorization while validation stayed floor-level",
    "precedent": "scripts/expressivity_1p3b.py make_batch fresh generate_batch"
  },
  "contingency": "X2 trigger state SUPERSEDED by this relaunch; X2 floor-level result was an artifact of the memorizable pool. p13/p31 remains separately authorized-if-triggered under corrected protocol; not launched here."
}
```

## Contingency

X2 trigger state SUPERSEDED by this relaunch; X2 floor-level result was an artifact of the memorizable pool. p13/p31 remains separately authorized-if-triggered under corrected protocol; not launched here.
Corrected-protocol trigger state: False; both three-seed mean best-val scores must be <=0.127% (floor0.026874%). No p13/p31 training launched here.

## Receipts and validation

The results root holds contracts, frozen-pool identities, all ten run logs, best/matched/cross-stop eval receipts, loss-window replay, selected checkpoint hashes, resource/command/exit receipts and final audit. Binary checkpoints are local ignored artifacts identified by SHA256, not pushed to git.
X2 receipts are unchanged. Local lambda01 execution only, exclusive scripts/gpu_lease.sh leases; no Slurm or WorksGood service. DiLoCo requirement matrix IDs are not applicable: no resilient training or scale behavior changed.
Concurrent-source disclosure: all ten X2b configs bind launched fb2c1b6e runner bytes (archived in receipts/runner_at_launch.py). HEAD advanced to X2c mid-run; no X2b runner edits after launch. The final audit verifies git-show launch bytes, not the current X2c-extended runner. Initial source-identity audit failure is retained and its correction disclosed in receipts/final_audit.json.
CPU validation output: receipts/cpu_tests.txt. Final commands and audit: receipts/validation_commands.txt and receipts/final_audit.json. Independent acceptance review remains required.

## Post-completion capacity prune

Executed 2026-10-16T19:47:56.267612+00:00 under the queued operator ruling, after all ten runs and seven cross-stop evaluations completed. No live training/evaluation process existed; no active checkpoint was touched.
Removed 237.265 GiB of checkpoint weights. Observed free space increased by 238.110 GiB to 527.655 GiB (volume accounting may differ from file-byte totals).
X2b retains the best-val chain, 10K grid, legacy fixed points, available broadcast stops, and own stop/latest. X2 completed runs retain ONLY their selected best-val and stop checkpoint. All original X2 non-checkpoint receipts/logs/manifests remain byte-identical; their historical ledgers are not assertions that pruned weights still exist.
The four weights in X2's failed/aborted alignment archive remain unchanged (those attempts have no completed best/stop contract). Full retained SHA256/byte inventories, removed steps, evidence hashes, and the interrupted-first-attempt disclosure are in receipts/postcompletion_capacity_prune.json; X2b per-run pruning.jsonl also records the action.

| Root | Run | Retained checkpoint steps | Removed GiB |
|---|---|---|---:|
| results_invertible_stage2b | calibration_e97_identity_seed0 | 0, 250, 500, 750, 1000, 1250, 2750, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 6000, 6500, 6750, 7000, 7750, 8000, 8250, 8750, 9000, 9250, 9500, 9750, 10000, 10250, 11000, 11750, 12000, 12250, 12750, 13750, 14500, 14750, 15000, 15750, 16000, 16750, 17250, 17500, 18269, 18500, 18750, 19250, 20000, 20250, 20500, 20750, 21000, 21250, 21750, 22750, 23250, 24000, 24750, 27000 | 0.000 |
| results_invertible_stage2b | calibration_e97_seed0 | 0, 250, 500, 750, 1000, 2250, 2500, 2750, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5250, 5500, 5750, 6000, 6250, 6750, 7000, 7250, 7750, 8000, 9000, 9250, 9500, 10000, 10500, 11000, 11250, 11500, 12000, 12250, 12500, 13500, 13750, 14500, 14750, 15750, 16000, 18000, 18269, 19000, 20000, 20250, 22000, 22750, 24000, 24250, 27000, 28000 | 0.000 |
| results_invertible_stage2b | calibration_gdn2_seed0 | 0, 250, 750, 1000, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4500, 4750, 5250, 5500, 5750, 6000, 6250, 6500, 7000, 7250, 8000, 8250, 8500, 8750, 9000, 9500, 10000, 10750, 11000, 11500, 12000, 12500, 13000, 13500, 14250, 15750, 16000, 17750, 18000, 18269, 20000, 20250 | 0.000 |
| results_invertible_stage2b | headline_e97_identity_seed0 | 0, 1000, 1250, 2250, 3750, 4000, 4500, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7750, 8000, 8500, 8750, 9250, 9500, 10000 | 4.793 |
| results_invertible_stage2b | headline_e97_seed0 | 0, 250, 1000, 1750, 2250, 4000, 4500, 5000, 5250, 5500, 5750, 6000, 6500, 6750, 7000, 7750, 8000, 8750, 9000, 9250, 10000 | 9.586 |
| results_invertible_stage2b | headline_e97_seed1 | 0, 500, 750, 1000, 3000, 3250, 4000, 4500, 5750, 6000, 6500, 7000, 7250, 7750, 8000, 8750, 9000, 9250, 10000 | 4.793 |
| results_invertible_stage2b | headline_e97_seed2 | 0, 250, 500, 1000, 1500, 1750, 2000, 3500, 4000, 4250, 4500, 4750, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 8000, 8500, 9000, 9500, 10000 | 7.190 |
| results_invertible_stage2b | headline_gdn2_seed0 | 0, 500, 750, 1000, 4000, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7500, 7750, 8000, 8500, 8750, 9000, 9250, 9750, 10000, 10250, 10500, 11250, 11500, 11750, 12000, 12250, 12500, 12750, 13000, 13500, 13750, 14250, 14500, 14750, 15250, 16000, 16250, 17000, 17250, 17750, 18250, 18269, 18750, 19000, 19500, 20000, 20250, 20750, 21000, 21500 | 0.000 |
| results_invertible_stage2b | headline_gdn2_seed1 | 0, 500, 750, 1000, 4000, 4500, 4750, 5500, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8500, 8750, 9000, 9500, 9750, 10000, 10250, 10500, 11000, 11250, 11500, 11750, 12000, 12250, 12750, 13000, 13250, 13500, 14000, 14250, 15500, 15750, 16000, 17250, 17750, 18000, 18250, 18269, 19000, 19750, 20000, 20250, 20500, 21250, 21750, 22250, 22500, 22750, 23000, 23500, 23750 | 0.000 |
| results_invertible_stage2b | headline_gdn2_seed2 | 0, 250, 500, 750, 1000, 1750, 4000, 5000, 5250, 5500, 5750, 6250, 6750, 7250, 7500, 7750, 8000, 8250, 8500, 9000, 9250, 9500, 10000, 10250, 10500, 10750, 11250, 11500, 11750, 12000, 12750, 13000, 13250, 13500, 13750, 14250, 14500, 14750, 15750, 16000, 16500, 16750, 17000, 18000, 18250, 18269, 19000, 19250, 19750, 20000, 20250, 20750, 21000, 21500, 22000, 22500, 23500 | 0.000 |
| results_invertible_stage2 | calibration_e97_identity_seed0 | 3500, 3750 | 33.551 |
| results_invertible_stage2 | calibration_e97_seed0 | 750, 2750 | 23.965 |
| results_invertible_stage2 | calibration_gdn2_seed0 | 3000, 3750 | 33.555 |
| results_invertible_stage2 | headline_e97_identity_seed0 | 0, 2000 | 16.776 |
| results_invertible_stage2 | headline_e97_seed0 | 1500, 2000 | 16.776 |
| results_invertible_stage2 | headline_e97_seed1 | 1000, 2000 | 16.776 |
| results_invertible_stage2 | headline_e97_seed2 | 0, 2000 | 16.776 |
| results_invertible_stage2 | headline_gdn2_seed0 | 1250, 2000 | 16.778 |
| results_invertible_stage2 | headline_gdn2_seed1 | 1000, 2000 | 16.778 |
| results_invertible_stage2 | headline_gdn2_seed2 | 2000 | 19.174 |

Headline state after pruning: all seven complete; E97 seeds0/1/2 and identity stopped10000, GDN2 seeds0/1/2 stopped21500/23750/23500. No training restarted; learned measurements are unchanged.
