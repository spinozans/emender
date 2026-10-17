# X5b — open-horizon growth curriculum: launch report

**LAUNCHED: six independent detached runs; completion watcher ARMED.**
Snapshot: 2026-10-17T14:51:13.659907+00:00, `lambda01`, from base `ad40bcebb8c0e70be898333e53ef0d4664854217`.
Registered implementation commit: `956045a8`. No completed growth-curve comparison or architecture superiority claim is made here.

## Live launch receipts and measured rates

| Arm | Seed | Exclusive GPU | Detached worker / trainer PID | Observed T=4 updates | Measured wall updates/s |
|---|---:|---:|---|---:|---:|
| e97 | 0 | 0 | 2067181 / 2067556 | 7,000 | 52.79 |
| gdn2 | 0 | 2 | 2067183 / 2067590 | 3,477 | 26.33 |
| e97 | 1 | 1 | 2067185 / 2067575 | 6,424 | 48.06 |
| gdn2 | 1 | 5 | 2067188 / 2067639 | 3,681 | 27.83 |
| e97 | 2 | 3 | 2067193 / 2067607 | 6,000 | 44.88 |
| gdn2 | 2 | 4 | 2067198 / 2067622 | 3,850 | 28.97 |

Rates use consecutive training receipts from update 100 to each listed update, including intervening validation/checkpoint overhead; startup/JIT before update 100 is excluded. These are **live operational rates**, not learning rates or growth-to-threshold rates. All six workers have PPID=1 and SID=their PID; each owns its own heartbeat lease and releases independently on exit. Six distinct leases and both process start identities were verified. No shell is held waiting for training completion.

Evidence root: `experiments/expressivity_tasks/results_invertible_stage5b_growth_8m` (abbreviated **G/**). `G/registration.json` and the 138-test prelaunch receipt were committed **before any calibration or curriculum launch**. `G/receipts/full_launch/` binds executed source snapshots, source commit, packages, GPU/disk snapshots and launch time. Per-run `G/receipts/NAME/` contains the exact command, worker/trainer PID + `/proc` start identity, and exclusive lease. `G/receipts/launch_verification.json` verifies source hashes against registration/configs, finite consecutive update streams, all six entry-checkpoint SHA-256 hashes, data-pool hashes, recipes/counts, detachment and live leases. `G/receipts/launch_snapshot/NAME/` freezes configs, validation, transitions, checkpoint metadata, pool arrays and sampled training-rate receipts; these are **launch snapshots**, not final results. The live output directories continue changing and are intentionally not committed mid-write.

## Registered curriculum and calibrated stop

- Scratch E97-MLP **9,900,800** parameters; real FLA GDN2 **9,887,056**, `allow_neg_eigval=True`; seeds 0/1/2 each. Exact X3 task-sized vocab 3721, batch 32, matched LR 1e-3, ScheduleFree AdamW betas (.9,.95), wd .01, clip 1, no warmup, bf16 autocast/fp32 parameters.
- Stages **4 → 8 → 16 → 32 → 64 → 128**. Fresh procedural data at the current depth; same weights, optimizer and training RNG across transitions. Frozen X3 validation pools, no test selection. **≥90% full-state validation accuracy advances**, on the 500-update observation grid.
- Supervisor-bound stagnation: nonoverlapping **2,000-update** windows. Compare cumulative best-validation accuracy at successive window endpoints and current-window mean training CE against the preceding window. A joint plateau requires **best-val gain <0.001 absolute (0.1pp)** AND **relative mean-loss decrease <0.001 (0.1%)**. Stop only after **two consecutive joint comparisons**; earliest eligibility **6,000 updates**. Either signal reaching its minimum resets the streak; worsening/negative improvement counts plateau. Detector resets at each new stage. Threshold crossing takes priority.
- **500,000 updates/stage** and **2,000,000 overall** are disclosed infinite-loop **safety ceilings**, labeled `stage_safety_ceiling` / `global_safety_cap`, never stagnation or an intrinsic capability finding. No 10K budget-exhaustion stop remains.
- Current and every lower registered depth are evaluated at entry, every 500 updates and every exit. `transitions.jsonl`, `stages.json`, and `growth_curve.json` receipt each stage's steps-to-threshold or stop step/accuracy, retention panel, cumulative updates, deepest threshold crossing and checkpoint. `training.jsonl` plus `validation.json` retain the loss trajectory, marginal deltas and explicit 2K-window calibration receipts. Final `result.json` supplies the run stop reason. Growth reports wait for terminal receipts, not a budgeted early summary.

**References:** uniform marginal `ln(3721)=8.221747728` nats; suffix-only full-state floor `1/3721=0.0268744961%`. For a proper driver suffix omitting at least two independent uniform drivers, this is an information bound, **not a model-capacity bound or inferred learning mechanism**. Both references accompany validation and stage-growth receipts.

## Checkpoints and capacity

Every-1,000-stage-update rolling checkpoint, best-validation chain per stage, and every stage-entry/exit checkpoint are armed. Payloads atomically save ScheduleFree eval-x weights, the full optimizer, and NumPy/Torch/CUDA RNG; checkpoint paths and SHA-256s are receipted. Existing X5 capacity pruning below **400GiB free** preserves all boundaries, each stage best, latest checkpoint and every10K checkpoint, auditing deletions. Latest checkpoint protection persists across evaluation-only boundaries. Launch had ~699GiB free, ~697GiB after startup; payloads remain local and Git-ignored. No automatic restart or scientific conclusion on crash is implied.

## Depth-calibrated cap projections — estimates, not walltime guarantees

`G/receipts/throughput/throughput.json` records a separately registered operational-only calibration, one exclusively leased GPU released before the six-run launch: each architecture, all six depths, 10 warmup + 30 timed updates, matched recipe, all-lower-depth evaluation time and checkpoint-save/hash time. No accuracy or growth outcome was retained from calibration. Isolated train-only T4/T128 rates: **E97 61.08 / 43.95 updates/s**, **GDN2 30.34 / 29.63 updates/s**.

Projections charge all-lower retention evaluation **and a checkpoint every500** (conservative versus rolling-only every1000), choose the slowest measured depth, and inflate by isolated-T4 / slowest-live-T4 slowdown if >1. Charging the full global cap at the slowest depth is conservative against the per-stage constraints, conditional on these measured costs.

| Arm | Parallel slowdown factor | 500K-stage projected maximum | 2M-global projected maximum |
|---|---:|---:|---:|
| E97 | 1.361× | 8.36 h | **33.44 h (1.39 d)** |
| GDN2 | 1.152× | 5.80 h | **23.21 h (0.97 d)** |

At the slowest live T4 rate alone, 500K/2M imply E97 3.09/12.38 h and GDN2 5.27/21.10 h. Six runs overlap, so do not sum their wall times. The depth-calibrated six-run horizon is about **33.44 hours**, subject to future CPU/GPU/disk contention, checkpoint best-save frequency and machine availability. These are planning projections, **not guaranteed bounds or projected steps-to-learning**. Training may run for days; no discretionary walltime kill is scheduled.

## Completion chain — supervisor wake armed

Supervisor confirmed its supervised watcher running (wrapper PID **2068744**):

```bash
cd /home/erikg/emender
.venv/bin/python scripts/watch_invertible_stage5b_growth.py
```

It polls every **900 seconds**, prints one-line remaining-run status and appends to `G/receipts/completion_watcher.log`. Each worker atomically publishes `G/receipts/NAME/terminal.json` and one terminal marker. The six success-marker paths are:

```
G/markers/e97_seed0.DONE
G/markers/gdn2_seed0.DONE
G/markers/e97_seed1.DONE
G/markers/gdn2_seed1.DONE
G/markers/e97_seed2.DONE
G/markers/gdn2_seed2.DONE
```

Crash markers use the same names with **`.FAIL`**. Exit 0 requires all six DONE receipts and result files; any FAIL or dead/reused worker/trainer PID without a marker exits nonzero at the next poll (2s terminal-publication grace). A host reboot also ends the supervised process; human re-arming is then required. **Runs → terminal markers → supervised watcher exit → native supervisor-session wake → terminal growth-curve report.** The supervisor explicitly confirmed this chain is live; no detached process attempts to call session-only tools. `--once` live checks returned 2 (healthy/pending), not a completed-curriculum claim.

## Validation and scope

**138 tests passed**, including 24 X5b tests: threshold inclusivity/priority, no 10K death, calibrated double-window eligibility, either-signal reset, worsening-loss treatment, stage reset, optimizer continuity and true stage lengths, retention panels, safety/stagnation/completion terminal paths, growth receipts, watcher DONE/FAIL/dead PID/PID-reuse checks. Existing X2/X3/X5/instrument/aggregation tests pass unchanged. Launcher `bash -n`, Python compile checks and `git diff --check` pass. Entry payload hashes and frozen-pool payloads were verified locally; later checkpoint payload audit requires the local Git-ignored files.

Reviewed authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` (ADR-003) and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`. These are local single-GPU task probes, **not resilient DiLoCo or production E97 training**: R16 evidence discipline only, no R/NDP/V21S/ISP systems gate, Frontier submission or Slurm Partition/QOS claim. Historical X5/X5D code and results remain unchanged. Independent reviewer gate is supervisor-owned; no training result is asserted accepted by this launch report.

**Next:** supervisor watcher handles completion/failure; inspect terminal receipts and aggregate the six stage-growth curves with retention and both references. Claim architectural separation only if those completed growth curves earn it.
