# X5c — replay-calibrated stagnation and fresh six-run launch

**Six fresh detached runs LIVE on lambda01.** Source commit **388b1644** was committed before launch; no X5b weights or optimizer state was resumed. Launch verification snapshot: **2026-10-17 15:09:12 UTC**. Evidence root: `experiments/expressivity_tasks/results_invertible_stage5c_growth_8m` (**C/**). This is a launch report, not a terminal growth comparison or superiority claim.

## Artifact #4: rejected rule kills a known-successful curve

The permanent suite gate is **`tests/test_replay_stagnation_rule.py`**. Every rule in `CANDIDATE_RULES` replays the original **per-update** `training.jsonl` and `validation.json` from `results_invertible_stage5_instrument/{e97,gdn2}_T1/`. It asserts that no stagnation stop fires before **either** the 90% curriculum threshold or 97% reproduction marker. The rejected no-floor rule is a negative control, not an admissible candidate. Synthetic boundary tests additionally prove that a completely flat curve cannot stop before 104K, that pre-floor evidence never contributes to the streak, and that each new stage resets the detector.

`C/receipts/artifact4/replay.json` independently replays the **actual archived X5b source**, records input/source SHA-256s, and corrects approximations in the assignment using the raw receipts:

| T1 curve | First ≥90% | First ≥97% | Archived X5b first stop before either event | Fixed X5c stop before either event |
|---|---:|---:|---:|---|
| E97 | 48,500 | **57,000** (97.8516%) | **10,000** | None |
| GDN2 | 13,000 | 13,750 (98.2422%) | None | None |

Thus the exact X5b rule fires at **10K, not the task's approximate 6K**, on this E97 receipt; 6K was its earliest theoretical eligibility. The measured ≥97% event is **57,000, not 56,750**. These corrections do not change the artifact: the old detector rejects a curve that later succeeds, during its low-accuracy pre-ignition period. The receipts describe delayed learning; they do not establish a cause or mechanism. The 100K floor is a calibration margin (~1.75× the measured 57K event), not a capability guarantee at deeper stages.

### X5b clean stop / preservation

At initial inspection **all six X5b runs were already cleanly terminal**, including the three GDN2 runs described as live in the task: all had `exit_code=0`, DONE markers, and old-rule `stop_reason=stagnation`. No worker/trainer PID was still present; GPUs were idle. Consequently **no signals were needed or sent**. `C/receipts/artifact4/stop_verification.json` records this no-op stop verification. The copied per-run terminal/result/validation/transition/growth receipts retain artifact #4. `x5b_preserved_sha256.json` binds every old non-checkpoint file, including original raw training logs; all hashes were rechecked unchanged after relaunch. Original X5b files and archived executed source remain untouched. Existing checkpoint payloads remain local/Git-ignored.

## Fixed rule — supervisor-bound floor semantics

The shared runner/launcher retain their discoverable `stage5b` filenames; their current default root and watcher now point to **X5c**. Executed source snapshots are in `C/receipts/full_launch/`; X5b's own source snapshots remain unchanged.

- Stage-local floor **100,000 updates**. Continuous disjoint 2K-window statistics may be logged before the floor, but the streak is **reset through the 100K boundary**. That boundary supplies the baseline; eligible comparisons end at **102K and 104K**, so earliest stagnation declaration is **104,000**, per the supervisor's explicit ruling. No pre-floor plateau counts.
- A joint plateau is cumulative **best-validation gain <0.001 absolute (0.1pp)** AND **relative mean-training-loss decrease <0.001 (0.1%)**. Two consecutive eligible comparisons are required. Either signal meeting its minimum resets the streak; negative gains count as plateau. A new stage gets a new detector/floor.
- **Threshold advancement always takes priority**, including a simultaneous plateau or safety boundary.
- Safety ceilings remain **500,000/stage** and **2,000,000 global**, honestly labeled `stage_safety_ceiling` and `global_safety_cap`, never stagnation/capacity findings.

## Fresh relaunch and measured wall rates

| Arm/seed | Exclusive GPU | Worker / trainer PID | Observed T4 updates | Wall updates/s |
|---|---:|---|---:|---:|
| E97/0 | 0 | 2080487 / 2080861 | 5,000 | 55.37 |
| GDN2/0 | 1 | 2080489 / 2080881 | 2,757 | 30.70 |
| E97/1 | 2 | 2080492 / 2080897 | 5,000 | 55.22 |
| GDN2/1 | 3 | 2080495 / 2080912 | 2,705 | 30.20 |
| E97/2 | 4 | 2080499 / 2080928 | 5,203 | 57.09 |
| GDN2/2 | 5 | 2080504 / 2080946 | 2,799 | 31.13 |

Rates use update 100 to the listed update, include intervening evaluation/checkpoint overhead, and exclude startup/JIT before update100. They are **operational throughput**, not steps-to-learning or superiority evidence. All six workers have **PPID=1, SID=worker PID**, own distinct live exclusive heartbeat leases, and retain worker/trainer `/proc` start identities. Trainers are children of their isolated workers. `C/receipts/launch_verification.json` verifies these identities, source/config/registration hashes, finite consecutive training streams, all six entry-checkpoint payload hashes and frozen validation payload hashes. `launch_snapshot/NAME/` freezes config, validation, transitions, checkpoints, data arrays and rate endpoints; live output directories are intentionally not committed mid-write.

Unchanged experiment: scratch E97-MLP **9,900,800** parameters and real FLA GDN2 **9,887,056**, `allow_neg_eigval=True`; seeds0/1/2, matched **lr1e-3**, X3 batch32, ScheduleFree AdamW betas(.9,.95), wd.01, clip1, no warmup, bf16 autocast/fp32 parameters, vocab3721. Stages **4→8→16→32→64→128**, ≥90% full-state validation advancement, frozen X3 validation pools, fresh procedural training. Model weights, full optimizer and training RNG carry across stages. Entry/every500/exit evals retain current and all lower depths; stage growth receipts retain steps-to-threshold or stagnation-at-depth. Uniform marginal **ln(3721)=8.221748 nats** and suffix-only floor **0.026874%** accompany printed validation/growth receipts, not an inferred recurrent-model capacity bound.

Every1000 rolling, per-stage best, and stage-entry/exit atomic checkpoints retain optimizer and NumPy/Torch/CUDA RNG. Existing below400GiB capacity pruning protects boundaries, stage best, latest and every10K, with audited deletions. Disk before launch was ~686GiB free. No automatic restart or discretionary walltime kill is scheduled. At current slowest T4 rates alone, 2M updates would take ~10.1h E97/~18.4h GDN2; deeper-stage costs/contention can differ, so these are not walltime guarantees or convergence projections.

## Repointed watcher — supervisor-side wake

**Watcher path: `scripts/watch_invertible_stage5b_growth.py`**, now defaulting to the new **X5c root**. Explicit command:

```bash
cd /home/erikg/emender
.venv/bin/python scripts/watch_invertible_stage5b_growth.py \
  --root experiments/expressivity_tasks/results_invertible_stage5c_growth_8m
```

Poll interval900s; log `C/receipts/completion_watcher.log`. Each isolated worker atomically publishes `receipts/NAME/terminal.json` and **`markers/{e97,gdn2}_seed{0,1,2}.DONE`** or **`.FAIL`**. Watcher exits0 only after six DONE+result receipts; FAIL/dead/reused PID exits nonzero after terminal-publication grace. `--once` returned **2 (healthy/pending, all six names)** at15:08:37 UTC. This worker notified the supervisor to re-arm the supervised watcher; **supervisor-session wake arming is not claimed without its confirmation**. A detached background watcher alone cannot supply the native session wake. Human re-arming is needed after host reboot.

## Validation and scope

**147 tests passed** before launch, including eight raw-receipt replay cases, post-floor boundary/reset/threshold priority tests, existing X2/X3/X5/instrument/aggregation/watcher coverage. Exact output: `C/receipts/tests_prelaunch.txt`. `bash -n`, Python compile checks and `git diff --check` passed. Registration and artifact evidence were committed in388b1644 before starting training; configs bind that exact commit and source digests. Launch verification rechecked the X5b preservation manifest after launch.

Reviewed architecture authority: `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and companion gap matrix. These are **local single-GPU task probes, not resilient DiLoCo or production E97 training**: **R16 evidence discipline only**; no R/NDP/V21S/ISP systems gate or Slurm Partition/QOS claim. Observations are not mechanisms. Independent reviewer acceptance remains supervisor-owned. Next: supervised watcher wake, terminal receipt verification, then the six matched growth/retention curves; no architectural superiority claim unless those completed curves earn it.
