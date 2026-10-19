# X8 — pay-for-depth T16: launch report

**LIVE: E97 on card2 and stock-kernel GDN2 on card7; native supervisor watcher armed.** Registration and operational sources were committed **before launch** in **`3ccc6252`**. Detached launch: **2026-10-19T15:03:22Z**, lambda01. Evidence root (**C/**): `experiments/expressivity_tasks/results_invertible_stage8_paydepth_8m/`. This report records launch, not a learning verdict. Four live X6 trainers and X7's GDN2 fp32 trainer retained their original PID/start identities; no existing trainer, source, lease or historical checkpoint was changed.

## Pre-launch predictions (verbatim)

(a) Incentive theory (X7's surviving reading): removing the free lunch breaks E97's position-1 attractor - Run 1 shows materially more position-2+ structure than X7's flat 512,2,0,... (threshold: any position >=2 reaching >=50/512 counts as material; report exact columns).

(b) If Run 1 still flat: E97's basin is initialization-structural, not incentive-structural - the next suspect is the optimizer (lr shaping) and/or scale (8M -> 1.3B).

(c) GDN2 prediction: deeper-weighted loss extends its frontier past P4 relative to X7 deep-first's 116/512 at P4 (or not - either is informative for the capacity question).

`C/registration.json` timestamps these exact predictions, records the GPU bindings and normalized weights, and hashes operational sources and preflight receipts. Launch gates require those source bytes to match both registration and committed HEAD. Optimizer/scale follow-ups are **not launched**.

## Recipe and intervention

Both runs start **T16 from scratch**, seed0, empty AdamWScheduleFree optimizer; no checkpoint load. **No ladder, no supervision/carry slots, no stagnation detector, no threshold stop.** The only stopping rule is the **500,000-update safety ceiling**. Core X7 recipe: p61/vocabulary3721, dim256, four layers, MLP ratio2, E97 expansion1 / GDN2 expansion8.625; **9,900,800 / 9,887,056 parameters**; batch32, lr0.001, betas(0.9,0.95), weight decay0.01, clip1, no warmup. Input length18 is causally right-padded to32 by the unchanged X3 helper, then sliced before loss/metrics. Training batches are fresh procedural draws, never validation examples.

**Training objective:** `mean_batch(sum(t/136 * CE_t, t=1..16))`. All16 ordered driver targets remain present. Weights sum to1, preserving the mean-loss scale of X7: P1 coefficient **1/136 = 0.735294%**, P16 **16/136 = 11.764706%**. The realized fraction of loss depends on each position's CE; the coefficient is not a hard bound on realized CE share. Only the training weighting changes versus the corresponding bf16 recipe. Validation remains **unweighted**, using X7's exact evaluator and X6D's `position_counts` instrument.

E97 retains X7 bf16's legacy state policy and `fused_inference=True` at evaluation (sequential Triton training/eval). Parameters and live state accumulation were already fp32; this is not an entirely bf16 accumulator. GDN2 uses **unmodified stock FLA bf16-autocast kernels**, fp32 parameters, negative eigenvalues enabled, **no X7 fp32 sequential adapter**. Thus the requested GDN2 comparison with X7's fp32 deep-first result also differs in recurrence precision/backend; do not attribute a difference solely to weighting. No precision intervention or installed kernel edits were made here.

## Frozen validation and verified launch

Every **500 updates**, including step0 and final500K, each run performs a complete **512-sequence T16** frozen validation pass: P1…P16 joint/x/y correct counts, per-position CE, aggregate/suffix/terminal metrics, and consecutive perfect-prefix depth. Counts reconcile exactly with aggregate totals. Both runs' archived arrays and array hashes match X7 seed0, and archive hashes are verified. No test-set selection.

`C/receipts/verify_launch.py` and `launch_verification.json` independently check source/dependency hashes, the pre-launch registration commit, worker/trainer PID/start identities, **worker PPID=1 / SID=worker PID**, trainer parent/session, fixed exclusive leases, frozen pools, initial checkpoint payload hashes, empty optimizer state, identical initial model weights to the corresponding X7 runs, and finite/consecutive T16-only training streams. Snapshot **2026-10-19T15:03:47Z**:

| Run | Card | Worker / trainer PID | Observed update | Operational updates/s | Remaining to500K at that rate |
|---|---:|---|---:|---:|---:|
| E97 bf16 | 2 | 2422923 / 2425214 | 778 | 52.90 | 2.62h |
| GDN2 bf16 stock | 7 | 2422934 / 2425373 | 366 | 28.94 | 4.80h |

Rates cover update100 to the observation, including eval/checkpoint cost. These are early operational observations, **not completion guarantees or learning predictions**. Both runs overlap; wall times are not additive. No discretionary walltime stop is scheduled.

**Step0: joint-state correct counts out of512 per position.** Both have 2/8192 aggregate and perfect-prefix depth0; high CE reflects scratch tied embeddings, not a learning finding.

| Run | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | P9 | P10 | P11 | P12 | P13 | P14 | P15 | P16 | Unweighted CE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| E97 bf16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 250.976103 |
| GDN2 bf16 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 243.962320 |

Immutable config/data/validation/checkpoint-index snapshots, frozen pool archives and finite training prefixes are in `C/receipts/launch_snapshot/`; live streams are not committed mid-write. `full_launch/` archives operational sources, installed FLA dependency bytes **with upstream MIT license**, packages, base commit/diff and GPU/disk snapshots; `full_launch_manifest.json` hashes them.

## Retention fix included from step0

Reuse the X6D-corrected discipline: the **immutable step0 entry path is stored separately from selected best**, never replaced when best changes. Below400GiB free, retain entry, selected aggregate-best and latest; final500K is latest/exit. `checkpoints.json` retains hashes of all checkpoint receipts; `pruning.jsonl` records removed paths and hashes still retained. Approximately267GiB was free at launch, so bounded pruning is active immediately. No historical files were deleted. A synthetic best-change regression exercises the actual pruner and proves entry survives; `C/receipts/retention_audit.json` and frozen pruning snapshots show both real entries survive post-launch pruning, with at most three checkpoints at the captured observation. This is a single-stage run, so entry/exit means T16 step0/final500K.

## Native watcher armed; growth report on wake

**Native supervisor watcher armed (PID2578383, started 2026-10-19T15:04:00Z)**; supervised process **`proc_4619`**, first healthy pending poll at the same timestamp. Command: `.venv/bin/python scripts/watch_invertible_stage8_paydepth.py --interval 300`. The supervisor confirms its exit wakes the session natively. `C/receipts/watcher_armed.json` records PID/starttime/command and confirmation; independent `--once` returned2 (healthy/pending).

Detached workers publish atomic `receipts/NAME/terminal.json` and `markers/NAME.DONE` or `.FAIL`; the watcher requires both successful DONE receipts and result files for zero exit, reports failure markers or dead/reused identities nonzero. It is not host-failure recovery; re-arm after host reboot. No WorksGood service/dispatcher/worker machinery was started (only the existing standalone local GPU lease script was reused).

**On watcher wake:** verify terminal/checkpoint/source identities; write the growth report against(a)/(b)/(c), with exact P1…P16 trajectories and positions reaching50/512. Report GDN2 P4 versus116/512 and positions beyondP4, without inventing an unregistered capacity threshold. A launch baseline cannot establish incentive-theory support or refutation.

## Validation and scope

**130 CPU tests passed (4 GPU skips), plus the X8 leased-GPU test passed**: six new X8 tests cover exact weights/loss/gradients/scale, invalid target masks, scratch initialization and stock GDN kernel identity, safety-only continuation despite perfect validation, immutable-entry retention through best changes with actual prune manifests, finite bf16 full-model backward and complete per-position validation, and pending/DONE/FAIL/dead-identity watcher states. Commands and exact output are in `C/receipts/tests_{cpu,gpu,prelaunch}.txt`, with preflight exit0 recorded. Existing X3/X5/X5B/X6/X7/task/position tests are included; the X7 GPU precision-pilot tests were not rerun. Python compilation, `bash -n`, source diff whitespace checks and launch verification pass.

Architecture authority: read `docs/RESILIENT_DILOCO_COMPUTE_POOL.md` and `docs/RESILIENT_DILOCO_GAP_MATRIX.md`; **R16 evidence discipline only** applies to this local probe. No resilient DiLoCo, production fixed-world training, Slurm Partition/QOS or systems-scale acceptance is claimed. Independent acceptance review remains supervisor-owned. Next scientific deliverable: growth report on native watcher wake.
