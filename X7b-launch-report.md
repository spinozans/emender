# X7b — matched stock-bf16 GDN2 uniform control (launched)

**Live on lambda01, card4; no learning verdict.** Registration, operational sources and implementation-test receipts were committed/pushed as **`d9785a65530fd8c720c26063541c89522d7124eb` before launch**. Root: `experiments/expressivity_tasks/results_invertible_stage7b_stock_8m/` (**C/**). Detached worker PID3915427/start253250211, trainer PID3915878/start253250584, with verified parent relation and exclusive card4 lease. No WorksGood service/dispatcher/worker machinery was used; only the pre-existing standalone GPU lease shell script.

## Prediction registered before launch

If this uniform stock-bf16 control reaches **P1–P3 simultaneously512/512 and P4 materially above X8's9/512 (>=50/512)**, the X8 GDN2 weighting refutation upgrades from descriptive to causal on matched evidence. Otherwise do not upgrade. This is a conditional one-seed matched comparison, not a general weighting theorem. P4 material acquisition is not perfect composition.

## Matched recipe

GDN2 **9,887,056 parameters**, p61/vocab3721, dim256/four layers, scratch T16, seed0. **Unmodified stock bf16 chunked recurrence, NO fp32 adapter**; fp32 parameters/bf16 autocast, batch32, uniform mean target CE, clip1. Exact X7 ScheduleFree recipe: constant1e-3, betas(.9,.95), eps1e-8, nominal decay.01 on all parameters, no warmup. No ladder, auxiliary carry loss, stagnation/threshold stop, restart or automatic retry. **500,000 updates** primary endpoint, full512-example frozen validation every500, P1–P16 joint-state counts and CE, max-anywhere separated from fixed-final. Fresh procedural batches never sample the frozen pool.

`C/registration.json` pins predictions, source/dependency hashes, pools/seeds, recipe, resources and readout command. `C/receipts/launch_verification_gdn2_uniform.json` proves the initial model tensors and frozen arrays match **X8's stock-bf16 GDN2** exactly, with empty optimizer state. All X10 controls additionally reproduce their historical initialization. Thus the matched comparison changes the loss weighting, not recurrent precision. Sequential fp32 X7 GDN2 remains a different comparison.

## Prelaunch checks and bounded resources

The shared source-named suite passed **30 CPU tests (eight GPU skips)**; the leased-device synthetic-seed99 suite passed **all four GPU tests**, including full per-position evaluation without confirmatory-pool selection. The disposable25-update card4 probe plus full-model serialization/next-update check had **max absolute difference0**, no frozen-task evaluation, peak two-model allocation561,805,312 bytes. Exact current-source probe is `C/receipts/probe_gdn2_uniform.json`.

Registered infrastructure ceilings: **314,679 seconds** (three times compile-inclusive probe projection, at least24h), **8GiB allocated GPU memory**. A timeout/resource failure is censored, not an optimizer loss; no retry. External worker deadline adds120s cleanup allowance, TERM then30s/KILL. Always retain **entry + aggregate-best + latest/final**, plus one atomic staging file; deletion receipts preserve hashes. No historical files are deleted.

Immutable launch snapshot at **2026-10-20T03:23:35Z**: **1,386 finite consecutive updates**, exact constant LR, validations0/500/1000, two retained checkpoints. Entry P1–P16 counts: **`1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0`**. Entry checkpoint SHA256 `041965345555028051cee6b91c4e89aca69ab18921e438fada20176cc6c037a8`. These are launch/integrity facts, not acquisition conclusions.

## Watcher and reproducibility

Supervisor arming command (cwd `/home/erikg/emender`): **`.venv/bin/python scripts/watch_invertible_stage7b_stock.py --interval 300`**. `--once` returned**2 (healthy/pending)**; parent was notified with exact command and worker/trainer identities. Native supervisor arming is parent-owned; no arming claim is inferred from a status poll. The watcher detects FAIL/dead/reused identities and exits0 only for DONE plus result. Host reboot requires re-arming, not automatic trainer recovery.

Shared read-only verification: `.venv/bin/python -m experiments.expressivity_tasks.results_invertible_stage10_optimizers_8m.receipts.verify_launch` (checks live PID/lease, source/registration/committed blobs, historical initialization/pool, empty entry optimizer, actual LR and finite consecutive rows). Immutable `C/receipts/launch_snapshot/` avoids committing live files. On completion: `.venv/bin/python -m scripts.invertible_stage10_control analyze --arm gdn2_uniform`, then validate terminal/checkpoint identities and adjudicate the conditional prediction.

Local single-card research only: architecture/gap-matrix **R16 evidence discipline by analogy**; no DiLoCo/Frontier Partition/QOS/resilience/production capability claim. Independent reviewer gate remains supervisor-owned.
