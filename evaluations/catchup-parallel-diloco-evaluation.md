# Evaluation: catchup-parallel-diloco

Task ID: `catchup-parallel-diloco`
Evaluator: `agent-1471`
Date: 2026-06-15

## Summary

Score: **0.16 / 1.00**
Confidence: **0.93**
Rubric underspecified: **No**

This task had an explicit validation checklist: run DiLoCo-native, DDP-naive,
and DDP-LR-rescaled configurations against the single-GPU
`run-ref-emender-mlp` answer-key curve; log durable held-out BPB-vs-token
curves; report measured throughput and days-to-100B; and answer the DDP
batch/LR confound question. The actor made a useful but incomplete partial
attempt: it added a DiLoCo-native detached launch harness, leased GPUs through
the broker, and briefly trained one 5-replica DiLoCo-native run with the native
per-replica batch and CMA LR. However, the run is no longer alive, produced no
held-out BPB points, did not use the rerouted `scripts/launch_detached_run.sh`
protocol, and did not run or report the DDP-naive or DDP-LR-rescaled controls.

## Dimension Scores

| Dimension | Score | Rationale |
| --- | ---: | --- |
| GPU leasing and execution discipline | 0.35 | The actor used `scripts/gpu_lease.sh` inside a detached supervisor and initially verified leases on GPUs 1-6 with active training on GPUs 2-6. This earns real partial credit. It loses most credit because the process is now dead, leases are gone, and the later coordinator guidance required using `scripts/launch_detached_run.sh`, which the committed harness did not do. |
| Configuration fidelity | 0.42 | The DiLoCo-native command used Pile data, `p50k_base`, bf16, schedule-free AdamW at CMA LR `0.0010071509461604343`, emender-mlp geometry, `--batch_size 4` per replica, and local-SGD DiLoCo (`outer_lr=1`, `outer_beta=0`, `k=250`). No DDP-naive or DDP-LR-rescaled configs were run, and the held-out evaluator used an absolute held-out path from another worktree without proving it matched the reference. |
| Measured BPB curves | 0.00 | `/mnt/nvme1n1/erikg/catchup_parallel/diloco_native_emender_mlp/heldout_curve.csv` contains only the header. The checkpoint evaluator failed with `ModuleNotFoundError: No module named 'ndm'`, so no held-out BPB-vs-token measurements exist for any config. |
| Reference comparison | 0.00 | No overlay or numerical comparison against the `run-ref-emender-mlp` curve was produced. |
| Throughput reporting | 0.20 | Training logs include transient throughput for the DiLoCo-native attempt, peaking around `global_tok/s` 40k after warmup and 37k at step 250. There is no consolidated report, no projected days-to-100B, and no throughput for the requested controls. |
| Confound conclusion | 0.00 | The actor did not run DDP-naive or DDP-LR-rescaled, so it could not state whether DDP-naive reproduces rollover or whether LR rescaling or DiLoCo-native removes it. |
| Completion behavior | 0.05 | The actor initially verified a live detached run and logged the PID, but the run was terminated by SIGTERM around step 425, the durable curve stayed empty, and the task explicitly required no done-on-launch and staying to completion. |

## Evidence Reviewed

- `wg show catchup-parallel-diloco` and `wg log catchup-parallel-diloco --list`: the actor added and launched a custom detached harness at 2026-06-15T17:04Z, then verified a live run at 2026-06-15T17:06Z. The task was later reset with explicit guidance to use `scripts/launch_detached_run.sh`.
- Commit `e19a4b0`: added `experiments/catchup_parallel/run_diloco_native_detached.sh`, `experiments/catchup_parallel/eval_emender_mlp_checkpoint_bpb.py`, and `experiments/catchup_parallel/README.md`.
- `/mnt/nvme1n1/erikg/catchup_parallel/diloco_native_emender_mlp/logs/supervisor.log`: broker granted GPUs 1 and 2-6; the evaluator attempted checkpoint `checkpoint_step_000250_loss_6.2467.pt` and failed.
- `/mnt/nvme1n1/erikg/catchup_parallel/diloco_native_emender_mlp/logs/train.log`: training ran from step 0 through step 425, completed a DiLoCo merge at step 250, saved one checkpoint, and then torchrun received SIGTERM at 2026-06-15T17:11:53Z.
- `/mnt/nvme1n1/erikg/catchup_parallel/diloco_native_emender_mlp/logs/eval.log`: `eval_emender_mlp_checkpoint_bpb.py` failed with `ModuleNotFoundError: No module named 'ndm'`.
- `ps`, `scripts/gpu_lease.sh status`, and `wc -l heldout_curve.csv`: the supervisor and training PIDs are no longer alive, there are no active leases, and the curve file has one line (header only).

## Calibration

A score of 0.16 reflects meaningful setup work and a real short-lived
multi-GPU DiLoCo-native launch, but almost none of the scientific deliverable.
The core task was not to write a harness or start one candidate; it was to
produce measured BPB curves and conclusions across three configurations. Since
the only run died before producing any held-out BPB point and both control
experiments are absent, this remains a low-scoring incomplete result rather
than a partially answered experiment.
