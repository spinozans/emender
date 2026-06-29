# Frontier E97 8-Node Avg Smoke

Date: 2026-06-24
Task: `run-8-node-e97`

## Verdict

PASS. The bounded 8-node E97-MLP DiLoCo regular-averaging smoke launched from
current `origin/main`, ran to the walltime finalization margin, completed cleanly
with Slurm exit `0:0`, wrote regular checkpoints every 10 steps, wrote a final
checkpoint at step 451, advanced `latest.pt` to that final checkpoint, and
retained only the bounded checkpoint set.

This is a launch/checkpoint/finalization smoke only. It is not an extended
performance conclusion.

## Upstream State

- Confirmed before launch that `origin/main` was `9948df6`.
- Confirmed `9948df6` includes the `merge-4-node-e97` integration context.
- The worktree was fast-forwarded to `origin/main` before submission.
- Runtime commit recorded by the job:
  `9948df67c97338389ed5dc4120787f0827442e0d`.

No 16-node, 32-node, or 64-node jobs were submitted. No schedule-free outer run
was submitted. No GDN2 control was submitted.

## Submitted Job

| Job | QOS | Nodes | Walltime | Max requested node-hours | State | Exit | Actual elapsed | Actual node-hours |
| --- | --- | ---: | --- | ---: | --- | --- | --- | ---: |
| `4893934` | debug | 8 | `00:30:00` | `4.000000` | `COMPLETED` | `0:0` | `00:20:17` | `2.704444` |

Scheduler accounting:

```text
4893934|e97-8n-avg-smoke|batch|debug|COMPLETED|0:0|00:20:17|00:30:00|8|billing=896,cpu=896,energy=10821192,mem=4000G,node=8|2026-06-24T05:25:39|2026-06-24T05:25:55|2026-06-24T05:46:12
4893934.batch|batch|||COMPLETED|0:0|00:20:17||1|cpu=56,mem=500G,node=1|2026-06-24T05:25:55|2026-06-24T05:25:55|2026-06-24T05:46:12
4893934.extern|extern|||COMPLETED|0:0|00:20:17||8|billing=896,cpu=896,mem=4000G,node=8|2026-06-24T05:25:55|2026-06-24T05:25:55|2026-06-24T05:46:12
4893934.0|bash|||COMPLETED|0:0|00:18:34||8|cpu=448,mem=4000G,node=8|2026-06-24T05:27:38|2026-06-24T05:27:38|2026-06-24T05:46:12
```

Actual node-hours are computed from the top-level allocation elapsed time:
`8 * (20m17s / 3600) = 2.704444`.

## Run Configuration

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z
```

Primary artifact paths:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/summaries/summary.md
```

Key launch settings:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=8
SCALEOUT_WALLTIME=00:30:00
TRAIN_MINUTES=20
DILOCO_K=10
DILOCO_ISLAND_SIZE=8
DILOCO_OUTER_OPTIMIZER=avg
DILOCO_OUTER_LR=1.0
DILOCO_OUTER_BETA=0.0
DILOCO_EXPORT_BASIS=x
BATCH_SIZE=1
CHUNK_SIZE=2048
LOG_EVERY=5
VAL_EVERY=10000
SAVE_EVERY=10
KEEP_CHECKPOINTS=4
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt
VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt
TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache
```

The job launched 64 ranks across 8 nodes. Rank 0 logged:

```text
[DiLoCo] world_size=64 backend=nccl; this is rank 0 on cuda:0
[DiLoCo-hybrid] 8 islands x 8 GPUs: per-step DDP gradient all-reduce WITHIN island + DiLoCo periodic averaging ACROSS islands every K=10 (subgroup comms warmed sequentially)
[DiLoCo] outer optimizer: avg (stateless periodic averaging)
```

All rank fused-guard samples showed the expected E97 Triton path:

```text
level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback
```

## Naming Verification

The integrated run-label scheme produced the expected E97/Emender label:

```text
Run label prefix: emender_E97_1.3B (params_arg=100m, total_params=1,299,726,652, trainable_params=1,299,726,652)
Output directory: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/train/emender_E97_1.3B_20260624_052918
Model: Level E97, 1,299,726,652 parameters
```

Run label:

```text
emender_E97_1.3B_20260624_052918
```

## Loss And Throughput Sanity

Losses remained finite through startup, steady-state training, and finalization.
Representative rank-0 metric lines:

```text
step      5 | loss 10.3781 | lr 9.00e-04 | grad 3.55 | tok/s 554 | global_tok/s 35460 | elapsed_h 0.005 | time 2026-06-24T09:30:01+00:00
step    100 | loss 5.9593 | lr 9.00e-04 | grad 1.54 | tok/s 1115 | global_tok/s 71336 | elapsed_h 0.061 | time 2026-06-24T09:33:21+00:00
step    250 | loss 5.1895 | lr 9.00e-04 | grad 1.28 | tok/s 1115 | global_tok/s 71354 | elapsed_h 0.149 | time 2026-06-24T09:38:39+00:00
step    410 | loss 4.7979 | lr 9.00e-04 | grad 0.97 | tok/s 1112 | global_tok/s 71176 | elapsed_h 0.244 | time 2026-06-24T09:44:19+00:00
step    450 | loss 5.5243 | lr 9.00e-04 | grad 1.00 | tok/s 1134 | global_tok/s 72571 | elapsed_h 0.268 | time 2026-06-24T09:45:46+00:00
```

Summary metrics:

```text
Training complete! Final step: 451
FINAL_LOSS_LAST100: 5.7391
DILOCO_MERGES: 46
DILOCO_K: 10
DILOCO_SYNC_TOTAL_S: 221.890
DILOCO_SYNC_AVG_MS: 4823.7
```

Representative global throughput after startup was usually about `53k` to
`72k` tokens/s depending on whether the log window included merge/checkpoint
work.

## Checkpoint And Retention Verification

Regular checkpoints were written every 10 steps through step 450. Finalization
triggered at step 451 due to the Slurm walltime margin. All ranks entered
finalization and rank 0 wrote the final checkpoint:

```text
[final-checkpoint] rank 0/64 entering finalization at step=451 reason=walltime:SLURM_JOB_END_TIME remaining_s=599.3 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 is_head=True
>>> [DiLoCo] FINAL merge #46 at step 451: consensus model averaged across 64 ranks (4796 ms)
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=451 loss=5.7391 remaining_s=594.2 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/64 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/train/emender_E97_1.3B_20260624_052918/checkpoint_step_000451_loss_5.7391.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/train/emender_E97_1.3B_20260624_052918/latest.pt model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/64 is_head=True
```

Final retained run-directory contents:

```text
checkpoint_step_000430_loss_5.2231.pt
checkpoint_step_000440_loss_5.4804.pt
checkpoint_step_000450_loss_5.5243.pt
checkpoint_step_000451_loss_5.7391.pt
latest.pt -> checkpoint_step_000451_loss_5.7391.pt
```

`readlink -f latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893934-20260624T092558Z/train/emender_E97_1.3B_20260624_052918/checkpoint_step_000451_loss_5.7391.pt
```

No checkpoint `.tmp` file remained after completion. Mid-run samples did observe
atomic-write temporary checkpoint files while saves were in progress; each was
removed when the corresponding save completed.

## Communication And Warning Scan

No NCCL/RCCL watchdog timeout, collective mismatch, traceback, runtime error,
OOM, or nonzero exit was found in the final scan. Slurm accounting reports
`COMPLETED` with exit `0:0`.

Non-blocking warnings:

- PyTorch/NCCL emitted the existing rank-to-GPU warning during process group
  startup: `using GPU 0 as device used by this process is currently unknown`.
  The run still completed 46 DiLoCo merges, final consensus averaging, final
  checkpoint write, and clean exit.
- Triton 3.2.0 emitted the existing recommendation warning for 3.3.0.
- c10d emitted transient IPv6 socket initialization warnings during rendezvous.
  The process group initialized and trained normally afterward.

## Recommendation

Next step should be a 4-node versus 8-node schedule-free outer comparison as a
separate bounded task, using the same launch/checkpoint/finalization reporting
discipline. Rationale: the avg ladder now has clean 4-node and 8-node launch,
checkpoint retention, final checkpoint, and clean finalization evidence. A
16-node avg smoke is also reasonable if the priority is systems scaling first,
but the immediate optimizer-risk question is now unblocked enough to compare
schedule-free outer separately.
