# Frontier E97 4-Node Avg Smoke

Date: 2026-06-24
Task: `run-4-node-e97`

## Verdict

PASS. The bounded 4-node E97-MLP DiLoCo regular-averaging smoke ran after the
upstream run-label and checkpoint-retention integration landed on `origin/main`,
completed cleanly, wrote regular checkpoints, wrote a walltime-triggered final
checkpoint, updated `latest.pt` to the final checkpoint, and retained only the
bounded checkpoint set.

This is not scaleout-conclusion evidence. It is a short launch/checkpoint/
finalization smoke for the E97-MLP avg arm.

## Upstream State

- `origin/main`: `d9e789b191664b591d99d89a66017cc8cdfb9705`
  (`fix: guard E97 checkpoint retention (emergency-e97-checkpoint)`)
- Worktree runtime commit recorded by the job:
  `42c6939ab74b6bdb2a4c14054e3aa93b24cd323a`
- `git diff --quiet HEAD origin/main` returned status `0` before reporting,
  so the submitted tree contents matched the integrated `origin/main` tree even
  though the local worktree commit hash was the WG integration commit.

## Submitted Jobs

| Job | QOS | Nodes | Walltime | Requested node-hours | State | Notes |
| --- | --- | ---: | --- | ---: | --- | --- |
| `4893867` | normal | 4 | `00:30:00` | `2.000000` | `CANCELLED by 19032` | Cancelled while pending for `Priority`; elapsed `00:00:00`, no allocation. |
| `4893868` | debug | 4 | `00:30:00` | `2.000000` | `COMPLETED`, exit `0:0` | Actual elapsed `00:20:18`; estimated actual node-hours `1.353333`. |

No 8-node or larger jobs were submitted.

Scheduler accounting captured after completion:

```text
4893867|e97-4n-avg-smoke|batch|normal|CANCELLED by 19032|0:0|00:00:00|4||None|2026-06-24T03:52:23
4893868|e97-4n-avg-smoke|batch|debug|COMPLETED|0:0|00:20:18|4|billing=448,cpu=448,energy=5877013,mem=2000G,node=4|2026-06-24T03:52:34|2026-06-24T04:12:52
```

## Run Configuration

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z
```

Primary artifact paths:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/artifacts/manifest.json
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/artifacts/env.txt
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/logs/train.log
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/summaries/summary.md
```

Key launch settings:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=4
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

The job launched 32 ranks across 4 nodes. Rank 0 logged:

```text
[DiLoCo] world_size=32 backend=nccl; this is rank 0 on cuda:0
[DiLoCo] outer optimizer: avg (stateless periodic averaging)
```

All sampled ranks logged the expected fused E97 path:

```text
level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback
```

## Naming Verification

The integrated naming scheme produced an unambiguous E97/Emender run label:

```text
Run label prefix: emender_E97_1.3B (params_arg=100m, total_params=1,299,726,652, trainable_params=1,299,726,652)
Output directory: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/train/emender_E97_1.3B_20260624_035347
Model: Level E97, 1,299,726,652 parameters
```

The per-run manifest confirms the same labels:

```json
{
  "run_name": "emender_E97_1.3B_20260624_035347",
  "run_label_prefix": "emender_E97_1.3B",
  "model": {
    "model_family": "emender",
    "level": "E97",
    "params_arg": "100m",
    "derived_param_slug": "1.3B",
    "total_params": 1299726652
  }
}
```

This avoids the prior confusing `levelE97_100m_*` path for a 1.3B E97 geometry.

## Loss And Throughput Sanity

The run reached finite losses from warmup through finalization. Selected lines:

```text
step      5 | loss 10.3462 | lr 9.00e-04 | grad 3.45 | tok/s 1282 | global_tok/s 41021 | elapsed_h 0.002 | time 2026-06-24T07:54:05+00:00
step    100 | loss 6.1379 | lr 9.00e-04 | grad 1.72 | tok/s 1230 | global_tok/s 39368 | elapsed_h 0.057 | time 2026-06-24T07:57:20+00:00
step    300 | loss 4.9835 | lr 9.00e-04 | grad 1.16 | tok/s 1235 | global_tok/s 39505 | elapsed_h 0.173 | time 2026-06-24T08:04:19+00:00
step    535 | loss 4.6998 | lr 9.00e-04 | grad 1.28 | tok/s 815 | global_tok/s 26071 | elapsed_h 0.310 | time 2026-06-24T08:12:34+00:00
```

Summary metrics:

```text
Training complete! Final step: 535
FINAL_LOSS_LAST100: 5.3741
PEAK_MEMORY_MB: 15587
RESERVED_MEMORY_MB: 22840
DILOCO_MERGES: 54
DILOCO_K: 10
DILOCO_SYNC_TOTAL_S: 221.293
DILOCO_SYNC_AVG_MS: 4098.0
```

Representative global throughput after startup alternated around `26k` to
`39k` tokens/s depending on whether the log window included merge/checkpoint
work.

## Checkpoint And Retention Verification

Regular checkpoints were written every 10 steps through step 530. The final
checkpoint was triggered by the walltime margin at step 535 and all 32 ranks
entered finalization:

```text
[final-checkpoint] rank 0/32 entering finalization at step=535 reason=walltime:SLURM_JOB_END_TIME remaining_s=599.2 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 is_head=True
>>> [DiLoCo] FINAL merge #54 at step 535: consensus model averaged across 32 ranks (4044 ms)
[final-checkpoint] START kind=final reason=walltime:SLURM_JOB_END_TIME step=535 loss=5.3741 remaining_s=594.9 model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/32 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/train/emender_E97_1.3B_20260624_035347/checkpoint_step_000535_loss_5.3741.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/train/emender_E97_1.3B_20260624_035347/latest.pt model_variant=level=E97,params_arg=100m,derived_params=1.3B,total_params=1299726652,mlp_ratio=1.5 rank=0/32 is_head=True
```

Final retained run-directory contents:

```text
checkpoint_step_000510_loss_5.2879.pt
checkpoint_step_000520_loss_4.6412.pt
checkpoint_step_000530_loss_4.9749.pt
checkpoint_step_000535_loss_5.3741.pt
latest.pt -> checkpoint_step_000535_loss_5.3741.pt
```

`readlink -f latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260624/e97-MLP/4893868-20260624T075236Z/train/emender_E97_1.3B_20260624_035347/checkpoint_step_000535_loss_5.3741.pt
```

No leftover checkpoint temp files were present after completion. A mid-run
sample during step 530 did observe an atomic-write temporary checkpoint name
while it was still in progress, and the final sample showed only completed
checkpoint files plus `latest.pt`.

Retention behavior was bounded and did not delete the active/latest checkpoint:
with `KEEP_CHECKPOINTS=4`, the final set retained the last three regular
checkpoints plus the final checkpoint, and `latest.pt` points to the final
checkpoint.

## Warnings And Non-Blocking Observations

- PyTorch/NCCL printed rank-to-GPU warning lines during process group startup
  (`using GPU 0 as device used by this process is currently unknown`). The job
  nevertheless reached 54 DiLoCo merges, final consensus averaging, final
  checkpoint completion, and clean exit.
- Triton 3.2.0 emitted the existing recommendation warning for 3.3.0. This did
  not block the fused E97 smoke.
- The run was fresh from scratch (`resume_checkpoint` empty). Resume from this
  new 4-node checkpoint was not launched in this task because the scope was a
  single bounded 4-node avg smoke and explicitly excluded follow-on larger or
  schedule-free-outer jobs.

## Recommendation

Next WG task should be an 8-node E97-MLP avg smoke, not schedule-free outer yet.
Reason: the 4-node avg path now has clean launch, finite loss/throughput,
checkpoint retention, final checkpoint, and naming evidence. Schedule-free outer
should wait until the avg ladder has at least one 8-node launch/checkpoint point,
unless a human explicitly prioritizes optimizer-variant risk over scaleout
systems risk.
