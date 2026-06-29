# Frontier E97 Two-Node Checkpoint Canary

Date: 2026-06-23
Task: `validate-e97-two-node-checkpoint-canary`
Worktree commit at first accepted runtime submission: `62d78bda4a4b0be8cefca4bf0eb6af8eff746c7f`

## Verdict

DEFECT FOUND: the bounded 2-node E97-MLP DiLoCo regular-averaging canary
launched, trained with finite loss, merged, wrote regular checkpoints, wrote a
final checkpoint, updated `latest.pt`, and exited cleanly. The required resume
canary loaded that `latest.pt`, resumed at step 140, continued to later finite
losses through step 195, and wrote post-resume regular checkpoints. It then
exposed a multi-node final-checkpoint coordination defect: ranks diverged into
incompatible NCCL all-reduces during finalization and the job timed out in the
collective watchdog.

No workaround was implemented here. Follow-up implementation task
`fix-multi-node` was created:
`Fix multi-node E97 DiLoCo final-checkpoint collective mismatch`.

No 4-node or 8-node job was submitted.

## Upstream Fix Evidence

This attempt ran after the checkpoint-fix and walltime-fix tasks completed and
after their integration reached `origin/main`.

- `fix-e97-mlp-checkpoint-finalization` completed and recorded live E97-MLP
  one-node checkpoint/resume jobs `4889966` and `4890029`.
- `implement-walltime-final-checkpoint` completed and added the walltime-aware
  final-checkpoint controller. That implementation task did not itself launch
  live Frontier runs because the worker shell lacked the project Python stack.
- `merge-e97-walltime-checkpoint-main` integrated both branches and pushed
  `8b4b7cf7bf65926ac788c6394e1d3f65ed479c63` to `origin/main`, preserving the
  E97 atomic checkpoint finalization and walltime metadata paths.
- `validate-cross-model-final-checkpoint` recorded post-merge GDN2 checkpoint
  and resume controls at jobs `4890166` and `4890189`.

The live upstream E97 checkpoint/resume precondition is satisfied for the E97
checkpoint finalization fix. The walltime task did not have independent live
E97 evidence before this canary; this canary therefore also served as the first
multi-node E97 exercise of the integrated walltime/final-checkpoint path.

## Submitted Jobs

| Job | Arm | Nodes | Walltime | QOS | Requested node-hours | Actual elapsed | Actual node-hours | Final state |
| --- | --- | ---: | --- | --- | ---: | --- | ---: | --- |
| `4891066` | E97-MLP regular averaging | 2 | `00:30:00` | normal | `1.000000` | `00:00:00` | `0.000000` | `CANCELLED`; pending with reason `Priority`, no allocation. |
| `4891072` | E97-MLP regular averaging | 2 | `00:30:00` | debug | `1.000000` | `00:00:00` | `0.000000` | `CANCELLED`; pending with reason `Priority`, no allocation. |
| `4891083` | E97-MLP regular averaging | 2 | `00:30:00` | debug | `1.000000` | `00:08:57` | `0.298333` | `COMPLETED`, exit `0:0`. |
| `4891149` | E97-MLP regular averaging resume | 2 | `00:20:00` | debug | `0.666667` | `00:14:09` | `0.471667` | `CANCELLED by 19032` after final-checkpoint NCCL timeout. |

Latest scheduler accounting captured after cancellation:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|AllocTRES|Start|End
4891083|emender-diloco-scaleout|batch|debug|COMPLETED|0:0|00:08:57|2|billing=224,cpu=224,energy=1036335,mem=1000G,node=2|2026-06-23T05:20:52|2026-06-23T05:29:49
4891149|emender-diloco-scaleout|batch|debug|CANCELLED by 19032|0:0|00:14:09|2|billing=224,cpu=224,mem=1000G,node=2|2026-06-23T05:31:39|2026-06-23T05:45:48
```

## Regular Averaging Canary: Job `4891083`

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z
```

Configuration:

```text
SCALEOUT_VARIANT=e97-MLP
SCALEOUT_NODES=2
SCALEOUT_WALLTIME=00:30:00
TRAIN_MINUTES=6
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
WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=120
WALLTIME_CHECK_EVERY=1
```

Launch and runtime sanity:

- 16 ranks launched across 2 nodes.
- E97 fused path was active on all ranks:
  `level=E97 bf16 use_triton=1 -> fused split-edit Triton kernel, NO eager fallback`.
- DiLoCo regular averaging initialized:
  `world_size=16 backend=nccl`, `K=10 outer_lr=1.0 outer_beta=0.0`.
- Finite losses were logged from step 5 through final step 140.
- Representative throughput range was approximately `8,796` to `17,992`
  global tok/s after the compile/startup phase.
- Peak memory was `15587` MiB per rank, reserved memory `22840` MiB.

Key training lines:

```text
step      5 | loss 10.4674 | lr 9.00e-04 | grad 3.59 | tok/s 1067 | global_tok/s 17073 | elapsed_h 0.003 | time 2026-06-23T09:23:31+00:00
  >>> [DiLoCo] merge #1 at step 10: averaged model weights across 16 ranks in 4398 ms (amortized over 10 steps)
step    100 | loss 6.0316 | lr 9.00e-04 | grad 1.45 | tok/s 1113 | global_tok/s 17806 | elapsed_h 0.069 | time 2026-06-23T09:27:29+00:00
step    140 | loss 5.4970 | lr 9.00e-04 | grad 1.26 | tok/s 853 | global_tok/s 13643 | elapsed_h 0.101 | time 2026-06-23T09:29:26+00:00
```

Checkpoint/final status:

- Regular checkpoints were written every 10 steps after the first merge.
- Retained regular checkpoints at completion:
  `checkpoint_step_000120_loss_5.7444.pt`,
  `checkpoint_step_000130_loss_6.8076.pt`,
  `checkpoint_step_000140_loss_5.4970.pt`.
- Final checkpoint was written at step 140 after detecting that the last step
  was already a consensus merge:

```text
>>> [DiLoCo] final merge SKIPPED at step 140: last step already merged (step % K == 0); checkpoint is already consensus (avoids spurious outer-momentum double-step)
[final-checkpoint] START kind=final reason=training_complete step=140 loss=6.6665 remaining_s=1276.1 model_variant=level=E97,params=100m,mlp_ratio=1.5 rank=0/16 is_head=True
[final-checkpoint] END path=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/checkpoint_step_000140_loss_6.6665.pt latest=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt model_variant=level=E97,params=100m,mlp_ratio=1.5 rank=0/16 is_head=True
Training complete! Final step: 140
FINAL_LOSS_LAST100: 6.6665
DILOCO_MERGES: 14
DILOCO_K: 10
DILOCO_SYNC_TOTAL_S: 62.407
DILOCO_SYNC_AVG_MS: 4457.7
```

Final checkpoint path:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/checkpoint_step_000140_loss_6.6665.pt
```

`latest.pt` resolved to that final checkpoint at resume-submission time.

## Resume Canary: Job `4891149`

Run root:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891149-20260623T093140Z
```

Resume source:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt
```

The source resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/checkpoint_step_000140_loss_6.6665.pt
```

Configuration matched the regular averaging canary except:

```text
SCALEOUT_WALLTIME=00:20:00
TRAIN_MINUTES=2
RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt
```

Resume status:

- All ranks printed the resume source path.
- The run printed `Resumed at step 140`.
- Training started from step 140 and advanced to later finite loss.
- Regular post-resume checkpoints were written through step 190.

Key resume lines:

```text
[DiLoCo] broadcast rank-0 W_0 to all 16 ranks (identical start)
Resuming from /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt
Resumed at step 140
[DiLoCo] outer optimizer: avg (stateless periodic averaging)
step    145 | loss 5.6714 | lr 9.00e-04 | grad 1.26 | tok/s 1828 | global_tok/s 29247 | elapsed_h 0.002 | time 2026-06-23T09:33:07+00:00
step    150 | loss 5.5963 | lr 9.00e-04 | grad 1.10 | tok/s 1047 | global_tok/s 16757 | elapsed_h 0.004 | time 2026-06-23T09:33:17+00:00
step    190 | loss 5.5159 | lr 9.00e-04 | grad 1.46 | tok/s 1113 | global_tok/s 17804 | elapsed_h 0.029 | time 2026-06-23T09:34:47+00:00
step    195 | loss 5.5702 | lr 9.00e-04 | grad 1.12 | tok/s 752 | global_tok/s 12038 | elapsed_h 0.033 | time 2026-06-23T09:35:00+00:00
```

Post-resume checkpoint path via `latest.pt` after cancellation:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891149-20260623T093140Z/train/levelE97_100m_20260623_053238/checkpoint_step_000190_loss_5.5159.pt
```

Final-checkpoint status:

- Regular checkpoint/resume semantics passed.
- Final-checkpoint completion failed in the resume job.
- At step 197, ranks 0-7 entered finalization due to
  `peer_final_checkpoint_request`; ranks 8-15 did not print the finalization
  entry lines before the watchdog timeout.
- NCCL watchdog reported collective sequence `218` with mismatched all-reduce
  sizes: rank 0 in an `ALLREDUCE` of `NumelIn=1`, while ranks 8-15 were in an
  `ALLREDUCE` of `NumelIn=1299726652`.
- The job was cancelled after the timeout evidence to avoid wasting allocation.

Representative failure lines:

```text
[final-checkpoint] rank 0/16 entering finalization at step=197 reason=peer_final_checkpoint_request remaining_s=996.2 model_variant=level=E97,params=100m,mlp_ratio=1.5 is_head=True
[final-checkpoint] rank 7/16 entering finalization at step=197 reason=peer_final_checkpoint_request remaining_s=996.2 model_variant=level=E97,params=100m,mlp_ratio=1.5 is_head=False
[rank8]:[E623 05:45:03.725058393 ProcessGroupNCCL.cpp:704] [Rank 8] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=218, OpType=ALLREDUCE, NumelIn=1299726652, NumelOut=1299726652, Timeout(ms)=600000) ran for 600001 milliseconds before timing out.
[rank0]:[E623 05:45:04.413715744 ProcessGroupNCCL.cpp:704] [Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=218, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600001 milliseconds before timing out.
```

## Schedule-Free Outer And GDN2 Status

Schedule-free outer was not launched. It was deferred because the regular
averaging arm exposed a final-checkpoint coordination defect during the required
2-node resume canary. Launching the schedule-free outer arm before fixing that
defect would not answer the checkpoint-correctness question cheaply.

GDN2-MLP was not launched by this task. Existing GDN2 evidence remains only the
post-merge one-node control/reference from
`docs/FRONTIER_CROSS_MODEL_FINAL_CHECKPOINT_20260623.md`.

## Boundary

No workaround was implemented in the scaleout layer. No 4-node or 8-node
experiment was submitted. The next valid action is the implementation follow-up
`fix-multi-node`, focused on the final-checkpoint collective mismatch exposed
by resume job `4891149`.
