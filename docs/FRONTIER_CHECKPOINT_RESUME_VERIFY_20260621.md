# Frontier Selected Arm Checkpoint/Resume Verification - 2026-06-21

Task: `verify-selected-frontier`

## Result

`gdn2-MLP` is the selected arm for this checkpoint/resume verification.
The verification passed on one Frontier node:

- checkpoint source job: `4881374`, `batch` partition, `debug` QOS, 1 node,
  00:30:00 requested, 00:05:32 elapsed;
- resume job: `4881380`, `batch` partition, `debug` QOS, 1 node, 00:20:00
  requested, 00:01:21 elapsed;
- checkpoint written:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/checkpoint_step_000262_loss_0.0766.pt`;
- `latest.pt` written and resolves to that checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/latest.pt`;
- resume loaded that `latest.pt`, printed `Resumed at step 262`, and reached
  later finite training losses at steps 263-267.

## Selected Arm Justification

Observed post-cache evidence selected `gdn2-MLP`, not a prior interpretation:

1. The post-cache shared tokenizer cache existed at
   `/lustre/orion/bif148/proj-shared/tiktoken_cache` and job `4881374`
   reached tokenizer initialization without the previous `p50k_base` download
   blocker.
2. Job `4881374` passed the GDN2 external fused-path guard on all ranks. The
   train log records the GDN2 checkout and fused chunk-op path:
   `/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py`.
3. Job `4881374` recorded finite training loss and throughput after corrected
   Frontier rank mapping, for example:
   `step 1 | loss 11.2562 | ... | global_tok/s 30911` and later
   `step 262 | loss 0.0363 | ... | global_tok/s 34870`.
4. The same job completed and wrote a checkpoint plus `latest.pt`.

The e97 path is not selected here. It did reach finite loss/throughput in job
`4881176`, but that generic debug-template run was cancelled after the train log
stopped at step 50 and no checkpoint or `latest.pt` appeared. That is useful
failure evidence for the debug template, not a checkpoint/resume pass.

## Checkpoint Job Evidence

Checkpoint-producing job:

| Field | Value |
| --- | --- |
| Job ID | `4881374` |
| WG source | `stage-p50k-cache` produced the post-cache debug job; this task uses its checkpoint as selected-arm evidence |
| Partition/QOS | `batch` / `debug` |
| Nodes | 1 |
| Requested walltime | 00:30:00 |
| Requested node-hours | 0.500000 |
| Elapsed | 00:05:32 |
| Consumed node-hours | 0.092222 |
| Start | 2026-06-21T11:20:27-04:00 (`sacct`; 2026-06-21T15:20:27Z) |
| End | 2026-06-21T11:25:59-04:00 (`sacct`; 2026-06-21T15:25:59Z) |
| State | `COMPLETED`, exit `0:0` |
| Variant | `gdn2-MLP` |
| Data | `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt` |
| Tokenizer cache | `/lustre/orion/bif148/proj-shared/tiktoken_cache` |
| Run root | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z` |

Observed checkpoint artifacts:

- checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/checkpoint_step_000262_loss_0.0766.pt`
  (`7,720,565,571` bytes);
- `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/latest.pt`
  (`37` bytes symlink, resolving to the checkpoint above);
- train log:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/logs/train.log`;
- manifest:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/artifacts/manifest.json`;
- scheduler stdout/stderr:
  `/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-54/logs/frontier/debug/emender-smoke-4881374.out`,
  `/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-54/logs/frontier/debug/emender-smoke-4881374.err`.

Relevant observed train-log lines:

```text
step    262 | loss 0.0363 | lr 9.00e-04 | grad 0.15 | tok/s 4359 | global_tok/s 34870 | elapsed_h 0.067 | time 2026-06-21T15:25:48+00:00
Training complete! Final step: 262
FINAL_LOSS_LAST100: 0.0766
PEAK_MEMORY_MB: 16847
RESERVED_MEMORY_MB: 17686
```

## Resume Job Evidence

Resume job:

| Field | Value |
| --- | --- |
| Job ID | `4881380` |
| Partition/QOS | `batch` / `debug` |
| Nodes | 1 |
| Requested walltime | 00:20:00 |
| Requested node-hours | 0.333333 |
| Elapsed | 00:01:21 |
| Consumed node-hours | 0.022500 |
| Start | 2026-06-21T11:27:07-04:00 (`sacct`; 2026-06-21T15:27:07Z) |
| End | 2026-06-21T11:28:28-04:00 (`sacct`; 2026-06-21T15:28:28Z) |
| State | `COMPLETED`, exit `0:0` |
| Variant | `gdn2-MLP` |
| Resume checkpoint | `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/latest.pt` |
| Run root | `frontier_runs/debug/20260621/gdn2-MLP/resume-20260621T152703Z` |

Observed resume artifacts:

- train log:
  `frontier_runs/debug/20260621/gdn2-MLP/resume-20260621T152703Z/logs/train.log`;
- env capture:
  `frontier_runs/debug/20260621/gdn2-MLP/resume-20260621T152703Z/logs/env.txt`;
- resume checkpoint written by the resume run:
  `frontier_runs/debug/20260621/gdn2-MLP/resume-20260621T152703Z/train/levelgdn2-mlp_100m_20260621_112753/checkpoint_step_000267_loss_0.0415.pt`;
- resume `latest.pt`:
  `frontier_runs/debug/20260621/gdn2-MLP/resume-20260621T152703Z/train/levelgdn2-mlp_100m_20260621_112753/latest.pt`;
- scheduler stdout/stderr:
  `logs/frontier/debug/gdn2-resume-4881380.out`,
  `logs/frontier/debug/gdn2-resume-4881380.err`.

Relevant observed resume lines:

```text
Resuming from /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/gdn2-MLP/4881374-20260621T152028Z/train/levelgdn2-mlp_100m_20260621_112210/latest.pt
Resumed at step 262
step    263 | loss 0.0383 | lr 9.00e-04 | grad 0.17 | tok/s 539 | global_tok/s 4314 | elapsed_h 0.016 | time 2026-06-21T15:28:16+00:00
step    267 | loss 0.0498 | lr 9.00e-04 | grad 0.17 | tok/s 4414 | global_tok/s 35309 | elapsed_h 0.017 | time 2026-06-21T15:28:18+00:00
Training complete! Final step: 267
FINAL_LOSS_LAST100: 0.0415
PEAK_MEMORY_MB: 16911
RESERVED_MEMORY_MB: 17934
```

This satisfies the resume criterion: the job resumed from `latest.pt` at step
262 and reached later finite losses at steps 263-267.

## Allocation Accounting

Rows added to `docs/FRONTIER_ALLOCATION_LEDGER_20260621.md` account for the two
successful jobs used for this verification:

| Run | Requested node-hours | Consumed node-hours |
| --- | ---: | ---: |
| `CHECKPOINT-RESUME-20260621-GDN2-CKPT` (`4881374`) | 0.500000 | 0.092222 |
| `CHECKPOINT-RESUME-20260621-GDN2-RESUME` (`4881380`) | 0.333333 | 0.022500 |
| Total | 0.833333 | 0.114722 |

Ledger remaining allocation moves from `19,999.802776` to `19,999.688054` for
these two successful verification jobs while holding the `4,928` node-hour
reserve. Earlier setup attempts observed during this task consumed additional
debug time but are not counted in these two success rows because they did not
produce the selected checkpoint/resume evidence:

- `4881173`: `gdn2-MLP`, failed before training because default `GDN2_PATH`
  pointed at a missing checkout path;
- `4881176`: `e97-MLP`, reached finite loss/throughput but was cancelled after
  no checkpoint or `latest.pt` appeared;
- `4881247`: custom DDP attempt failed because `LOCAL_RANK` was mapped to
  `SLURM_LOCALID` under one-visible-GPU-per-task binding.

## Evidence Boundary

Evidence in this report is limited to observed scheduler accounting, logs, and
filesystem artifacts. It does not claim that `gdn2-MLP` is better than `e97-MLP`
as a model. It only establishes that, after the shared tokenizer cache and
corrected GDN2/Slurm rank mapping, `gdn2-MLP` produced finite one-node loss and
throughput, wrote a checkpoint plus `latest.pt`, and resumed from that
checkpoint to later finite loss.

Hypotheses that remain unverified:

- whether this result is representative beyond the 1 MiB smoke dataset;
- whether the same checkpoint/resume path remains stable in 8/16/32-node
  DiLoCo scaleout;
- whether e97 can pass the same checkpoint/resume gate after its debug template
  finalization behavior is repaired.

