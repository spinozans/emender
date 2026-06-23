# Rerun 2-node E97 resume canary

Task: `rerun-2-node-e97`
Date: 2026-06-23

## Status

Submitted, scheduler-pending. The bounded 2-node E97-MLP regular-averaging
resume canary was submitted as Frontier job `4891298`, but it remained pending
for priority during this agent turn and produced no allocation/runtime evidence
yet. Do not treat the resume/final-checkpoint validation as passed until
`4891298` reaches a terminal state and its logs are harvested.

No 4-node or 8-node jobs were submitted.

## Merge prerequisite

Dependency context records that `merge-multi-node` landed on main:

- `origin/main=ab30fd70fd911d02a8f2d865ca9cd6c8a5c48d38`
- merge task final state: pushed to `origin/main`, evaluator passed.

This worktree was on:

```text
c1e733ccb6a6f2e4ac7bba273b4a8197eba05939
c1e733c feat: merge-multi-node (agent-113)
```

That commit contains the merge-multi-node history, including the completed
`fix-multi-node` ancestry.

## Submitted job

Job: `4891298`
Name: `emender-e97-resume-canary`
Partition/QOS: `batch` / `debug`
Nodes: `2`
Walltime: `00:20:00`
Max requested exposure: `2 nodes * 20 minutes = 0.666667 node-hours`
Actual elapsed at handoff: `00:00:00`
Actual node-hours at handoff: `0.000000`

Submission command:

```text
sbatch -A bif148 -p batch --qos=debug -N 2 -t 00:20:00 --job-name emender-e97-resume-canary --export=ALL,WG_TASK_ID=rerun-2-node-e97,TASK_ID=rerun-2-node-e97,SCALEOUT_VARIANT=e97-MLP,SCALEOUT_NODES=2,SCALEOUT_WALLTIME=00:20:00,TRAIN_MINUTES=2,DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt,VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt,TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache,DILOCO_K=10,DILOCO_OUTER_OPTIMIZER=avg,DILOCO_OUTER_LR=1.0,DILOCO_OUTER_BETA=0.0,DILOCO_EXPORT_BASIS=x,DILOCO_ISLAND_SIZE=8,BATCH_SIZE=1,CHUNK_SIZE=2048,LOG_EVERY=5,VAL_EVERY=10000,SAVE_EVERY=10,KEEP_CHECKPOINTS=4,HUMAN_APPROVAL_RECORD='WG task rerun-2-node-e97 validation: bounded 2-node avg resume canary only; max requested exposure 2 nodes * 20 min = 0.666667 node-hours',RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt scripts/frontier/diloco_scaleout_readiness.sbatch
```

## Checkpoint source

The job resumes from the clean 2-node E97-MLP avg checkpoint produced by job
`4891083`:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt
```

At submission time, `latest.pt` resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/checkpoint_step_000140_loss_6.6665.pt
```

Source file check:

```text
-rw------- 1 erikgarrison bif148 7.3G Jun 23 05:29 checkpoint_step_000140_loss_6.6665.pt
lrwxrwxrwx 1 erikgarrison bif148   37 Jun 23 05:29 latest.pt -> checkpoint_step_000140_loss_6.6665.pt
```

## Scheduler state at handoff

Captured after the job remained pending through repeated polls:

```text
JOBID              PARTITION NAME                         USER     ST TIME       NODES NODELIST(REASON)
4891298            batch     emender-e97-resume-canary    erikgarr PD 0:00       2     (Priority)
```

Accounting snapshot:

```text
JobID|JobName|Partition|QOS|State|ExitCode|Elapsed|NNodes|AllocTRES|Start|End
4891298|emender-e97-resume-canary|batch|debug|PENDING|0:0|00:00:00|2||Unknown|Unknown
```

`scontrol show job 4891298` confirmed:

```text
JobState=PENDING Reason=Priority
TimeLimit=00:20:00
NumNodes=2-2 NumCPUs=112 NumTasks=16 CPUs/Task=7
ReqTRES=cpu=112,mem=1000G,node=2,billing=112
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.err
```

## Pending harvest checklist

When `4891298` starts and exits, harvest:

- terminal `sacct` row for job id, nodes, walltime, elapsed, node-hours, and
  exit status.
- run root from `logs/frontier/scaleout/emender-e97-resume-canary-4891298.out`
  or the generated manifest.
- output path, final train log, and generated summary under the run root.
- finite post-resume losses after `Resumed at step 140`.
- final-checkpoint `START` / `END` status.
- whether `latest.pt` advances to the final post-resume checkpoint.
- absence or presence of NCCL/RCCL collective mismatch, watchdog timeout,
  non-finite loss, traceback, or OOM evidence.

If the final-checkpoint NCCL mismatch persists, create a focused follow-up with
the exact sequence-number/collective evidence and do not scale out.

## Resume poll: 2026-06-23T06:56:47-04:00

The wait condition fired after the initial 15-minute handoff, but Frontier had
not yet started the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|2||2026-06-23T06:24:52|Unknown|Unknown
```

`scontrol show job 4891298` still reported:

```text
JobState=PENDING Reason=Priority
RunTime=00:00:00 TimeLimit=00:20:00
StartTime=Unknown EndTime=Unknown
NumNodes=2-2 NumCPUs=112 NumTasks=16 CPUs/Task=7
ReqTRES=cpu=112,mem=1000G,node=2,billing=112
AllocTRES=(null)
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.err
```

The recorded stdout/stderr files did not exist yet, which is consistent with a
job that has not allocated:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. Do not submit replacement, 4-node, or 8-node jobs
unless `4891298` fails before starting or the task is explicitly re-scoped.
