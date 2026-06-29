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

## Resume poll: 2026-06-23T07:13:43-04:00

The second 15-minute wait condition fired, but Frontier still had not started
the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T07:30:22-04:00

The third 15-minute wait condition fired, but Frontier still had not started
the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned||2026-06-23T06:24:52|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T07:46:54-04:00

The fourth 15-minute wait condition fired, but Frontier still had not started
the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned||2026-06-23T06:24:52|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T08:03:39-04:00

The fifth 15-minute wait condition fired, but Frontier still had not started
the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned||2026-06-23T06:24:52|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T08:21:50-04:00

The sixth 15-minute wait condition fired, but Frontier still had not started
the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID STATE TIME NODES NODELIST(REASON)
4891298 PENDING 0:00 2 (Priority)
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|AllocTRES|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|2||Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T08:41:17-04:00

The seventh post-submission wait condition fired, but Frontier still had not
started the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LEFT|NODES|NODELIST(REASON)
4891298|PENDING|0:00|20:00|2|(Priority)
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T08:59:10-04:00

The eighth post-submission wait condition fired, but Frontier still had not
started the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LEFT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned||2026-06-23T06:24:52|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T09:18:14-04:00

The ninth post-submission wait condition fired, but Frontier still had not
started the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned||2026-06-23T06:24:52|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T09:36:14-04:00

The tenth post-submission wait condition fired, but Frontier still had not
started the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|NODES|NODELIST(REASON)|START_TIME|TIME_LIMIT|USER|PARTITION
4891298|PENDING|0:00|2|(Priority)|N/A|20:00|erikgarrison|batch
```

Accounting snapshot:

```text
JobIDRaw|JobName|State|ExitCode|Elapsed|NNodes|NodeList|Submit|Start|End|Timelimit
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|2|None assigned|2026-06-23T06:24:52|Unknown|Unknown|00:20:00
```

`scontrol show job 4891298` still reported:

```text
JobState=PENDING Reason=Priority
RunTime=00:00:00 TimeLimit=00:20:00
StartTime=Unknown EndTime=Unknown
LastSchedEval=2026-06-23T09:35:52
NumNodes=2-2 NumCPUs=112 NumTasks=16 CPUs/Task=7
ReqTRES=cpu=112,mem=1000G,node=2,billing=112
AllocTRES=(null)
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.err
```

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T09:54:03-04:00

The eleventh post-submission wait condition fired, but Frontier still had not
started the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|NODES|NODELIST(REASON)|START_TIME|TIME_LIMIT|USER|PARTITION
4891298|PENDING|0:00|2|(Priority)|N/A|20:00|erikgarrison|batch
```

Accounting snapshot:

```text
JobIDRaw|JobName|State|ExitCode|Elapsed|NNodes|NodeList|Submit|Start|End|Timelimit
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|2|None assigned|2026-06-23T06:24:52|Unknown|Unknown|00:20:00
```

`scontrol show job 4891298` still reported:

```text
JobState=PENDING Reason=Priority
RunTime=00:00:00 TimeLimit=00:20:00
StartTime=Unknown EndTime=Unknown
LastSchedEval=2026-06-23T09:53:23
NumNodes=2-2 NumCPUs=112 NumTasks=16 CPUs/Task=7
ReqTRES=cpu=112,mem=1000G,node=2,billing=112
AllocTRES=(null)
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.err
```

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T10:12:20-04:00

The twelfth post-submission wait condition fired, but Frontier still had not
started the bounded 2-node canary.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned|2026-06-23T06:24:52|Unknown|Unknown
```

`scontrol show job 4891298` still reported:

```text
JobState=PENDING Reason=Priority
RunTime=00:00:00 TimeLimit=00:20:00
StartTime=Unknown EndTime=Unknown
LastSchedEval=2026-06-23T10:06:08
NumNodes=2-2 NumCPUs=112 NumTasks=16 CPUs/Task=7
ReqTRES=cpu=112,mem=1000G,node=2,billing=112
AllocTRES=(null)
QOS=debug Partition=batch
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.err
SubmitLine=sbatch -A bif148 -p batch --qos=debug -N 2 -t 00:20:00 --job-name emender-e97-resume-canary ... RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt scripts/frontier/diloco_scaleout_readiness.sbatch
```

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending. No replacement job, 4-node job, or 8-node job was
submitted during this poll.

## Resume poll: 2026-06-23T10:31:17-04:00

The thirteenth resume poll found the bounded 2-node canary still waiting in
the scheduler with no allocation or runtime. The submitted job remains the only
job from this task; no replacement, 4-node, or 8-node jobs were submitted.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891298|PENDING|0:00|20:00|2|(Priority)|N/A|2026-06-23T06:24:52
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned||2026-06-23T06:24:52|Unknown|Unknown
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

The recorded stdout/stderr files still did not exist:

```text
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.out': No such file or directory
ls: cannot access 'logs/frontier/scaleout/emender-e97-resume-canary-4891298.err': No such file or directory
```

Validation remains pending because there are no post-resume losses, final
checkpoint records, `latest.pt` updates, or finalization/NCCL logs to inspect
until the job starts and exits.

## Resume poll: 2026-06-23T10:51:03-04:00

The fourteenth resume poll found the bounded 2-node canary still pending in
the scheduler with no allocation and no runtime. Job `4891298` remains the only
submitted job for this task; no replacement, 4-node, or 8-node jobs were
submitted during this poll.

Current scheduler snapshot:

```text
JOBID|ST|TIME|NODES|NODELIST(REASON)|START_TIME|TIME_LEFT
4891298|PD|0:00|2|(Priority)|N/A|20:00
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|AllocTRES|Submit|Start|End
4891298|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2||2026-06-23T06:24:52|Unknown|Unknown
```

`scontrol show job 4891298` reported:

```text
JobState=PENDING Reason=Priority
RunTime=00:00:00 TimeLimit=00:20:00
StartTime=Unknown EndTime=Unknown
LastSchedEval=2026-06-23T10:19:40
NumNodes=2-2 NumCPUs=112 NumTasks=16 CPUs/Task=7
ReqTRES=cpu=112,mem=1000G,node=2,billing=112
AllocTRES=(null)
QOS=debug Partition=batch
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891298.err
SubmitLine=sbatch -A bif148 -p batch --qos=debug -N 2 -t 00:20:00 --job-name emender-e97-resume-canary ... RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt scripts/frontier/diloco_scaleout_readiness.sbatch
```

Validation remains pending because the canary has not started and therefore
has no post-resume losses, final-checkpoint status, `latest.pt` behavior, clean
exit status, or NCCL finalization logs to inspect.

## Failed launch harvest: 2026-06-23T11:03:00-04:00

The original bounded 2-node canary job `4891298` reached a terminal state, but
it did not launch training. This is recorded as a failed launch attempt, not as
evidence for or against the multi-node final-checkpoint fix.

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|NodeList|Submit|Start|End
4891298|emender-e97-resume-canary|FAILED|1:0|00:00:06|2|frontier[07171,07200]|2026-06-23T06:24:52|2026-06-23T10:59:46|2026-06-23T10:59:52
4891298.batch|batch|FAILED|1:0|00:00:06|1|frontier07171|2026-06-23T10:59:46|2026-06-23T10:59:46|2026-06-23T10:59:52
4891298.extern|extern|COMPLETED|0:0|00:00:06|2|frontier[07171,07200]|2026-06-23T10:59:46|2026-06-23T10:59:46|2026-06-23T10:59:52
```

The stdout file exists but is empty:

```text
logs/frontier/scaleout/emender-e97-resume-canary-4891298.out 0 bytes
```

The stderr file contains only the default `OUTPUT_ROOT` permission failure:

```text
mkdir: cannot create directory '/lustre/orion/scratch/erikgarrison/emender': Permission denied
mkdir: cannot create directory '/lustre/orion/scratch/erikgarrison/emender': Permission denied
mkdir: cannot create directory '/lustre/orion/scratch/erikgarrison/emender': Permission denied
```

Conclusion: `4891298` failed before writing the run manifest, environment
record, post-resume losses, final-checkpoint records, or `latest.pt` updates.
No NCCL collective mismatch, timeout, traceback, non-finite loss, or finalizer
evidence was produced.

## Replacement submission: job 4891784

Submitted replacement bounded 2-node E97-MLP regular-averaging resume canary
job `4891784` with the same checkpoint and exposure limits, plus an explicit
writable output root:

```text
OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout
```

Job: `4891784`
Name: `emender-e97-resume-canary`
Partition/QOS: `batch` / `debug`
Nodes: `2`
Walltime: `00:20:00`
Max requested exposure: `2 nodes * 20 minutes = 0.666667 node-hours`
Actual elapsed at submission handoff: `00:00:00`
Actual node-hours at submission handoff: `0.000000`

Submission command:

```text
sbatch -A bif148 -p batch --qos=debug -N 2 -t 00:20:00 --job-name emender-e97-resume-canary --export=ALL,WG_TASK_ID=rerun-2-node-e97,TASK_ID=rerun-2-node-e97,SCALEOUT_VARIANT=e97-MLP,SCALEOUT_NODES=2,SCALEOUT_WALLTIME=00:20:00,TRAIN_MINUTES=2,OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout,DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt,VAL_DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt,TIKTOKEN_CACHE_DIR=/lustre/orion/bif148/proj-shared/tiktoken_cache,DILOCO_K=10,DILOCO_OUTER_OPTIMIZER=avg,DILOCO_OUTER_LR=1.0,DILOCO_OUTER_BETA=0.0,DILOCO_EXPORT_BASIS=x,DILOCO_ISLAND_SIZE=8,BATCH_SIZE=1,CHUNK_SIZE=2048,LOG_EVERY=5,VAL_EVERY=10000,SAVE_EVERY=10,KEEP_CHECKPOINTS=4,HUMAN_APPROVAL_RECORD='WG task rerun-2-node-e97 validation retry: bounded 2-node avg resume canary only after job 4891298 launch failure; explicit OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout; max requested exposure 2 nodes * 20 min = 0.666667 node-hours',RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt scripts/frontier/diloco_scaleout_readiness.sbatch
```

Checkpoint source remains the clean 2-node E97-MLP avg checkpoint from job
`4891083`:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt
```

At replacement submission time, `latest.pt` still resolved to:

```text
/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/checkpoint_step_000140_loss_6.6665.pt
```

Source file check:

```text
-rw------- 1 erikgarrison bif148 7798508283 Jun 23 05:29 /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/checkpoint_step_000140_loss_6.6665.pt
lrwxrwxrwx 1 erikgarrison bif148         37 Jun 23 05:29 /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt -> checkpoint_step_000140_loss_6.6665.pt
```

Current scheduler snapshot immediately after replacement submission:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891784|PENDING|0:00|20:00|2|(Resources)|N/A|2026-06-23T11:05:18
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|Submit|Start|End
4891784|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned|2026-06-23T11:05:18|Unknown|Unknown
```

`scontrol show job 4891784` reported:

```text
JobState=PENDING Reason=Resources
RunTime=00:00:00 TimeLimit=00:20:00
StartTime=Unknown EndTime=Unknown
NumNodes=2-2 NumCPUs=112 NumTasks=16 CPUs/Task=7
ReqTRES=cpu=112,mem=1000G,node=2,billing=112
AllocTRES=(null)
StdOut=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891784.out
StdErr=/lustre/orion/bif148/scratch/erikgarrison/emender/.wg-worktrees/agent-117/logs/frontier/scaleout/emender-e97-resume-canary-4891784.err
SubmitLine=sbatch -A bif148 -p batch --qos=debug -N 2 -t 00:20:00 --job-name emender-e97-resume-canary ... OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout ... RESUME_CHECKPOINT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891083-20260623T092053Z/train/levelE97_100m_20260623_052300/latest.pt scripts/frontier/diloco_scaleout_readiness.sbatch
```

No stdout/stderr files exist yet for `4891784`, which is consistent with a
job that has not allocated.

Validation remains pending. When `4891784` starts and exits, harvest the run
root, post-resume losses, final-checkpoint `START` / `END` status,
`latest.pt` behavior, clean exit status, and finalization/NCCL evidence. No
4-node or 8-node jobs were submitted.

## Replacement poll: 2026-06-23T11:11:00-04:00

Job `4891784` remains pending with no allocation or runtime evidence yet. No
stdout/stderr files exist for the replacement job.

Current scheduler snapshot:

```text
JOBID|STATE|TIME|TIME_LIMIT|NODES|NODELIST(REASON)|START_TIME|SUBMIT_TIME
4891784|PENDING|0:00|20:00|2|(Priority)|2026-06-23T11:56:00|2026-06-23T11:05:18
```

Accounting snapshot:

```text
JobID|JobName|State|ExitCode|Elapsed|Timelimit|NNodes|NodeList|Submit|Start|End
4891784|emender-e97-resume-canary|PENDING|0:0|00:00:00|00:20:00|2|None assigned|2026-06-23T11:05:18|Unknown|Unknown
```

Validation remains pending because there are still no post-resume losses,
final-checkpoint records, `latest.pt` updates, clean exit status, or NCCL
finalization logs to inspect.

## Replacement terminal harvest: 2026-06-23T11:27:22-04:00

Job `4891784` allocated and ran the bounded 2-node E97-MLP regular-averaging
resume canary from the clean `4891083` checkpoint with explicit writable
`OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout`.

Important git state nuance: the submitted job used worktree commit
`e397a1a9b450335395de16b9dbbc29c67cb6e2b9`, recorded in the run manifest. That
commit contains the multi-node final-checkpoint code merge commit `c1e733c`
and has no `train.py` / `tests/test_walltime_final_checkpoint.py` diff from
`origin/main=ab30fd70fd911d02a8f2d865ca9cd6c8a5c48d38`; it is only missing the
later `logs/merge-multi-node.md` documentation commit from `origin/main`.

Accounting:

```text
JobID|JobName|State|ExitCode|Elapsed|NNodes|NodeList|Submit|Start|End
4891784|emender-e97-resume-canary|FAILED|137:0|00:15:32|2|frontier[07200,07210]|2026-06-23T11:05:18|2026-06-23T11:11:50|2026-06-23T11:27:22
4891784.batch|batch|FAILED|137:0|00:15:32|1|frontier07200|2026-06-23T11:11:50|2026-06-23T11:11:50|2026-06-23T11:27:22
4891784.extern|extern|COMPLETED|0:0|00:15:32|2|frontier[07200,07210]|2026-06-23T11:11:50|2026-06-23T11:11:50|2026-06-23T11:27:22
4891784.0|bash|CANCELLED|0:6|00:14:58|2|frontier[07200,07210]|2026-06-23T11:12:24|2026-06-23T11:12:24|2026-06-23T11:27:22
```

Actual node exposure:

```text
2 nodes * 15.533333 minutes = 0.517778 node-hours
```

Run output paths:

```text
Slurm stdout: logs/frontier/scaleout/emender-e97-resume-canary-4891784.out
Slurm stderr: logs/frontier/scaleout/emender-e97-resume-canary-4891784.err
Run root: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891784-20260623T151204Z
Train directory: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891784-20260623T151204Z/train/levelE97_100m_20260623_111301
Manifest: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891784-20260623T151204Z/artifacts/manifest.json
Summary: /lustre/orion/bif148/proj-shared/emender/frontier_runs/scaleout/20260623/e97-MLP/4891784-20260623T151204Z/summaries/summary.md
```

Finite post-resume losses were observed after resuming from step `140`:

```text
step 145 loss 5.1694
step 150 loss 5.1284
step 155 loss 4.3432
step 160 loss 4.4035
step 165 loss 3.6829
step 170 loss 3.4195
step 175 loss 2.5726
step 180 loss 3.4492
step 185 loss 2.8669
step 190 loss 2.6149
step 195 loss 1.8267
step 200 loss 1.6622
step 205 loss 1.1961
```

Regular DiLoCo averaging completed through merge `#6` at step `200`, with
checkpoints saved at steps `150`, `160`, `170`, `180`, `190`, and `200`.

Final-checkpoint status:

```text
[final-checkpoint] armed: source=SLURM_JOB_END_TIME remaining_s=1128.9 margin_s=600.0 check_every=1 model_variant=level=E97,params=100m,mlp_ratio=1.5 head_rank=0/16
```

No final-checkpoint `START`, `END`, or success record appears before the NCCL
watchdog failure. The final-checkpoint path therefore did not complete cleanly.

`latest.pt` behavior in the replacement run:

```text
latest.pt -> checkpoint_step_000200_loss_1.6622.pt
```

The train directory retained only the last four checkpoints under
`KEEP_CHECKPOINTS=4`:

```text
checkpoint_step_000170_loss_3.4195.pt
checkpoint_step_000180_loss_3.4492.pt
checkpoint_step_000190_loss_2.6149.pt
checkpoint_step_000200_loss_1.6622.pt
```

Exact failure evidence from Slurm stdout shows the bug persists as a
finalization/collective mismatch or hang class, not as a launch environment
failure:

```text
[rank14]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank15]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank12]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank13]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank8]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank9]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank11]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank10]: Watchdog caught collective operation timeout: WorkNCCL(SeqNum=167, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
[rank14]: failure detected by watchdog at work sequence id: 167 PG status: last enqueued work: 167, last completed work: 166
[rank2]: Received a dump signal due to a collective timeout from rank 10 ... Last enqueued NCCL work: 166, last completed NCCL work: 166.
[rank0]: Received a dump signal due to a collective timeout from rank 10 ... Last enqueued NCCL work: 166, last completed NCCL work: 166.
srun: error: frontier07210: task 13: Aborted (core dumped)
srun: error: frontier07210: tasks 8-12,14-15: Aborted (core dumped)
srun: error: frontier07200: tasks 0-7: Killed
```

The `stderr` file also contains shell `command not found` lines from the batch
script's post-run summary block after the failed step:

```text
/var/spool/slurmd/job4891784/slurm_script: line 310: rerun-2-node-e97: command not found
/var/spool/slurmd/job4891784/slurm_script: line 310: e97-MLP: command not found
/var/spool/slurmd/job4891784/slurm_script: line 310: 4891784: command not found
/var/spool/slurmd/job4891784/slurm_script: line 310: 2: command not found
/var/spool/slurmd/job4891784/slurm_script: line 310: 0.666667: command not found
/var/spool/slurmd/job4891784/slurm_script: line 310: WG: command not found
/var/spool/slurmd/job4891784/slurm_script: line 310: 137: command not found
```

These summary-block errors occurred after the NCCL task failure and do not
explain the collective timeout, but they are recorded for the follow-up.

Validation conclusion: the bounded 2-node E97-MLP avg resume canary did not
pass. It produced finite post-resume losses and checkpoints through step `200`,
then failed at finalization/shutdown with NCCL `ALLREDUCE` timeouts and no clean
exit. Per task instructions, no 4-node or 8-node jobs were submitted; a focused
follow-up was created instead of scaling out.
