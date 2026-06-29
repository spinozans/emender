# Frontier Cross-Model Final Checkpoint Validation

Date: 2026-06-23
Task: `validate-cross-model-final-checkpoint`
Worktree commit: `8b4b7cf7bf65926ac788c6394e1d3f65ed479c63`
Post-merge gate: `merge-e97-walltime-checkpoint-main` pushed this commit to
`origin/main` before the GDN2 control validation below completed.

## Verdict

PASS for the current one-node Frontier launch harness checkpoint/resume gate:

- E97-MLP passed using the upstream live checkpoint/resume evidence from
  `fix-e97-mlp-checkpoint-finalization`.
- GDN2-MLP passed with fresh live checkpoint/resume jobs from integrated commit
  `8b4b7cf7bf65926ac788c6394e1d3f65ed479c63`.
- `e97-linear-MLP` remains deferred because it is explicitly quarantined from
  Frontier readiness decisions until the chunked E97 ROCm kernel blocker is fixed.
- No multi-node scaleout job was launched. All live work here used one node in
  debug QOS.

## E97-MLP Evidence

The E97 primary arm is satisfied by the immediately upstream live validation in
`docs/FRONTIER_E97_CHECKPOINT_FINALIZATION_20260623.md`. That run validated the
same final-checkpoint and resume semantics that this task checks:

- checkpoint job: `4889966`, `COMPLETED`, 1 node, debug QOS, elapsed `00:04:49`;
- resume job: `4890029`, `COMPLETED`, 1 node, debug QOS, elapsed `00:03:48`;
- checkpoint path:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/train/levelE97_100m_20260623_034728/checkpoint_step_000162_loss_0.6950.pt`;
- checkpoint `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/train/levelE97_100m_20260623_034728/latest.pt`,
  resolving to `checkpoint_step_000162_loss_0.6950.pt`;
- resume source:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/train/levelE97_100m_20260623_034728/latest.pt`;
- resume load evidence: all ranks logged `Resumed at step 162`;
- first resumed finite loss: `step 163 | loss 0.0401`;
- later resumed finite loss: `step 179 | loss 0.3122`;
- resume final checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/train/levelE97_100m_20260623_035919/checkpoint_step_000179_loss_0.1739.pt`;
- resume `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/train/levelE97_100m_20260623_035919/latest.pt`,
  resolving to `checkpoint_step_000179_loss_0.1739.pt`;
- artifacts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/logs/train.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/artifacts/manifest.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/logs/train.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/artifacts/manifest.json`.

Boundary: a redundant post-merge E97 rerun was submitted as job `4890461` after
the GDN2 resume completed, then canceled while still pending once the coordinator
confirmed that the upstream E97 evidence plus fresh GDN2 evidence was sufficient.
It consumed no node time: `CANCELLED`, elapsed `00:00:00`.

## GDN2-MLP Evidence

### Failed Preflight Attempt

Job `4890149` was the first GDN2 submission. It failed before training because
the script's default `OUTPUT_ROOT` expanded through `MEMBERWORK` to
`/lustre/orion/scratch/erikgarrison/emender/...`, which this account could not
create:

```text
mkdir: cannot create directory '/lustre/orion/scratch/erikgarrison/emender': Permission denied
```

This did not exercise model import, training, checkpointing, or resume. The
corrected jobs below used explicit `OUTPUT_ROOT=/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug`.

### Checkpoint Job

GDN2 checkpoint job:

- job id: `4890166`;
- script: `scripts/frontier/debug_smoke_one_node.slurm`;
- submit command essentials:
  `SMOKE_VARIANT=gdn2-MLP`, `TRAIN_MINUTES=1`,
  `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=120`,
  `WALLTIME_CHECK_EVERY=1`,
  `GDN2_PATH=/lustre/orion/bif148/scratch/erikgarrison/emender/src/GatedDeltaNet-2`,
  `DATA=/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt`;
- worktree commit from manifest:
  `8b4b7cf7bf65926ac788c6394e1d3f65ed479c63`;
- partition/QOS: `batch` / `debug`;
- requested nodes/walltime: 1 node / `00:30:00`;
- requested node-hours: `0.500000`;
- sacct result: `COMPLETED`, exit `0:0`, elapsed `00:02:18`;
- run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z`;
- first finite training metric:
  `step     10 | loss 9.4174 | lr 9.00e-04 | grad 7.97 | tok/s 3925 | global_tok/s 31396`;
- later finite training metrics:
  `step     20 | loss 7.0241`, `step     30 | loss 5.3633`,
  `step     40 | loss 5.2154`;
- final checkpoint log:
  `[final-checkpoint] START kind=final reason=training_complete step=41 loss=6.7550 remaining_s=1671.3 model_variant=level=gdn2-mlp,params=100m,gdn2_mlp_ratio=3.258732449079677 rank=0/8 is_head=True`;
- final checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/train/levelgdn2-mlp_100m_20260623_044300/checkpoint_step_000041_loss_6.7550.pt`;
- `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/train/levelgdn2-mlp_100m_20260623_044300/latest.pt`,
  resolving to `checkpoint_step_000041_loss_6.7550.pt`;
- checkpoint metadata inspected with the Frontier module Python:
  `step=41`, `loss=6.755042111873627`, `kind=final`,
  `reason=training_complete`, `rank=0`, `world_size=8`, `is_head=true`,
  `walltime_deadline_source=SLURM_JOB_END_TIME`,
  `walltime_margin_s=120.0`, `walltime_remaining_s=1671.290699005127`;
- artifacts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/logs/train.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/logs/kernel_smoke.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/artifacts/manifest.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/artifacts/env.txt`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/summaries/summary.md`.

### Resume Job

GDN2 resume job:

- job id: `4890189`;
- script: `scripts/frontier/debug_smoke_one_node.slurm`;
- worktree commit from manifest:
  `8b4b7cf7bf65926ac788c6394e1d3f65ed479c63`;
- partition/QOS: `batch` / `debug`;
- requested nodes/walltime: 1 node / `00:30:00`;
- requested node-hours: `0.500000`;
- sacct result: `COMPLETED`, exit `0:0`, elapsed `00:03:53`;
- resume source:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/train/levelgdn2-mlp_100m_20260623_044300/latest.pt`;
- run root:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z`;
- load evidence: all ranks logged `Resumed at step 41`;
- first resumed finite metric:
  `step     50 | loss 3.7054 | lr 9.00e-04 | grad 1.59 | tok/s 3917 | global_tok/s 31335`;
- later resumed finite metric:
  `step     60 | loss 3.6255 | lr 9.00e-04 | grad 1.54 | tok/s 3958 | global_tok/s 31660`;
- final checkpoint log:
  `[final-checkpoint] START kind=final reason=training_complete step=62 loss=3.6654 remaining_s=1579.3 model_variant=level=gdn2-mlp,params=100m,gdn2_mlp_ratio=3.258732449079677 rank=0/8 is_head=True`;
- final checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/train/levelgdn2-mlp_100m_20260623_045429/checkpoint_step_000062_loss_3.6654.pt`;
- `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/train/levelgdn2-mlp_100m_20260623_045429/latest.pt`,
  resolving to `checkpoint_step_000062_loss_3.6654.pt`;
- checkpoint metadata inspected with the Frontier module Python:
  `step=62`, `loss=3.665433883666992`, `kind=final`,
  `reason=training_complete`, `rank=0`, `world_size=8`, `is_head=true`,
  `walltime_deadline_source=SLURM_JOB_END_TIME`,
  `walltime_margin_s=120.0`, `walltime_remaining_s=1579.2813472747803`;
- artifacts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/logs/train.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/logs/kernel_smoke.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/artifacts/manifest.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/artifacts/env.txt`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/summaries/summary.md`.

## Other Runnable Variants

The launch harness currently exposes three `SMOKE_VARIANT` values:

1. `e97-MLP`: tested by upstream live checkpoint/resume jobs and accepted for
   this validation gate.
2. `gdn2-MLP`: tested here by live checkpoint/resume jobs from integrated
   commit `8b4b7cf7bf65926ac788c6394e1d3f65ed479c63`.
3. `e97-linear-MLP`: deferred. The active quarantine record is
   `docs/FRONTIER_E97_LINEAR_ROCM_QUARANTINE_20260621.md`, which states that
   `e97-linear-MLP` is quarantined from Frontier readiness decisions until a
   successor task fixes the chunked E97 ROCm kernel blocker and records new
   passing artifacts. `docs/FRONTIER_DILOCO_SCALEOUT_READINESS_20260621.md`
   also states that `e97-linear-MLP` is not a scaleout candidate, fallback
   candidate, or source of supporting evidence while the quarantine is active.

No other `SMOKE_VARIANT` branch is accepted by
`scripts/frontier/debug_smoke_one_node.slurm`.

## Allocation Boundary

Only one-node debug-QOS jobs were launched by this task:

- `4890149`: failed before training due output-root permission, elapsed
  `00:00:05`;
- `4890166`: GDN2 checkpoint validation, elapsed `00:02:18`;
- `4890189`: GDN2 resume validation, elapsed `00:03:53`;
- `4890461`: redundant E97 rerun canceled while pending, elapsed `00:00:00`.

No multi-node scaleout job was launched. The successful GDN2 evidence requested
`1.000000` node-hours and consumed approximately `0.103056` node-hours by
elapsed time. Including the failed pre-training attempt adds approximately
`0.001389` node-hours of elapsed allocation.

## Validation Commands

Commands used after the live jobs:

```text
sacct -j 4890149,4890166,4890189,4890461 --format=JobID,JobName%24,Partition,State,ExitCode,Elapsed,NNodes,AllocTRES%40 -P
readlink -f /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/train/levelgdn2-mlp_100m_20260623_044300/latest.pt
readlink -f /lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/train/levelgdn2-mlp_100m_20260623_045429/latest.pt
module load PrgEnv-gnu/8.7.0 cpe/26.03 miniforge3/23.11.0-0 rocm/7.1.1 craype-accel-amd-gfx90a
python - <<'PY'
import json, torch
for label,p in {
 'gdn2_checkpoint':'/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890166-20260623T084133Z/train/levelgdn2-mlp_100m_20260623_044300/latest.pt',
 'gdn2_resume':'/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/gdn2-MLP/4890189-20260623T085129Z/train/levelgdn2-mlp_100m_20260623_045429/latest.pt',
}.items():
    ckpt=torch.load(p,map_location='cpu')
    print(label)
    print(' step', ckpt.get('step'))
    print(' loss', ckpt.get('loss'))
    print(' checkpoint_metadata', json.dumps(ckpt.get('checkpoint_metadata'), sort_keys=True))
PY
```
