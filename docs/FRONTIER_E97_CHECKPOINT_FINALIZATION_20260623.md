# Frontier E97 Checkpoint Finalization

Date: 2026-06-23
Task: `fix-e97-mlp-checkpoint-finalization`

## Failure Mode

The relevant post-cache E97 evidence is Frontier job `4881258`.
That job reached finite E97-MLP training loss, throughput, peak memory, and a
checkpoint:

- first finite loss: `step 1 | loss 11.1347 | tok/s 2972 | global_tok/s 2972`;
- last observed metric: `step 50 | loss 6.0547 | tok/s 3202 | global_tok/s 3202`;
- memory: `PEAK_MEMORY_MB: 13115`, `RESERVED_MEMORY_MB: 15504`;
- checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/train/levelE97_100m_20260621_110302/checkpoint_step_000050_loss_7.7509.pt`;
- `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260621/e97-MLP/4881258-20260621T150136Z/train/levelE97_100m_20260621_110302/latest.pt`.

The blocker was not E97 numerical instability or tokenizer download. The launch
script started eight Slurm tasks without exporting `RANK`, `WORLD_SIZE`, and a
rank-local `LOCAL_RANK`. `train.py` therefore treated every task as a
single-process rank 0, so all tasks entered checkpoint/finalization ownership
paths and raced on the same checkpoint directory and `latest.pt`.

## Fix

The implementation has two parts:

1. `train.py` now derives missing distributed environment variables from Slurm
   before deciding `dist_enabled` and rank-0 ownership. When `SLURM_NTASKS > 1`
   and `WORLD_SIZE` is not exported, it fills `WORLD_SIZE=SLURM_NTASKS`,
   `RANK=SLURM_PROCID`, and `LOCAL_RANK=0` when each task sees one rank-local
   GPU.
2. Checkpoint finalization now writes checkpoints through a temporary file and
   atomically replaces both the checkpoint target and the `latest.pt` symlink.
   Distributed runs still save only from rank 0; the atomic update prevents
   readers from observing a missing or half-updated latest pointer.

The remaining Frontier templates that launch with `--gpus-per-task=1
--gpu-bind=closest` were also aligned to export `LOCAL_RANK=0`, matching the
already-correct one-node debug smoke template.

## Local Validation

Focused local validation under Frontier's module Python:

```text
python -m pytest tests/test_checkpoint_finalization.py \
  tests/test_diloco_merge.py::test_diloco_checkpoint_roundtrip_preserves_outer_and_inner_sf_state \
  tests/test_rocm_e97_runtime_config.py -q
```

Result: `7 passed`; latest rerun completed in `20.37s`.

## Live Frontier Validation

### Checkpoint Job

One-node E97-MLP debug job after the fix:

- job id: `4889966`;
- script: `scripts/frontier/debug_smoke_one_node.slurm`;
- partition/QOS: `batch` / `debug`;
- requested nodes/walltime: 1 node / 00:30:00;
- requested node-hours: 0.500000;
- elapsed: 00:04:49;
- consumed node-hours: 0.080278;
- data:
  `/lustre/orion/bif148/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt`;
- tokenizer cache:
  `/lustre/orion/bif148/proj-shared/tiktoken_cache`;
- state: `COMPLETED`, exit `0:0`;
- first finite loss:
  `step 1 | loss 11.1347 | tok/s 2445 | global_tok/s 19564`;
- final finite training line:
  `step 162 | loss 0.0708 | tok/s 2622 | global_tok/s 20973`;
- final summary: `Training complete! Final step: 162`,
  `FINAL_LOSS_LAST100: 0.6950`, `PEAK_MEMORY_MB: 15586`;
- checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/train/levelE97_100m_20260623_034728/checkpoint_step_000162_loss_0.6950.pt`;
- `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/train/levelE97_100m_20260623_034728/latest.pt`,
  resolving to `checkpoint_step_000162_loss_0.6950.pt`;
- logs/artifacts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/logs/train.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/logs/kernel_smoke.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/artifacts/manifest.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/summaries/summary.md`.

### Resume Job

One-node E97-MLP resume job from the checkpoint above:

- job id: `4890029`;
- script: `scripts/frontier/debug_smoke_one_node.slurm`;
- partition/QOS: `batch` / `debug`;
- requested nodes/walltime: 1 node / 00:30:00;
- requested node-hours: 0.500000;
- elapsed: 00:03:48;
- consumed node-hours: 0.063333;
- state: `COMPLETED`, exit `0:0`;
- resume checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4889966-20260623T074559Z/train/levelE97_100m_20260623_034728/latest.pt`;
- load evidence: `Resumed at step 162` appeared on all ranks;
- first resumed finite loss:
  `step 163 | loss 0.0401 | tok/s 589 | global_tok/s 4712`;
- later resumed finite loss:
  `step 179 | loss 0.3122 | tok/s 2631 | global_tok/s 21052`;
- final summary: `Training complete! Final step: 179`,
  `FINAL_LOSS_LAST100: 0.1739`, `PEAK_MEMORY_MB: 15595`;
- resume checkpoint:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/train/levelE97_100m_20260623_035919/checkpoint_step_000179_loss_0.1739.pt`;
- resume `latest.pt`:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/train/levelE97_100m_20260623_035919/latest.pt`,
  resolving to `checkpoint_step_000179_loss_0.1739.pt`;
- logs/artifacts:
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/logs/train.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/logs/kernel_smoke.log`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/artifacts/manifest.json`,
  `/lustre/orion/bif148/proj-shared/emender/frontier_runs/debug/20260623/e97-MLP/4890029-20260623T075622Z/summaries/summary.md`.

### Accounting and Boundary

Successful checkpoint/resume validation requested 1.000000 node-hours and
consumed 0.143611 node-hours.

An earlier live attempt, job `4889715`, reached finite E97 loss through step 50
but included optional validation (`VAL_DATA`, `val_every=50`) and stalled before
checkpoint finalization. It was cancelled at 00:08:10 to avoid unnecessary
debug allocation burn. It requested 0.500000 node-hours and consumed 0.136111
node-hours. This was not used as checkpoint/resume success evidence.

No GDN2-MLP control was launched for this task. Existing GDN2 checkpoint/resume
evidence from prior tasks was sufficient as a reference, and the task-specific
fix was validated directly on E97-MLP.

Remaining blocker: none for single-node E97-MLP checkpoint finalization and
resume. Optional rank-0 validation under DDP may deserve a separate follow-up
if that flow is needed, but it did not affect the checkpoint-only gate.
