# Frontier Walltime Final Checkpoint

Date: 2026-06-23
Task: `implement-walltime-final-checkpoint`

## Implementation

`train.py` now has a model-agnostic final checkpoint controller. It arms itself
from `--walltime_minutes`, `SLURM_JOB_END_TIME`, or `SLURM_TIMELIMIT`, and also
handles `SIGTERM`, `SIGINT`, `SIGHUP`, and `SIGUSR1` as graceful-shutdown
requests. The normal finalization path remains shared by all model families:
DiLoCo ranks perform the existing final consensus merge when needed, then only
rank 0 writes the authoritative checkpoint and updates `latest.pt`.

New CLI controls:

- `--walltime_minutes`: optional explicit scheduler budget in minutes.
- `--walltime_final_checkpoint_margin_seconds`: remaining-time threshold for
  stopping training and entering final checkpoint finalization. Default: 600.
- `--walltime_check_every`: optimizer-step interval for coordinated checks when
  walltime/shutdown detection is active. Default: 1.
- `--disable_walltime_final_checkpoint`: disables walltime deadline arming while
  leaving signal-triggered graceful checkpointing active.

Checkpoint files now include `checkpoint_metadata` with `kind`, `reason`,
`model_variant`, `rank`, `world_size`, `is_head`, walltime source/remaining
time, and shutdown-signal details. `latest.pt` is refreshed after an atomic
temporary-file save.

## Frontier Launch Coverage

The Frontier SLURM templates pass the new controls:

- `scripts/frontier/debug_smoke_one_node.slurm` uses
  `WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS` and `WALLTIME_CHECK_EVERY`.
- `scripts/frontier/e97_extended_64x24.sbatch` uses the same knobs, defaulting
  to a larger 900 second margin for extended jobs.

The debug template covers the runnable Frontier variants currently exposed by
that harness: `e97-MLP`, `e97-linear-MLP`, and `gdn2-MLP`. All use the same
`train.py` finalization path; no model-specific checkpoint adapter is required.

## Validation Status

Local syntax validation passed:

```bash
python3 -m py_compile train.py tests/test_walltime_final_checkpoint.py
```

Focused unit tests were added in `tests/test_walltime_final_checkpoint.py`, but
this worker shell did not have project test dependencies installed:

```text
/usr/bin/python3: No module named pytest
ModuleNotFoundError: No module named 'torch'
```

Live Frontier short-walltime E97-MLP and GDN2-MLP control runs were not launched
from this implementation worker because the active login-node Python environment
lacked `torch`/`pytest` and no `EMENDER_CONDA_ENV` was configured in the task
context. The exact intended smoke path is:

```bash
mkdir -p logs/frontier/debug
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt \
TRAIN_MINUTES=999999 \
WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=300 \
SMOKE_VARIANT=e97-MLP \
sbatch scripts/frontier/debug_smoke_one_node.slurm
```

For the GDN2 control:

```bash
mkdir -p logs/frontier/debug
DATA=/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt \
TRAIN_MINUTES=999999 \
WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS=300 \
SMOKE_VARIANT=gdn2-MLP \
sbatch scripts/frontier/debug_smoke_one_node.slurm
```

Downstream validation should assert the logs contain:

- `[final-checkpoint] armed`
- `entering finalization`
- `[final-checkpoint] START`
- `[final-checkpoint] END path=... latest=...`
- model variant and rank/head identity
- `checkpoint_metadata.reason` beginning with `walltime:` or `signal:`
- successful resume from `latest.pt` with finite post-load loss
