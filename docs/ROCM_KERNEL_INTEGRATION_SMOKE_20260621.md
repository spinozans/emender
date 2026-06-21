# ROCm Kernel Integration Smoke - 2026-06-21

Task: `rocm-kernel-integration-smoke`

This note records the Frontier debug-queue integration gate for `e97-MLP`,
`e97-linear-MLP`, and `gdn2-MLP`.

## Committed Smoke Commands

Reusable one-node template:

```bash
scripts/frontier/debug_smoke_one_node.slurm
```

Sequential submit helper:

```bash
DATA=/path/to/smoke_train.txt \
VAL_DATA=/path/to/smoke_val.txt \
TRAIN_MINUTES=20 \
scripts/frontier/submit_kernel_integration_smokes.sh
```

Manual variant commands:

```bash
mkdir -p logs/frontier/debug

DATA=/path/to/smoke_train.txt VAL_DATA=/path/to/smoke_val.txt \
  TRAIN_MINUTES=20 SMOKE_VARIANT=e97-MLP \
  sbatch scripts/frontier/debug_smoke_one_node.slurm

DATA=/path/to/smoke_train.txt VAL_DATA=/path/to/smoke_val.txt \
  TRAIN_MINUTES=20 SMOKE_VARIANT=e97-linear-MLP \
  sbatch scripts/frontier/debug_smoke_one_node.slurm

DATA=/path/to/smoke_train.txt VAL_DATA=/path/to/smoke_val.txt \
  TRAIN_MINUTES=20 SMOKE_VARIANT=gdn2-MLP \
  sbatch scripts/frontier/debug_smoke_one_node.slurm
```

The template maps the three variants to separate train arguments, runs a
one-rank kernel/import smoke before the eight-rank train smoke, captures git
commit, module and Python package state, `rocm-smi`, a one-rank GPU visibility
probe, stdout/stderr paths, manifest JSON, train/kernel logs, final loss or
throughput lines when available, and first actionable error lines.

## Observed Frontier State

Observed from `login04` at capture time:

- This worktree was not a verified integrated predecessor tree. Exact
  predecessor branch tips `36531c6` (`rocm-e97-mlp-port`), `6bdc650`
  (`rocm-gdn2-mlp-port`), and `050fd95` (`frontier-env-debug-recipe`) were
  not ancestors of `HEAD` at smoke-preparation time. Treat all artifacts below
  as pre-merge blocker evidence only. Rerun from the output of
  `frontier-merge-rocm-debug-main` before admitting any variant to the benchmark
  matrix.
- `sbatch`, `squeue`, and `sacct` were available.
- Existing debug-QOS job for the same user blocked sequential submission:
  `4880568 stage-commapile-mainmix`, state `RUNNING`, partition `batch`, QOS
  `debug`, account `bif148`.
- Default staged paths were absent:
  `/lustre/orion/scratch/erikgarrison/emender/data/commapile_mainmix_smoke.txt`,
  `/lustre/orion/scratch/erikgarrison/emender/data/commapile_mainmix_val_smoke.txt`,
  `/lustre/orion/scratch/erikgarrison/emender/conda/emender-rocm711`, and
  `/lustre/orion/scratch/erikgarrison/emender/src/GatedDeltaNet-2`.
- A tiny fallback data pair was written under
  `frontier_runs/debug/20260621/rocm-kernel-integration-smoke/artifacts/` only
  to make retry commands concrete. It is not benchmark-matrix data.

Because the integrated tree was not yet confirmed, the debug slot was occupied,
and the staged environment/data paths were missing, no variant job was submitted
in this attempt. This is a launch-configuration/environment/integration blocker,
not ROCm/Triton numerical evidence.

## Pass/Fail Table

| Variant | Job ID | Kernel smoke | Train smoke | Evidence | Benchmark-matrix recommendation |
| --- | --- | --- | --- | --- | --- |
| `e97-MLP` | not submitted | not run | not run | Pre-merge worktree plus debug queue occupied by job `4880568`; DATA/VAL_DATA/env paths absent. Artifact: `frontier_runs/debug/20260621/rocm-kernel-integration-smoke/e97-MLP/summaries/summary.md`. | **FAIL / do not enter** until the integrated tree runs a one-node debug job that records GPU visibility, runtime path, loss sanity, and throughput or a kernel blocker. |
| `e97-linear-MLP` | not submitted | not run | not run | Same integration, scheduler, and environment blocker. Artifact: `frontier_runs/debug/20260621/rocm-kernel-integration-smoke/e97-linear-MLP/summaries/summary.md`. | **FAIL / do not enter** until the integrated-tree run confirms `use_chunked_e97=1` on Frontier and records loss/throughput or a kernel blocker. |
| `gdn2-MLP` | not submitted | not run | not run | Same integration, scheduler, and environment blocker; additionally needs valid `GDN2_PATH`. Artifact: `frontier_runs/debug/20260621/rocm-kernel-integration-smoke/gdn2-MLP/summaries/summary.md`. | **FAIL / do not enter** until the integrated tree has the external GDN2 checkout imports and the one-rank fwdbwd preflight plus train smoke record logs. |

## Retry Boundary

Retry only after `frontier-merge-rocm-debug-main` produces the integrated tree
and the staging/debug prerequisite clears:

```bash
squeue -u "$USER" -o "%.18i %.9P %.8q %.25j %.2t %.10M %.6D %R"
```

Then provide real staged paths:

```bash
export DATA="$MEMBERWORK/emender/data/commapile_mainmix_smoke.txt"
export VAL_DATA="$MEMBERWORK/emender/data/commapile_mainmix_val_smoke.txt"
export EMENDER_CONDA_ENV="$MEMBERWORK/emender/conda/emender-rocm711"
export GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2"
scripts/frontier/submit_kernel_integration_smokes.sh
```

Readiness should be updated from the resulting `summary.md` and
`manifest.json` artifacts, not inherited from this blocked attempt.
