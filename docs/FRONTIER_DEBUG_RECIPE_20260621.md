# Frontier Debug Queue Recipe - 2026-06-21

Task: `frontier-env-debug-recipe`

This recipe is the minimal Frontier launch path for short ROCm/Triton smoke
validation of `e97-MLP`, `e97-linear-MLP`, and `gdn2-MLP`. It is intentionally
neutral: a smoke result is evidence about a specific job/environment, not a
go/no-go decision for later extended runs.

## Frontier Scheduler Shape

Use the `batch` partition with the `debug` QOS for short non-production checks:

```bash
sbatch scripts/frontier/debug_smoke_one_node.slurm
```

Project/account:

```text
account:   bif148
partition: batch
qos:       debug
nodes:     1
walltime:  00:30:00 by project convention, never more than 02:00:00 in debug
```

OLCF documents the debug QOS as a short, non-production lane on Frontier. It
has higher priority than same-size production jobs, allows only one debug-QOS
job per user in any state, rejects walltimes above two hours, and prohibits
production work or chained debug jobs. This project uses a stricter 30-minute
one-node default so smoke runs consume at most about 0.5 node-hour when they
finish at the requested walltime, and less if they exit early.

Relevant OLCF references:

- Frontier User Guide, Running Jobs and queue policy:
  https://docs.olcf.ornl.gov/systems/frontier_user_guide.html
- PyTorch on Frontier:
  https://docs.olcf.ornl.gov/software/analytics/pytorch_frontier.html

## Environment Setup

The template loads the current OLCF-recommended PyTorch/ROCm module family for
Frontier:

```bash
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
```

For a first-time environment, create a persistent conda/pip environment outside
the job and then point the job at it with `EMENDER_CONDA_ENV`:

```bash
module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

conda create -p "$MEMBERWORK/emender/conda/emender-rocm711" python=3.12 -c conda-forge
conda activate "$MEMBERWORK/emender/conda/emender-rocm711"
python -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/rocm7.1
python -m pip install -e . --no-deps
python -m pip install numpy einops tqdm schedulefree triton==3.5.1 \
  datasets tiktoken pytest pytest-timeout
```

Notes:

- `pyproject.toml` currently pins `torch==2.9.1` and `triton==3.5.1`; the
  Frontier module/wheel recommendation may be newer than local development
  pins. Record the actual installed versions in every smoke artifact and do
  not compare runs across environments without noting the difference.
- `gdn2-MLP` requires the external GatedDeltaNet-2 fused path. Set
  `GDN2_PATH` to the Frontier checkout before submission if it is not at the
  template default.
- Frontier PyTorch guidance recommends `srun` for rank/GPU mapping instead of
  nesting `torchrun`. The template follows that guidance and runs one Python
  process per MI250X GCD: `srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest`.

## Location Convention

Use this directory layout for all Frontier debug smoke work:

```text
scripts/frontier/                         committed reusable SLURM templates
frontier_runs/debug/YYYYMMDD/<variant>/   uncommitted job scripts and run state
frontier_runs/debug/YYYYMMDD/<variant>/logs/
frontier_runs/debug/YYYYMMDD/<variant>/artifacts/
frontier_runs/debug/YYYYMMDD/<variant>/summaries/
```

The `frontier_runs/` tree is for generated Frontier artifacts and should not be
committed unless a later task explicitly asks to promote a small summary. Record
the committed template path with `wg artifact` and record generated run
directories in WG logs for later synthesis.

Each run directory should contain:

- `logs/<job-name>-<jobid>.out` and `logs/<job-name>-<jobid>.err`.
- `artifacts/env.txt` with module list, `rocm-smi`, Python executable, package
  versions, git commit/status, SLURM variables, and host/GPU mapping.
- `artifacts/manifest.json` with ledger fields: task id, variant, job id,
  account, partition, QOS, nodes, requested walltime, start/end timestamps,
  stdout/stderr paths, git commit, and requested node-hours.
- `summaries/summary.md` with observations, failures, stderr/stdout paths,
  final loss or smoke metric if available, peak memory lines, and next-step
  interpretation.

Workers should also log the same high-level fields to:

```bash
wg log rocm-kernel-integration-smoke "Frontier smoke: variant=<...> job=<...> ..."
wg artifact rocm-kernel-integration-smoke frontier_runs/debug/YYYYMMDD/<variant>/summaries/summary.md
```

For allocation accounting, also append the job to the ledger artifact produced
by `frontier-allocation-ledger` when that task publishes its final path. Until
then, keep the `manifest.json` and `summary.md` in the run directory.

## Submit A Sub-30-Minute Smoke

From a Frontier login node:

```bash
cd /path/to/emender
mkdir -p logs/frontier/debug

export EMENDER_CONDA_ENV="$MEMBERWORK/emender/conda/emender-rocm711"
export DATA="$MEMBERWORK/emender/data/commapile_mainmix_smoke.txt"
export VAL_DATA="$MEMBERWORK/emender/data/commapile_mainmix_val_smoke.txt"
export GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2"

export SMOKE_VARIANT=e97-MLP
sbatch scripts/frontier/debug_smoke_one_node.slurm

export SMOKE_VARIANT=e97-linear-MLP
sbatch scripts/frontier/debug_smoke_one_node.slurm

export SMOKE_VARIANT=gdn2-MLP
sbatch scripts/frontier/debug_smoke_one_node.slurm
```

Submit only one debug-QOS job at a time. Because Frontier rejects multiple
debug-QOS jobs per user in any state, wait for completion before submitting the
next variant:

```bash
squeue -u "$USER" -o "%.18i %.9P %.8q %.25j %.2t %.10M %.6D %R"
sacct -j <jobid> --format=JobID,JobName,Partition,QOS,Account,State,Elapsed,AllocNodes,NNodes,NodeList,ExitCode
```

The template defaults are deliberately small:

```text
TRAIN_MINUTES=20
SLURM walltime=30 minutes
node-hours requested at full walltime = 1 node * 0.5 h = 0.5 node-hour
```

Use the debug queue only to answer launch, import, fused-kernel, memory, and
first-loss-sanity questions. Do not run production-length training, CMA loops,
or chained matrices through `debug`.

## Template Controls

The reusable template is:

```text
scripts/frontier/debug_smoke_one_node.slurm
```

Common overrides:

```bash
SMOKE_VARIANT=e97-MLP              # e97-MLP | e97-linear-MLP | gdn2-MLP
TRAIN_MINUTES=20                   # keep below SLURM walltime
DATA=/path/to/smoke_train.txt
VAL_DATA=/path/to/smoke_val.txt
OUTPUT_ROOT="$MEMBERWORK/emender/frontier_runs/debug"
EMENDER_CONDA_ENV="$MEMBERWORK/emender/conda/emender-rocm711"
GDN2_PATH="$MEMBERWORK/emender/src/GatedDeltaNet-2"
```

The template captures these required fields every time:

- `rocm-smi` output.
- Python executable and version.
- `torch`, `triton`, `schedulefree`, `numpy`, and editable package versions.
- Full `python -m pip freeze`.
- `git rev-parse HEAD`, branch, status, and diff summary.
- `SLURM_JOB_ID`, `SLURM_JOB_NAME`, `SLURM_JOB_NUM_NODES`, partition, QOS,
  account, requested walltime, nodelist, stdout path, stderr path.
- Start/end timestamps and requested node-hours.

## Variant Commands

The template maps variants to these train arguments:

```text
e97-MLP:
  --level E97 --linear_state 0 --use_triton 1
  --n_state 32 --n_heads 323 --dim 1536 --depth 10 --mlp_ratio 1.5

e97-linear-MLP:
  --level E97 --linear_state 1 --use_triton 1 --use_chunked_e97 1
  --e97_chunk_size 32
  --n_state 32 --n_heads 128 --dim 1536 --depth 21 --mlp_ratio 2.0

gdn2-MLP:
  --level gdn2-mlp --dim 2176 --depth 12 --n_heads 30 --expansion 1.0
  --gdn2_mlp_ratio 3.258732449079677
```

For E97 variants the template runs a one-GCD kernel-only smoke before the
training command and records `kernel_smoke_status`, `train_status`, and log paths
in the manifest:

```bash
# e97-MLP sequential split-edit Triton path
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference -q -s

# e97-linear-MLP chunked E97 Triton path
HIP_VISIBLE_DEVICES=0 python -m pytest \
  tests/test_e97_chunked.py::test_fused_triton_forward_parity_fp32 \
  tests/test_e97_chunked.py::test_fused_triton_backward_parity_bf16 \
  tests/test_e97_chunked.py::test_strong_decay_cluster_backward_finite \
  tests/test_e97_chunked.py::test_glog_floor_drift_overflow_finite -q -s
```

All variants run bf16, ScheduleFree AdamW, `p50k_base`, one process per GCD,
and time-boxed training via `--train_minutes`.

For `gdn2-MLP`, the SLURM template first runs a one-rank import and kernel
smoke before the 8-rank training command:

```bash
srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
  python -u scripts/frontier/gdn2_rocm_preflight.py \
    --run-fwdbwd --bf16 --dim 128 --n-heads 4 --head-dim 16 \
    --seq-len 16 --batch-size 1 --expansion 1.0 --use-conv 1 \
    --d-conv 4 --mlp-ratio 2.0
```

The same check can be run without a GPU launch to verify only the external
checkout, `fla`, and `lit_gpt.gdn2_ops.chunk_gdn2` import path:

```bash
python -u scripts/frontier/gdn2_rocm_preflight.py
```

The preflight emits JSON including `torch.version.hip`, `GDN2_PATH`, the loaded
external module, the chunk-op module path, and finite forward/backward checks
when `--run-fwdbwd` is used. If this step fails, the template records
`logs/gdn2_rocm_preflight.log`, skips training, and writes the preflight status
into the run manifest.

## Observation Summary Format

Write `summaries/summary.md` in this form:

```markdown
# Frontier Debug Smoke Summary

- task: rocm-kernel-integration-smoke
- variant:
- job_id:
- account:
- partition:
- qos:
- nodes:
- requested_walltime:
- elapsed:
- requested_node_hours:
- actual_node_hours:
- git_commit:
- stdout:
- stderr:
- artifacts:

## Observation
What ran, what completed, and the exact final metrics or failure line observed.

## Failure Mode
If failed, include import errors, ROCm/Triton errors, non-finite loss, OOM,
RCCL/NCCL errors, or scheduler/environment issues. If passed, write "none
observed in this smoke".

## Interpretation Boundary
State only what this short run supports. Do not extrapolate to extended
allocation readiness without benchmark matrix and ledger synthesis.
```
