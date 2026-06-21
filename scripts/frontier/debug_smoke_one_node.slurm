#!/bin/bash
# Frontier one-node debug smoke template for Emender ROCm/Triton validation.
# Submit from the repo root after creating logs/frontier/debug.
# Example:
#   mkdir -p logs/frontier/debug
#   SMOKE_VARIANT=e97-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm
#   SMOKE_VARIANT=e97-linear-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm
#   SMOKE_VARIANT=gdn2-MLP sbatch scripts/frontier/debug_smoke_one_node.slurm

#SBATCH -A bif148
#SBATCH -J emender-smoke
#SBATCH -p batch
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/debug/%x-%j.out
#SBATCH -e logs/frontier/debug/%x-%j.err

set -euo pipefail

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
TASK_ID=${WG_TASK_ID:-rocm-kernel-integration-smoke}
ACCOUNT=${SLURM_JOB_ACCOUNT:-bif148}
SMOKE_VARIANT=${SMOKE_VARIANT:-e97-MLP}
TRAIN_MINUTES=${TRAIN_MINUTES:-20}
DATA=${DATA:?Set DATA to a short Frontier-readable smoke training text file}
VAL_DATA=${VAL_DATA:-}
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
  if [[ -n "${MEMBERWORK:-}" ]]; then
    OUTPUT_ROOT="${MEMBERWORK}/emender/frontier_runs/debug"
  else
    OUTPUT_ROOT="${REPO}/frontier_runs/debug"
  fi
fi
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${SMOKE_VARIANT}/${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
SUMMARY_DIR="${RUN_ROOT}/summaries"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"
SUMMARY_FILE="${SUMMARY_DIR}/summary.md"
STDOUT_PATH="logs/frontier/debug/${SLURM_JOB_NAME:-emender-smoke}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/debug/${SLURM_JOB_NAME:-emender-smoke}-${SLURM_JOB_ID:-manual}.err"

mkdir -p "$LOG_DIR" "$ARTIFACT_DIR" "$SUMMARY_DIR"
cd "$REPO"

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load miniforge3/23.11.0-0
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"

if [[ -n "${EMENDER_CONDA_ENV:-}" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$EMENDER_CONDA_ENV"
fi

export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -i | awk '{print $1}')}
export MASTER_PORT=${MASTER_PORT:-3442}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}
if [[ -z "${GDN2_PATH:-}" ]]; then
  if [[ -n "${MEMBERWORK:-}" ]]; then
    export GDN2_PATH="${MEMBERWORK}/emender/src/GatedDeltaNet-2"
  else
    export GDN2_PATH="${REPO}/../GatedDeltaNet-2"
  fi
else
  export GDN2_PATH
fi

COMMON_ARGS=(
  --data "$DATA"
  --bf16
  --batch_size "${BATCH_SIZE:-1}"
  --chunk_size "${CHUNK_SIZE:-2048}"
  --tokenizer "${TOKENIZER:-p50k_base}"
  --optimizer schedulefree
  --seed "${SEED:-42}"
  --lr "${LR:-9e-4}"
  --train_minutes "$TRAIN_MINUTES"
  --compile_warmup_steps "${COMPILE_WARMUP_STEPS:-1}"
  --log_every "${LOG_EVERY:-10}"
  --save_every 999999
  --keep_checkpoints 1
  --output "${RUN_ROOT}/train"
)

if [[ -n "$VAL_DATA" ]]; then
  COMMON_ARGS+=(--val_data "$VAL_DATA" --val_every "${VAL_EVERY:-50}")
fi

case "$SMOKE_VARIANT" in
  e97-MLP)
    MODEL_ARGS=(
      --level E97
      --linear_state 0
      --use_triton 1
      --n_state 32
      --n_heads 323
      --dim 1536
      --depth 10
      --expansion 1.0
      --use_gate 1
      --gate_activation silu
      --mlp_ratio 1.5
      --mlp_multiple 64
    )
    ;;
  e97-linear-MLP)
    MODEL_ARGS=(
      --level E97
      --linear_state 1
      --use_triton 1
      --use_chunked_e97 1
      --e97_chunk_size "${E97_CHUNK_SIZE:-32}"
      --n_state 32
      --n_heads 128
      --dim 1536
      --depth 21
      --expansion 1.0
      --use_gate 1
      --gate_activation silu
      --mlp_ratio 2.0
      --mlp_multiple 64
    )
    ;;
  gdn2-MLP)
    MODEL_ARGS=(
      --level gdn2-mlp
      --dim 2176
      --depth 12
      --n_heads 30
      --expansion 1.0
      --use_conv 1
      --d_conv 4
      --gdn2_mlp_ratio 3.258732449079677
    )
    ;;
  *)
    echo "Unknown SMOKE_VARIANT='$SMOKE_VARIANT' (expected e97-MLP, e97-linear-MLP, or gdn2-MLP)" >&2
    exit 2
    ;;
esac

case "$SMOKE_VARIANT" in
  e97-MLP)
    KERNEL_SMOKE_CMD=(
      python -m pytest
      tests/test_e88_triton.py::test_e97_split_edit_triton_matches_reference
      -q -s
    )
    ;;
  e97-linear-MLP)
    KERNEL_SMOKE_CMD=(
      python -m pytest
      tests/test_e97_chunked.py::test_fused_triton_forward_parity_fp32
      tests/test_e97_chunked.py::test_fused_triton_backward_parity_bf16
      tests/test_e97_chunked.py::test_strong_decay_cluster_backward_finite
      tests/test_e97_chunked.py::test_glog_floor_drift_overflow_finite
      -q -s
    )
    ;;
  gdn2-MLP)
    KERNEL_SMOKE_CMD=(
      python -u scripts/frontier/gdn2_rocm_preflight.py
      --run-fwdbwd
      --bf16
      --dim 128
      --n-heads 4
      --head-dim 16
      --seq-len 16
      --batch-size 1
      --expansion 1.0
      --use-conv 1
      --d-conv 4
      --mlp-ratio 2.0
    )
    ;;
  *)
    KERNEL_SMOKE_CMD=()
    ;;
esac

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse HEAD)
REQUESTED_NODES=${SLURM_JOB_NUM_NODES:-1}
REQUESTED_SECONDS=$(python - <<'PY'
import os
t = os.environ.get("SLURM_TIMELIMIT", "30")
if "-" in t:
    d, t = t.split("-", 1)
    day_s = int(d) * 86400
else:
    day_s = 0
parts = [int(p) for p in t.split(":")]
if len(parts) == 1:
    sec = parts[0] * 60
elif len(parts) == 2:
    sec = parts[0] * 3600 + parts[1] * 60
else:
    sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
print(day_s + sec)
PY
)
REQUESTED_NODE_HOURS=$(python - <<PY
nodes = float("${REQUESTED_NODES}")
seconds = float("${REQUESTED_SECONDS}")
print(f"{nodes * seconds / 3600.0:.6f}")
PY
)

{
  echo "=== frontier smoke env ==="
  echo "task_id=$TASK_ID"
  echo "variant=$SMOKE_VARIANT"
  echo "start_utc=$START_UTC"
  echo "repo=$REPO"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo
  echo "=== slurm ==="
  env | sort | grep -E '^(SLURM|MPICH|NCCL|MASTER|OMP|ROCR|HIP|HSA|GDN2|NDM|DATA|VAL_DATA|TRAIN_MINUTES|SMOKE_VARIANT)=' || true
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo
  echo "=== modules ==="
  module list 2>&1
  echo
  echo "=== rocm-smi ==="
  rocm-smi || true
  echo
  echo "=== python ==="
  which python
  python --version
  python - <<'PY'
import importlib.metadata as md
for name in ["torch", "triton", "schedulefree", "numpy", "ndm"]:
    try:
        print(f"{name}=={md.version(name)}")
    except md.PackageNotFoundError:
        print(f"{name}=NOT_INSTALLED")
try:
    import torch
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
except Exception as exc:
    print(f"torch_import_error={exc!r}")
PY
  echo
  echo "=== pip freeze ==="
  python -m pip freeze
  echo
  echo "=== git ==="
  git rev-parse --show-toplevel
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git status --short
  git diff --stat
  echo
  echo "=== command ==="
  if [[ ${#KERNEL_SMOKE_CMD[@]} -gt 0 ]]; then
    printf '%q ' srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest "${KERNEL_SMOKE_CMD[@]}"
    echo
  fi
  printf '%q ' srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest python -u train.py "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}"
  echo
} | tee "$ENV_FILE"

set +e
KERNEL_SMOKE_STATUS=0
if [[ ${#KERNEL_SMOKE_CMD[@]} -gt 0 ]]; then
  srun -N1 -n1 -c7 --gpus-per-task=1 --gpu-bind=closest \
    "${KERNEL_SMOKE_CMD[@]}" \
    2>&1 | tee "${LOG_DIR}/kernel_smoke.log"
  KERNEL_SMOKE_STATUS=${PIPESTATUS[0]}
else
  echo "no kernel-only smoke configured for ${SMOKE_VARIANT}" | tee "${LOG_DIR}/kernel_smoke.log"
fi
if [[ "$KERNEL_SMOKE_STATUS" -eq 0 ]]; then
  srun -N1 -n8 -c7 --gpus-per-task=1 --gpu-bind=closest \
    python -u train.py "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" \
    2>&1 | tee "${LOG_DIR}/train.log"
  TRAIN_STATUS=${PIPESTATUS[0]}
  STATUS=$TRAIN_STATUS
else
  TRAIN_STATUS=77
  STATUS=$KERNEL_SMOKE_STATUS
  echo "kernel smoke failed; skipping training smoke" | tee "${LOG_DIR}/train.log"
fi
set -e

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python - <<PY
import json
manifest = {
    "task_id": "${TASK_ID}",
    "variant": "${SMOKE_VARIANT}",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "job_name": "${SLURM_JOB_NAME:-emender-smoke}",
    "account": "${ACCOUNT}",
    "partition": "${SLURM_JOB_PARTITION:-batch}",
    "qos": "${SLURM_JOB_QOS:-debug}",
    "nodes": int("${REQUESTED_NODES}"),
    "requested_walltime": "${SLURM_TIMELIMIT:-00:30:00}",
    "requested_node_hours": float("${REQUESTED_NODE_HOURS}"),
    "start_utc": "${START_UTC}",
    "end_utc": "${END_UTC}",
    "exit_status": int("${STATUS}"),
    "kernel_smoke_status": int("${KERNEL_SMOKE_STATUS}"),
    "train_status": int("${TRAIN_STATUS}"),
    "git_commit": "${GIT_COMMIT}",
    "stdout": "${STDOUT_PATH}",
    "stderr": "${STDERR_PATH}",
    "run_root": "${RUN_ROOT}",
    "env_file": "${ENV_FILE}",
    "kernel_smoke_log": "${LOG_DIR}/kernel_smoke.log",
    "train_log": "${LOG_DIR}/train.log",
}
with open("${MANIFEST_FILE}", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY

cat > "$SUMMARY_FILE" <<EOF
# Frontier Debug Smoke Summary

- task: ${TASK_ID}
- variant: ${SMOKE_VARIANT}
- job_id: ${SLURM_JOB_ID:-manual}
- account: ${ACCOUNT}
- partition: ${SLURM_JOB_PARTITION:-batch}
- qos: ${SLURM_JOB_QOS:-debug}
- nodes: ${REQUESTED_NODES}
- requested_walltime: ${SLURM_TIMELIMIT:-00:30:00}
- elapsed: fill from sacct after completion
- requested_node_hours: ${REQUESTED_NODE_HOURS}
- actual_node_hours: fill from sacct after completion
- git_commit: ${GIT_COMMIT}
- stdout: ${STDOUT_PATH}
- stderr: ${STDERR_PATH}
- artifacts: ${RUN_ROOT}
- kernel_smoke_status: ${KERNEL_SMOKE_STATUS}
- train_status: ${TRAIN_STATUS}
- exit_status: ${STATUS}

## Observation
Fill in final loss, peak memory, fused-kernel guard lines, and whether the job
completed the requested smoke window.

## Failure Mode
If failed, paste the first actionable error line from stderr/stdout. If passed,
write "none observed in this smoke".

## Interpretation Boundary
This is one short debug-QOS run. Treat it as launch/import/kernel/memory/loss
sanity evidence only, not as extended-allocation readiness.
EOF

echo "manifest=$MANIFEST_FILE"
echo "summary=$SUMMARY_FILE"
echo "train_log=${LOG_DIR}/train.log"

exit "$STATUS"
