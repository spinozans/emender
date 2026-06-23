#!/bin/bash
# Guarded Frontier extended-queue launch package for the first e97-MLP 64x24h
# candidate. This file is an auditable package, not approval to spend allocation.
#
# Required human gate:
#   WG task frontier-extended-e97-readiness (or successor approval task) must
#   record explicit approval before submission, then submit with:
#     HUMAN_APPROVAL_RECORD='<WG approval message/artifact id>' \
#     CONFIRM_EXTENDED_SUBMISSION='I_ACCEPT_1536_NODE_HOURS' \
#     sbatch scripts/frontier/e97_extended_64x24.sbatch

#SBATCH -A bif148
#SBATCH -J e97-64x24-01
#SBATCH -p batch
#SBATCH -q extended
#SBATCH -N 64
#SBATCH -t 24:00:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH -o logs/frontier/extended/%x-%j.out
#SBATCH -e logs/frontier/extended/%x-%j.err

set -euo pipefail

if [[ "${CONFIRM_EXTENDED_SUBMISSION:-}" != "I_ACCEPT_1536_NODE_HOURS" ]]; then
  echo "Refusing to run: set CONFIRM_EXTENDED_SUBMISSION=I_ACCEPT_1536_NODE_HOURS after WG human approval." >&2
  exit 3
fi
if [[ -z "${HUMAN_APPROVAL_RECORD:-}" ]]; then
  echo "Refusing to run: HUMAN_APPROVAL_RECORD must cite the WG approval message or committed approval artifact." >&2
  exit 3
fi

REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
ACCOUNT=${SLURM_JOB_ACCOUNT:-bif148}
TASK_ID=${WG_TASK_ID:-frontier-extended-e97-readiness}
RUN_ID=${RUN_ID:-E97-64x24-01}
VARIANT=${VARIANT:-e97-MLP}
DATA=${DATA:-/lustre/orion/bif148/proj-shared/commapile/commapile_mainmix_v0.1_1tb.txt}
VAL_DATA=${VAL_DATA:-}
TOKENIZER=${TOKENIZER:-p50k_base}
TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/lustre/orion/bif148/proj-shared/tiktoken_cache}
OUTPUT_ROOT=${OUTPUT_ROOT:-/lustre/orion/bif148/proj-shared/emender/frontier_runs/extended}
RUN_DATE=$(date -u +%Y%m%d)
RUN_STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_ROOT="${OUTPUT_ROOT}/${RUN_DATE}/${RUN_ID}-${SLURM_JOB_ID:-manual}-${RUN_STAMP}"
LOG_DIR="${RUN_ROOT}/logs"
ARTIFACT_DIR="${RUN_ROOT}/artifacts"
TRAIN_ROOT="${RUN_ROOT}/train"
ENV_FILE="${ARTIFACT_DIR}/env.txt"
MANIFEST_FILE="${ARTIFACT_DIR}/manifest.json"
STDOUT_PATH="logs/frontier/extended/${SLURM_JOB_NAME:-e97-64x24-01}-${SLURM_JOB_ID:-manual}.out"
STDERR_PATH="logs/frontier/extended/${SLURM_JOB_NAME:-e97-64x24-01}-${SLURM_JOB_ID:-manual}.err"

mkdir -p logs/frontier/extended "$LOG_DIR" "$ARTIFACT_DIR" "$TRAIN_ROOT"
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

export TIKTOKEN_CACHE_DIR
export MPICH_GPU_SUPPORT_ENABLED=${MPICH_GPU_SUPPORT_ENABLED:-1}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-hsn0}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname -i | awk '{print $1}')}
export MASTER_PORT=${MASTER_PORT:-3442}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-7}
export NDM_PIN_TRITON_AUTOTUNE=${NDM_PIN_TRITON_AUTOTUNE:-1}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZER

[[ -r "$DATA" ]] || { echo "DATA is not readable: $DATA" >&2; exit 4; }
[[ -d "$TIKTOKEN_CACHE_DIR" ]] || { echo "TIKTOKEN_CACHE_DIR is missing: $TIKTOKEN_CACHE_DIR" >&2; exit 4; }

python - <<'PY'
import os
import tiktoken

name = os.environ.get("TOKENIZER", "p50k_base")
cache = os.environ["TIKTOKEN_CACHE_DIR"]
before = set(os.listdir(cache))
enc = tiktoken.get_encoding(name)
after = set(os.listdir(cache))
if after != before:
    raise SystemExit(
        f"tokenizer cache changed while resolving {name}; pre-populate {cache} "
        "from a login/service node before extended submission"
    )
print(f"tokenizer={name} n_vocab={enc.n_vocab} cache_dir={cache} entries={len(after)}")
PY

START_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_COMMIT=$(git rev-parse HEAD)
REQUESTED_NODES=${SLURM_JOB_NUM_NODES:-64}
REQUESTED_NODE_HOURS=$(python - <<PY
print(f"{float('${REQUESTED_NODES}') * 24.0:.6f}")
PY
)

COMMON_ARGS=(
  --data "$DATA"
  --bf16
  --batch_size "${BATCH_SIZE:-1}"
  --chunk_size "${CHUNK_SIZE:-2048}"
  --tokenizer "$TOKENIZER"
  --optimizer schedulefree
  --seed "${SEED:-42}"
  --lr "${LR:-9e-4}"
  --train_minutes "${TRAIN_MINUTES:-1410}"
  --timer_after_compile_warmup
  --compile_warmup_steps "${COMPILE_WARMUP_STEPS:-1}"
  --log_every "${LOG_EVERY:-25}"
  --save_every "${SAVE_EVERY:-500}"
  --keep_checkpoints "${KEEP_CHECKPOINTS:-12}"
  --walltime_final_checkpoint_margin_seconds "${WALLTIME_FINAL_CHECKPOINT_MARGIN_SECONDS:-900}"
  --walltime_check_every "${WALLTIME_CHECK_EVERY:-1}"
  --heldout_eval_mode y
  --output "$TRAIN_ROOT"
)
if [[ -n "$VAL_DATA" ]]; then
  [[ -r "$VAL_DATA" ]] || { echo "VAL_DATA is not readable: $VAL_DATA" >&2; exit 4; }
  COMMON_ARGS+=(--val_data "$VAL_DATA" --val_every "${VAL_EVERY:-500}")
fi
if [[ -n "${HELDOUT_TENSOR:-}" ]]; then
  [[ -r "$HELDOUT_TENSOR" ]] || { echo "HELDOUT_TENSOR is not readable: $HELDOUT_TENSOR" >&2; exit 4; }
  COMMON_ARGS+=(
    --heldout_tensor "$HELDOUT_TENSOR"
    --final_heldout_eval
    --heldout_curve_every "${HELDOUT_CURVE_EVERY:-500}"
    --heldout_curve_path "${RUN_ROOT}/heldout_curve.csv"
  )
fi
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  [[ -r "$RESUME_CHECKPOINT" ]] || { echo "RESUME_CHECKPOINT is not readable: $RESUME_CHECKPOINT" >&2; exit 4; }
  COMMON_ARGS+=(--resume "$RESUME_CHECKPOINT")
fi

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

{
  echo "task_id=$TASK_ID"
  echo "run_id=$RUN_ID"
  echo "human_approval_record=$HUMAN_APPROVAL_RECORD"
  echo "variant=$VARIANT"
  echo "start_utc=$START_UTC"
  echo "repo=$REPO"
  echo "run_root=$RUN_ROOT"
  echo "stdout=$STDOUT_PATH"
  echo "stderr=$STDERR_PATH"
  echo "requested_node_hours=$REQUESTED_NODE_HOURS"
  echo "data=$DATA"
  echo "val_data=${VAL_DATA:-}"
  echo "tiktoken_cache_dir=$TIKTOKEN_CACHE_DIR"
  echo "resume_checkpoint=${RESUME_CHECKPOINT:-}"
  echo
  echo "=== slurm/env ==="
  env | sort | grep -E '^(SLURM|MPICH|NCCL|MASTER|OMP|ROCR|HIP|HSA|NDM|DATA|VAL_DATA|TOKENIZER|TIKTOKEN|RUN_ID|TRAIN_MINUTES|SAVE_EVERY|KEEP_CHECKPOINTS|RESUME_CHECKPOINT|HUMAN_APPROVAL_RECORD)=' || true
  echo
  echo "=== modules ==="
  module list 2>&1
  echo
  echo "=== python ==="
  which python
  python --version
  python -m pip freeze
  echo
  echo "=== git ==="
  git rev-parse --show-toplevel
  git rev-parse --abbrev-ref HEAD
  git rev-parse HEAD
  git status --short
  echo
  echo "=== command ==="
  printf '%q ' srun -N "$REQUESTED_NODES" -n "$((REQUESTED_NODES * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest \
    bash -c 'export RANK="$SLURM_PROCID"; export WORLD_SIZE="$SLURM_NTASKS"; export LOCAL_RANK="$SLURM_LOCALID"; exec python -u train.py "$@"' \
    _ "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}"
  echo
} | tee "$ENV_FILE"

set +e
srun -N "$REQUESTED_NODES" -n "$((REQUESTED_NODES * 8))" -c7 --gpus-per-task=1 --gpu-bind=closest \
  bash -c 'export RANK="$SLURM_PROCID"; export WORLD_SIZE="$SLURM_NTASKS"; export LOCAL_RANK="$SLURM_LOCALID"; exec python -u train.py "$@"' \
  _ "${COMMON_ARGS[@]}" "${MODEL_ARGS[@]}" \
  2>&1 | tee "${LOG_DIR}/train.log"
STATUS=${PIPESTATUS[0]}
set -e

END_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
python - <<PY
import json
manifest = {
    "task_id": "${TASK_ID}",
    "run_id": "${RUN_ID}",
    "human_approval_record": "${HUMAN_APPROVAL_RECORD}",
    "variant": "${VARIANT}",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "job_name": "${SLURM_JOB_NAME:-e97-64x24-01}",
    "account": "${ACCOUNT}",
    "partition": "${SLURM_JOB_PARTITION:-batch}",
    "qos": "${SLURM_JOB_QOS:-extended}",
    "nodes": int("${REQUESTED_NODES}"),
    "requested_walltime": "${SLURM_TIMELIMIT:-24:00:00}",
    "requested_node_hours": float("${REQUESTED_NODE_HOURS}"),
    "start_utc": "${START_UTC}",
    "end_utc": "${END_UTC}",
    "exit_status": int("${STATUS}"),
    "git_commit": "${GIT_COMMIT}",
    "data": "${DATA}",
    "val_data": "${VAL_DATA}",
    "tiktoken_cache_dir": "${TIKTOKEN_CACHE_DIR}",
    "resume_checkpoint": "${RESUME_CHECKPOINT:-}",
    "stdout": "${STDOUT_PATH}",
    "stderr": "${STDERR_PATH}",
    "run_root": "${RUN_ROOT}",
    "env_file": "${ENV_FILE}",
    "train_log": "${LOG_DIR}/train.log",
    "heldout_curve": "${RUN_ROOT}/heldout_curve.csv",
}
with open("${MANIFEST_FILE}", "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\\n")
PY

echo "manifest=$MANIFEST_FILE"
echo "train_log=${LOG_DIR}/train.log"
exit "$STATUS"
