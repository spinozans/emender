#!/bin/bash
#SBATCH -p gpu22
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -J rwkv7_train
#SBATCH -o rwkv.log
 
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"

BASE="/tmp/${USER}/rwkv7_train_${SLURM_JOB_ID}"
VENV_DIR="${BASE}/venv"
export TMPDIR="${BASE}/pip-tmp"
export PIP_CACHE_DIR="${BASE}/pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

echo "[Host] $(hostname)"
echo "[BASE] $BASE"
echo "[VENV] $VENV_DIR"

rm -rf "$VENV_DIR"
$PYTHON_BIN -m venv "$VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# -----------------------
# Pip tooling + basics
# -----------------------
python -m pip install --no-cache-dir -U pip setuptools wheel packaging ninja
python -m pip install --no-cache-dir -U numpy einops tqdm

# -----------------------
# Torch (CUDA 11.8 wheels)
# torchvision/torchaudio are not imported by train_rwkv7.py, and their pinned
# versions here have no cp313 wheel -- install torch only.
# -----------------------
python -m pip install --no-cache-dir \
  torch==2.5.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# (Optional) HF utilities
python -m pip install --no-cache-dir -U transformers accelerate safetensors

# -----------------------
# FLA backend (RWKV7Attention)
# -----------------------
# NOTE: keeping your pins; if you hit Triton-related issues, see the env vars below.
# fla.modules.convolution imports triton unconditionally -- required, not optional.
python -m pip install --no-cache-dir -U triton
python -m pip install --no-cache-dir --no-deps flash-linear-attention==0.4.1
python -m pip install --no-cache-dir fla-core==0.4.1

echo
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

# -----------------------
# RWKV7 stability knobs
# -----------------------
# 0) torch 2.5.1's Dynamo doesn't support this node's Python 3.13, but fla's
#    activations.py unconditionally decorates a function with @torch.compile
#    at import time. train_rwkv7.py's patch #0 makes torch.compile an identity
#    no-op when this is set, so the import succeeds (runs eager, no perf loss
#    expected here since chunk-mode RWKV7 doesn't rely on torch.compile).
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

# 1) Disable the flaky Triton fused_addcmul path used by some FLA builds
export DISABLE_RWKV7_FUSED_ADDCMUL="${DISABLE_RWKV7_FUSED_ADDCMUL:-1}"

# 2) Make Triton/Inductor caches local to /tmp (avoid home/NFS issues)
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$BASE/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$BASE/torchinductor-cache}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# (Optional) reduce thread oversubscription on CPU-heavy preload
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

# -----------------------
# Run training
# -----------------------
DATA_DIR="${DATA_DIR:-data/mm_nomod_binary_T100}"
SAVE_PATH="${SAVE_PATH:-ckpt_rwkv7_tf_final_nomod_binary.pt}"
ALPHABET="${ALPHABET:-pm1}"          # pm1 (entries in {-1,0,1}) / 01 (entries in {0,1})

# 0 (default) => input-only tokenization (matches Mamba/Transformer baselines)
# 1 => leaky STATE-token teacher forcing ablation (see train_rwkv7.py --teacher_force_state)
TEACHER_FORCE_STATE="${TEACHER_FORCE_STATE:-0}"

# weight on the dense per-step auxiliary BCE loss (0 disables, matches old behavior)
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.0}"

CMD=(
python3 -u train_rwkv7.py \
  --data_dir "$DATA_DIR" \
  --alphabet "$ALPHABET" \
  --cuda --amp --amp_dtype bf16 \
  --aux_loss_weight "$AUX_LOSS_WEIGHT" \
  --d_model 256 --rwkv7_depth 2 --rwkv7_head_dim 64 --rwkv7_mode chunk --dropout 0.1 \
  --batch_size 256 --lr 3e-4 --weight_decay 1e-3 --grad_clip 1.0 --early_stop finalAcc \
  --max_steps 60000 --patience 20 \
  --save_path "$SAVE_PATH"
)
if [[ "$TEACHER_FORCE_STATE" == "1" ]]; then
  CMD+=(--teacher_force_state)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

