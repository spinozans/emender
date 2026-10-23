#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J rnn
#SBATCH -o rnn.log
 
set -euo pipefail

# ============================================================
# User knobs (env overrides)
# ============================================================
CU_INDEX="${CU_INDEX:-cu126}"

# Put venv + pip cache in /tmp (avoid home quota)
TMP_BASE="${TMP_BASE:-/tmp/$USER/mm_tf_rnn}"
VENV_DIR="${VENV_DIR:-$TMP_BASE/venv}"

# Project paths (defaults assume you submit from repo root)
WORKDIR="${WORKDIR:-$PWD}"
DATA_DIR="${DATA_DIR:-$PWD/data/mm_nomod_binary_T100}"
TRAIN_PY="${TRAIN_PY:-$WORKDIR/train_rnn.py}"        # <-- the TF-RNN script you saved
ALPHABET="${ALPHABET:-pm1}"          # pm1 (entries in {-1,0,1}) / 01 (entries in {0,1})

# 0 (default) => input-only tokenization (matches Mamba/Transformer baselines)
# 1 => leaky STATE-token teacher forcing ablation (see train_rnn.py --teacher_force_state)
TEACHER_FORCE_STATE="${TEACHER_FORCE_STATE:-0}"

# Training hyperparams (sane defaults)
RNN_TYPE="${RNN_TYPE:-gru}"          # gru / rnn_tanh / rnn_relu
D_MODEL="${D_MODEL:-256}"
LAYERS="${LAYERS:-2}"
# 0.1/1e-3 let the GRU hit 99.8% train acc vs ~55% val/test (pure
# memorization). Stronger dropout + weight decay to close that gap.
DROPOUT="${DROPOUT:-0.3}"

MLP_HMULT="${MLP_HMULT:-4}"
MLP_ACT="${MLP_ACT:-gelu}"           # gelu / relu

BATCH_SIZE="${BATCH_SIZE:-256}"
# epochs=200 was silently the binding stop condition (~15800/30000 steps,
# bad=85/100 patience) -- raise it so max_steps/patience actually govern.
EPOCHS="${EPOCHS:-500}"
MAX_STEPS="${MAX_STEPS:-30000}"

LR="${LR:-3e-4}"
WD="${WD:-1e-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"

# In-distribution (test_bin0) accuracy was only ~54-58% even after the
# dropout/WD/epochs fixes -- the bottleneck is the sparse single-bit,
# up-to-T=100-step credit assignment, not regularization. Dense per-step
# auxiliary supervision targets this directly.
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.3}"
# ramp train sequence length T=5->100 over the first 150 epochs instead of
# exposing the full 100-step credit-assignment horizon from epoch 1.
CURRICULUM_MIN_T="${CURRICULUM_MIN_T:-5}"
CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-150}"

# TF tokenization knobs
# max_len=0 => auto-infer from train_src (recommended)
MAX_LEN="${MAX_LEN:-0}"
# state_cap=0 => no clipping (recommended if your gen.py used --value_cap)
STATE_CAP="${STATE_CAP:-0}"

# DataLoader
NUM_WORKERS="${NUM_WORKERS:-2}"

# Output checkpoint
SAVE_PATH="${SAVE_PATH:-$WORKDIR/ckpt_mm_tf_rnn.pt}"

# ============================================================
# 1) Prepare /tmp dirs
# ============================================================
echo "[1/8] Prepare /tmp directories: $TMP_BASE"
mkdir -p "$TMP_BASE" "$TMP_BASE/pip-cache" "$TMP_BASE/pip-tmp"

# ============================================================
# 2) Create/reuse venv
# ============================================================
echo "[2/8] Create or reuse venv: $VENV_DIR"
if [[ -x "$VENV_DIR/bin/python" ]]; then
  echo "  [reuse] venv exists"
else
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# ============================================================
# 3) Force pip cache/temp to /tmp
# ============================================================
echo "[3/8] Configure pip cache/temp to /tmp"
export PIP_CACHE_DIR="$TMP_BASE/pip-cache"
export TMPDIR="$TMP_BASE/pip-tmp"

# ============================================================
# 4) Upgrade pip tooling
# ============================================================
echo "[4/8] Upgrade pip tooling"
python -m pip install --no-cache-dir -U pip setuptools wheel

# ============================================================
# 5) Install PyTorch (CUDA wheels) if missing
# ============================================================
echo "[5/8] Ensure torch installed (CUDA wheels: $CU_INDEX)"
if python - <<'PY'
import importlib
try:
    importlib.import_module("torch")
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
then
  echo "  [skip] torch already installed"
else
  python -m pip install --no-cache-dir \
    --index-url "https://download.pytorch.org/whl/${CU_INDEX}" \
    torch torchvision torchaudio
fi

# ============================================================
# 6) Install deps
# ============================================================
echo "[6/8] Install deps"
python -m pip install --no-cache-dir -U numpy tqdm

# ============================================================
# 7) Environment summary + sanity checks
# ============================================================
echo
echo "===================="
echo "[7/8] Environment"
echo "===================="
python - <<'PY'
import sys
import torch
import numpy as np
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("NumPy :", np.__version__)
PY

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "ERROR: training script not found: $TRAIN_PY"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir not found: $DATA_DIR"
  exit 1
fi

# quick presence check for expected files
for sp in train val_bin0 test_bin0 test_bin1 test_bin2; do
  [[ -f "$DATA_DIR/${sp}_src.txt" ]] || { echo "ERROR: missing $DATA_DIR/${sp}_src.txt"; exit 1; }
  [[ -f "$DATA_DIR/${sp}_tgt.txt" ]] || { echo "ERROR: missing $DATA_DIR/${sp}_tgt.txt"; exit 1; }
done

# ============================================================
# 8) Run training
# ============================================================
echo
echo "===================="
echo "[8/8] Run training"
echo "===================="

CMD=(
python3 -u "$TRAIN_PY" \
  --data_dir "$DATA_DIR" \
  --alphabet "$ALPHABET" \
  --rnn_type "$RNN_TYPE" --d_model "$D_MODEL" --hidden 256 --layers "$LAYERS" --dropout "$DROPOUT" \
  --cuda --amp --amp_dtype bf16 \
  --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay "$WD" --grad_clip "$GRAD_CLIP" \
  --warmup_steps "$WARMUP_STEPS" --aux_loss_weight "$AUX_LOSS_WEIGHT" \
  --curriculum_min_T "$CURRICULUM_MIN_T" --curriculum_epochs "$CURRICULUM_EPOCHS" \
  --epochs "$EPOCHS" --max_steps "$MAX_STEPS" --eval_every 2 --patience 100 --early_stop finalAcc \
  --num_workers "$NUM_WORKERS" --prefetch_factor 4 --persistent_workers \
  --save_path "$SAVE_PATH"
)
if [[ "$MAX_LEN" != "0" ]]; then
  CMD+=(--max_len "$MAX_LEN")
fi
if [[ "$TEACHER_FORCE_STATE" == "1" ]]; then
  CMD+=(--teacher_force_state)
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo
echo "Done."
echo "Venv: $VENV_DIR"
echo "Checkpoint: $SAVE_PATH"
