#!/bin/bash
#SBATCH -p gpu22
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J deltanet
#SBATCH -o delta.log
set -euo pipefail

# -----------------------
# User knobs (env overrides)
# -----------------------
CU_INDEX="${CU_INDEX:-cu126}"

TMP_BASE="${TMP_BASE:-/tmp/$USER/deltanet_tf_nomod}"
VENV_DIR="${VENV_DIR:-$TMP_BASE/venv}"

DATA_DIR="${DATA_DIR:-$PWD/data/mm_nomod_binary_T100}"
TRAIN_PY="${TRAIN_PY:-$PWD/train_deltanet.py}"   # <-- your deltanet TF-nomod script
SAVE_PATH="${SAVE_PATH:-$PWD/ckpt_deltanet_tf_final_nomod_binary.pt}"

# 0 (default) => input-only tokenization (matches Mamba/Transformer baselines)
# 1 => leaky STATE-token teacher forcing ablation (see train_deltanet.py --teacher_force_state)
TEACHER_FORCE_STATE="${TEACHER_FORCE_STATE:-0}"

# 0 (default) => beta_t in (0,1): cannot represent negative-eigenvalue
# transitions (e.g. parity, odd permutations/transpositions).
# 1 => beta_t in (0,2), see Grazzi et al. 2025.
ALLOW_NEG_EIGVAL="${ALLOW_NEG_EIGVAL:-0}"
ALPHABET="${ALPHABET:-pm1}"          # pm1 (entries in {-1,0,1}) / 01 (entries in {0,1})

# model / train hyperparams
D_MODEL="${D_MODEL:-256}"
HEADS="${HEADS:-4}"
LAYERS="${LAYERS:-1}"
DROPOUT="${DROPOUT:-0.2}"

MLP_HIDDEN_MULT="${MLP_HIDDEN_MULT:-4}"
MLP_ACT="${MLP_ACT:-gelu}"

BATCH_SIZE="${BATCH_SIZE:-256}"
# epochs=200 (old default) capped training at ~15800/30000 steps before
# max_steps or patience could bind -- raise it so max_steps is the real cap.
EPOCHS="${EPOCHS:-500}"
MAX_STEPS="${MAX_STEPS:-30000}"
LR="${LR:-3e-4}"
WD="${WD:-0.005}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
WARMUP_STEPS="${WARMUP_STEPS:-1000}"

# In-distribution (test_bin0) accuracy was only ~54% even with more patience
# budget (run 2: train finalAcc 58%->73.5% but test barely moved) -- the
# bottleneck is the sparse single-bit, up-to-T=100-step credit assignment,
# not training budget. Dense per-step auxiliary supervision targets this
# directly; curriculum ramps train T=5->100 over the first 150 epochs
# instead of exposing the full horizon from epoch 1.
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.3}"
CURRICULUM_MIN_T="${CURRICULUM_MIN_T:-5}"
CURRICULUM_EPOCHS="${CURRICULUM_EPOCHS:-150}"

# patience=20 on raw loss stopped DeltaNet at step ~6800/30000 while still
# underfit (train finalAcc ~58%). Give it more room, and stop on the metric
# we actually care about (finalAcc) instead of BCE loss, matching rnn.sh.
PATIENCE="${PATIENCE:-60}"
EARLY_STOP="${EARLY_STOP:-finalAcc}"   # loss / stepAcc / finalAcc (depends on your script)

# CausalLinearAttentionVec chunk length: memory scales with this, not L.
# Fixes the near-OOM seen evaluating test_bin1/test_bin2 (L up to ~301)
# with the old whole-sequence cumsum.
CHUNK_SIZE="${CHUNK_SIZE:-64}"

# tokenization / safety
MAX_LEN="${MAX_LEN:-0}"            # 0 => auto infer (recommended)
STATE_CAP="${STATE_CAP:-0}"        # 0 disables

# perf
NUM_WORKERS="${NUM_WORKERS:-2}"

# AMP
AMP="${AMP:-1}"                    # 1 => enable --amp
AMP_DTYPE="${AMP_DTYPE:-bf16}"     # bf16 or fp16 (must match your script)

# Install fla?
INSTALL_FLA="${INSTALL_FLA:-1}"    # 1 => pip install flash-linear-attention

echo "[1/8] Prepare /tmp: $TMP_BASE"
mkdir -p "$TMP_BASE"

echo "[2/8] Create/reuse venv: $VENV_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[3/8] Pip cache/tmp -> /tmp"
export PIP_CACHE_DIR="$TMP_BASE/pip-cache"
export TMPDIR="$TMP_BASE/pip-tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

echo "[4/8] Upgrade pip tooling"
python -m pip install --no-cache-dir -U pip setuptools wheel

echo "[5/8] Install torch if missing (CUDA wheels: ${CU_INDEX})"
if python - <<'PY'
import importlib, sys
try:
    importlib.import_module("torch")
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
  echo "  [skip] torch already installed"
else
  python -m pip install --no-cache-dir --index-url "https://download.pytorch.org/whl/${CU_INDEX}" \
    torch torchvision torchaudio
fi

echo "[6/8] Install deps"
python -m pip install --no-cache-dir -U numpy tqdm

if [[ "$INSTALL_FLA" == "1" ]]; then
  echo "[6.5/8] Install flash-linear-attention (fla)"
  # If your cluster needs a specific version, pin it here.
  python -m pip install --no-cache-dir -U flash-linear-attention
fi

echo "[7/8] Environment"
python - <<'PY'
import sys
import torch, numpy as np
print("Python:", sys.version.split()[0])
print("Torch :", torch.__version__)
print("CUDA  :", torch.version.cuda)
print("CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("NumPy :", np.__version__)
try:
    import fla
    print("fla   :", getattr(fla, "__version__", "unknown"))
except Exception as e:
    print("fla   : not importable:", e)
PY

echo "[8/8] Run training"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "ERROR: TRAIN_PY not found: $TRAIN_PY" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: DATA_DIR not found: $DATA_DIR" >&2
  exit 1
fi

CMD=(
python3 -u "$TRAIN_PY" \
  --data_dir "$DATA_DIR" --alphabet "$ALPHABET" \
  --cuda \
  --d_model "$D_MODEL" --layers "$LAYERS" --heads "$HEADS" --dropout "$DROPOUT" \
  --chunk_size "$CHUNK_SIZE" \
  --batch_size "$BATCH_SIZE" --lr "$LR" --weight_decay "$WD" --grad_clip "$GRAD_CLIP" \
  --warmup_steps "$WARMUP_STEPS" --aux_loss_weight "$AUX_LOSS_WEIGHT" \
  --curriculum_min_T "$CURRICULUM_MIN_T" --curriculum_epochs "$CURRICULUM_EPOCHS" \
  --epochs "$EPOCHS" --max_steps "$MAX_STEPS" --eval_every 2 --patience "$PATIENCE" --early_stop "$EARLY_STOP" \
  --num_workers "$NUM_WORKERS" --prefetch_factor 4 --persistent_workers \
  --save_path "$SAVE_PATH"
)

# optional args
if [[ "$MAX_LEN" != "0" ]]; then
  CMD+=(--max_len "$MAX_LEN")
fi
if [[ "$TEACHER_FORCE_STATE" == "1" ]]; then
  CMD+=(--teacher_force_state)
fi
if [[ "$ALLOW_NEG_EIGVAL" == "1" ]]; then
  CMD+=(--allow_neg_eigval)
fi

# AMP flags MUST be inside CMD (this fixes your error)
if [[ "$AMP" == "1" ]]; then
  CMD+=(--amp --amp_dtype "$AMP_DTYPE")
fi

echo "Command: ${CMD[*]}"
"${CMD[@]}"

echo "Done. Saved: $SAVE_PATH"
