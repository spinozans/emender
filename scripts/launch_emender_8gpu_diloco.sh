#!/usr/bin/env bash
# Launch the from-scratch 8-GPU (I=8 islands) DiLoCo emender arm.
#
# Recipe is the PROVEN plain_i6b launch_manifest.json (212 in-basin merges),
# edited ONLY for this run:
#   - nproc_per_node = 8 (I=8 islands; one notch past the validated I<=6 ceiling
#     -- the in-basin tripwire is the guard)
#   - FROM SCRATCH (no --resume; random init; beta=0 measured in-basin from
#     random init at K=250/I<=4 -- I=8 is the open question the tripwire answers)
#   - new --output dir under /mnt/nvme1n1/erikg/diloco_8gpu/emender
#     (train.py derives the visible run slug from the instantiated model, so
#     this E97 geometry writes emender_E97_1.3B_<timestamp> rather than the
#     stale params-arg label levelE97_100m_<timestamp>)
#   - save_every = 500 (multiple of diloco_k=250 -> checkpoints land on a MERGE
#     boundary, capturing post-merge consensus, never mid-cycle island divergence)
# Everything else (env, geometry, optimizer, data) is byte-identical to the
# proven manifest. See task roll-emender-8-gpu.
#
# Usage:
#   scripts/launch_emender_8gpu_diloco.sh                 # fresh from-scratch
#   RESUME=/path/to/ckpt.pt scripts/launch_emender_8gpu_diloco.sh   # resume
#   LOGDIR=/alt/dir scripts/launch_emender_8gpu_diloco.sh           # alt run dir
#
# Prints the detached run PID on stdout (see launch_detached_run.sh).
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

LOGDIR="${LOGDIR:-/mnt/nvme1n1/erikg/diloco_8gpu/emender}"
OUTPUT="${OUTPUT:-$LOGDIR/runs}"
NAME="${NAME:-diloco_emender_8gpu_i8}"
RESUME="${RESUME:-}"

mkdir -p "$OUTPUT"

# Proven env (byte-identical to plain_i6b manifest).
ENV_ARGS=(
  env
  NCCL_P2P_DISABLE=1
  TORCH_NCCL_ENABLE_MONITORING=0
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

TRAIN_ARGS=(
  torchrun --standalone --nproc_per_node=8 train.py
)
# --resume FIRST when resuming (mirrors manifest ordering; load happens before
# the loop and optimizer.train() rebuilds y from the merged x/z pair).
if [ -n "$RESUME" ]; then
  TRAIN_ARGS+=(--resume "$RESUME")
fi
TRAIN_ARGS+=(
  --level E97
  --dim 1792
  --n_heads 216
  --n_state 32
  --depth 11
  --expansion 1.0
  --use_gate 1
  --gate_activation silu
  --mlp_ratio 2.2623
  --mlp_multiple 64
  --use_triton 1
  --optimizer schedulefree
  --lr 0.001007
  --bf16
  --batch_size 4
  --chunk_size 2048
  --data /home/erikg/elman/data/pile.txt
  --tokenizer p50k_base
  --diloco
  --diloco_k 250
  --diloco_outer_lr 1.0
  --diloco_outer_beta 0.0
  --steps 100000000
  --save_every 500
  --keep_checkpoints 20
  --log_every 25
  --output "$OUTPUT"
)

exec scripts/launch_detached_run.sh \
  --name "$NAME" \
  --gpus 8 \
  --logdir "$LOGDIR" \
  -- "${ENV_ARGS[@]}" "${TRAIN_ARGS[@]}"
