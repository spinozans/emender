#!/usr/bin/env bash
# launch_sf_diloco_sfsgd_smoke.sh  (task sf-diloco-p3)
#
# Short real-data ScheduleFree-DiLoCo smoke for the separate OUTER sfsgd state
# machine. Uses the same fused E97/p50k/bf16 setup as the P1 plain-average
# baseline, but routes the DiLoCo boundary through:
#
#   --diloco_outer_optimizer sfsgd --diloco_export_basis x
#
# Self-leasing: acquires N exclusive GPUs via scripts/gpu_lease.sh (--no-wait)
# and auto-releases on exit. REAL data, REAL fused Triton kernel.
#
# Usage:
#   scripts/launch_sf_diloco_sfsgd_smoke.sh
#   OUTER_LR=0.5 OUTER_BETA=0.05 STEPS=350 scripts/launch_sf_diloco_sfsgd_smoke.sh
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

K="${K:-100}"
STEPS="${STEPS:-350}"
GPUS="${GPUS:-2}"
OUTER_LR="${OUTER_LR:-1.0}"
OUTER_BETA="${OUTER_BETA:-0.1}"
EXPORT_BASIS="${EXPORT_BASIS:-x}"
DATA="${DATA:-/home/erikg/elman/data/pile.txt}"
OUTPUT="${OUTPUT:-/tmp/sf_diloco_sfsgd_smoke/k${K}_lr${OUTER_LR}_b${OUTER_BETA}_${EXPORT_BASIS}}"
LOGFILE="${LOGFILE:-/tmp/sf_diloco_sfsgd_smoke_k${K}_lr${OUTER_LR}_b${OUTER_BETA}_${EXPORT_BASIS}.log}"
mkdir -p "$OUTPUT"

echo "[sfsgd-smoke] acquiring $GPUS GPU lease (--no-wait) ..."
eval "$(scripts/gpu_lease.sh acquire "$GPUS" --no-wait)"
echo "[sfsgd-smoke] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES K=$K STEPS=$STEPS OUTER_LR=$OUTER_LR OUTER_BETA=$OUTER_BETA EXPORT_BASIS=$EXPORT_BASIS LOG=$LOGFILE"

ENV_ARGS=(
  env
  NCCL_P2P_DISABLE=1
  TORCH_NCCL_ENABLE_MONITORING=0
  TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
)

"${ENV_ARGS[@]}" \
  torchrun --standalone --nproc_per_node="$GPUS" train.py \
  --level E97 \
  --dim 1792 \
  --n_heads 216 \
  --n_state 32 \
  --depth 11 \
  --expansion 1.0 \
  --use_gate 1 \
  --gate_activation silu \
  --mlp_ratio 2.2623 \
  --mlp_multiple 64 \
  --use_triton 1 \
  --optimizer schedulefree \
  --lr 0.001007 \
  --bf16 \
  --batch_size 4 \
  --chunk_size 2048 \
  --data "$DATA" \
  --tokenizer p50k_base \
  --diloco \
  --diloco_k "$K" \
  --diloco_outer_optimizer sfsgd \
  --diloco_export_basis "$EXPORT_BASIS" \
  --diloco_outer_lr "$OUTER_LR" \
  --diloco_outer_beta "$OUTER_BETA" \
  --steps "$STEPS" \
  --save_every "$K" \
  --keep_checkpoints 2 \
  --log_every 25 \
  --output "$OUTPUT" 2>&1 | tee "$LOGFILE"

echo "[sfsgd-smoke] DONE K=$K OUTER_LR=$OUTER_LR OUTER_BETA=$OUTER_BETA EXPORT_BASIS=$EXPORT_BASIS -> $LOGFILE"
