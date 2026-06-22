#!/usr/bin/env bash
# launch_sf_diloco_p4.sh  (task sf-diloco-p4)
#
# Sequential clean comparison of the four ScheduleFree-DiLoCo outer regimes:
#   A: avg outer
#   B: geometry-fixed fixed-momentum outer, using P2's best matched-gain config
#      among momentum arms (outer_beta=0.5, outer_lr=0.5)
#   C: sfsgd outer, export x
#   D: sfsgd outer, export y
#
# Each arm is from scratch, matched tokens, bf16 E97 fused Triton only, real Pile
# data, and leases GPUs via scripts/gpu_lease.sh acquire --no-wait. Arms run
# sequentially to avoid co-contention.
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

K="${K:-250}"
STEPS="${STEPS:-1100}"
GPUS="${GPUS:-2}"
DATA="${DATA:-/home/erikg/elman/data/pile.txt}"
HELDOUT_TENSOR="${HELDOUT_TENSOR:-experiments/lb_compare_20260613/heldout_p50k_2048.pt}"
OUT_ROOT="${OUT_ROOT:-/mnt/nvme1n1/erikg/sf_diloco_p4_outer_regimes}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
HELDOUT_EVERY="${HELDOUT_EVERY:-250}"
SFSGD_OUTER_LR="${SFSGD_OUTER_LR:-1.0}"
SFSGD_OUTER_BETA="${SFSGD_OUTER_BETA:-0.1}"
MOM_OUTER_LR="${MOM_OUTER_LR:-0.5}"
MOM_OUTER_BETA="${MOM_OUTER_BETA:-0.5}"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

if [[ ! -r "$DATA" ]]; then
  echo "DATA is not readable: $DATA" >&2
  exit 2
fi
if [[ ! -r "$HELDOUT_TENSOR" ]]; then
  echo "HELDOUT_TENSOR is not readable: $HELDOUT_TENSOR" >&2
  exit 2
fi

run_arm() {
  local label="$1"
  shift
  local output="$OUT_ROOT/$label"
  local logfile="$LOG_DIR/${label}.log"
  local curve="$OUT_ROOT/${label}_heldout_curve.csv"
  rm -rf "$output"
  mkdir -p "$output"

  echo "[p4] acquiring $GPUS GPU lease for $label (--no-wait) ..."
  (
    set -euo pipefail
    eval "$(scripts/gpu_lease.sh acquire "$GPUS" --no-wait)"
    echo "[p4] arm=$label CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES K=$K STEPS=$STEPS LOG=$logfile"
    env \
      NCCL_P2P_DISABLE=1 \
      TORCH_NCCL_ENABLE_MONITORING=0 \
      TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      HELDOUT_EVAL_BS="${HELDOUT_EVAL_BS:-8}" \
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
        "$@" \
        --steps "$STEPS" \
        --save_every 100000000 \
        --keep_checkpoints 1 \
        --log_every 25 \
        --heldout_tensor "$HELDOUT_TENSOR" \
        --heldout_eval_mode x \
        --heldout_curve_every "$HELDOUT_EVERY" \
        --heldout_curve_path "$curve" \
        --final_heldout_eval \
        --output "$output" 2>&1 | tee "$logfile"
  )
  echo "[p4] DONE arm=$label -> $logfile"
}

run_arm "A_avg" \
  --diloco_outer_optimizer avg \
  --diloco_outer_lr 1.0 \
  --diloco_outer_beta 0.0

run_arm "B_momentum_beta05_lr05" \
  --diloco_outer_optimizer momentum \
  --diloco_outer_lr "$MOM_OUTER_LR" \
  --diloco_outer_beta "$MOM_OUTER_BETA"

run_arm "C_sfsgd_export_x" \
  --diloco_outer_optimizer sfsgd \
  --diloco_export_basis x \
  --diloco_outer_lr "$SFSGD_OUTER_LR" \
  --diloco_outer_beta "$SFSGD_OUTER_BETA"

run_arm "D_sfsgd_export_y" \
  --diloco_outer_optimizer sfsgd \
  --diloco_export_basis y \
  --diloco_outer_lr "$SFSGD_OUTER_LR" \
  --diloco_outer_beta "$SFSGD_OUTER_BETA"

python scripts/analyze_sf_diloco_p4.py "$LOG_DIR" --out "$OUT_ROOT/summary.json"

echo "[p4] summary: $OUT_ROOT/summary.json"
