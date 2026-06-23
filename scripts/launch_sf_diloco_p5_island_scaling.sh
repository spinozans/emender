#!/usr/bin/env bash
# SF-DiLoCo P5/P6 island-count scaling launcher.
#
# Runs the P5 frozen matrix for task sf-diloco-p6:
#   W in {2,4,8}, seeds 7000..7005, arms avg and sfsgd_y.
# W=2 uses up to four concurrent 2-GPU partitions, W=4 uses up to two
# concurrent 4-GPU partitions, and W=8 uses one true 8-GPU run.
set -euo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

K="${K:-250}"
STEPS="${STEPS:-1500}"
DATA="${DATA:-/home/erikg/elman/data/pile.txt}"
HELDOUT_TENSOR="${HELDOUT_TENSOR:-experiments/lb_compare_20260613/heldout_p50k_2048.pt}"
OUT_ROOT="${OUT_ROOT:-/mnt/nvme1n1/erikg/sf_diloco_p5_island_scaling}"
LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs}"
RUN_DIR="${RUN_DIR:-$OUT_ROOT/runs}"
CURVE_DIR="${CURVE_DIR:-$OUT_ROOT/curves}"
HELDOUT_EVERY="${HELDOUT_EVERY:-250}"
HELDOUT_EVAL_BS="${HELDOUT_EVAL_BS:-8}"
SEEDS="${SEEDS:-7000 7001 7002 7003 7004 7005}"
WORLD_SIZES="${WORLD_SIZES:-2 4 8}"
ALLOW_OVERWRITE="${ALLOW_OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"

mkdir -p "$OUT_ROOT" "$LOG_DIR" "$RUN_DIR" "$CURVE_DIR"

if [[ ! -r "$DATA" ]]; then
  echo "DATA is not readable: $DATA" >&2
  exit 2
fi
if [[ ! -r "$HELDOUT_TENSOR" ]]; then
  echo "HELDOUT_TENSOR is not readable: $HELDOUT_TENSOR" >&2
  exit 2
fi

parallelism_for_w() {
  case "$1" in
    2) echo 4 ;;
    4) echo 2 ;;
    8) echo 1 ;;
    *) echo "unsupported W=$1" >&2; return 2 ;;
  esac
}

arms_for_seed() {
  local seed="$1"
  if (( seed % 2 == 0 )); then
    echo "avg sfsgd_y"
  else
    echo "sfsgd_y avg"
  fi
}

arm_args() {
  case "$1" in
    avg)
      printf '%s\n' \
        --diloco_outer_optimizer avg \
        --diloco_outer_lr 1.0 \
        --diloco_outer_beta 0.0
      ;;
    sfsgd_y)
      printf '%s\n' \
        --diloco_outer_optimizer sfsgd \
        --diloco_export_basis y \
        --diloco_outer_lr 1.0 \
        --diloco_outer_beta 0.1
      ;;
    *)
      echo "unknown arm: $1" >&2
      return 2
      ;;
  esac
}

write_plan() {
  python - "$OUT_ROOT/planned_matrix.json" "$SEEDS" "$WORLD_SIZES" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]
world_sizes = [int(x) for x in sys.argv[3].split()]
runs = []
for w in world_sizes:
    for seed in seeds:
        arms = ["avg", "sfsgd_y"] if seed % 2 == 0 else ["sfsgd_y", "avg"]
        for order, arm in enumerate(arms, start=1):
            label = f"W{w:02d}_seed{seed}_{arm}"
            runs.append({
                "world_size": w,
                "seed": seed,
                "arm": arm,
                "order_within_seed": order,
                "label": label,
                "log": f"logs/{label}.log",
                "curve": f"curves/{label}_heldout_curve.csv",
                "run_dir": f"runs/{label}",
            })
out.write_text(json.dumps({
    "task": "sf-diloco-p6",
    "source_plan": "docs/experiments/SF_DILOCO_P5_8GPU_REPLICATION_MATRIX.md",
    "world_sizes": world_sizes,
    "seeds": seeds,
    "arms": ["avg", "sfsgd_y"],
    "runs": runs,
}, indent=2) + "\n")
print(out)
PY
}

check_target() {
  local label="$1"
  local output="$RUN_DIR/$label"
  local logfile="$LOG_DIR/$label.log"
  local curve="$CURVE_DIR/${label}_heldout_curve.csv"
  if [[ "$ALLOW_OVERWRITE" != "1" ]]; then
    if [[ -e "$output" || -e "$logfile" || -e "$curve" ]]; then
      echo "Refusing to overwrite existing target for $label. Set ALLOW_OVERWRITE=1 to replace." >&2
      exit 3
    fi
  fi
}

run_one() {
  local w="$1"
  local seed="$2"
  local arm="$3"
  local label="W$(printf '%02d' "$w")_seed${seed}_${arm}"
  local output="$RUN_DIR/$label"
  local logfile="$LOG_DIR/$label.log"
  local curve="$CURVE_DIR/${label}_heldout_curve.csv"
  local -a extra_args
  mapfile -t extra_args < <(arm_args "$arm")

  check_target "$label"
  if [[ "$ALLOW_OVERWRITE" == "1" ]]; then
    rm -rf "$output"
    rm -f "$logfile" "$curve"
  fi
  mkdir -p "$output"

  local -a cmd=(
    torchrun --standalone --nproc_per_node="$w" train.py
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
    --data "$DATA"
    --tokenizer p50k_base
    --diloco
    --diloco_k "$K"
    "${extra_args[@]}"
    --steps "$STEPS"
    --seed "$seed"
    --save_every 100000000
    --keep_checkpoints 1
    --log_every 25
    --heldout_tensor "$HELDOUT_TENSOR"
    --heldout_eval_mode x
    --heldout_curve_every "$HELDOUT_EVERY"
    --heldout_curve_path "$curve"
    --final_heldout_eval
    --output "$output"
  )

  {
    echo "[p5] task=sf-diloco-p6 label=$label W=$w seed=$seed arm=$arm"
    echo "[p5] output=$output"
    echo "[p5] curve=$curve"
    echo "[p5] log=$logfile"
    echo "[p5] planned WORLD_SIZE=$w"
    echo "[p5] acquiring $w GPU lease (--no-wait)"
  } | tee "$logfile"

  (
    set -euo pipefail
    eval "$(scripts/gpu_lease.sh acquire "$w" --no-wait)"
    {
      echo "[p5] acquired CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
      echo "[p5] env NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
      printf '[p5] command:'
      printf ' %q' "${cmd[@]}"
      printf '\n'
    } | tee -a "$logfile"
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "[p5] DRY_RUN=1, not executing $label" | tee -a "$logfile"
      return 0
    fi
    env \
      NCCL_P2P_DISABLE=1 \
      TORCH_NCCL_ENABLE_MONITORING=0 \
      TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      HELDOUT_EVAL_BS="$HELDOUT_EVAL_BS" \
      "${cmd[@]}" 2>&1 | tee -a "$logfile"
  )
}

run_group() {
  local -a specs=("$@")
  local -a pids=()
  local spec
  for spec in "${specs[@]}"; do
    IFS=: read -r w seed arm <<<"$spec"
    run_one "$w" "$seed" "$arm" &
    pids+=("$!")
  done

  local failures=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  if (( failures > 0 )); then
    echo "[p5] group completed with $failures failed run(s)" >&2
    return 1
  fi
}

write_plan

for w in $WORLD_SIZES; do
  par="$(parallelism_for_w "$w")"
  echo "[p5] Starting W=$w with concurrency=$par"
  pending=()
  for seed in $SEEDS; do
    for arm in $(arms_for_seed "$seed"); do
      pending+=("$w:$seed:$arm")
      if (( ${#pending[@]} == par )); then
        run_group "${pending[@]}"
        pending=()
      fi
    done
  done
  if (( ${#pending[@]} > 0 )); then
    run_group "${pending[@]}"
  fi
done

python scripts/analyze_sf_diloco_p5.py "$OUT_ROOT"

echo "[p5] summary: $OUT_ROOT/summary.json"
echo "[p5] pairs: $OUT_ROOT/paired_by_w.json"
echo "[p5] scaling decision: $OUT_ROOT/scaling_decision.json"
