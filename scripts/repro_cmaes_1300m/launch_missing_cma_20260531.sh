#!/usr/bin/env bash
set -euo pipefail

cd /home/erikg/emender

ROOT=experiments/local/cmaes_redo_1300m_20260529
LOG_DIR="$ROOT/logs"
ANCHORS="$ROOT/anchors_missing_20260531.json"
mkdir -p "$LOG_DIR"

export XMA_PATH="${XMA_PATH:-/home/erikg/xma}"

COMMON=(
  --phase cmaes
  --params 1300M
  --param_tolerance 0.03
  --train_minutes 15
  --popsize 8
  --sigma 0.8
  --chunk_size 2048
  --tokenizer p50k_base
  --data /home/erikg/elman/data/pile.txt
  --min_generations 8
  --anchor_configs "$ANCHORS"
  --anchor_only_cmaes
)

setsid env XMA_PATH="$XMA_PATH" uv run python -u scripts/cmaes_search_v2.py \
  --model e88-raw --gpus 4 --output "$ROOT/e88-raw" \
  "${COMMON[@]}" --use_triton_e88 \
  > "$LOG_DIR/e88-raw.log" 2>&1 < /dev/null &
echo "e88-raw $!" >> "$ROOT/launch_pids.txt"

setsid env XMA_PATH="$XMA_PATH" uv run python -u scripts/cmaes_search_v2.py \
  --model e88-linear --gpus 5 --output "$ROOT/e88-linear" \
  "${COMMON[@]}" --use_triton_e88 \
  > "$LOG_DIR/e88-linear.log" 2>&1 < /dev/null &
echo "e88-linear $!" >> "$ROOT/launch_pids.txt"

setsid env XMA_PATH="$XMA_PATH" uv run python -u scripts/cmaes_search_v2.py \
  --model fla-gdn --gpus 6 --output "$ROOT/fla-gdn" \
  "${COMMON[@]}" \
  > "$LOG_DIR/fla-gdn.log" 2>&1 < /dev/null &
echo "fla-gdn $!" >> "$ROOT/launch_pids.txt"

setsid env XMA_PATH="$XMA_PATH" uv run python -u scripts/cmaes_search_v2.py \
  --model m2rnn --gpus 7 --output "$ROOT/m2rnn" \
  "${COMMON[@]}" \
  > "$LOG_DIR/m2rnn.log" 2>&1 < /dev/null &
echo "m2rnn $!" >> "$ROOT/launch_pids.txt"
