#!/usr/bin/env bash
set -euo pipefail

cd /home/erikg/emender

export CMAES_MAX_VALID_ATTEMPTS=80

ROOT="experiments/local/cmaes_redo_1300m_20260529"
ANCHORS="$ROOT/anchors_corrected.json"

for model in e97 e97-raw e97-linear; do
  echo "[e97-redo] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting $model"
  uv run python -u scripts/cmaes_search_v2.py \
    --model "$model" \
    --gpus 4,5 \
    --output "$ROOT/$model" \
    --phase cmaes \
    --params 1300M \
    --param_tolerance 0.03 \
    --train_minutes 15 \
    --popsize 8 \
    --sigma 0.8 \
    --chunk_size 2048 \
    --tokenizer p50k_base \
    --data /home/erikg/elman/data/pile.txt \
    --min_generations 8 \
    --use_triton_e88 \
    --anchor_configs "$ANCHORS" \
    --anchor_only_cmaes
  status=$?
  echo "[e97-redo] $(date -u +%Y-%m-%dT%H:%M:%SZ) $model exit=$status"
done
