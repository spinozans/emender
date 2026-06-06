#!/usr/bin/env bash
set -euo pipefail

cd /home/erikg/emender

export CMAES_MAX_VALID_ATTEMPTS=80

ROOT="experiments/local/cmaes_redo_1300m_20260529"
ANCHORS="$ROOT/anchors_gdn2_primary.json"

echo "[gdn2-redo] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting gdn2"
uv run python -u scripts/cmaes_search_v2.py \
  --model gdn2 \
  --gpus 6,7 \
  --output "$ROOT/gdn2" \
  --phase cmaes \
  --params 1300M \
  --param_tolerance 0.03 \
  --train_minutes 15 \
  --popsize 8 \
  --sigma 0.6 \
  --chunk_size 2048 \
  --tokenizer p50k_base \
  --data /home/erikg/elman/data/pile.txt \
  --min_generations 12 \
  --anchor_configs "$ANCHORS" \
  --anchor_only_cmaes
status=$?
echo "[gdn2-redo] $(date -u +%Y-%m-%dT%H:%M:%SZ) gdn2 exit=$status"
