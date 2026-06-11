#!/usr/bin/env bash
# emender-1p3b-cma — STANDARD-driver 1.3B CMA-ES over the typed-gdn2 Emender.
# SYMMETRIC budget to the gdn2 (launch_gdn2.sh) and m2rnn (launch_missing_cma_20260531.sh)
# 1.3B searches: popsize 8, train_minutes 15, params 1300M, param_tolerance 0.03,
# chunk_size 2048, tokenizer p50k_base, sigma 0.8, min_generations 8, anchor-warm-started.
set -euo pipefail
cd /home/erikg/ndm/.wg-worktrees/agent-1380

ROOT="experiments/emender_1p3b_cma"
ANCHORS="$ROOT/anchors_emender.json"
GPU_FILE="$ROOT/gpus.txt"

# Lease all 8 idle GPUs via the broker (auto-release on shell exit).
eval "$(scripts/gpu_lease.sh 8)"
echo "leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# The driver uses ABSOLUTE GPU ids; with a full-box lease they are 0..7. Write them
# to the gpu_file so the driver schedules one candidate per leased GPU.
echo "$CUDA_VISIBLE_DEVICES" > "$GPU_FILE"

export CMAES_MAX_VALID_ATTEMPTS=80
export CMAES_PARAM_TOLERANCE=0.03

echo "[emender-cma] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
python -u scripts/cmaes_search_v2.py \
  --model emender \
  --gpu_file "$GPU_FILE" \
  --output "$ROOT/search" \
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
  --anchor_configs "$ANCHORS" \
  --anchor_only_cmaes
echo "[emender-cma] $(date -u +%Y-%m-%dT%H:%M:%SZ) done exit=$?"
