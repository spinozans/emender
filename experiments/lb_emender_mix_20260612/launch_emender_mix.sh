#!/usr/bin/env bash
# lb-emender-mix — STANDARD-driver 1.3B CMA-ES over the typed-gdn2 Emender with the
# FULL-RANGE mixture axis: 0% nonlinear (pure gdn2_recall) -> 100% (pure e97 split-edit).
# Symmetric budget to the gdn2/m2rnn/e97 1.3B searches: popsize 8, train_minutes 15,
# params 1300M, param_tolerance 0.03, chunk_size 2048, tokenizer p50k_base, sigma 0.8,
# min_generations 13 (>=104 evals). n_state PINNED 32, expansion 1.0. bf16 uniform + fused.
# SAME data slice as ~/emender/experiments/local/cmaes_redo_1300m_20260529 (pile.txt).
set -euo pipefail
cd /home/erikg/ndm/.wg-worktrees/agent-1393

ROOT="experiments/lb_emender_mix_20260612"
ANCHORS="$ROOT/anchors_emender.json"
GPU_FILE="$ROOT/gpus.txt"

# Lease all 8 idle GPUs via the broker (auto-release on shell exit).
eval "$(scripts/gpu_lease.sh 8)"
echo "leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "$CUDA_VISIBLE_DEVICES" > "$GPU_FILE"

export CMAES_MAX_VALID_ATTEMPTS=80
export CMAES_PARAM_TOLERANCE=0.03

echo "[emender-mix] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
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
  --min_generations 13 \
  --anchor_configs "$ANCHORS" \
  --anchor_only_cmaes
echo "[emender-mix] $(date -u +%Y-%m-%dT%H:%M:%SZ) done exit=$?"
