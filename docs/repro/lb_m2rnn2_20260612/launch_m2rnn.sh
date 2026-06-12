#!/usr/bin/env bash
# lb-m2rnn2 — CLEAN re-run of the STANDARD-driver 1.3B CMA-ES for the M2RNN
# baseline (matrix-to-matrix nonlinear RNN, Mishra/Tan/Stoica/Gonzalez/Dao).
# Supersedes lb-m2rnn (declared done 25s after launch; ran CONTENDED with
# emender-mlp -> corrupted). This run owns the WHOLE 8-GPU box, no concurrency.
#
# Symmetric budget to the gdn2-mlp / gdn2 / e97 / emender 1.3B searches: popsize 8,
# train_minutes 15, params 1300M, param_tolerance 0.03, chunk_size 2048,
# tokenizer p50k_base, sigma 0.8, min_generations 13 (>=104 evals). bf16 uniform
# (worker --bf16) + XMA FUSED Triton M2RNN kernel (--require_m2rnn_xma, NO eager;
# both built into the driver's m2rnn worker command). n_state is SWEPT as a free
# CMA dimension over {16,32} (NATIVE _E88 space — NOT pinned). SAME data slice as
# the gdn2-mlp sibling (pile.txt, seed 42) so avg-loss is directly comparable to
# the OLD leaderboard (m2rnn 6.1161). Anchor = old XMA long-racer dim1920/nh370/
# n16/dep21 (1.307B, +0.55%).
set -euo pipefail
cd /home/erikg/ndm/.wg-worktrees/agent-1405

ROOT="experiments/lb_m2rnn2_20260612"
ANCHORS="$ROOT/anchors_m2rnn.json"
GPU_FILE="$ROOT/gpus.txt"

# Lease all 8 idle GPUs via the broker (auto-release on shell exit).
eval "$(scripts/gpu_lease.sh 8)"
echo "leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "$CUDA_VISIBLE_DEVICES" > "$GPU_FILE"

export CMAES_MAX_VALID_ATTEMPTS=80
export CMAES_PARAM_TOLERANCE=0.03

echo "[m2rnn] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting"
python -u scripts/cmaes_search_v2.py \
  --model m2rnn \
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
echo "[m2rnn] $(date -u +%Y-%m-%dT%H:%M:%SZ) done exit=$?"
