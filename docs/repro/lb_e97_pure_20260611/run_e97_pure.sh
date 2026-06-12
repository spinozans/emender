#!/usr/bin/env bash
# lb-e97-pure: PURE E97 (--level E97, split erase/write edit) through the STANDARD
# 1.3B CMA driver, SAME data slice/offset/anchors as the OLD leaderboard
# (~/emender/experiments/local/cmaes_redo_1300m_20260529) so the best avg-loss is
# DIRECTLY comparable. Reproduction gate: pure-E97 must land near 5.95-5.97
# (old pure-e97 best = 5.9733, n_state=32). n_state PINNED 32. bf16 + fused Triton
# split-edit (--use_triton 1 via --use_triton_e88; E97 AUTO-resolves to Triton too).
set -euo pipefail
cd /home/erikg/ndm/.wg-worktrees/agent-1390

# Reserve all 8 GPUs (atomic, auto-release on shell exit, heartbeat keeper).
eval "$(scripts/gpu_lease.sh acquire 8)"
echo "[lb-e97-pure] leased GPUs: ${CUDA_VISIBLE_DEVICES:-<none>}"
# The driver sets CUDA_VISIBLE_DEVICES per train.py worker to the PHYSICAL gpu id
# from --gpus; unset the parent mask so those physical ids are not double-remapped.
unset CUDA_VISIBLE_DEVICES

export CMAES_MAX_VALID_ATTEMPTS=80   # match the old e97 queue

OUT=experiments/local/lb_e97_pure_20260611/e97
ANCHORS=docs/repro/cmaes_1300m_anchors/anchors_corrected.json

python -u scripts/cmaes_search_v2.py \
  --model e97 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output "$OUT" \
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
  --fixed_n_state 32 \
  --use_triton_e88 \
  --anchor_configs "$ANCHORS" \
  --anchor_only_cmaes
echo "[lb-e97-pure] e97 exit=$?"
