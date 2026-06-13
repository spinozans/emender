#!/usr/bin/env bash
# Launch the high-p temporal separation sweep on N broker-leased GPUs.
# Waits (round-robins) until N GPUs are free, then runs the 40-job sweep.
# Lease auto-releases when this shell exits.
set -euo pipefail
cd /home/erikg/ndm/.wg-worktrees/agent-1415
# NB: do NOT name a var GROUPS — it is a bash special array (supplementary gids).
N="${1:-4}"
STEPS="${2:-50000}"
GRPS="${3:-main,ldepth,width,wdsweep}"
echo "[launch] $(date -u +%FT%TZ) requesting $N GPUs via broker (waits until free)..."
eval "$(scripts/gpu_lease.sh acquire "$N" --wait)"
echo "[launch] $(date -u +%FT%TZ) leased CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
python experiments/grok_highp_temporal/orchestrate_temporal.py \
    --gpus "$CUDA_VISIBLE_DEVICES" \
    --steps "$STEPS" --eval_interval 500 --patience_evals 40 \
    --groups "$GRPS" \
    --outdir experiments/grok_highp_temporal/runs
echo "[launch] $(date -u +%FT%TZ) sweep complete"
