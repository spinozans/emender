#!/usr/bin/env bash
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-1411
eval "$(scripts/gpu_lease.sh 6)"
echo "SEP_LEASED: $CUDA_VISIBLE_DEVICES"
python3 experiments/lb_compare_20260613/run_separators.py --gpus "$CUDA_VISIBLE_DEVICES" --seeds 0,1 \
   > experiments/lb_compare_20260613/sep_driver2.log 2>&1 &
P=$!
wait $P; echo "SEP2_DONE rc=$?"
scripts/gpu_lease.sh release >/dev/null 2>&1
echo "SEP_ALL_DONE_LEASE_RELEASED"
