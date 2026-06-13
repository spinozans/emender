#!/usr/bin/env bash
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-1411
eval "$(scripts/gpu_lease.sh 2)"
echo "BPB2_LEASED: $CUDA_VISIBLE_DEVICES"
export HELDOUT_EVAL_BS=8
python3 experiments/lb_compare_20260613/run_bpb.py --gpus "$CUDA_VISIBLE_DEVICES" --train_minutes 15 \
   > experiments/lb_compare_20260613/bpb_driver2.log 2>&1 &
P=$!
wait $P; echo "BPB2_DONE rc=$?"
scripts/gpu_lease.sh release >/dev/null 2>&1
echo "BPB2_ALL_DONE_LEASE_RELEASED"
