#!/usr/bin/env bash
# lb-compare orchestrator: ONE process owns the GPU lease set (no cross-agent
# contention). Held-out BPB (5 GPUs, 15-min train each) + formal separators
# (3 GPUs) run concurrently. Waits on the two child PIDs (NOT bare `wait`, which
# hangs on the lease heartbeat-keeper).
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-1411
eval "$(scripts/gpu_lease.sh 8)"
echo "LEASED: $CUDA_VISIBLE_DEVICES"
export HELDOUT_EVAL_BS=8
IFS=',' read -ra G <<< "$CUDA_VISIBLE_DEVICES"
BPB_GPUS="${G[0]},${G[1]},${G[2]},${G[3]},${G[4]}"
SEP_GPUS="${G[5]},${G[6]},${G[7]}"
echo "BPB on $BPB_GPUS ; SEP on $SEP_GPUS"

python3 experiments/lb_compare_20260613/run_bpb.py --gpus "$BPB_GPUS" --train_minutes 15 \
   > experiments/lb_compare_20260613/bpb_driver.log 2>&1 &
BPB_PID=$!
python3 experiments/lb_compare_20260613/run_separators.py --gpus "$SEP_GPUS" --seeds 0,1 \
   > experiments/lb_compare_20260613/sep_driver.log 2>&1 &
SEP_PID=$!
echo "BPB_PID=$BPB_PID SEP_PID=$SEP_PID"

wait $BPB_PID; echo "BPB_DONE rc=$?"
wait $SEP_PID; echo "SEP_DONE rc=$?"
scripts/gpu_lease.sh release >/dev/null 2>&1
echo "ALL_DONE_LEASE_RELEASED"
