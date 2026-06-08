#!/bin/bash
cd /home/erikg/ndm/.wg-worktrees/agent-1255
echo "waiting for a free GPU (util<15 and mem<4000MB)..."
for attempt in $(seq 1 90); do
  line=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | awk -F', ' '$2<4000 && $3<15 {print $1; exit}')
  if [ -n "$line" ]; then
    echo "FREE GPU FOUND: $line at attempt $attempt"
    CUDA_VISIBLE_DEVICES=$line python experiments/hetero_kernel/final_bench.py 2>&1 | grep -v -i "warning\|fla\b\|Variable._exec"
    exit 0
  fi
  sleep 20
done
echo "NO FREE GPU after 30 min; running on least-loaded GPU under contention (LOWER BOUND)"
gpu=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | awk -F',' '{print $1}')
echo "using GPU $gpu (contended)"
CUDA_VISIBLE_DEVICES=$gpu python experiments/hetero_kernel/final_bench.py 2>&1 | grep -v -i "warning\|fla\b\|Variable._exec"
