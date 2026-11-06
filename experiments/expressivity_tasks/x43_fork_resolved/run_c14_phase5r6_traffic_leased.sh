#!/bin/bash
# Physical traffic counters, deliberately separate from CUPTI timing collection.
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round6
[[ ! -e "$r/traffic_terminal.json" ]]
cp "$GPU_LEASE_DIR/4" "$r/traffic_lease.txt"
rc=0
timeout 240 /usr/local/cuda-12.8/bin/ncu --target-processes all --csv \
 --metrics dram__bytes_read.sum,dram__bytes_write.sum --launch-count 1 \
 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r5_probe \
 --only1024 --matched-r4 --output "$r/traffic_probe_sweep.json" \
 > "$r/traffic_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/traffic_terminal.json"
exit "$rc"
