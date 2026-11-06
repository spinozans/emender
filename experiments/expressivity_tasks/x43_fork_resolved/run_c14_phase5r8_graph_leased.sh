#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
label=${1:-graph}
[[ "$label" =~ ^[a-z0-9_]+$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round8
[[ ! -e "$r/${label}_terminal.json" ]]
cp "$GPU_LEASE_DIR/4" "$r/${label}_lease.txt"
rc=0
timeout 240 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r8_graph_probe "$r/${label}.json" > "$r/${label}_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/${label}_terminal.json"
exit "$rc"
