#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round2
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/lease_probe.txt"
label=${1:-sweep}
[[ "$label" =~ ^[a-zA-Z0-9_-]+$ ]]
[[ ! -e "$r/${label}.json" && ! -e "$r/${label}_terminal.json" ]]
rc=0
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r2_probe --output "$r/${label}.json" > "$r/${label}_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/${label}_terminal.json"
exit "$rc"
