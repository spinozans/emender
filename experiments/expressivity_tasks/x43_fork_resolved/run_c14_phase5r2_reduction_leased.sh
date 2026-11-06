#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=5 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 5 ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase5_round2
[[ ! -e "$r/reduction.json" && ! -e "$r/reduction_terminal.json" ]]
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/lease_reduction.txt"
rc=0
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase5r2_reduction_probe --output "$r/reduction.json" > "$r/reduction_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/reduction_terminal.json"
exit "$rc"
