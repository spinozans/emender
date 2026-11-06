#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" == 4 ]]
mode=${1:?}; label=${2:-$mode}
[[ "$label" =~ ^[a-z0-9_]+$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/gap_decomposition
[[ ! -e "$r/${label}_terminal.json" ]]
cp "$GPU_LEASE_DIR/4" "$r/${label}_lease.txt"
cp experiments/expressivity_tasks/x43_fork_resolved/c14_gap_receipts.py "$r/${label}_source.py.txt"
rc=0
timeout 600 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_gap_receipts "$mode" --output "$r/${label}.json" > "$r/${label}_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/${label}_terminal.json"
exit "$rc"
