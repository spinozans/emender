#!/bin/bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
lease_exports=$(scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$lease_exports"
[[ "$GPU_LEASE_HELD" == 4 ]]
mode=${1:?}; context=${2:?}; label=${mode}_t${context}${3:-}
[[ "$mode" =~ ^stress(48|2049)$ && "$context" =~ ^(128|256)$ ]]
[[ "$label" =~ ^[a-z0-9_]+$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/protocol_gap
mkdir -p "$r"
[[ ! -e "$r/${label}_terminal.json" ]]
cp "$GPU_LEASE_DIR/4" "$r/${label}_lease.txt"
cp experiments/expressivity_tasks/x43_fork_resolved/c14_gap_receipts.py "$r/${label}_source.py.txt"
rc=0
timeout 600 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_gap_receipts "$mode" --context "$context" --output "$r/${label}.json" > "$r/${label}_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/${label}_terminal.json"
exit "$rc"
