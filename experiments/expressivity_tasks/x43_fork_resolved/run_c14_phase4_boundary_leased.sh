#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase4
while [[ ! -f "$r/terminal.json" ]]; do sleep 10; done
export GPU_LEASE_VISIBLE=4 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --wait --timeout 60)"
[[ "$GPU_LEASE_HELD" == 4 ]]
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "$r/boundary_lease_initial.txt"
rc=0
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase4_boundary > "$r/boundary_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/boundary_terminal.json"
exit "$rc"
