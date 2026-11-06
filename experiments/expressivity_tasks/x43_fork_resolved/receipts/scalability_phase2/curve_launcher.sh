#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=3,5,6 GPU_LEASE_HB_INTERVAL=10 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" =~ ^[356]$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase2
printf '%s\n' "$GPU_LEASE_HELD" > "$r/curve_lease.txt"
rc=0
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_curve --variant rowkeys --output "$r/curve_rowkeys" --gates "$r" --registration "$r/registration.json" > "$r/curve_console.txt" 2>&1 || rc=$?
printf '{"exit_code":%d}\n' "$rc" > "$r/curve_terminal.json"
exit "$rc"
