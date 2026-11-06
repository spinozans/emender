#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=3,5,6 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" =~ ^[356]$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase1
printf '%s\n' "$GPU_LEASE_HELD" > "$r/gradient_lease.txt"
set +e
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_shape_gradient_parity --output "$r/gradient_parity.json" > "$r/gradient_parity_console.txt" 2>&1
status=$?
printf '{"exit_code":%d}\n' "$status" > "$r/gradient_parity_terminal.json"
exit "$status"
