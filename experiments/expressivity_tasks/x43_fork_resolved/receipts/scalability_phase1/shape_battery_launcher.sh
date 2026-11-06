#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=3,5,6 OMP_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" =~ ^[356]$ ]]
r=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase1
printf '%s\n' "$GPU_LEASE_HELD" > "$r/shape_battery_lease.txt"
set +e
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_shape_parity --output "$r/shape_parity.json" > "$r/shape_parity_console.txt" 2>&1
printf '{"exit_code":%d}\n' "$?" > "$r/shape_parity_terminal.json"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_shape --warps 8 --output "$r/unused" --battery > "$r/shape_battery.txt" 2>&1
printf '{"exit_code":%d}\n' "$?" > "$r/shape_battery_terminal.json"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_shape --warps 8 --output "$r/shape_final" --compare "$r/baseline/fingerprint.json" > "$r/shape_final_console.txt" 2>&1
printf '{"exit_code":%d}\n' "$?" > "$r/shape_final_terminal.json"
