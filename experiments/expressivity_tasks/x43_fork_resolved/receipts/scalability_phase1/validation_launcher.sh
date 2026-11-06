#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE=3,5,6 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
[[ "$GPU_LEASE_HELD" =~ ^[356]$ ]]
receipt=experiments/expressivity_tasks/x43_fork_resolved/receipts/scalability_phase1
mkdir -p "$receipt"
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt/lease.txt"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_validation --warps 8 --output "$receipt/baseline" > "$receipt/baseline_console.txt" 2>&1
set +e
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_validation --warps 16 --output "$receipt/warps16" --compare "$receipt/baseline/fingerprint.json" > "$receipt/warps16_console.txt" 2>&1
printf '{"exit_code":%d}\n' "$?" > "$receipt/warps16_terminal.json"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_validation --warps 32 --output "$receipt/warps32" --compare "$receipt/baseline/fingerprint.json" > "$receipt/warps32_console.txt" 2>&1
printf '{"exit_code":%d}\n' "$?" > "$receipt/warps32_terminal.json"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_validation --warps 16 --output "$receipt/unused" --battery > "$receipt/battery16.txt" 2>&1
printf '{"exit_code":%d}\n' "$?" > "$receipt/battery16_terminal.json"
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_launch_curve --output "$receipt/fixed_curve" > "$receipt/fixed_curve_console.txt" 2>&1
printf '{"exit_code":%d}\n' "$?" > "$receipt/fixed_curve_terminal.json"
