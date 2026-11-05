#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export TORCHINDUCTOR_COMPILE_THREADS=1 GPU_LEASE_VISIBLE=2,4 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
lease_start=$SECONDS
eval "$(scripts/gpu_lease.sh acquire 1 --no-wait)"
export C14_B_LEASE_WAIT_SECONDS=$((SECONDS-lease_start))
[[ "${GPU_LEASE_HELD:-}" =~ ^[24]$ ]] || exit 90
receipt=experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_matched
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt/train_lease.txt"
set +e
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x43_fork_resolved/c14_b_matched.py train "$@" >> "$receipt/train_console.txt" 2>&1
status=$?
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt/train_terminal.json"
exit "$status"
