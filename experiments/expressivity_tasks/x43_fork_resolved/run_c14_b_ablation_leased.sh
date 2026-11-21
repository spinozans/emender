#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
control="${1:?control required}"; shift
[[ "$control" == append_only || "$control" == unbudgeted ]] || exit 91
export GPU_LEASE_VISIBLE=2,4 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
lease_start=$SECONDS
lease_exports="$(scripts/gpu_lease.sh acquire 1 --no-wait)"
eval "$lease_exports"
export C14_B_LEASE_WAIT_SECONDS=$((SECONDS-lease_start))
[[ "${GPU_LEASE_HELD:-}" =~ ^[24]$ ]] || exit 90
receipt="experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_${control}"
mkdir -p "$receipt"
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt/lease.txt"
set +e
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_b_ablation "$control" train "$@" > "$receipt/training_console.txt" 2>&1
status=$?
if [[ "$status" == 0 ]]; then
 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c14_b_ablation "$control" eval > "$receipt/eval_console.txt" 2>&1
 status=$?
fi
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt/terminal.json"
exit "$status"
