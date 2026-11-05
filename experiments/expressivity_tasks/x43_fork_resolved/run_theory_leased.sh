#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
phase="${1:?rounds or forced}"
case "$phase" in
 rounds) export GPU_LEASE_VISIBLE=3,5; module=theory_rounds_sweep; receipt=theory_rounds ;;
 forced) export GPU_LEASE_VISIBLE=5,6; module=theory_forced_span; receipt=theory_forced_span ;;
 *) exit 64 ;;
esac
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --no-wait)"
eval "$lease_exports"
output="experiments/expressivity_tasks/x43_fork_resolved/receipts/$receipt"
mkdir -p "$output"
printf '%s\n' "$GPU_LEASE_HELD" > "$output/lease.txt"
set +e
/home/erikg/emender/.venv/bin/python -m "experiments.expressivity_tasks.x43_fork_resolved.$module" > "$output/console.txt" 2>&1
status=$?
set -e
printf '{"exit_code":%d}\n' "$status" > "$output/terminal.json"
exit "$status"
