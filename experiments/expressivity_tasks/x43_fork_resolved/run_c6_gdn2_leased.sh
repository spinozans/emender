#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export GPU_LEASE_VISIBLE=2,4 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --no-wait)"
eval "$lease_exports"
[[ "${GPU_LEASE_HELD:-}" =~ ^[24]$ ]] || exit 90
receipt=experiments/expressivity_tasks/x43_fork_resolved/receipts/c6
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt/lease_gdn2.txt"
set +e
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x43_fork_resolved.c6_gdn2 > "$receipt/console_gdn2.txt" 2>&1
status=$?
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt/terminal_gdn2.json"
exit "$status"
