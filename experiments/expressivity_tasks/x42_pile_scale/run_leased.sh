#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GPU_LEASE_VISIBLE=1,5,6
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait)"
eval "$lease_exports"
[[ "${GPU_LEASE_HELD:-}" =~ ^[156]$ ]] || exit 90
# Broker installs release traps and starts its existing heartbeat keeper.
printf 'X42 physical card %s\n' "$GPU_LEASE_HELD"
receipt_dir="${X42_LAUNCH_RECEIPTS:-experiments/expressivity_tasks/x42_pile_scale/runs/launch_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
mkdir -p "$receipt_dir"
git rev-parse HEAD > "$receipt_dir/base_commit.txt"
nvidia-smi > "$receipt_dir/gpu_before.txt"
printf '%s\n' "$GPU_LEASE_HELD" > "$receipt_dir/lease.txt"
printf '%q ' "$@" > "$receipt_dir/command.txt"
set +e
/home/erikg/emender/.venv/bin/python "$@" 2>&1 | tee "$receipt_dir/console.txt"
status=${PIPESTATUS[0]}
set -e
printf '{"exit_code":%d}\n' "$status" > "$receipt_dir/terminal.json"
exit "$status"
