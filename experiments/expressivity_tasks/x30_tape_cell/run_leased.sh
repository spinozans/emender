#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export GPU_LEASE_VISIBLE="0,1,2,3,4,5,6"
# The broker installs release traps and a heartbeat; it waits for a genuinely
# idle, unleased card. Card 7 is excluded even if physically idle.
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
[[ "$GPU_LEASE_HELD" =~ ^[0-6]$ ]] || exit 90
printf 'X30 physical card %s\n' "$GPU_LEASE_HELD"
.venv/bin/python -m "experiments.expressivity_tasks.x30_tape_cell.$1" "${@:2}" --device cuda
