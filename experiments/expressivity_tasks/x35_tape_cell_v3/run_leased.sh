#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export GPU_LEASE_VISIBLE="0,1,2,3,4,5,6"
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --wait --timeout 60)"
[[ "$GPU_LEASE_HELD" =~ ^[0-6]$ ]] || exit 90
printf 'X35 physical card %s\n' "$GPU_LEASE_HELD"
if [[ "${1:-}" == "--module" ]]; then
    shift
    [[ "${1:-}" == experiments.expressivity_tasks.x35_tape_cell_v3.* ]] || exit 91
    /home/erikg/emender/.venv/bin/python -m "$@"
else
    /home/erikg/emender/.venv/bin/python -m pytest experiments/expressivity_tasks/x35_tape_cell_v3/tests "$@"
fi
