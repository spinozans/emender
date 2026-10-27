#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GPU_LEASE_VISIBLE=0,1,5,6
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --wait --timeout 60)"
[[ "$GPU_LEASE_HELD" =~ ^[0156]$ ]] || exit 90
printf 'X38 physical card %s\n' "$GPU_LEASE_HELD"
if [[ "${1:-}" == test ]]; then
 shift
 /home/erikg/emender/.venv/bin/python -m pytest experiments/expressivity_tasks/x38_diversity_objective/tests "$@"
 exit
fi
if [[ "${1:-}" == parity ]]; then
 shift
 /home/erikg/emender/.venv/bin/python -m pytest -p experiments.expressivity_tasks.x36_tape_perf.oracle_adapter experiments/expressivity_tasks/x31_tape_cell_fused/tests experiments/expressivity_tasks/x36_tape_perf/tests "$@"
 exit
fi
/home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x38_diversity_objective/receipts/resume_pinned.py "$@"
