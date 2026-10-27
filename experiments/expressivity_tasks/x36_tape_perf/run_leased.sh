#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 GPU_LEASE_VISIBLE=0,1,5,6
source scripts/gpu_lease.sh
eval "$(scripts/gpu_lease.sh acquire 1 --wait --timeout 60)"
[[ "$GPU_LEASE_HELD" =~ ^[0156]$ ]] || exit 90
printf 'X36 physical card %s\n' "$GPU_LEASE_HELD"
if [[ "${1:-}" == part2-gates ]]; then
  shift
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x36_tape_perf.part2_gates "$@"
  exit
fi
if [[ "${1:-}" == compare ]]; then
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x36_tape_perf.compare
  exit
fi
if [[ "${1:-}" == train-slice ]]; then
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x36_tape_perf.train_slice
  exit
fi
if [[ "${1:-}" == validate ]]; then
  shift
  /home/erikg/emender/.venv/bin/python -m pytest -p experiments.expressivity_tasks.x36_tape_perf.oracle_adapter experiments/expressivity_tasks/x31_tape_cell_fused/tests experiments/expressivity_tasks/x36_tape_perf/tests "$@"
  exit
fi
if [[ "${1:-}" == parity ]]; then
  shift
  /home/erikg/emender/.venv/bin/python -m pytest experiments/expressivity_tasks/x31_tape_cell_fused/tests "$@"
else
  /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x36_tape_perf.measure "$@"
fi
