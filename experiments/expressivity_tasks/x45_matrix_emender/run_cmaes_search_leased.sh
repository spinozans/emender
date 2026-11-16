#!/usr/bin/env bash
# Launch the CMA-ES hyperparameter search under an exclusive single-GPU lease,
# detached.  The arms lane holds cards 0/1; this search leases from 2/3.
#   usage: run_cmaes_search_leased.sh [extra search args]
# Mirrors run_matched_arms_leased.sh (the registered discipline): acquire inside
# the detached wrapper so the lease owner and heartbeat keeper survive the
# attended session; release on exit.
set -euo pipefail
cd /home/erikg/emender
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
echo "leased card: $GPU_LEASE_HELD"
status=0
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x45_matrix_emender.cmaes_search search \
  --evals experiments/expressivity_tasks/x45_matrix_emender/receipts/cmaes_evals.jsonl \
  --result experiments/expressivity_tasks/x45_matrix_emender/receipts/cmaes_result.json \
  "$@" || status=$?
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
exit "$status"
