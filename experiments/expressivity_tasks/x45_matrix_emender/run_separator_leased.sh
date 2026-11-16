#!/usr/bin/env bash
# Detached leased driver for the GRANULARITY SEPARATOR sweep (cards 0/1, the
# matched-arms lane's pool -- cards 2/3 are the CMA-ES lane's and 4/5 the
# extension lane's, never touched here).  Runs the whole best-effort sweep
# (SEP-AR headline + SEP-EVICT corroboration) in its own process and COMMITS
# the receipts when they land, so the sweep survives an attended-session
# ceiling.  Run this DETACHED (setsid + redirected stdio, the long-session
# survival playbook): the sweep and the lease owner survive the session that
# launched them.
#
#   usage: run_separator_leased.sh [STEPS]
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender

STEPS=${1:-3000}

export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none}"

steps_opt=""
if [ "$STEPS" != "3000" ]; then
  steps_opt="--steps $STEPS"
fi

if $PY -m experiments.expressivity_tasks.x45_matrix_emender.granularity_separator \
    --task both $steps_opt; then
  attempt=0
  until git add "$X45/granularity_separator.py" \
        "$X45/test_granularity_separator.py" \
        "$X45/run_separator_leased.sh" \
        "$X45/RESULT_GRANULARITY_SEPARATOR.md" \
        "$X45/receipts"/separator_*.json \
        "$X45/receipts"/separator_*.pt 2>/dev/null \
      && git commit -m "X45 granularity separator: the sweep's receipts and results (pre-stated predictions in RESULT_GRANULARITY_SEPARATOR.md commit precedes this one)" \
         >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 12 ]; then
      echo "commit failed after $attempt attempts; artifacts remain on disk"
      exit 1
    fi
    sleep 10   # index.lock contention with the concurrent lanes
  done
  echo "committed: separator receipts and results"
  git push origin main >/dev/null 2>&1 || echo "push deferred (offline?)"
else
  echo "the separator sweep FAILED; artifacts remain on disk"
  exit 1
fi
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
