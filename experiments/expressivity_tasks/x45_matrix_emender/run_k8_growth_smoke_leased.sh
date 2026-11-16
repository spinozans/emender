#!/usr/bin/env bash
# Detached leased driver for the K8 growth SMOKE (the operator's short-run
# confirmation BEFORE the long arms are committed to): a few real steps at
# probe shape on the grow=True arm, verifying the trigger fires, re-dispatch
# executes, the zero rows are exact and the backward survives rank changes.
# Runs in its own process and COMMITS the receipt when it lands.  Run this
# DETACHED (setsid + redirected stdio, the long-session survival playbook).
#
#   usage: run_k8_growth_smoke_leased.sh [STEPS]
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
STEPS=${1:-"6"}

export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-4,5}"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
echo "leased card: ${GPU_LEASE_HELD:-none}"

if $PY -m experiments.expressivity_tasks.x45_matrix_emender.k8_probe_growth \
    smoke --steps "$STEPS"; then
  attempt=0
  until git add "$X45/k8_probe_growth.py" \
        "$X45/run_k8_growth_smoke_leased.sh" \
        "$X45/run_k8_probe_leased.sh" \
        "$X45/receipts/k8_smoke.json" 2>/dev/null \
      && git commit -m "X45 K8 growth smoke: the trigger fires and the machinery holds at probe shape (receipt)" \
         >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 12 ]; then
      echo "commit failed after $attempt attempts; artifacts remain on disk"
      exit 1
    fi
    sleep 10   # index.lock contention with the concurrent lanes
  done
  echo "committed: the smoke receipt"
  git push origin main >/dev/null 2>&1 || echo "push deferred (offline?)"
else
  echo "the smoke FAILED; artifacts remain on disk"
  exit 1
fi
GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
