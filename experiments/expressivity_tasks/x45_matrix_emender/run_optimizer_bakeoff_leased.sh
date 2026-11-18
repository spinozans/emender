#!/usr/bin/env bash
# Detached leased driver for the optimizer/schedule bake-off (cards 2/3, the
# diagnosis lane's pool; the side-by-side lane owns cards 0/1 -- untouched).
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook) so the arm, its lease owner and the heartbeat keeper survive the
# attended session.
#
#   usage: run_optimizer_bakeoff_leased.sh LABEL [smoke]
#
# Runs ONE bake-off arm (LABEL: sf | muon1x | muon3x | sf_prior | sf_muon |
# adamw) at the exact probe protocol on ONE leased card and commits the run's
# receipts + the lane's sources as they land (commit after each arm).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
LABEL=${1:?usage: run_optimizer_bakeoff_leased.sh LABEL [smoke]}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/bakeoff_runs
mkdir -p "$RUNS"
eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
echo "leased card: ${GPU_LEASE_HELD:-none}"
trap 'GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true' EXIT

commit_run() {
  local msg="$1"; shift
  local attempt=0
  until git add -- "$@" 2>/dev/null \
      && git commit -m "$msg" -- "$@" >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 12 ]; then
      echo "commit failed after $attempt attempts; artifacts remain on disk"
      return 1
    fi
    sleep 10   # index.lock contention with the concurrent lanes
  done
  echo "committed: $msg"
  git push origin main >/dev/null 2>&1 || echo "push deferred (offline?)"
}

status=0
if [ "${2:-}" = "smoke" ]; then
  EXTRA="--smoke"
  OUT="$RUNS/smoke_${LABEL}_seed3401"
  MSGSUF=" (smoke)"
else
  EXTRA=""
  OUT="$RUNS/${LABEL}_seed3401"
  MSGSUF=""
fi

$PY -m experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
  arm --label "$LABEL" --output "$OUT" $EXTRA || status=$?
commit_run "X45 optimizer bake-off: the $LABEL arm landed (the checkpoint-eval curve at the probe protocol, the approved conventions receipted in the run)$MSGSUF" \
  "$OUT" "$X45/optimizer_bakeoff.py" "$X45/bakeoff_optimizers.py" \
  "$X45/run_optimizer_bakeoff_leased.sh" tests/test_x45_optimizer_bakeoff.py || true

exit "$status"
