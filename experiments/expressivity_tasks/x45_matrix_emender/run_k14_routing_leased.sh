#!/usr/bin/env bash
# Detached leased driver for the K14 learned-management-routing probe (cards
# 5/6, this lane's own pool per the operator's task; the K13 lane owns cards
# 0/1, the X46 sweep cards 2/3, the detached emender card 4, and card 7 is
# forbidden -- none of their files or cards are touched).  Run this DETACHED
# (setsid + redirected stdio, the long-session survival playbook) so the arm,
# its lease owner and the heartbeat keeper survive the attended session.
#
#   usage: run_k14_routing_leased.sh LABEL [smoke]
#
# LABEL (the registration's arms, seeds 3401/3402; the seed-3402 arms read
#       their step-1/falsifier bars against baseline_seed3402's own log --
#       run baseline_seed3402 FIRST):
#   routing_learned_seed3401    the learned-router arm (the router born at
#                              exact zeros, trainable via the admitted
#                              straight-through estimator)
#   routing_frozen_seed3401    the FROZEN-ROUTER falsifier control (the
#                              K11B 2.4-class bar; full-trajectory
#                              bit-identity expected and receipted per-step)
#   baseline_seed3402          the fresh seed-3402 baseline (run FIRST at
#                              3402)
#   routing_learned_seed3402 / routing_frozen_seed3402   the twins
#   baseline_seed3401          runnable but normally REUSED from the K12
#                              lane's re-scored baseline receipt (the
#                              supervisor's ruling, after the in-code
#                              config+protocol identity assert)
#
# Runs ONE arm at the exact probe protocol + the corrected scoring on ONE
# leased card and commits the run's receipts + the lane's sources as they
# land (commit after each arm).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-5,6}"
LABEL=${1:?usage: run_k14_routing_leased.sh LABEL [smoke]}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k14_runs
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

if [ "${2:-}" = "smoke" ]; then
  EXTRA="--smoke"
  OUT="$RUNS/smoke/${LABEL}"
  MSGSUF=" (smoke)"
else
  EXTRA=""
  OUT="$RUNS/${LABEL}"
  MSGSUF=""
fi

status=0
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k14_probe_routing \
  arm --label "$LABEL" $EXTRA || status=$?
commit_run "X45 K14 learned management routing: the $LABEL arm landed (the LEARNED op router on the K10 lineage at the matched-exposure probe protocol under the corrected cooldown-primary scoring, born at the banked thresholds via the supervisor-approved branch-active-margin construction, the op mix as the headline instrument, M-M1..M-M4 pre-registered with the M-M2 rescope verbatim; the frozen-router falsifier receipted per-step against the seed-matched baseline)$MSGSUF" \
  "$OUT" "$X45/k14_probe_routing.py" "$X45/run_k14_routing_leased.sh" \
  "$X45/receipts/k14_${LABEL}.json" || true

exit "$status"
