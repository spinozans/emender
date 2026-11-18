#!/usr/bin/env bash
# Detached leased driver for the K12 learned-placement probe (cards 2/3, this
# lane's own pool per the operator's task; the K11 lane owns cards 0/1 and is
# untouched).  Run this DETACHED (setsid + redirected stdio, the long-session
# survival playbook) so the arm, its lease owner and the heartbeat keeper
# survive the attended session.
#
#   usage: run_k12_placement_leased.sh LABEL [smoke]
#
# LABEL (the registration's arms, seed 3401; second seeds unlocked by the
#       pre-stated >= 0.01-nat win bar):
#   baseline_seed3401            the banked K10-keys config re-scored under
#                                the corrected protocol (the banked
#                                comparability row; its constant-180 eval
#                                must reproduce the banked 1.841756020273481)
#   placement_learned_seed3401   the same config + the learned placement
#   placement_fixed0p25_seed3401 the supervisor-ruled FIXED-w bracket
#   placement_fixed0p5_seed3401
#   placement_fixed1p0_seed3401
#
# Runs ONE arm at the exact probe protocol + the corrected scoring on ONE
# leased card and commits the run's receipts + the lane's sources as they
# land (commit after each arm).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
LABEL=${1:?usage: run_k12_placement_leased.sh LABEL [smoke]}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k12_runs
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
  OUT="$RUNS/smoke_${LABEL}"
  MSGSUF=" (smoke)"
else
  EXTRA=""
  OUT="$RUNS/${LABEL}"
  MSGSUF=""
fi

status=0
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k12_probe_placement \
  arm --label "$LABEL" $EXTRA || status=$?
commit_run "X45 K12 learned placement: the $LABEL arm landed (the learned free-slot policy on the dead-key residue at the matched-exposure probe protocol under the corrected cooldown-primary scoring; the zero-gradient finding receipted, the supervisor-approved straight-through estimator with the fixed-w bracket {0.25, 0.5, 1.0} as estimator-free controls, the three-way read pre-stated: (i) the estimator works, (ii) the surrogate's bias is binding -> option (c) named follow-up, (iii) THE HONEST NULL -- free-slot placement decorative on natural text)$MSGSUF" \
  "$OUT" "$X45/k12_probe_placement.py" "$X45/run_k12_placement_leased.sh" \
  "$X45/receipts/k12_${LABEL}.json" || true

exit "$status"
