#!/usr/bin/env bash
# Detached leased driver for THE SYMMETRIC ROUTING EXTENSION (cards 2/3, the
# diagnosis lane's pool; the side-by-side lane owns cards 0/1 -- untouched).
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook) so the arm, its lease owner and the heartbeat keeper survive the
# attended session.
#
#   usage: run_routing_symmetry_leased.sh LABEL [SEED] [smoke]
#
# LABEL:
#   muon3x_max            THE A-ANALOG (max-Muon: every 2D matrix under Muon,
#                         the direct analog of GDN2's option A)
#   muon3x_gates_adamw    the broad-mapping appendix (M2: decay + the address
#                         producers key/query_key on AdamW)
#   muon2x_max|muon4x_max           the LR bracket at the max-Muon routing
#   muon2x_gates_adamw|muon4x_gates_adamw   ... at the broad-mapping routing
#   muon2x_banked|muon4x_banked      ... at the banked (B-analog) routing
# SEED: 3401 (default) or 3402 (the second-seed arm of the best routing)
#
# Runs ONE arm at the exact probe protocol on ONE leased card and commits the
# run's receipts + the lane's sources as they land (commit after each arm).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
LABEL=${1:?usage: run_routing_symmetry_leased.sh LABEL [SEED] [smoke]}
SEEDARG=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/routing_symmetry_runs
mkdir -p "$RUNS"
eval "$(GPU_LEASE_VISIBLE=2,3 scripts/gpu_lease.sh acquire 1 --wait)"
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

if [ "${3:-}" = "smoke" ]; then
  EXTRA="--smoke"
  OUT="$RUNS/smoke_${LABEL}_seed${SEEDARG}"
  MSGSUF=" (smoke)"
else
  EXTRA=""
  OUT="$RUNS/${LABEL}_seed${SEEDARG}"
  MSGSUF=""
fi

status=0
$PY -m experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
  arm --label "$LABEL" --seed "$SEEDARG" --output "$OUT" $EXTRA || status=$?
commit_run "X45 symmetric routing extension: the $LABEL arm landed at seed $SEEDARG (our winner's missing Muon routings: the A-analog max-Muon and the broad-mapping appendix, the checkpoint-eval curve at the probe protocol)$MSGSUF" \
  "$OUT" "$X45/routing_symmetry.py" "$X45/run_routing_symmetry_leased.sh" \
  tests/test_x45_routing_symmetry.py || true

exit "$status"
