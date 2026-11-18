#!/usr/bin/env bash
# Detached leased driver for the K11 LEARNED-GATES probe lane (cards 0/1,
# this lane's own pool; the K12 placement lane owns 2/3, the C3 lane 4/5,
# the growth+POP lane 6 -- all untouched).  Run this DETACHED (setsid +
# redirected stdio, the long-session survival playbook) so the arm, its
# lease owner and the heartbeat keeper survive the attended session.
#
#   usage: run_k11_learned_gates_leased.sh ARM [SEED] [smoke]
#
# ARM:  g1 (write-strength + read-out gates) | g2 (dense gated write
#       channel) | g3 (combined)
# SEED: 3401 (the primary arms) | 3402 (the SECOND seed, on the BEST arm
#       only -- the post-cooldown primary score decides which arm; do not
#       launch 3402 on a non-best arm)
#
# Runs ONE arm at the exact probe protocol (the 20,022,866-target
# matched-exposure landing, 180 constant-LR steps) under the corrected
# scoring (24-step linear cooldown PRIMARY, EMA-0.99 secondary, constant-180
# retained for banked comparability) on ONE leased card, then commits the
# run's receipts + the lane's sources as they land (commit after each arm:
# the ceiling is real, a committed partial beats an uncommitted complete).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
ARM=${1:?usage: run_k11_learned_gates_leased.sh ARM [SEED] [smoke]}
SEED=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k11_runs
mkdir -p "$RUNS"
eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
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
  OUT="$RUNS/smoke_${ARM}_seed${SEED}"
  MSGSUF=" (smoke)"
else
  EXTRA=""
  OUT="$RUNS/${ARM}_seed${SEED}"
  MSGSUF=""
fi

status=0
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k11_probe_learned_gates \
  arm --arm "$ARM" --seed "$SEED" --output "$OUT" $EXTRA || status=$?
commit_run "X45 K11 learned gates: the ${ARM} arm landed (seed ${SEED}) -- the matched-exposure probe under the corrected cooldown-primary scoring vs the banked winner config (hybrid Muon 4x lineage), the step-1 loss asserted bit-identical to the banked same-seed row (the 0-flip-at-birth receipt at probe scale), the mechanism instrument (identity share, read entropy/effective rows, occupancy, op mix, gate telemetry) at both the step-180 and post-cooldown states, the read-mask sweep at both states, and the pre-registered M-1..M-5 predictions evaluated against the amended anchors (constant-LR banked 1.4899/1.4907, corrected post-cooldown 1.4338, both quoted)$MSGSUF" \
  "$OUT" "$X45/k11_probe_learned_gates.py" \
  "$X45/run_k11_learned_gates_leased.sh" || true

exit "$status"
