#!/usr/bin/env bash
# Detached leased driver for the K13 ADDITIVE-READ probe lane (cards 0/1,
# this lane's own pool; the signed X46 sweep owns 2/3, the detached emender
# run owns 4 -- all untouched; card 7 forbidden).  Run this DETACHED (setsid
# + redirected stdio, the long-session survival playbook) so the arm, its
# lease owner and the heartbeat keeper survive the attended session.
#
#   usage: run_k13_additive_read_leased.sh ARM [SEED] [smoke]
#
# ARM:  additive (the one arm -- the banked winner + the additive read + the
#       zero-born g_proj gate)
# SEED: 3401 | 3402  (BOTH seeds primary evidence -- no birth parity is
#       possible, a different function; run both)
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
ARM=${1:?usage: run_k13_additive_read_leased.sh ARM [SEED] [smoke]}
SEED=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k13_runs
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
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k13_probe_additive_read \
  arm --arm "$ARM" --seed "$SEED" --output "$OUT" $EXTRA || status=$?
commit_run "X45 K13 additive read: the ${ARM} arm landed (seed ${SEED}) -- the machine-variant arm (GDN2's additive read transplanted: w_j = the banked score channel raw, live-gated; softmax and ||S_j|| dropped; the k11b g_proj gate zero-born) at the matched-exposure probe under the corrected cooldown-primary scoring vs the banked winner config, the DECISIONS-BIT-IDENTICAL birth receipt + the birth-magnitude diagnostic (R13-5) + the no-birth-parity disclosure, the mechanism instrument (identity share, |w|-mass effective rows + the collapse detector, occupancy, op mix, gate telemetry) and the decorative-store sweep at both the step-180 and post-cooldown states, the pre-registered M-R1..M-R3 predictions evaluated against the matched anchors (both anchors quoted: constant 1.4899/1.4907, corrected post-cooldown 1.4338, GDN2 1.3522 the gap reference)$MSGSUF" \
  "$OUT" "$X45/k13_probe_additive_read.py" \
  "$X45/run_k13_additive_read_leased.sh" || true

exit "$status"
