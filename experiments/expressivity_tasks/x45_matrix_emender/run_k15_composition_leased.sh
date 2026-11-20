#!/usr/bin/env bash
# Detached leased driver for the K15 COMPOSITION probe lane (cards 0/1,
# this lane's own pool; the signed X46 sweep owns 2/3, the detached emender
# run owns 4 -- all untouched; cards 5/6 left as buffer; card 7 forbidden).
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook) so the arm, its lease owner and the heartbeat keeper survive
# the attended session.
#
#   usage: run_k15_composition_leased.sh ARM [SEED] [smoke]
#
# ARM:  composition (the one arm -- the banked winner + the K13 additive
#       read + the K11B p3 per-channel gate stack at the mimetic-neutral
#       birth)
# SEED: 3401 | 3402  (BOTH seeds primary evidence -- the composition's own
#       read-form baseline, the K13 additive rows, exists at both; run both)
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
ARM=${1:?usage: run_k15_composition_leased.sh ARM [SEED] [smoke]}
SEED=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k15_runs
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
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k15_probe_composition \
  arm --arm "$ARM" --seed "$SEED" --output "$OUT" $EXTRA || status=$?
commit_run "X45 K15 THE COMPOSITION: the ${ARM} arm landed (seed ${SEED}) -- the two proven-positive levers composed (the K13 additive read: banked score channel verbatim, softmax removed, strength dropped, g_proj gate zero-born; under the K11B p3 per-channel decay/write/read gate stack born at the registered scalar equivalents) at the matched-exposure probe under the corrected cooldown-primary scoring, the BORN-AS-THE-ADDITIVE-MACHINE birth receipt (every channel vs the K13 kernel at fixed inputs, the nearest-kin step-1 parity asserted) + the no-banked-birth-parity disclosure, the mechanism instrument (the |w|-mass effective rows + the collapse detector UNDER the per-channel gates, identity share via the per-channel replay, occupancy, op mix, all three gate channels' telemetry) and the decorative-store sweep at both the step-180 and post-cooldown states, the pre-registered C-1/C-2/C-3 clauses evaluated against the composition's OWN additive baseline (K13's 1.3928/1.3845 post-cooldown) with the composition-vs-GDN2 1.3522 rematch the headline (both anchors quoted: banked corrected 1.4338, the K11B p3 softmax rows 1.42353/1.42343)$MSGSUF" \
  "$OUT" "$X45/k15_probe_composition.py" \
  "$X45/run_k15_composition_leased.sh" || true

exit "$status"
