#!/usr/bin/env bash
# Detached leased driver for the K11B PER-CHANNEL-GATES probe lane (cards
# 0/1, this lane's own pool; the K11 lane is complete; card 4 is the
# detached emender run -- untouched; card 7 forbidden).  Run this DETACHED
# (setsid + redirected stdio, the long-session survival playbook) so the
# arm, its lease owner and the heartbeat keeper survive the attended
# session.
#
#   usage: run_k11b_channel_gates_leased.sh ARM [SEED] [smoke|frozen]
#
# ARM:  p1 (per-channel input-dependent DECAY, the f_proj analog -- the
#         priority, the audit's "strictly poorer" row) |
#       p2 (+ per-channel write admission, the b_proj analog) |
#       p3 (+ per-channel read gates, the g_proj analog, combined)
# SEED: 3401 (the primary arms) | 3402 (the SECOND seed, on any arm whose
#         primary score beats the seed-matched comparison by >= 0.01)
# frozen: the FROZEN-GATES falsifier control (registered on p1 only): the
#         gate parameters frozen at birth; step-1 bit-identity + a parity
#         landing, the per-step divergence measured (registration 2.4)
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
ARM=${1:?usage: run_k11b_channel_gates_leased.sh ARM [SEED] [smoke|frozen]}
SEED=${2:-3401}
MODE=${3:-}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k11b_runs
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

if [ "$MODE" = "smoke" ]; then
  EXTRA="--smoke"
  OUT="$RUNS/smoke_${ARM}_seed${SEED}"
  LABEL="smoke"
elif [ "$MODE" = "frozen" ]; then
  if [ "$ARM" != "p1" ]; then
    echo "the frozen-gates falsifier control is registered on the priority arm p1 only" >&2
    exit 2
  fi
  EXTRA="--frozen"
  OUT="$RUNS/${ARM}_frozen_seed${SEED}"
  LABEL="frozen-gates falsifier control"
else
  EXTRA=""
  OUT="$RUNS/${ARM}_seed${SEED}"
  LABEL="live arm"
fi

status=0
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k11b_probe_channel_gates \
  arm --arm "$ARM" --seed "$SEED" --output "$OUT" $EXTRA || status=$?
commit_run "X45 K11B per-channel gates: the ${ARM} arm landed (${LABEL}, seed ${SEED}) -- GDN2's gate granularity (per-channel, per-token, input-dependent DECAY/write/read) through the recorded-schedule channels at the matched-exposure probe under the corrected cooldown-primary scoring vs the banked winner config, the step-1 loss asserted bit-identical to the banked same-seed row (the 0-flip-at-birth receipt at probe scale), the mechanism instrument (identity share via the per-channel trajectory replay, read entropy/effective rows, occupancy, op mix) plus the born-identity gate telemetry at both the step-180 and post-cooldown states, the read-mask sweep at both states, and the pre-stated two-way question evaluated (gap-closure bar 0.02, win bar 0.01 for the second seed; constant-LR banked 1.4899/1.4907, corrected post-cooldown 1.4338, GDN2 1.3522, all quoted)" \
  "$OUT" "$X45/k11b_probe_channel_gates.py" \
  "$X45/run_k11b_channel_gates_leased.sh" || true

exit "$status"
