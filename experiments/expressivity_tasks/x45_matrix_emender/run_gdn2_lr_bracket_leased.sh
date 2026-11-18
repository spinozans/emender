#!/usr/bin/env bash
# Detached leased driver for GDN2'S LR BRACKET -- the last config-symmetry
# arm (cards 2/3, the diagnosis lane's pool; the side-by-side lane and the
# K11 lane own cards 0/1 -- untouched).  Run this DETACHED (setsid +
# redirected stdio, the long-session survival playbook) so the arm, its
# lease owner and the heartbeat keeper survive the attended session.
#
#   usage: run_gdn2_lr_bracket_leased.sh LABEL [SEED] [smoke]
#
# LABEL:
#   gdn2_muon4x_gates_adamw  THE UPWARD BRACKET POINT: GDN2 + hybrid Muon at
#                            4x its own banked AdamW LR, at its best routing
#                            (option B, gates-on-AdamW)
#   gdn2_muon5x_gates_adamw  THE FAR-SIDE BRACKET: the same arm at 5x
# SEED: 3401 (default) or 3402 (the second seed a clear new best owes)
#
# Runs ONE arm at the exact probe protocol on ONE leased card and commits the
# run's receipts + the lane's sources as they land (commit after each arm).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
LABEL=${1:?usage: run_gdn2_lr_bracket_leased.sh LABEL [SEED] [smoke]}
SEEDARG=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/gdn2_lr_bracket_runs
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
$PY -m experiments.expressivity_tasks.x45_matrix_emender.gdn2_lr_bracket \
  arm --label "$LABEL" --seed "$SEEDARG" --output "$OUT" $EXTRA || status=$?
commit_run "X45 GDN2 LR bracket: the $LABEL arm landed at seed $SEEDARG (GDN2's missing LR bracket at its best routing, option B gates-on-AdamW: 4x/5x its own banked AdamW LR, the checkpoint-eval curve at the probe protocol -- the last config-symmetry arm of the symmetric claim rule)$MSGSUF" \
  "$OUT" "$X45/gdn2_lr_bracket.py" "$X45/run_gdn2_lr_bracket_leased.sh" \
  tests/test_x45_gdn2_lr_bracket.py || true

exit "$status"
