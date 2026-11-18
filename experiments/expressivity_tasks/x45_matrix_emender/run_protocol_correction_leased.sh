#!/usr/bin/env bash
# Detached leased driver for THE PROTOCOL CORRECTION lane (cards 4/5, this
# lane's own pool; the K11 lane owns cards 0/1 and the op-audit lane 2/3 --
# both untouched).  Run this DETACHED (setsid + redirected stdio, the
# long-session survival playbook) so the arm, its lease owner and the
# heartbeat keeper survive the attended session.
#
#   usage: run_protocol_correction_leased.sh LABEL [smoke]
#
# LABEL:
#   winner_speedrun_rescored  DELIVERABLE 1, our side: the banked best row
#                             (B-analog routing, 4x, seed 3402, the 1.4899
#                             row's own config) re-scored under the corrected
#                             instrument (24-step linear cooldown + EMA-0.99)
#   gdn2_speedrun_rescored    DELIVERABLE 1, GDN2's side: the banked best row
#                             (option B, 5x, seed 3401, the 1.3951 row's own
#                             config) re-scored the same way
#   winner_moonlight          DELIVERABLE 2, our side: the SAME RMS-matched
#                             Muon at the Moonlight LR point (1x the tuned
#                             AdamW LR), own betas/weight-decay/warmup
#   gdn2_moonlight            DELIVERABLE 2, GDN2's side: the same at 1x
#                             GDN2's banked AdamW LR
#
# Each arm's own seed is part of its label (the banked best rows' own:
# ours 3402, GDN2 3401 -- the approved machine-seed pairing); the driver
# rejects any other seed.  Runs ONE arm at the exact probe protocol + the
# corrected scoring on ONE leased card and commits the run's receipts + the
# lane's sources as they land (commit after each arm).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-4,5}"
LABEL=${1:?usage: run_protocol_correction_leased.sh LABEL [smoke]}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/protocol_correction_runs
mkdir -p "$RUNS"
eval "$(GPU_LEASE_VISIBLE=4,5 scripts/gpu_lease.sh acquire 1 --wait)"
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
$PY -m experiments.expressivity_tasks.x45_matrix_emender.protocol_correction \
  arm --label "$LABEL" --output "$OUT" $EXTRA || status=$?
commit_run "X45 protocol correction: the $LABEL arm landed (the literature-backed scoring fixes applied to the velocity probe: a 24-step linear cooldown after the constant-LR budget + EMA-0.99 scoring, and the recipe-correct Moonlight Muon arms at 1x the AdamW-tuned LR -- both machines' banked best speedrun rows re-scored and the Moonlight recipe run on both machines, RMS-matching receipted by measured update RMS on both groups)$MSGSUF" \
  "$OUT" "$X45/protocol_correction.py" "$X45/run_protocol_correction_leased.sh" \
  tests/test_x45_protocol_correction.py || true

exit "$status"
