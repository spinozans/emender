#!/usr/bin/env bash
# Detached leased driver for THE SYMMETRIC MUON CONTROL (cards 2/3, the
# diagnosis lane's pool; the side-by-side lane owns cards 0/1 -- untouched).
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook) so the arm, its lease owner and the heartbeat keeper survive the
# attended session.
#
#   usage: run_muon_control_leased.sh LABEL [smoke]
#
# LABEL:
#   gdn2_muon3x             THE SYMMETRIC CONTROL (option A, the verdict arm)
#   gdn2_muon3x_gates_adamw the informative convention-mirror appendix (option B)
#   muon3x_seed3402         the winner's second-seed arm -- the BANKED
#                           optimizer_bakeoff.py driver, run UNMODIFIED with
#                           --seed 3402 (receipts under bakeoff_runs/)
#
# Runs ONE arm at the exact probe protocol on ONE leased card and commits the
# run's receipts + the lane's sources as they land (commit after each arm).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
LABEL=${1:?usage: run_muon_control_leased.sh LABEL [smoke]}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
mkdir -p "$X45/muon_control_runs" "$X45/bakeoff_runs"
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
case "$LABEL" in
  muon3x_seed3402)
    # the banked bake-off driver, UNMODIFIED, at the second seed
    if [ "${2:-}" = "smoke" ]; then
      echo "the seed arm has no smoke mode (the banked driver's own smoke is receipted)"
      exit 90
    fi
    OUT="$X45/bakeoff_runs/muon3x_seed3402"
    MSGSUF=""
    $PY -m experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
      arm --label muon3x --seed 3402 --output "$OUT" || status=$?
    commit_run "X45 symmetric Muon control: the winner+muon3x second-seed (3402) arm landed (the banked driver unmodified, the single-seed caveat on the 1.5146 number)" \
      "$OUT" || true
    ;;
  gdn2_muon3x|gdn2_muon3x_gates_adamw)
    if [ "${2:-}" = "smoke" ]; then
      EXTRA="--smoke"
      OUT="$X45/muon_control_runs/smoke_${LABEL}_seed3401"
      MSGSUF=" (smoke)"
    else
      EXTRA=""
      OUT="$X45/muon_control_runs/${LABEL}_seed3401"
      MSGSUF=""
    fi
    $PY -m experiments.expressivity_tasks.x45_matrix_emender.muon_control \
      arm --label "$LABEL" --output "$OUT" $EXTRA || status=$?
    commit_run "X45 symmetric Muon control: the $LABEL arm landed (GDN2 + the bake-off's hybrid Muon at 3x its own AdamW LR, the checkpoint-eval curve at the probe protocol)$MSGSUF" \
      "$OUT" "$X45/muon_control.py" "$X45/run_muon_control_leased.sh" \
      tests/test_x45_muon_control.py || true
    ;;
  *)
    echo "unknown LABEL: $LABEL" >&2
    exit 91
    ;;
esac

exit "$status"
