#!/usr/bin/env bash
# Detached leased driver for the side-by-side gap dissection's armed runs
# (cards 0/1, the arms/kernel lane's pool).  Run this DETACHED (setsid +
# redirected stdio, the long-session survival playbook) so the runs, their
# lease owner and the heartbeat keeper survive the attended session.
#
#   usage: run_side_by_side_leased.sh [smoke]
#
# Runs, sequentially on ONE leased card (each committed as it lands):
#   1. the tuned winner (K10 + the CMA-ES incumbent) at the probe protocol,
#      with the checkpoint curve, the per-layer telemetry and the read-mask
#      sweep -- asserted bit-identical to the banked confirm cell;
#   2. the gdn2 arm at the probe protocol, with the checkpoint curve and the
#      gate telemetry -- asserted bit-identical to the banked gdn2 row;
#   3. the K10 arm at the transferred (untuned) config, with the checkpoint
#      curve only (the C1 config-shape contrast) -- asserted bit-identical to
#      the banked untuned probe.
#
# The C1 fair-tuning extension is NOT driven here: it runs detached on the
# OTHER card through the banked run_velocity_ladder_leased.sh (see
# RESULT_SIDE_BY_SIDE_0239.md 0.1 for the pre-stated protocol).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/side_by_side_runs
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
    sleep 10   # index.lock contention with the concurrent C1 search lane
  done
  echo "committed: $msg"
  git push origin main >/dev/null 2>&1 || echo "push deferred (offline?)"
}

TUNED_CFG='{"lr": 0.00033427141575371284, "weight_decay": 0.17422187309224696, "warmup_steps": 45, "read_scale": 2.7870458737509636, "tau_match": 0.45, "tau_pop": 0.1, "tau_merge": 0.85}'
TRANSFERRED_CFG='{"lr": 0.0005420185524740253, "weight_decay": 0.20059436743465126, "warmup_steps": 69, "read_scale": 2.0, "tau_match": 0.45, "tau_pop": 0.1, "tau_merge": 0.85}'

if [ "${1:-}" = "smoke" ]; then
  EXTRA="--smoke"
  OUT_TAG=smoke
else
  EXTRA=""
  OUT_TAG=""
fi

status=0

# 1. the tuned winner, armed
$PY -m experiments.expressivity_tasks.x45_matrix_emender.side_by_side k10-armed \
  --config "$TUNED_CFG" --label k10_tuned_armed \
  --output "$RUNS/${OUT_TAG}k10_tuned_armed_seed3401" \
  --banked-log "$X45/velocity_runs/k10_tuned_confirm_seed3401/training.jsonl" \
  $EXTRA || status=$?
commit_run "X45 side-by-side: the tuned-winner armed run (the gap-vs-phase checkpoint curve, per-layer telemetry at every checkpoint, the C4 read-mask sweep), protocol-equivalent bit-for-bit to the banked confirm cell" \
  "$RUNS/${OUT_TAG}k10_tuned_armed_seed3401" "$X45/side_by_side.py" \
  "$X45/run_side_by_side_leased.sh" || true

# 2. gdn2, armed
$PY -m experiments.expressivity_tasks.x45_matrix_emender.side_by_side gdn2-armed \
  --label gdn2_armed \
  --output "$RUNS/${OUT_TAG}gdn2_armed_seed3401" \
  --banked-log "$X45/velocity_runs/gdn2_probe_seed3401/training.jsonl" \
  $EXTRA || status=$?
commit_run "X45 side-by-side: the gdn2 armed run (the checkpoint curve + its own gate/state telemetry), protocol-equivalent bit-for-bit to the banked gdn2 row" \
  "$RUNS/${OUT_TAG}gdn2_armed_seed3401" || true

# 3. the transferred config, armed (the C1 config-shape contrast)
if [ -z "$EXTRA" ]; then
  $PY -m experiments.expressivity_tasks.x45_matrix_emender.side_by_side k10-armed \
    --config "$TRANSFERRED_CFG" --label k10_transferred_armed \
    --output "$RUNS/k10_transferred_armed_seed3401" \
    --banked-log "$X45/k10_runs/separated_keys_seed3401/training.jsonl" \
    || status=$?
  commit_run "X45 side-by-side: the transferred-config armed run (the C1 config-shape contrast curve), protocol-equivalent bit-for-bit to the banked untuned probe" \
    "$RUNS/k10_transferred_armed_seed3401" || true
fi

exit "$status"
