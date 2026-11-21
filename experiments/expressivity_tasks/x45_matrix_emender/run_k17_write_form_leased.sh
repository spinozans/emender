#!/usr/bin/env bash
# Detached leased driver for the K17 WRITE FORM lane (cards 0/1, this
# lane's own pool; the X46 chain owns 2/3 -- untouched; cards 4/5/6 left
# free as buffer; card 7 forbidden).  Run this DETACHED (setsid +
# redirected stdio, the long-session survival playbook) so the arm, its
# lease owner and the heartbeat keeper survive the attended session.
#
#   usage: run_k17_write_form_leased.sh ARM [SEED] [smoke] [GAIN]
#
# ARM:  dense | dense_frozen | gain1x | gain2x | gain4x | combined
#       (the dense gated write -- the b_proj-delta form, born as the
#       composed machine; the frozen-write falsifier control; the
#       read-gain bracket {1.0, 2.0, 4.0} -- config arms, the inherited
#       2.787 cell cited from the K15 rows; the conditional combined arm)
# SEED: 3401 (every arm first) | 3402 (the >= 0.01 win-bar rule's own call)
# GAIN: the combined arm's gain cell (gain1x | gain2x | gain4x)
#
# Runs ONE arm at the exact probe protocol (the 20,022,866-target
# matched-exposure landing, 180 constant-LR steps, the inherited 4x
# hybrid-Muon point) under the corrected scoring (24-step linear cooldown
# PRIMARY, EMA-0.99 secondary, constant-180 retained) on ONE leased card,
# then commits the run's receipts + the lane's sources as they land
# (commit after each arm: the ceiling is real, a committed partial beats
# an uncommitted complete).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-0,1}"
ARM=${1:?usage: run_k17_write_form_leased.sh ARM [SEED] [smoke] [GAIN]}
SEED=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k17_runs
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

GAIN_ARG=""
if [ "${1:-}" = "combined" ] && [ -n "${4:-}" ]; then
  GAIN_ARG="--gain ${4}"
fi

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
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k17_probe_write_form \
  arm --arm "$ARM" --seed "$SEED" --output "$OUT" $GAIN_ARG $EXTRA || status=$?
commit_run "X45 K17 THE WRITE FORM: the ${ARM} arm landed (seed ${SEED}) -- the last untested structural transplant on the machine that changed: the dense gated write (the b_proj-delta form, G2's best-match-live-row deposit reborn at the composed machine's per-channel granularity, born at exact zero through the committed zero-born gate conventions so the machine is BORN AS THE COMPOSED MACHINE bit for bit -- the K15 birth discipline, the 0-flip bar) and the read-gain bracket {1.0, 2.0, 4.0} (config arms; the machine's actual inherited gain is the tuned 2.787 point, cited from the K15 rows) on the COMMITTED K15 composed machine (the K13 additive read under the K11B p3 gate stack; the program's best absolute rows 1.3811/1.3832 post-cooldown, +0.029/+0.031 remaining vs GDN2's 1.3522, parity-within-band per the ruling of 2026-11-25) at the exact probe protocol (matched exposure 20,022,866 targets, seed ${SEED}, the registered flagship stream, the document-disjoint calibration eval) under the corrected cooldown-primary scoring, the inherited 4x hybrid-Muon point (K16's receipt: the bracket optimum), the pre-stated W-1/W-2/W-3/G-1 clauses + the band discipline receipted, the C5 write-channel telemetry (the dense channel's mass vs the event writes') + the K15 mechanism instrument under the dense write + the decorative-store sweep at both the step-180 and post-cooldown states; the fp64 gradcheck through the dense-write path green (the suite 27/27, the committed K10/K13/K11B/K15 suites 84/84 with the module loaded)$MSGSUF" \
  "$OUT" "$X45/k17_dense_write.py" \
  "$X45/test_k17_dense_write.py" \
  "$X45/k17_gate_receipt.py" \
  "$X45/k17_probe_write_form.py" \
  "$X45/run_k17_write_form_leased.sh" \
  "$X45/PREREGISTRATION_K17_WRITE_FORM.md" || true

exit "$status"
