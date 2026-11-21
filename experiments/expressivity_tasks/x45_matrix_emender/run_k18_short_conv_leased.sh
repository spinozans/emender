#!/usr/bin/env bash
# Detached leased driver for the K18 SHORT-CONV TRANSPLANT lane (cards 0/1,
# this lane's own pool; the X46 chain owns 2/3 -- untouched; cards 4/5/6
# left free as buffer; card 7 forbidden).  Run this DETACHED (setsid +
# redirected stdio, the long-session survival playbook) so the arm, its
# lease owner and the heartbeat keeper survive the attended session.
#
#   usage: run_k18_short_conv_leased.sh ARM [SEED] [smoke]
#
# ARM:  conv | conv_frozen | shift_fixed
#       (GDN2's registered short depthwise conv, kernel 4, on the composed
#       machine's payload path at GDN2's own position -- the v_conv1d
#       analog, the escalated and supervisor-approved design: born at the
#       identity kernel [0,0,0,1] in the STANDARD nn.Conv1d layout so the
#       machine is BORN AS THE COMPOSED MACHINE bit for bit, then learns
#       to deviate; the frozen-conv falsifier control; the fixed-shift
#       control [0,0,0.5,0.5] -- the published mechanism decomposition's
#       own recommendation, separating ANY LOCAL MIXING from a LEARNED
#       FILTER)
# SEED: 3401 (every arm first) | 3402 (the >= 0.01 win-bar rule's own call)
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
ARM=${1:?usage: run_k18_short_conv_leased.sh ARM [SEED] [smoke]}
SEED=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k18_runs
mkdir -p "$RUNS"
eval "$(GPU_LEASE_VISIBLE=${GPU_LEASE_VISIBLE} scripts/gpu_lease.sh acquire 1 --wait)"
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
$PY -m experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv \
  arm --arm "$ARM" --seed "$SEED" --output "$OUT" $EXTRA || status=$?
commit_run "X45 K18 THE SHORT-CONV TRANSPLANT: the ${ARM} arm landed (seed ${SEED}) -- the missing-component candidate with published backing: GDN2's registered short depthwise convolution (conv_size=4, default ON, the delta-rule family's ablations say load-bearing locally, 0.05-13.5 ppl heterogeneous across the lineage) that our comparison machine has NEVER had, transplanted onto the COMMITTED K15 composed machine's payload path at GDN2's own position (post-projection, pre-recurrence; the v_conv1d analog -- the escalated, supervisor-approved single-side design: our key/query projections already consume the 4-token causal context window so the q/k side has the locality GDN2's q/k convs buy, while the payload path had NO temporal receptive field; the conv COMPLETES the receptive-field parity), born at the identity kernel [0,0,0,1] in the STANDARD nn.Conv1d/fla tap layout (the escalated correction of the [1,0,0,0] spec; the silu NOT transplanted -- GDN2 silus q/k/v even with the conv OFF, so the temporal filter is the ablated object and ours isolates it) so the machine is BORN AS THE COMPOSED MACHINE bit for bit (the K15 birth discipline, the 0-flip bar, the frozen-conv falsifier that must be bit-identical forever, the fixed-shift control [0,0,0.5,0.5] separating any local mixing from a learned filter), a VALUE-CHANNEL change (no conv value feeds op selection; P1/P2/P3 trivially held) at the exact probe protocol (matched exposure 20,022,866 targets, seed ${SEED}, the registered flagship stream, the document-disjoint calibration eval) under the corrected cooldown-primary scoring, the inherited 4x hybrid-Muon point, the pre-stated THREE-OUTCOME rule (SC-1 local-features >= 0.01 owes seed 3402 / SC-2 parity the uniformity attribution stands with one more receipt / SC-3 hurts the slot-semantics machine-family difference) + the falsifier's own bar receipted, the C5 instrument on the conved payload (identity share, effective rows, occupancy, the decorative sweep -- does the local conv make more layers load-bearing? the uniformity hypothesis's direct test), the gate telemetry unmoved, the LEARNED KERNEL receipts (the per-tap means/spreads, the distance from the identity birth -- the mechanism's direct read), the fp64 gradcheck through the conv path green$MSGSUF" \
  "$OUT" "$X45/k18_short_conv.py" \
  "$X45/test_k18_short_conv.py" \
  "$X45/k18_gate_receipt.py" \
  "$X45/k18_probe_short_conv.py" \
  "$X45/run_k18_short_conv_leased.sh" \
  "$X45/PREREGISTRATION_SHORT_CONV.md" \
  tests/test_k18_short_conv_probe.py || true

exit "$status"
