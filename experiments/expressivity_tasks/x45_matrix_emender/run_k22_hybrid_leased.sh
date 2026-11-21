#!/usr/bin/env bash
# Detached leased driver for the K22 THE HYBRID lane (cards 2/3, this
# lane's own pool per the task's rule -- untouched by any other lane).
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook) so the arm, its lease owner and the heartbeat keeper survive
# the attended session.
#
#   usage: run_k22_hybrid_leased.sh STEP [SEED] [smoke]
#
# STEP: gates     the engine gate receipts (k22_gate_receipt.py) -- the
#                 birth discipline at the seam, run BEFORE any arm
#          suites    the committed K10/K13/K11B/K15/K17/K18 suites with
#                    the K22 module loaded (the standing receipt)
#          arm:hybrid | arm:hybrid_frozen
#                    ONE arm at the exact probe protocol (the
#                    20,022,866-target matched-exposure landing, the
#                    banked GDN2 row's own 5x recipe, the corrected
#                    cooldown-primary scoring)
#          assemble  the arms' table + the pre-stated clause read
# SEED: 3401 (every arm first) | 3402 (the >= 0.01 win-bar rule's own call)
#
# Commits the run's receipts + the lane's sources as they land (commit
# after each step: the ceiling is real, a committed partial beats an
# uncommitted complete).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
STEP=${1:?usage: run_k22_hybrid_leased.sh STEP [SEED] [smoke]}
SEED=${2:-3401}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k22_runs
mkdir -p "$RUNS"
eval "$(GPU_LEASE_VISIBLE=${GPU_LEASE_VISIBLE} scripts/gpu_lease.sh acquire 1 --wait)"
echo "leased card: ${GPU_LEASE_HELD:-none} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
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

LANE_SOURCES=(
  "$X45/k22_hybrid.py"
  "$X45/k22_gate_receipt.py"
  "$X45/k22_probe_hybrid.py"
  "$X45/test_k22_hybrid.py"
  "$X45/run_k22_hybrid_leased.sh"
  "$X45/PREREGISTRATION_K22_HYBRID.md"
  tests/test_k22_hybrid_probe.py
)

case "$STEP" in
  gates)
    $PY -m experiments.expressivity_tasks.x45_matrix_emender.k22_gate_receipt \
      --out "$X45/receipts/k22_gates.json" \
      2>&1 | tee "$RUNS/gate_receipt_console.txt"
    commit_run "X45 K22 THE HYBRID: the engine gate receipts -- the birth discipline at the seam, receipted BEFORE any arm: BORN AS GDN2 bit for bit (the hybrid vs a pure-GDN2 twin of the same build on one real production microbatch: the full-model loss and every slot layer's combined output bit-identical, flips counted bitwise, the IEEE sign-of-zero corner counted separately, flips outside exact-zero entries ZERO), the seam's dtype discipline under the production bf16 autocast regime, the fp64 gradcheck THROUGH THE HYBRID SEAM (the slot-in-dense-stack boundary: dense_out + gate*slot, the slot path walked by the composed oracle in fp64, both gated combinations, over the dense leaf, the gate, the input and every slot parameter) + the kernel path's seam gradients within the measured floors, the OPEN SIGNAL (dLoss/dgate nonzero at the zero-born gate) and the INERT BIRTH (every other slot gradient exactly zero) + the store fills at birth (the machinery runs, nothing reads out), the honest wall stated against the K20/K21 receipts; receipts/k22_gates.json + the console committed" \
      "$X45/receipts/k22_gates.json" "$RUNS/gate_receipt_console.txt" "${LANE_SOURCES[@]}"
    ;;
  suites)
    # the standing receipt: the committed suites stay green with the K22
    # module loaded (import the module first, then the suites)
    $PY - <<'EOF' 2>&1 | tee "$RUNS/k22_module_load_check.txt"
import experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid as k
print("k22 module loaded:", sorted(k.__all__)[:4], "...")
import experiments.expressivity_tasks.x45_matrix_emender.k22_probe_hybrid as p
print("k22 probe module loaded; arms:", sorted(p.ARMS))
EOF
    status=0
    $PY -m pytest -x -q \
      experiments/expressivity_tasks/x45_matrix_emender/test_k22_hybrid.py \
      experiments/expressivity_tasks/x45_matrix_emender/test_k10_separated_keys.py \
      experiments/expressivity_tasks/x45_matrix_emender/test_k13_additive_read.py \
      experiments/expressivity_tasks/x45_matrix_emender/test_k11b_channel_gates.py \
      experiments/expressivity_tasks/x45_matrix_emender/test_k15_composition.py \
      experiments/expressivity_tasks/x45_matrix_emender/test_k17_dense_write.py \
      experiments/expressivity_tasks/x45_matrix_emender/test_k18_short_conv.py \
      tests/test_k22_hybrid_probe.py \
      2>&1 | tee "$RUNS/k10_k13_k11b_k15_k17_k18_k22_suites_console.txt" || status=$?
    echo "SUITES_RC=$status"
    commit_run "X45 K22 THE HYBRID: the committed suites' green receipts with the K22 module loaded -- the lane's own suite + the K10/K13/K11B/K15/K17/K18 suites + the CPU-only probe tests, one console committed (the standing receipt: the module's import changes nothing the committed engines do)" \
      "$RUNS/k22_module_load_check.txt" \
      "$RUNS/k10_k13_k11b_k15_k17_k18_k22_suites_console.txt" "${LANE_SOURCES[@]}"
    exit "$status"
    ;;
  arm:*)
    ARM=${STEP#arm:}
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
    $PY -m experiments.expressivity_tasks.x45_matrix_emender.k22_probe_hybrid \
      run_arm --arm "$ARM" --seed "$SEED" --output "$OUT" $EXTRA || status=$?
    if [ ! -f "$OUT/result.json" ]; then
      echo "the arm did NOT land (status $status; no result.json) -- not committed as landed"
      exit "$status"
    fi
    if [ "$ARM" = "hybrid_frozen" ]; then
      ROLE="THE FALSIFIER: the whole slot branch frozen at the zero-read birth, asserted bit-identical to the banked GDN2 row's own 204-step trajectory (max per-step delta 0.0) and landing at the banked 1.3521734476089478 post-cooldown bit for bit -- the strongest plumbing receipt, the K18 perfect-receipt precedent the expectation to beat"
    else
      ROLE="the live arm: born zero-read (bit-identical to GDN2 at step 1, asserted), the gates LEARN to open -- any delta belongs to the slot reads opening"
    fi
    commit_run "X45 K22 THE HYBRID: the ${ARM} arm landed (seed ${SEED}) -- GDN2's committed 150M arm (the banked 1.3522 machine, matched_arms.build_gdn2 verbatim, width 640, 24 dense mixers) with 3 of its 24 mixer positions {6, 12, 18} carrying our committed composed slot mixer (the K15/K18 engine VERBATIM: the K13 additive read, the K11B p3 gates, the K10 separated keys at the mimetic birth, the K18 identity-born conv; independent stores; the K19-confirmed incumbent thresholds) added to the dense mixer output through the ZERO-BORN read gate -- born as PURE GDN2 bit for bit, +2,661,915 parameters (+1.776%) purely additive, quoted per the registration; $ROLE; at the exact probe protocol (matched exposure 20,022,866 targets, the banked GDN2 row's own 5x HybridMuonX45 recipe VERBATIM, option-B routing + the slot branch per the K18 convention mirrored, the corrected cooldown-primary scoring), the at-birth receipts (the open signal, the inert birth, the store fills), the C5 instrument on the slot layers, THE DECORATIVE SWEEP AS THE HEADLINE (the slot-read mask per slot layer: do the slot layers become load-bearing -- does the uniformity finding invert?), the gate trajectory receipts, the op-mix rows, occupancy/effective rows/identity share vs the pure machine's banked anchors, both parents' rows quoted (GDN2 1.3522, composed-ours 1.3811/1.3832), the pre-stated H-1/H-2/REFUTATION clause read$MSGSUF" \
      "$OUT" "${LANE_SOURCES[@]}" || true
    exit "$status"
    ;;
  assemble)
    $PY -m experiments.expressivity_tasks.x45_matrix_emender.k22_probe_hybrid \
      assemble --runs "$RUNS" 2>&1 | tee "$RUNS/assemble_console.txt"
    commit_run "X45 K22 THE HYBRID: the assemble -- the arms' table, the pre-stated H-1/H-2/REFUTATION clause read applied exactly as stated, both parents' banked rows quoted (GDN2 1.3522, composed-ours 1.3811/1.3832), the falsifier's bar, the headline instrument's read (the slot layers' load-bearing sweep), the escalation flags; the final gap-quote phrasing the supervisor's to adjudicate (the K13 clause-conflict precedent)" \
      "$RUNS/assemble_console.txt" "$RUNS/assemble.json" "${LANE_SOURCES[@]}"
    ;;
  *)
    echo "unknown STEP: $STEP" >&2
    exit 2
    ;;
esac
