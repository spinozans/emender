#!/usr/bin/env bash
# Detached leased driver for the K24 LONG-WINDOW lane (cards 2/3, this
# lane's own pool per the task's rule -- untouched by any other lane).
# Run this DETACHED (setsid + redirected stdio, the long-session survival
# playbook) so the arms, their lease owner and the heartbeat keeper survive
# the attended session.  Before relying on the run, verify PPID=1, the
# live lease owner/starttime, at least one heartbeat renewal, and the
# committed receipts on disk.
#
#   usage: run_k24_long_window_leased.sh STEP
#
# STEP: registration   the window-arithmetic receipt (committed with the
#                      registration BEFORE any arm)
#        gates          the long-window engine gate receipts
#                      (k24_gate_receipt.py) -- the scan path's disciplines
#                      at the longest window, run BEFORE any arm
#        suites         the committed K10/K13/K11B/K15/K17/K18/K22 suites
#                      with the K24 module loaded (the standing receipt)
#        smoke:OURS|GDN2|HYBRID   the pipeline shakedown at 4x for one
#                      machine (tiny exposure; never a ladder row)
#        arm:ours_1x | arm:gdn2_1x | arm:ours_2x | arm:gdn2_2x |
#        arm:ours_4x | arm:gdn2_4x | arm:hybrid_4x
#                      ONE arm at the exact probe protocol (matched
#                      exposure, the corrected cooldown-primary scoring);
#                      the 1x cells are the ANCHORS -- the driver fails
#                      closed if either anchor has not landed bit-for-bit
#                      before any long window arm runs
#        assemble       the curves + the pre-stated clause adjudications
#        chain          the whole lane in order: registration, gates,
#                      suites, the smokes, the two anchors (bit-for-bit
#                      bars asserted by the runners themselves), the five
#                      long arms, the assemble -- one commit per step, one
#                      leased card at a time (released at each step's end)
#
# Commits the run's receipts + the lane's sources as they land (commit
# after each step: the ceiling is real, a committed partial beats an
# uncommitted complete).
set -euo pipefail
cd /home/erikg/emender
export OMP_NUM_THREADS=4
export GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-2,3}"
STEP=${1:?usage: run_k24_long_window_leased.sh STEP}
X45=experiments/expressivity_tasks/x45_matrix_emender
PY=/home/erikg/emender/.venv/bin/python
RUNS=$X45/k24_runs
mkdir -p "$RUNS"

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
  "$X45/k24_long_window.py"
  "$X45/k24_gate_receipt.py"
  "$X45/test_k24_long_window.py"
  "$X45/run_k24_long_window_leased.sh"
  "$X45/PREREGISTRATION_K24_LONG_WINDOW.md"
  "$X45/receipts/k24_window_arithmetic.json"
  tests/test_k24_long_window_probe.py
)

# One leased card for the duration of ONE step, in a subshell: the lease
# owner is the subshell's $$, the heartbeat keeper dies with it, and the
# EXIT trap releases at the step's end (the next step acquires fresh --
# the pool is 2 cards and the lane is sequential).
with_lease() {
  (
    eval "$(GPU_LEASE_VISIBLE=${GPU_LEASE_VISIBLE} scripts/gpu_lease.sh acquire 1 --wait)"
    echo "leased card: ${GPU_LEASE_HELD:-none} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
    "$@"
  )
}

anchor_ok() {
  # fail closed: no long window arm runs before BOTH 1x anchors have
  # landed their bit-for-bit bars (the anchor convention is non-negotiable)
  local label="$1"
  for arm in ours_1x gdn2_1x; do
    if [ ! -f "$RUNS/${arm}_seed3401/result.json" ]; then
      echo "REFUSING to run ${label}: the ${arm} anchor has not landed -- the anchors MUST reproduce the banked rows bit for bit FIRST (STOP and escalate)" >&2
      return 1
    fi
    if ! $PY - "$RUNS/${arm}_seed3401/result.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
checks = (r.get("anchor_receipt") or {}).get("checks") or {}
ok = checks.get("landing_bit_identical") and checks.get("trajectory_bit_identical") \
     and checks.get("constant_bit_identical", True) and checks.get("ema_bit_identical", True)
raise SystemExit(0 if ok else 1)
EOF
    then
      echo "REFUSING to run ${label}: the ${arm} anchor's bit-for-bit bar is not green in its receipt (STOP and escalate)" >&2
      return 1
    fi
  done
}

run_step() {
  local step="$1"
  case "$step" in
    registration)
      $PY -m experiments.expressivity_tasks.x45_matrix_emender.k24_long_window \
        window_receipt 2>&1 | tee "$RUNS/window_receipt_console.txt"
      commit_run "X45 K24 THE LONG-WINDOW PROBE: the REGISTRATION DRAFT + the engine + the suites, committed BEFORE any arm (the standing lesson) -- the window-arithmetic receipt computed from the flagship manifest and committed with it: the banked 1x landing reproduced by the arithmetic itself (step 180 / 20,022,866 targets / 11,520 windows -- the anchor protocol's own numbers, computed not hardcoded), the longer arms' step shape holding the ~111K-targets/step and ~180-step landing (windows/step 64/38/27) and the memory-matched 16,384 tokens/microbatch (8x2048/4x4096/2x8192), the honest effective-window distribution (means 1738/2902/4180 B, full-length fractions 0.718/0.530/0.323 -- the 4x arm's ACHIEVED mean is 2.405x the banked mean, not 4x; the lever is only as strong as what was actually achieved); the seven arms ({1x,2x,4x}x{ours-composed,gdn2} + {4x}x{hybrid}, seed 3401, matched 20M-target exposure, the corrected cooldown-primary scoring); THE ANCHOR CONVENTION (ours_1x must reproduce 1.3811104467936925 and gdn2_1x 1.3521734476089478 bit for bit -- full trajectories, constant-180, EMA-180 and the landing, every digit -- before any long window runs; any mismatch STOPS the lane); the eval rule (the first 7 FULL windows of the arm's own context in calibration order; set-equal to the banked L.evaluate set at 1x, asserted by the tests); the pre-state (L-1 the horizon hypothesis: the load-bearing count RISES with window length; L-2: the quality gap SHRINKS; the REFUTATION clause: mask dark + gap flat at 4x kills the horizon hypothesis at probe scale -- no band-worship, every delta against the 0.0119 seed band); the engine gates at the longest window (the composed scan vs the eager oracle at 8192 tokens, the fp64 gradchecks with longer sequences, the honest walls at the three window shapes); the engines imported VERBATIM, never modified (k15 composed engine, the GDN2 committed arm, the k22 hybrid engine)" \
        "${LANE_SOURCES[@]}"
      ;;
    gates)
      with_lease $PY -m \
        experiments.expressivity_tasks.x45_matrix_emender.k24_gate_receipt \
        --out "$X45/receipts/k24_gates.json" \
        2>&1 | tee "$RUNS/gate_receipt_console.txt"
      commit_run "X45 K24 THE LONG-WINDOW PROBE: the long-window engine gate receipts, BEFORE any arm -- the committed composed scan at the 4x window shape (batch 2 x 8192, the 4x microbatch's own scan shape) BIT-EQUAL to the eager per-token oracle on every schedule channel (slot/erase/op/live_after/weights/live_pre/state/live/keys/payload) with the reads at the measured fp32-vs-fp64 floor, the fp64 gradchecks with LONGER sequences (the composed path's every adjoint branch at a 5-segment 320-token fixture vs the banked gate's own 48; the hybrid seam at a 2-segment 128-token fixture vs the banked gate's own 12), the kernel path's seven gradients within the measured floors AT THE 4X SCAN SHAPE, the segment-recompute bit-equality at 128 segments, and the honest walls at all three window shapes (the same 16,384 tokens per microbatch, the sequential walk's own scaling stated from measurement); receipts/k24_gates.json + the console committed" \
        "$X45/receipts/k24_gates.json" "$RUNS/gate_receipt_console.txt" "${LANE_SOURCES[@]}"
      ;;
    suites)
      $PY - <<'EOF' 2>&1 | tee "$RUNS/k24_module_load_check.txt"
import experiments.expressivity_tasks.x45_matrix_emender.k24_long_window as k
print("k24 module loaded:", sorted(k.ARMS))
import experiments.expressivity_tasks.x45_matrix_emender.k24_gate_receipt as g
print("k24 gate module loaded")
EOF
      status=0
      with_lease $PY -m pytest -x -q \
        experiments/expressivity_tasks/x45_matrix_emender/test_k24_long_window.py \
        experiments/expressivity_tasks/x45_matrix_emender/test_k22_hybrid.py \
        experiments/expressivity_tasks/x45_matrix_emender/test_k10_separated_keys.py \
        experiments/expressivity_tasks/x45_matrix_emender/test_k13_additive_read.py \
        experiments/expressivity_tasks/x45_matrix_emender/test_k11b_channel_gates.py \
        experiments/expressivity_tasks/x45_matrix_emender/test_k15_composition.py \
        experiments/expressivity_tasks/x45_matrix_emender/test_k17_dense_write.py \
        experiments/expressivity_tasks/x45_matrix_emender/test_k18_short_conv.py \
        tests/test_k24_long_window_probe.py \
        tests/test_k22_hybrid_probe.py \
        2>&1 | tee "$RUNS/k10_k13_k11b_k15_k17_k18_k22_k24_suites_console.txt" || status=$?
      echo "SUITES_RC=$status"
      commit_run "X45 K24 THE LONG-WINDOW PROBE: the committed suites' green receipts with the K24 module loaded -- the lane's own suite + the K10/K13/K11B/K15/K17/K18/K22 suites + the CPU-only probe tests, one console committed (the standing receipt: the module's import changes nothing the committed engines do)" \
        "$RUNS/k24_module_load_check.txt" \
        "$RUNS/k10_k13_k11b_k15_k17_k18_k22_k24_suites_console.txt" "${LANE_SOURCES[@]}"
      [ "$status" -eq 0 ]
      ;;
    smoke:*)
      local machine="${step#smoke:}"
      local arm
      case "$machine" in
        OURS) arm=ours_4x ;;
        GDN2) arm=gdn2_4x ;;
        HYBRID) arm=hybrid_4x ;;
        *) echo "unknown smoke machine: $machine" >&2; return 2 ;;
      esac
      local out="$RUNS/smoke_${arm}"
      with_lease $PY -m \
        experiments.expressivity_tasks.x45_matrix_emender.k24_long_window \
        arm --arm "$arm" --output "$out" --smoke \
        2>&1 | tee "$RUNS/smoke_${arm}.console.txt"
      [ -f "$out/result.json" ] || { echo "the smoke did NOT land (no result.json)"; return 1; }
      commit_run "X45 K24 THE LONG-WINDOW PROBE: the ${machine} 4x pipeline smoke (tiny exposure, never a ladder row) -- the long-window plumbing shakedown: the longer-context stream construction, the context-aware grouping, the 8192-token scan, the own-window eval, the cross-eval grid and the instruments at the long window, all exercised end to end at tiny exposure" \
        "$out" "$RUNS/smoke_${arm}.console.txt" "${LANE_SOURCES[@]}"
      ;;
    arm:*)
      local arm="${step#arm:}"
      local mult="${arm##*_}"; mult="${mult%x}"
      if [ "$mult" != "1" ]; then
        anchor_ok "$arm" || return 3
      fi
      local out="$RUNS/${arm}_seed3401"
      with_lease $PY -m \
        experiments.expressivity_tasks.x45_matrix_emender.k24_long_window \
        arm --arm "$arm" --seed 3401 --output "$out" \
        2>&1 | tee "$RUNS/${arm}_seed3401.console.txt"
      [ -f "$out/result.json" ] || { echo "the arm did NOT land (no result.json) -- not committed as landed"; return 1; }
      local role=""
      if [ "$mult" = "1" ]; then
        role="THE ANCHOR BAR: the banked row reproduced bit for bit (every training-log row, the exposure landing, the constant-180, the EMA-180 and the post-cooldown eval, every digit -- the runner raises and the lane stops on any mismatch)"
      fi
      local instruments=""
      case "$arm" in
        ours_*) instruments="THE DECORATIVE SWEEP AS THE HEADLINE (the load-bearing mask at the ${mult}x window -- the curve this lane exists for), the C5 instrument (effective rows, occupancy, the op mix, the identity share, the p3 gate telemetry) at the arm's own window length" ;;
        gdn2_*) instruments="GDN2's own state-saturation telemetry at the ${mult}x window (the retention exp(g), the implied memory horizon, the erase/write gates per layer)" ;;
        hybrid_*) instruments="THE DECORATIVE SWEEP AS THE HEADLINE on the slot layers at 4x (do the slot reads become load-bearing when the regime demands state?), the C5 instrument on the slot layers, the gate trajectories (do the gates open FURTHER than the banked 1x hybrid's |gate| ~ 0.13?), the at-birth receipts (born as GDN2 vs a pure twin at the 4x microbatch, the open signal, the inert birth, the store fills)" ;;
      esac
      commit_run "X45 K24 THE LONG-WINDOW PROBE: the ${arm} arm landed (seed 3401) -- the ${mult}x window-length cell of the registered bracket: the committed machine VERBATIM (engines imported, never modified), matched 20M-target exposure, the corrected cooldown-primary scoring, the step shape per the registration (windows/step $( [ "$mult" = 1 ] && echo 64 || ([ "$mult" = 2 ] && echo 38 || echo 27) ), microbatch $( [ "$mult" = 1 ] && echo 8 || ([ "$mult" = 2 ] && echo 4 || echo 2) ) x ${mult}x2048B = 16,384 tokens/microbatch), the honest effective-window distribution receipted in the run (the achieved mean/p50/p90 and the full-length fraction -- the lever is only as strong as what was actually achieved), the own-window eval + the cross-eval grid at all three window lengths, ${instruments}, the honest wall first-class (matched-exposure: same bytes, different clocks); ${role}" \
        "$out" "${LANE_SOURCES[@]}"
      ;;
    assemble)
      $PY -m experiments.expressivity_tasks.x45_matrix_emender.k24_long_window \
        assemble --runs "$RUNS" 2>&1 | tee "$RUNS/assemble_console.txt"
      commit_run "X45 K24 THE LONG-WINDOW PROBE: the assemble -- the gap-vs-window curve, the load-bearing-count-vs-window curve (THE headline instrument: does the decorative sweep light up as windows grow?), the effective-window telemetry, the mechanism rows (the C5 rows, GDN2's state-saturation telemetry, the hybrid's gate trajectories), the L-1/L-2/REFUTATION clause adjudications with every delta against the 0.0119 seed band, and the escalation flags (the operator hears any 4x inversion from the receipts)" \
        "$RUNS/assemble_console.txt" "$RUNS/assemble.json" "${LANE_SOURCES[@]}"
      ;;
    chain)
      run_step registration
      run_step gates
      run_step suites
      run_step smoke:OURS
      run_step smoke:GDN2
      run_step smoke:HYBRID
      run_step arm:ours_1x
      run_step arm:gdn2_1x
      anchor_ok "the chain" || return 3
      run_step arm:ours_2x
      run_step arm:gdn2_2x
      run_step arm:ours_4x
      run_step arm:gdn2_4x
      run_step arm:hybrid_4x
      run_step assemble
      ;;
    *)
      echo "unknown STEP: $step" >&2
      return 2
      ;;
  esac
}

run_step "$STEP"
