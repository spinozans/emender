#!/usr/bin/env bash
# Detached leased driver for the X46 growth separator (cards 2/3, this
# lane's pool; the C3 lane's card 4 -- the detached emender run -- is never
# touched; card 7 is forbidden).  Each command runs in its own process under
# a broker lease and COMMITS its own artifacts when it lands, so the run
# survives an attended-session ceiling (the long-session survival playbook:
# setsid + redirected stdio; the lease owner and heartbeat keeper live in
# the detached process).
#
#   usage: run_x46_leased.sh COMMAND [ARGS...]
#     COMMAND in { selftest, smoke, calibrate, arm --machine NAME [--smoke],
#                  sweep --n N, assemble --n N }
#
#   examples:
#     run_x46_leased.sh selftest
#     run_x46_leased.sh smoke --n 320 --steps 30
#     run_x46_leased.sh calibrate --n 320
#     run_x46_leased.sh arm --machine ours-growable --n 320 --lr 1e-3
#     run_x46_leased.sh assemble --n 320
#
#   The arm/sweep/assemble commands are FAIL-CLOSED until
#   PREREGISTRATION_X46_GROWTH_SEPARATOR.md is OPERATOR-SIGNED.
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
MOD=$X45/x46_growth_separator.py
CMD=${1:?"usage: run_x46_leased.sh COMMAND [ARGS...]"; }
shift || true

# this lane's pool: cards 2/3 (never 0/1/5/6, never the C3 lane's 4, never 7)
if [ -z "${GPU_LEASE_VISIBLE:-}" ]; then
  export GPU_LEASE_VISIBLE="2,3"
fi
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
trap 'GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "${GPU_LEASE_HELD}" >/dev/null 2>&1 || true' EXIT
echo "leased card: ${GPU_LEASE_HELD:-none} (x46 $CMD $*)"

export OMP_NUM_THREADS=4
export NDM_PIN_TRITON_AUTOTUNE=0

if $PY -m experiments.expressivity_tasks.x45_matrix_emender.x46_growth_separator \
    "$CMD" "$@"; then
  case "$CMD" in
    selftest)
      FILES="$MOD $X45/test_x46_growth_separator.py $X45/DESIGN_X46_GROWTH_SEPARATOR.md $X45/PREREGISTRATION_X46_GROWTH_SEPARATOR.md $X45/RESULT_X46_GROWTH_SEPARATOR.md $X45/run_x46_leased.sh $X45/receipts/x46_selftest.json"
      MSG="X46 selftest re-run green (the derivation checks: slot-count, aliasing swap, Welch/crosstalk, the registered trigger on overflow, the construction invariants)";;
    smoke)
      FILES="$X45/receipts/x46_smoke.json"
      MSG="X46 smoke landed: all four arms at tiny budget, the growable arms' growth events receipted on the constructed stream";;
    calibrate)
      FILES="$X45/receipts/x46_gdn2_calibration.json"
      MSG="X46 gdn2 tau calibration (pre-registration evidence): the trained fixed machine's per-layer smoothed-PR ceiling on the constructed stream, tau_cal committed BEFORE any growth arm";;
    arm)
      MACHINE=""
      N=""
      SMOKE=""
      prev=""
      for a in "$@"; do
        case "$prev" in
          --machine) MACHINE="$a";;
          --n) N="$a";;
        esac
        case "$a" in --smoke) SMOKE="_smoke";; esac
        prev="$a"
      done
      LABEL="${MACHINE:-arm}_n${N:-320}${SMOKE}"
      FILES="$X45/receipts/x46_arm_${LABEL}*.json"
      FILES=$(ls $FILES 2>/dev/null || echo "$X45/receipts")
      MSG="X46 arm landed: ${MACHINE} at n=${N:-320} (matched exposure, the ceiling record and the mechanistic receipts in the cell receipt)";;
    sweep)
      N=""
      prev=""
      for a in "$@"; do
        [ "$prev" = "--n" ] && N="$a"
        prev="$a"
      done
      FILES="$X45/receipts/x46_sweep_n${N:-320}_seed*.json"
      FILES=$(ls $FILES 2>/dev/null || echo "$X45/receipts")
      MSG="X46 sweep landed at n=${N:-320} (the LR grid, seed 3401)";;
    assemble)
      N=""
      prev=""
      for a in "$@"; do
        [ "$prev" = "--n" ] && N="$a"
        prev="$a"
      done
      FILES="$X45/receipts/x46_summary_n${N:-320}.json"
      MSG="X46 assemble: the ceiling adjudication at n=${N:-320} against the pre-stated band";;
    *)
      FILES="$MOD"
      MSG="X46 $CMD";;
  esac
  git add $FILES $MOD 2>/dev/null || git add $MOD
  git commit -q -m "X46 $MSG" || true
else
  echo "x46 $CMD FAILED; no commit" >&2
  exit 1
fi
