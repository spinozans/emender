#!/usr/bin/env bash
# Detached leased driver for the C3 arms (cards 4/5, this lane's pool -- the
# matched-arms/kernel lane's 0/1, the CMA-ES lane's 2/3 and the growth+POP
# lane's 6 are never touched).  Each command runs in its own process under a
# broker lease, and COMMITS its own artifacts when it lands, so the run
# survives an attended-session ceiling (the long-session survival playbook:
# setsid + redirected stdio; the lease owner and heartbeat keeper live in the
# detached process).
#
#   usage: run_gdn2_growth_c3_leased.sh COMMAND [ARGS...]
#     COMMAND in { kshapes, calibrate, arm --arm NAME [--smoke], assemble }
#
#   examples:
#     run_gdn2_growth_c3_leased.sh kshapes
#     run_gdn2_growth_c3_leased.sh calibrate
#     run_gdn2_growth_c3_leased.sh arm --arm gdn2_growth
set -uo pipefail
cd /home/erikg/emender
PY=/home/erikg/emender/.venv/bin/python
X45=experiments/expressivity_tasks/x45_matrix_emender
CMD=${1:?"usage: run_gdn2_growth_c3_leased.sh COMMAND [ARGS...]"}
shift || true

# the C3 lane's pool: prefer card 4, fall back to 5
if [ -z "${GPU_LEASE_VISIBLE:-}" ]; then
  export GPU_LEASE_VISIBLE="4,5"
fi
eval "$(scripts/gpu_lease.sh acquire 1 --wait)" || exit 1
trap 'GPU_LEASE_PID="$$" scripts/gpu_lease.sh release "${GPU_LEASE_HELD}" >/dev/null 2>&1 || true' EXIT
echo "leased card: ${GPU_LEASE_HELD:-none} (c3 $CMD $*)"

export OMP_NUM_THREADS=4
export NDM_PIN_TRITON_AUTOTUNE=0

if $PY -m experiments.expressivity_tasks.x45_matrix_emender.gdn2_growth_c3 \
    "$CMD" "$@"; then
  attempt=0
  case "$CMD" in
    kshapes)
      FILES="$X45/gdn2_growth_c3.py $X45/run_gdn2_growth_c3_leased.sh $X45/test_gdn2_growth_c3.py $X45/receipts/gdn2_growth_c3_kshapes.json"
      MSG="X45 C3 kernel-shape receipts at the regime's long horizons (T 8192/16384, H 2/4/8, bf16 autocast fwd+bwd finite; the probe trajectory finite at every shape) -- no kernel impossibility on the heads axis (the ruling's item 4: receipt, never patch)";;
    calibrate)
      FILES="$X45/gdn2_growth_c3.py $X45/receipts/gdn2_growth_c3_calibration_8192.json $X45/receipts/gdn2_growth_c3_calibration_16384.json $X45/receipts/gdn2_growth_c3_regime.json $X45/c3_runs"
      MSG="X45 C3 tau_sat calibration scan COMPLETE (both horizons) + the regime lock: the fixed GDN2 arm's per-boundary PR receipted on the C3 stream at 8192 AND 16384 with the iota co-diagnostic alongside per boundary; the SHORTEST reachable horizon locked with tau_sat=96, or the escalation receipt if neither horizon reaches it (the ruling's own branch); the pre-stated P-G1/P-G2/P-G3 predictions committed BEFORE any growth arm runs";;
    arm)
      ARM=""
      prev=""
      for a in "$@"; do
        if [ "$prev" = "--arm" ]; then ARM="$a"; fi
        case "$a" in --arm=*) ARM="${a#--arm=}";; esac
        prev="$a"
      done
      SMOKE=$(echo "$*" | grep -c -- --smoke || true)
      LABEL="$ARM$([ "$SMOKE" -gt 0 ] && echo _smoke)"
      FILES="$X45/gdn2_growth_c3.py $X45/run_gdn2_growth_c3_leased.sh $X45/test_gdn2_growth_c3.py $X45/receipts/gdn2_growth_c3_${LABEL}.json $X45/c3_runs"
      MSG="X45 C3 arm landed: $ARM at the locked regime (matched 20M exposure, constant phase + the 24-step cooldown PRIMARY, EMA secondary, the saturation/interference divergence receipts, growth telemetry)";;
    assemble)
      FILES="$X45/gdn2_growth_c3.py $X45/receipts/gdn2_growth_c3_summary.json"
      MSG="X45 C3 assemble: the four-arm table + the P-G1/P-G2/P-G3 adjudication against the pre-stated 0.075 band";;
    *)
      FILES="$X45/gdn2_growth_c3.py"; MSG="X45 C3 $CMD artifacts";;
  esac
  until git add $FILES 2>/dev/null && git commit -m "$MSG" >/dev/null; do
    attempt=$((attempt + 1))
    if [ "$attempt" -ge 12 ]; then
      echo "commit failed after $attempt attempts; artifacts remain on disk"
      exit 1
    fi
    sleep 10   # index.lock contention with the concurrent lanes
  done
  echo "committed: $CMD $*"
  git push origin main >/dev/null 2>&1 || echo "push deferred (offline?)"
else
  echo "the $CMD command FAILED; artifacts remain on disk"
  exit 1
fi
