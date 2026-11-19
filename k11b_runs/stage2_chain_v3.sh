#!/usr/bin/env bash
# K11B stage-2 detached chain, v2 (the long-session survival playbook).
#
# v1 ran the arms sequentially; v2 parallelizes the independent arms across
# the lane's two cards (0/1) because the whole program -- falsifier + three
# arms + the 3402 confirmations + the assemble + the docs -- must land
# inside one attended session, and the falsifier's ~1h sequential prefix is
# already half the margin.  The FROZEN-GATES falsifier stays STRICTLY FIRST
# and gates the whole chain (the ruling): its result.json is awaited and
# mechanically gated on the registration 2.4 bars -- (a) step-1 train loss
# BIT-IDENTICAL to the banked same-seed row, (b) the post-cooldown landing
# inside the 0.075 endpoint-variance band of the corrected anchor
# 1.4338248457227434.  Full-trajectory bit-identity is not the bar (the
# registration's honest limitation: ULP-scale reduction-order divergence
# beyond step 1 by construction); the per-step divergence is quoted in the
# receipt for adjudication at harvest.  On failure the chain STOPS and the
# harvest escalates with the receipt.
#
# The falsifier was launched by chain v1 and is already running detached
# (run_k11b_channel_gates_leased.sh p1 3401 frozen, self-committing); this
# chain only WAITS for it.  Then: p1 and p2 in parallel, then p3 plus any
# 3402 confirmations (win bar 0.01) in parallel, at most 2 concurrent (the
# lane's two cards; each run acquires 1 leased card with --wait).
set -uo pipefail
cd /home/erikg/emender
X45=experiments/expressivity_tasks/x45_matrix_emender
RUNS=experiments/expressivity_tasks/x45_matrix_emender/k11b_runs
PY=/home/erikg/emender/.venv/bin/python
export OMP_NUM_THREADS=4

log(){ echo "[chain2 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

arm_line(){   # arm seed -> "win|nowin|missing status pc=... const=..."
  $PY - "$1" "$2" <<'PYEOF'
import json, sys
arm, seed = sys.argv[1], int(sys.argv[2])
try:
    r = json.load(open(f"experiments/expressivity_tasks/x45_matrix_emender/k11b_runs/{arm}_seed{seed}/result.json"))
except FileNotFoundError:
    print("missing"); raise SystemExit
pc = r.get("final_eval_nll_post_cooldown")
const = r.get("final_eval_nll_constant_lr")
banked_const = (1.4907409463609969 if seed == 3401 else 1.4899048124040877)
anchor = 1.4338248457227434
win = ((pc is not None and anchor - pc >= 0.01)
       or (const is not None and banked_const - const >= 0.01))
print("win" if win else "nowin",
      "completed" if r.get("status") == "completed" else r.get("status"),
      f"pc={pc}", f"const={const}")
PYEOF
}

# ---- phase A: WAIT for the falsifier (already running, pid $FALSIFIER_PID
# from chain v1), then gate on the registration 2.4 bars
log "phase A: waiting for the FROZEN-GATES falsifier (p1_frozen_seed3401)"
while [ ! -f "$RUNS/p1_frozen_seed3401/result.json" ]; do
  if [ -n "${FALSIFIER_PID:-}" ] && ! kill -0 "$FALSIFIER_PID" 2>/dev/null; then
    log "FALSIFIER PROCESS DEAD WITHOUT A RECEIPT -- CHAIN STOPPED, escalate"
    exit 10
  fi
  sleep 30
done
verdict=$($PY - <<'PYEOF'
import json
r = json.load(open("experiments/expressivity_tasks/x45_matrix_emender/k11b_runs/p1_frozen_seed3401/result.json"))
fd = r.get("frozen_divergence_receipt") or {}
step1 = bool(r.get("step1_bit_identical_to_banked"))
pc = r.get("final_eval_nll_post_cooldown")
parity = pc is not None and abs(pc - 1.4338248457227434) <= 0.075
print("PASS" if (step1 and parity) else "FAIL", step1, pc,
      fd.get("steps_bit_identical"), fd.get("max_abs_delta"))
PYEOF
)
log "falsifier receipt: $verdict"
case "$verdict" in
  PASS*) ;;
  *) log "FALSIFIER CONTROL FAILED ($verdict) -- CHAIN STOPPED, escalate with the receipt"
     exit 11 ;;
esac

# ---- phase B: p1 and p2 in parallel (independent arms, two cards)
log "phase B: the live p1 and p2 arms in parallel (seed 3401)"
bash "$X45/run_k11b_channel_gates_leased.sh" p1 3401 \
  > "$RUNS/p1_seed3401.console.txt" 2>&1 &
b1=$!
bash "$X45/run_k11b_channel_gates_leased.sh" p2 3401 \
  > "$RUNS/p2_seed3401.console.txt" 2>&1 &
b2=$!
wait $b1; log "p1 exit $? : $(arm_line p1 3401)"
wait $b2; log "p2 exit $? : $(arm_line p2 3401)"

# ---- phase C: p3 plus any 3402 confirmations that the win bar fired on
declare -a c_jobs=()
run_arm_job(){   # arm seed
  bash "$X45/run_k11b_channel_gates_leased.sh" "$1" "$2" \
    > "$RUNS/${1}_seed${2}.console.txt" 2>&1 &
  c_jobs+=($!)
}
log "phase C: the live p3 arm (+ any 3402 confirmations)"
run_arm_job p3 3401
for arm in p1 p2; do
  line=$(arm_line "$arm" 3401)
  case "$line" in
    win*)  log "$arm cleared the 0.01 win bar ($line) -- the second seed 3402"
           run_arm_job "$arm" 3402 ;;
    *)     log "$arm did not clear the win bar ($line)" ;;
  esac
done
for j in "${c_jobs[@]}"; do
  wait "$j"; log "phase C job $j exit $?"
done
for arm in p1 p2 p3; do log "$arm@3401: $(arm_line "$arm" 3401)"; done
for arm in p1 p2; do
  if [ -f "$RUNS/${arm}_seed3402/result.json" ]; then
    log "$arm@3402: $(arm_line "$arm" 3402)"
  fi
done

# ---- phase D: p3's own win bar -> its 3402 confirmation
line=$(arm_line p3 3401)
case "$line" in
  win*) log "phase D: p3 cleared the 0.01 win bar ($line) -- the second seed 3402"
         bash "$X45/run_k11b_channel_gates_leased.sh" p3 3402 \
           > "$RUNS/p3_seed3402.console.txt" 2>&1
         log "p3 seed 3402 exit $?: $(arm_line p3 3402)" ;;
  *)    log "phase D: p3 did not clear the win bar ($line) -- no second seed" ;;
esac

log "stage 2 chain v2 complete -- harvest and assemble"
