#!/usr/bin/env bash
# K11B stage-2 arms chain, v4 (detached, the long-session survival playbook).
#
# The falsifier control has PASSED on the fixed engine (step-1 BIT-IDENTICAL
# to the banked row, post-cooldown landing at parity, committed at f7051b72)
# and the re-verification bill is green -- the arms may start, exactly as
# registered.  v4 = chain v3's phases B/C/D only:
#   phase B: p1 and p2 (seed 3401) in parallel, one leased card each;
#   phase C: p3 (seed 3401) plus the 3402 confirmation of any arm that
#     cleared the 0.01 win bar (post-cooldown <= 1.4238 vs the corrected
#     anchor 1.4338248457227434, or constant-180 <= 1.4807 vs the banked
#     seed-3401 row 1.4907409463609969);
#   phase D: p3's own win bar -> its 3402 confirmation.
# Each run self-commits via run_k11b_channel_gates_leased.sh as it lands.
set -uo pipefail
cd /home/erikg/emender
X45=experiments/expressivity_tasks/x45_matrix_emender
RUNS=experiments/expressivity_tasks/x45_matrix_emender/k11b_runs
PY=/home/erikg/emender/.venv/bin/python
export OMP_NUM_THREADS=4

log(){ echo "[chain4 $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

arm_line(){   # arm seed -> "win|nowin|missing status pc=... const=..."
  $PY - "$1" "$2" <<'PYEOF'
import json, sys
arm, seed = sys.argv[1], int(sys.argv[2])
try:
    r = json.load(open(f"experiments/expressivity_tasks/x45_matrix_emender/"
                       f"k11b_runs/{arm}_seed{seed}/result.json"))
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

log "phase B: the live p1 and p2 arms in parallel (seed 3401)"
bash "$X45/run_k11b_channel_gates_leased.sh" p1 3401 \
  > k11b_runs/p1_seed3401.console.txt 2>&1 &
b1=$!
bash "$X45/run_k11b_channel_gates_leased.sh" p2 3401 \
  > k11b_runs/p2_seed3401.console.txt 2>&1 &
b2=$!
wait $b1; log "p1 exit $? : $(arm_line p1 3401)"
wait $b2; log "p2 exit $? : $(arm_line p2 3401)"

declare -a c_jobs=()
run_arm_job(){   # arm seed
  bash "$X45/run_k11b_channel_gates_leased.sh" "$1" "$2" \
    > "k11b_runs/${1}_seed${2}.console.txt" 2>&1 &
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

line=$(arm_line p3 3401)
case "$line" in
  win*) log "phase D: p3 cleared the 0.01 win bar ($line) -- the second seed 3402"
         bash "$X45/run_k11b_channel_gates_leased.sh" p3 3402 \
           > k11b_runs/p3_seed3402.console.txt 2>&1
         log "p3 seed 3402 exit $?: $(arm_line p3 3402)" ;;
  *)    log "phase D: p3 did not clear the win bar ($line) -- no second seed" ;;
esac

log "stage 2 arms chain v4 complete -- harvest and assemble"
