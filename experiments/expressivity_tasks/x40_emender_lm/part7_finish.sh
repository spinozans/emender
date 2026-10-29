#!/usr/bin/env bash
# Attended sequential completion; preserve the inherited active surface process.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
receipt="$root/receipts/part6"
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() {
  local label=$1; shift
  "$@" > "$receipt/part7_${label}.log" 2>&1
  local result=$?
  printf '%s exit=%s\n' "$label" "$result" | tee -a "$receipt/part7_commands.log"
  return "$result"
}
while kill -0 2204473 2>/dev/null; do sleep 5; done
run token_equivalence /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_protocol
run baselines bash "$root/part6_baselines.sh"
run memory_variants bash "$root/part6_memory_variants.sh"
for suite in x31_tape_cell_fused x36_tape_perf; do
  run "$suite" "$root/run_leased.sh" -m pytest -q -p experiments.expressivity_tasks.x40_emender_lm.oracle_adapter "experiments/expressivity_tasks/$suite/tests" --junitxml="$receipt/$suite.xml"
done
if run curve "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_curve; then
  label=pilot_B128_part7_registers0_chunk1024
  if run pilot "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --batch 128 --no-checkpoint --profile --label "$label"; then
    run matched_pilot "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_batch_curve --receipt "$label.json"
  fi
  for physical_batch in ${X40_PART7_OCCUPANCY_BATCHES:-}; do
    label="pilot_B${physical_batch}_part7_packed_registers0_chunk1024"
    if run "pilot_B${physical_batch}" "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --batch "$physical_batch" --packed --profile-if-gate --label "$label"; then
      run "matched_B${physical_batch}" "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_batch_curve --receipt "$label.json"
    fi
  done
fi
if run fusion_gate "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pool_fusion; then
  certified=1
  for suite in x31_tape_cell_fused x36_tape_perf; do
    run "fusion_$suite" "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pool_fusion --action tests --test-label "$suite" -q -p experiments.expressivity_tasks.x40_emender_lm.oracle_adapter "experiments/expressivity_tasks/$suite/tests" --junitxml="$receipt/fused_pool_$suite.xml" || certified=0
  done
  if (( certified )); then
    if run fusion_curve "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pool_fusion --action curve --output "$receipt/fused_pool_curve" --bounded-loss --steps 64; then
      for fusion in off on; do
        options=(); [[ "$fusion" == on ]] && options+=(--fused-pool)
        label="pilot_B128_fusion_${fusion}_clear_registers0_chunk1024"
        if run "fusion_${fusion}_pilot" "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --batch 128 --no-checkpoint --profile --label "$label" "${options[@]}"; then
          run "fusion_${fusion}_matched" "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_batch_curve --receipt "$label.json"
        fi
      done
    fi
  fi
fi
run memory_report /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_memory_report
run fusion_report /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_fusion_report
run summary /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x40_emender_lm.part6_summary
