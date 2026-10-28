#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
receipt=experiments/expressivity_tasks/x40_emender_lm/receipts/part5
runner=experiments/expressivity_tasks/x40_emender_lm/run_leased.sh
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1
for batch in 128 256; do
  "$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_pilot --batch "$batch" --label "pilot_B${batch}" > "$receipt/pilot_B${batch}.log" 2>&1
  printf 'B%s %s\n' "$batch" "$?" >> "$receipt/occupancy_exit_codes.txt"
done
export X40_FUSED_INIT=1
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_gate > "$receipt/fused_parity.log" 2>&1 || exit 1
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_pilot --label pilot_fused > "$receipt/pilot_fused.log" 2>&1
