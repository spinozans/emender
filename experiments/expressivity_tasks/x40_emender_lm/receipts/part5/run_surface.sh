#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
receipt=experiments/expressivity_tasks/x40_emender_lm/receipts/part5
runner=experiments/expressivity_tasks/x40_emender_lm/run_leased.sh
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_gate > "$receipt/production_parity.log" 2>&1 || exit 1
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_gate --batch 1 --context 2048 --receipt production_T2048.json > "$receipt/production_T2048.log" 2>&1 || exit 1
for scale in 22M 150M 1.3B; do
  for context in 512 2048; do
    for batch in 64 128; do
      for arm in pure transformer; do
        name="${scale}_${arm}_B${batch}_T${context}"
        "$runner" -m experiments.expressivity_tasks.x40_emender_lm.scale_surface --scale "$scale" --arm "$arm" --batch "$batch" --context "$context" --steps 20 > "$receipt/${name}.log" 2>&1
        code=$?
        printf '%s %s\n' "$name" "$code" >> "$receipt/surface_exit_codes.txt"
      done
    done
  done
done
