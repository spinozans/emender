#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
x36_runner=experiments/expressivity_tasks/x36_tape_perf/run_leased.sh
x36_receipts=experiments/expressivity_tasks/x36_tape_perf/receipts
for x36_live in 256 1024 2800; do
  bash "$x36_runner" continued-capacity --live "$x36_live" --output "part6_hot_live${x36_live}.json" > "$x36_receipts/part6_hot_live${x36_live}.txt" 2>&1
done
bash "$x36_runner" continued-capacity --near-budget --output part6_hot_near_budget.json > "$x36_receipts/part6_hot_near_budget.txt" 2>&1
bash "$x36_runner" continued-capacity --live 256 --slots 11776 --profile --output part6_hot_profile.json > "$x36_receipts/part6_hot_profile.txt" 2>&1
bash "$x36_runner" part2-gates --profile --traffic --output part6_final_fresh.json > "$x36_receipts/part6_final_fresh.txt" 2>&1
for x36_slots in 8192 8224; do
  bash "$x36_runner" part2-gates --tokens 512 --slots "$x36_slots" --profile --traffic --output "part6_final_fresh_s${x36_slots}.json" > "$x36_receipts/part6_final_fresh_s${x36_slots}.txt" 2>&1
done
bash "$x36_runner" train-slice --output part6_final_training_slice.json > "$x36_receipts/part6_final_training_slice.txt" 2>&1
