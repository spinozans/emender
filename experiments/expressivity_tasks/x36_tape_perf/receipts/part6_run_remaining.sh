#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
x36_runner=experiments/expressivity_tasks/x36_tape_perf/run_leased.sh
x36_receipts=experiments/expressivity_tasks/x36_tape_perf/receipts
for x36_live in 17 1024 2800; do
  bash "$x36_runner" continued-capacity --live "$x36_live" --output "part6_continued_live${x36_live}.json" > "$x36_receipts/part6_continued_live${x36_live}.txt" 2>&1
done
bash "$x36_runner" continued-capacity --near-budget --output part6_continued_near_budget.json > "$x36_receipts/part6_continued_near_budget.txt" 2>&1
bash "$x36_runner" part2-gates --profile --traffic --output part6_fresh.json > "$x36_receipts/part6_fresh.txt" 2>&1
for x36_slots in 8192 8224; do
  bash "$x36_runner" part2-gates --tokens 512 --slots "$x36_slots" --profile --traffic --output "part6_fresh_s${x36_slots}.json" > "$x36_receipts/part6_fresh_s${x36_slots}.txt" 2>&1
done
bash "$x36_runner" train-slice --output part6_training_slice.json > "$x36_receipts/part6_training_slice.txt" 2>&1
