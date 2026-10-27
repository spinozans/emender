#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
x36_runner=experiments/expressivity_tasks/x36_tape_perf/run_leased.sh
x36_receipts=experiments/expressivity_tasks/x36_tape_perf/receipts
bash "$x36_runner" validate -q > "$x36_receipts/part7_final_parity.txt" 2>&1
for x36_live in 17 256 1024 2800; do
  bash "$x36_runner" continued-capacity --live "$x36_live" --output "part7_live${x36_live}.json" > "$x36_receipts/part7_live${x36_live}.txt" 2>&1
done
bash "$x36_runner" continued-capacity --live 256 --slots 11776 --profile --output part7_profile.json > "$x36_receipts/part7_profile.txt" 2>&1
for x36_tokens in 512 4096; do
  bash "$x36_runner" part2-gates --tokens "$x36_tokens" --slots 2944 --profile --traffic --output "part7_fresh_t${x36_tokens}.json" > "$x36_receipts/part7_fresh_t${x36_tokens}.txt" 2>&1
done
