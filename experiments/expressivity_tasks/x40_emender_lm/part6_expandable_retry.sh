#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
# This installed CUDA allocator reads the legacy CUDA-specific variable.
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --context 2048 --packed --require-expandable --tag _expandable > "$root/receipts/part6/expandable_gate_T2048.log" 2>&1
for context in 512 2048; do
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part5_gate --batch 1 --context "$context" --receipt "/home/erikg/emender/$root/receipts/part6/production_T${context}_expandable.json" > "$root/receipts/part6/production_T${context}_expandable.log" 2>&1
  echo "verified expandable production T$context gate passed"
done
for batch in 64 128; do
  name="1.3B_pure_B${batch}_T2048"
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_surface --scale 1.3B --arm pure --batch "$batch" --context 2048 --packed --require-expandable > "$root/receipts/part6/surface/${name}.log" 2>&1
  echo "$name verified expandable exit=0"
done
