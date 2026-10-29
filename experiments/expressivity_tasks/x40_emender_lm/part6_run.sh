#!/usr/bin/env bash
# Sequential attended-session driver. No dispatchers or workers.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for context in 512 2048; do
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --context "$context" --packed --require-expandable --tag _expandable > "$root/receipts/part6/checkpoint_gate_T${context}_packed_bf16.log" 2>&1 || exit 1
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part5_gate --batch 1 --context "$context" --receipt "/home/erikg/emender/$root/receipts/part6/production_T${context}_expandable.json" > "$root/receipts/part6/production_T${context}.log" 2>&1 || exit 1
  echo "context $context gates passed"
done
mkdir -p "$root/receipts/part6/surface"
failures=0
for scale in 22M 150M 1.3B; do
  for context in 512 2048; do
    for batch in 64 128; do
      for arm in pure transformer; do
        name="${scale}_${arm}_B${batch}_T${context}"
        "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_surface --scale "$scale" --arm "$arm" --context "$context" --batch "$batch" --packed --require-expandable > "$root/receipts/part6/surface/${name}.log" 2>&1
        arm_status=$?
        echo "$name exit=$arm_status"
        if (( arm_status != 0 )); then failures=$((failures+1)); fi
      done
    done
  done
done

if (( failures != 0 )); then exit 1; fi
