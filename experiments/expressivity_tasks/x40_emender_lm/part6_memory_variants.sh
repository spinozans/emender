#!/usr/bin/env bash
# Two complete updates include Adam states on the second forward/backward.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for context in 512 2048; do
  cp "$root/receipts/part6/checkpoint_gate_B1_T${context}_packed_fp32.json" "$root/receipts/part6/checkpoint_gate_B1_T${context}_packed_fp32_numeric_equal.json"
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --context "$context" --packed --fp32 > "$root/receipts/part6/checkpoint_gate_T${context}_packed_fp32_bytes.log" 2>&1 || exit 1
  cp "$root/receipts/part6/checkpoint_gate_B1_T${context}_packed.json" "$root/receipts/part6/checkpoint_gate_B1_T${context}_packed_numeric_equal.json"
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --context "$context" --packed > "$root/receipts/part6/checkpoint_gate_T${context}_packed_bytes.log" 2>&1 || exit 1
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --context "$context" > "$root/receipts/part6/checkpoint_gate_T${context}_unpacked_bf16.log" 2>&1 || exit 1
  for precision in bf16 fp32; do
    options=(); if [[ "$precision" == fp32 ]]; then options+=(--fp32); fi
    "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --batch 32 --context "$context" --packed --require-expandable --tag _expandable "${options[@]}" > "$root/receipts/part6/checkpoint_gate_B32_T${context}_${precision}.log" 2>&1 || exit 1
  done
done
"$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --context 2048 --packed --offload --require-expandable --tag _expandable > "$root/receipts/part6/offload_expandable_gate_T2048.log" 2>&1 || exit 1
for scale in 150M 1.3B; do
  for context in 512 2048; do
    for batch in 64 128; do
      variants=(legacy recorded_physical accumulated packed)
      if [[ "${X40_MEMORY_INCLUDE_OFFLOAD:-0}" == 1 && "$scale" == 1.3B && "$context" == 2048 ]]; then variants+=(offloaded); fi
      for variant in "${variants[@]}"; do
        options=()
        microbatch=$batch
        case "$variant" in
          legacy) options+=(--legacy-checkpoint) ;;
          accumulated) microbatch=32 ;;
          packed) microbatch=32; options+=(--packed) ;;
          offloaded) microbatch=32; options+=(--packed --offload-checkpoint --require-expandable) ;;
        esac
        label="memory_${variant}"
        name="${scale}_pure_B${batch}_T${context}"
        "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_surface --scale "$scale" --arm pure --batch "$batch" --context "$context" --microbatch "$microbatch" --warmups 0 --steps 2 --label "$label" "${options[@]}" > "$root/receipts/part6/${label}_${name}.log" 2>&1
        echo "$variant $name exit=$?"
      done
    done
  done
done
