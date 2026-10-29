#!/usr/bin/env bash
# Run only after part7_finish.sh returns; finish pilot occupancy and small OOM controls.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
receipt="$root/receipts/part6"
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for context in 512 2048; do
  for batch in 64 128; do
    [[ "$context/$batch" == 512/64 ]] && continue
    for variant in legacy recorded_physical accumulated packed; do
      options=(); microbatch=$batch
      case "$variant" in
        legacy) options+=(--legacy-checkpoint) ;;
        accumulated) microbatch=32 ;;
        packed) microbatch=32; options+=(--packed) ;;
      esac
      name="22M_pure_B${batch}_T${context}"
      "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_surface --scale 22M --arm pure --batch "$batch" --context "$context" --microbatch "$microbatch" --warmups 0 --steps 2 --label "memory_$variant" "${options[@]}" > "$receipt/memory_${variant}_${name}.log" 2>&1
      echo "$variant $name exit=$?" >> "$receipt/part7_extra_commands.log"
    done
  done
done
for scale in 22M 150M 1.3B; do
  for context in 512 2048; do
    for batch in 64 128; do
      name="${scale}_transformer_B${batch}_T${context}"
      if ! /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]=="passed")' "$root/receipts/part5/surface/$name.json"; then continue; fi
      for variant in physical accumulated; do
        microbatch=$batch; [[ "$variant" == accumulated ]] && microbatch=32
        "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_surface --scale "$scale" --arm transformer --batch "$batch" --context "$context" --microbatch "$microbatch" --warmups 0 --steps 2 --label "memory_transformer_$variant" > "$receipt/memory_transformer_${variant}_${name}.log" 2>&1
        echo "transformer_$variant $name exit=$?" >> "$receipt/part7_extra_commands.log"
      done
    done
  done
done
for physical_batch in ${X40_PART7_OCCUPANCY_BATCHES:-}; do
  label="pilot_B${physical_batch}_part7_packed_registers0_chunk1024"
  [[ -f "$receipt/$label.json" ]] && continue
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_pilot_checkpoint --batch "$physical_batch" --packed --profile-if-gate --label "$label" > "$receipt/part7_pilot_B${physical_batch}.log" 2>&1
  result=$?; echo "pilot_B$physical_batch exit=$result" >> "$receipt/part7_extra_commands.log"
  if (( result == 0 )); then
    "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_batch_curve --receipt "$label.json" > "$receipt/part7_matched_B${physical_batch}.log" 2>&1
    echo "matched_B$physical_batch exit=$?" >> "$receipt/part7_extra_commands.log"
  fi
done
