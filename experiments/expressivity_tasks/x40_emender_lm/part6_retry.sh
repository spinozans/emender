#!/usr/bin/env bash
# Largest-context pure arms need additional input-storage relief; no precision cut.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
for batch in 1 2; do
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_gate --context 2048 --batch "$batch" --packed --offload > "$root/receipts/part6/offload_gate_B${batch}_T2048.log" 2>&1
  echo "offload B$batch byte gate passed"
done
mkdir "$root/receipts/part6/surface_before_offload"
for batch in 64 128; do
  name="1.3B_pure_B${batch}_T2048"
  mv "$root/receipts/part6/surface/${name}.json" "$root/receipts/part6/surface_before_offload/"
  mv "$root/receipts/part6/surface/${name}.log" "$root/receipts/part6/surface_before_offload/"
  "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_surface --scale 1.3B --arm pure --batch "$batch" --context 2048 --packed --offload-checkpoint > "$root/receipts/part6/surface/${name}.log" 2>&1
  echo "$name offloaded exit=0"
done
