#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
root=experiments/expressivity_tasks/x40_emender_lm
export X40_BATCHED_READS=0 X40_FRESH_POOL=0 X40_STABLE_MAPPING=0 X40_FUSED_INIT=0
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
for scale in 22M 150M 1.3B; do
  for context in 512 2048; do
    for batch in 64 128; do
      for arm in pure transformer; do
        name="${scale}_${arm}_B${batch}_T${context}"
        if /home/erikg/emender/.venv/bin/python -c 'import json,sys;sys.exit(json.load(open(sys.argv[1]))["status"]=="passed")' "$root/receipts/part5/surface/${name}.json"; then
          "$root/run_leased.sh" -m experiments.expressivity_tasks.x40_emender_lm.part6_baseline --scale "$scale" --arm "$arm" --context "$context" --batch "$batch" --steps 1 > "$root/receipts/part6/baseline_${name}.log" 2>&1
          echo "$name baseline exit=$?"
        fi
      done
    done
  done
done
