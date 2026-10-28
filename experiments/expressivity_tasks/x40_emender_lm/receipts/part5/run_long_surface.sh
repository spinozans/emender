#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
receipt=experiments/expressivity_tasks/x40_emender_lm/receipts/part5
runner=experiments/expressivity_tasks/x40_emender_lm/run_leased.sh
/home/erikg/emender/.venv/bin/python -c 'import json; from pathlib import Path; p=Path("experiments/expressivity_tasks/x40_emender_lm/receipts/part5/production_T2048.json"); assert json.loads(p.read_text())["status"]=="passed"' || exit 1
mkdir -p "$receipt/surface_pre_long_gate"
for scale in 22M 150M 1.3B; do
  for batch in 64 128; do
    for arm in pure transformer; do
      name="${scale}_${arm}_B${batch}_T2048"
      cp "$receipt/surface/$name.json" "$receipt/surface_pre_long_gate/$name.json"
      "$runner" -m experiments.expressivity_tasks.x40_emender_lm.scale_surface --scale "$scale" --arm "$arm" --batch "$batch" --context 2048 --steps 20 > "$receipt/${name}_after_gate.log" 2>&1
      printf '%s %s\n' "$name" "$?" >> "$receipt/long_surface_exit_codes.txt"
    done
  done
done
