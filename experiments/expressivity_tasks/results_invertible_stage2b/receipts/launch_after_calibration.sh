#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
export OUT="$PWD/experiments/expressivity_tasks/results_invertible_stage2b"
export PYTHON_BIN=/usr/bin/python3
while :; do
  complete=0
  for arm in e97 gdn2 e97_identity; do
    receipt="$OUT/receipts/calibration_${arm}_seed0/exit.txt"
    if test -f "$receipt"; then
      grep -qx 'exit_code=0' "$receipt" || exit 1
      complete=$((complete+1))
    fi
  done
  test "$complete" -eq 3 && break
  sleep 30
done
bash experiments/expressivity_tasks/launch_invertible_stage2.sh headline
