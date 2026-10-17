#!/usr/bin/env bash
# X5D only: six SERIAL diagnostic probes on one leased local GPU. Not an X5 relaunch.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-$PWD/experiments/expressivity_tasks/results_invertible_stage5_instrument}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT/receipts"
test -f "$OUT/data_verification.json"
mkdir "$OUT/receipts/probes"
receipt="$OUT/receipts/probes"
git rev-parse HEAD > "$receipt/base_commit.txt"
git diff --binary > "$receipt/tracked_diff.patch"
cp experiments/expressivity_tasks/verify_invertible_stage5_instrument.py "$receipt/instrument_at_launch.py"
cp "$0" "$receipt/launcher_at_launch.sh"
"$PYTHON_BIN" -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions())))' > "$receipt/packages.txt"
lease_code="$(bash scripts/gpu_lease.sh acquire 1 --wait)"
eval "$lease_code"
printf 'host=%s\npid=%s\ngpu=%s\nstart=%s\n' "$(hostname)" "$$" "$CUDA_VISIBLE_DEVICES" "$(date -Is)" > "$receipt/lease.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
df -B1 . > "$receipt/disk_before.txt"
for depth in 1 2 4; do
  for arm in e97 gdn2; do
    name="${arm}_T${depth}"
    mkdir "$OUT/receipts/$name"
    cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.verify_invertible_stage5_instrument --arm "$arm" --depth "$depth" --out "$OUT/$name")
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES" > "$OUT/receipts/$name/command.txt"
    printf '%q ' "${cmd[@]}" >> "$OUT/receipts/$name/command.txt"; echo >> "$OUT/receipts/$name/command.txt"
    printf 'start=%s\n' "$(date -Is)" > "$OUT/receipts/$name/exit.txt"
    "${cmd[@]}" > "$OUT/receipts/$name/console.txt" 2>&1
    printf 'exit_code=0\nend=%s\n' "$(date -Is)" >> "$OUT/receipts/$name/exit.txt"
    if [[ "$depth" == 1 ]]; then
      # Fail closed before proceeding to deeper probes if one-step does not match.
      "$PYTHON_BIN" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["stopping_accuracy"] >= .97' "$OUT/$name/result.json"
    fi
  done
done
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$receipt/gpu_after.txt"
df -B1 . > "$receipt/disk_after.txt"
printf 'exit_code=0\nend=%s\n' "$(date -Is)" > "$receipt/exit.txt"
