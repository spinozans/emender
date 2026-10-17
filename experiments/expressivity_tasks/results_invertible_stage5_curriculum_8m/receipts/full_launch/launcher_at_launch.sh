#!/usr/bin/env bash
# Six independent X5 runs, one exclusively leased local GPU per run.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-$PWD/experiments/expressivity_tasks/results_invertible_stage5_curriculum_8m}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$OUT/receipts"
test -f "$OUT/registration.json"
test -f "$OUT/receipts/tests_prelaunch.txt"
mkdir "$OUT/receipts/full_launch"
receipt="$OUT/receipts/full_launch"
cp experiments/expressivity_tasks/run_invertible_stage5_curriculum_8m.py "$receipt/runner_at_launch.py"
cp experiments/expressivity_tasks/run_invertible_stage3_8m.py "$receipt/x3_runner_at_launch.py"
cp experiments/expressivity_tasks/run_invertible_stage2.py "$receipt/x2_runner_at_launch.py"
cp "$0" "$receipt/launcher_at_launch.sh"
git rev-parse HEAD > "$receipt/base_commit.txt"
git diff --binary > "$receipt/tracked_diff_at_launch.patch"
"$PYTHON_BIN" -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in m.distributions())))' > "$receipt/packages.txt"
df -B1 . > "$receipt/disk_before.txt"
lease_code="$(bash scripts/gpu_lease.sh acquire 6 --wait)"
eval "$lease_code"
printf '%s\n' "host=$(hostname)" "pid=$$" "gpus=$CUDA_VISIBLE_DEVICES" "start=$(date -Is)" > "$receipt/lease.txt"
IFS=, read -ra cards <<< "$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
pids=(); index=0
for seed in 0 1 2; do
  for arm in e97 gdn2; do
    name="${arm}_seed${seed}"
    run_receipt="$OUT/receipts/$name"
    mkdir "$run_receipt"
    gpu="${cards[$index]}"
    cp "${GPU_LEASE_DIR:-$PWD/.wg/gpu_leases}/$gpu" "$run_receipt/lease.txt"
    cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.run_invertible_stage5_curriculum_8m --arm "$arm" --seed "$seed" --out "$OUT/$name")
    printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu" > "$run_receipt/command.txt"
    printf '%q ' "${cmd[@]}" >> "$run_receipt/command.txt"; echo >> "$run_receipt/command.txt"
    (
      export CUDA_VISIBLE_DEVICES="$gpu"
      printf 'start=%s\ngpu=%s\nhost=%s\n' "$(date -Is)" "$gpu" "$(hostname)" > "$run_receipt/start.txt"
      set +e
      "${cmd[@]}" > "$run_receipt/console.txt" 2>&1
      rc=$?
      set -e
      printf 'exit_code=%s\nend=%s\n' "$rc" "$(date -Is)" > "$run_receipt/exit.txt"
      exit "$rc"
    ) & pids+=("$!")
    index=$((index + 1))
  done
done
failed=0
for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$receipt/gpu_after.txt"
df -B1 . > "$receipt/disk_after.txt"
printf 'exit_code=%s\nend=%s\n' "$failed" "$(date -Is)" > "$receipt/exit.txt"
exit "$failed"
