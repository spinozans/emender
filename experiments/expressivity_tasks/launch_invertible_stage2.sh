#!/usr/bin/env bash
# Local GPU experiment processes only; does not start WorksGood or use Slurm.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT="$PWD"
OUT="$ROOT/experiments/expressivity_tasks/results_invertible_stage2"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
run_arm() (
    local arm="$1" seed="$2" drivers="$3" kind="$4"
    local name="${kind}_${arm}_seed${seed}"
    local receipt="$OUT/receipts/$name"
    mkdir -p "$receipt"
    eval "$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"
    printf '%s\n' "host=$(hostname)" "pid=$$" "gpu=$CUDA_VISIBLE_DEVICES" "start=$(date -Is)" > "$receipt/lease.txt"
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
    { printf '%q' "$PYTHON_BIN"; printf ' %q' -m experiments.expressivity_tasks.run_invertible_stage2 --arm "$arm" --seed "$seed" --drivers "$drivers" --out "$OUT/$name"; printf '\n'; } > "$receipt/command.txt"
    set +e
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$PYTHON_BIN" -m experiments.expressivity_tasks.run_invertible_stage2 \
        --arm "$arm" --seed "$seed" --drivers "$drivers" --out "$OUT/$name" > "$receipt/console.txt" 2>&1
    local rc=$?
    set -e
    printf '%s\n' "exit_code=$rc" "end=$(date -Is)" > "$receipt/exit.txt"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$receipt/gpu_after.txt"
    return "$rc"
)
wait_wave() {
    local failed=0 pid
    for pid in "$@"; do
        if ! wait "$pid"; then failed=1; fi
    done
    return "$failed"
}
mkdir -p "$OUT/receipts"
case "${1:?choose calibration or headline}" in
  calibration)
    pids=()
    for arm in e97 gdn2 e97_identity; do
        run_arm "$arm" 0 1 calibration & pids+=("$!")
    done
    wait_wave "${pids[@]}"
    ;;
  headline)
    # Failed calibration arms remain published; no silent skip or inference.
    for arm in e97 gdn2 e97_identity; do
        test -f "$OUT/calibration_${arm}_seed0/result.json" || {
            echo "Calibration incomplete/failed for $arm: fail closed" >&2; exit 1;
        }
    done
    pids=()
    for arm in e97 gdn2; do
        for seed in 0 1 2; do
            run_arm "$arm" "$seed" 128 headline & pids+=("$!")
        done
    done
    run_arm e97_identity 0 128 headline & pids+=("$!")
    wait_wave "${pids[@]}"
    ;;
  *) echo 'choose calibration or headline' >&2; exit 2;;
esac
