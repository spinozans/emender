#!/usr/bin/env bash
# Local amendment #5: six explicitly approved exclusive single-GPU processes.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-$PWD/experiments/expressivity_tasks/results_invertible_stage4_13b}"
WEIGHTS="${WEIGHTS:-/mnt/nvme1n1/erikg/invertible_stage4_13b}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export OUT WEIGHTS PYTHON_BIN
SELF="$PWD/experiments/expressivity_tasks/launch_invertible_stage4_13b.sh"
mkdir -p "$OUT/receipts"
case "${1:?choose train, eval, or worker}" in
  worker)
    mode="$2"; arm="$3"; seed="$4"; name="headline_${arm}_seed${seed}"
    receipt="$OUT/receipts/$name"
    if [[ "$mode" == eval ]]; then receipt="$OUT/receipts/matched_$name"; fi
    mkdir "$receipt"
    printf 'queued=%s\n' "$(date -Is)" > "$receipt/queued.txt"
    lease_code="$(bash scripts/gpu_lease.sh acquire 1 --wait)"
    eval "$lease_code"
    printf '%s\n' "host=$(hostname)" "pid=$$" "gpu=$CUDA_VISIBLE_DEVICES" "start=$(date -Is)" > "$receipt/lease.txt"
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
    if [[ "$mode" == train ]]; then
        cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.run_invertible_stage2 --arm "$arm" --seed "$seed" --drivers 128 --lr 0.001 --protocol-amendment 5 --out "$OUT/$name" --checkpoint-dir "$WEIGHTS/$name")
    else
        cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.eval_invertible_matched --root "$OUT" --run "$name")
    fi
    { printf '%q' "${cmd[0]}"; printf ' %q' "${cmd[@]:1}"; printf '\n'; } > "$receipt/command.txt"
    set +e
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${cmd[@]}" > "$receipt/console.txt" 2>&1
    rc=$?
    set -e
    printf '%s\n' "exit_code=$rc" "end=$(date -Is)" > "$receipt/exit.txt"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$receipt/gpu_after.txt"
    exit "$rc"
    ;;
  train|eval)
    mode="$1"
    test -f "$OUT/amendment5.json"
    if [[ "$mode" == train ]]; then
        test -f "$OUT/receipts/launch_commit.json"
        cmp experiments/expressivity_tasks/run_invertible_stage2.py "$OUT/receipts/runner_at_launch.py"
    fi
    pids=()
    for arm in e97 gdn2; do
        for seed in 0 1 2; do
            bash "$SELF" worker "$mode" "$arm" "$seed" & pids+=("$!")
        done
    done
    failed=0
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    printf 'mode=%s\nexit_code=%s\nend=%s\n' "$mode" "$failed" "$(date -Is)" > "$OUT/receipts/${mode}_wave_exit.txt"
    exit "$failed"
    ;;
  *) exit 2;;
esac
