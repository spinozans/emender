#!/usr/bin/env bash
# Local amendment #3 only. One exclusive lease/process; X2b is never cancelled.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-$PWD/experiments/expressivity_tasks/results_invertible_stage2c}"
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
export OUT PYTHON_BIN
SELF="$PWD/experiments/expressivity_tasks/launch_invertible_stage2c.sh"
mkdir -p "$OUT/receipts"
wait_wave() {
    local failed=0 pid
    for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
    return "$failed"
}
case "${1:?choose pilot, full, or worker}" in
  worker)
    name="$2"; arm="$3"; seed="$4"; drivers="$5"; lr="$6"; mode="$7"
    receipt="$OUT/receipts/$name"
    mkdir "$receipt"
    printf 'queued=%s\n' "$(date -Is)" > "$receipt/queued.txt"
    lease_code="$(bash scripts/gpu_lease.sh acquire 1 --wait)"
    eval "$lease_code"
    printf '%s\n' "host=$(hostname)" "pid=$$" "gpu=$CUDA_VISIBLE_DEVICES" "start=$(date -Is)" > "$receipt/lease.txt"
    nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
    cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.run_invertible_stage2 --arm "$arm" --seed "$seed" --drivers "$drivers" --out "$OUT/$name" --lr "$lr" --protocol-amendment 3)
    if [[ "$mode" == pilot ]]; then cmd+=(--pilot); fi
    { printf '%q' "${cmd[0]}"; printf ' %q' "${cmd[@]:1}"; printf '\n'; } > "$receipt/command.txt"
    set +e
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "${cmd[@]}" > "$receipt/console.txt" 2>&1
    rc=$?
    set -e
    printf '%s\n' "exit_code=$rc" "end=$(date -Is)" > "$receipt/exit.txt"
    nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$receipt/gpu_after.txt"
    exit "$rc"
    ;;
  pilot)
    test -f "$OUT/amendment3.json"
    tag="${2:?choose 2e-4 or 1e-3}"
    case "$tag" in
      2e-4) test ! -f "$OUT/pilot_gate_2e-4.json"; lr=0.0002;;
      1e-3)
        "$PYTHON_BIN" -c 'import json,sys; from pathlib import Path; r=Path(sys.argv[1]); assert not json.loads((r/"pilot_gate_2e-4.json").read_text())["e97_passed"]; assert not (r/"lr_decision.json").exists()' "$OUT"
        lr=0.001;;
      *) exit 2;;
    esac
    pids=()
    for arm in e97 gdn2; do
        bash "$SELF" worker "pilot_${tag}_${arm}_seed0" "$arm" 0 128 "$lr" pilot & pids+=("$!")
    done
    wait_wave "${pids[@]}"
    "$PYTHON_BIN" -m experiments.expressivity_tasks.invertible_stage2c gate --root "$OUT" --tag "$tag"
    ;;
  full)
    lr="$("$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="passed"; print(d["lr"])' "$OUT/lr_decision.json")"
    pids=()
    # Seven primary comparisons start first; diagnostics queue on the same broker.
    for arm in e97 gdn2; do
        for seed in 0 1 2; do
            bash "$SELF" worker "headline_${arm}_seed${seed}" "$arm" "$seed" 128 "$lr" full & pids+=("$!")
        done
    done
    bash "$SELF" worker headline_e97_identity_seed0 e97_identity 0 128 "$lr" full & pids+=("$!")
    for arm in e97 gdn2 e97_identity; do
        bash "$SELF" worker "calibration_${arm}_seed0" "$arm" 0 1 "$lr" full & pids+=("$!")
    done
    wait_wave "${pids[@]}"
    ;;
  *) exit 2;;
esac
