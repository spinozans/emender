#!/usr/bin/env bash
# Standalone attended research, no WorksGood service/dispatcher/worker machinery.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ "${1:-}" != --worker ]]; then
  exec "$PYTHON_BIN" -m scripts.watch_recipe_controlled_separators --launch
fi
arm="$2"; task="$3"; card="$4"; name="${arm}_${task}_seed0"
case "$arm:$task:$card" in
  e97:graph:0|e97softplus:graph:7|gdn2:graph:1|e97:integer:2|gdn2:integer:3) ;;
  *) exit 64 ;;
esac
root="$PWD/experiments/expressivity_tasks/results_stage25_recipe_controlled"
receipt="$root/receipts/$name"
train_pid=""
finish() {
  rc=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "$train_pid" ]]; then kill "$train_pid" 2>/dev/null; wait "$train_pid" 2>/dev/null; fi
  if declare -F gpu_lease_release >/dev/null; then gpu_lease_release; fi
  if [[ -n "${GPU_LEASE_HB_PID:-}" ]]; then kill "$GPU_LEASE_HB_PID" 2>/dev/null; fi
  "$PYTHON_BIN" -m scripts.watch_recipe_controlled_separators --terminal "$name" --exit-code "$rc"
}
trap finish EXIT
trap 'exit 143' INT TERM
export GPU_LEASE_VISIBLE="$card"
lease_code="$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"
eval "$lease_code"
trap finish EXIT
trap 'exit 143' INT TERM
cp "${GPU_LEASE_DIR:-$PWD/.wg/gpu_leases}/$CUDA_VISIBLE_DEVICES" "$receipt/lease.txt"
cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.run_recipe_controlled_separators --arm "$arm" --task "$task")
{ printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"; printf '%q ' "${cmd[@]}"; printf '\n'; } > "$receipt/command.txt"
"${cmd[@]}" > "$receipt/console.txt" 2>&1 &
train_pid=$!
"$PYTHON_BIN" - "$train_pid" "$receipt/trainer.json" <<'PY'
import json, pathlib, socket, sys
pid = int(sys.argv[1])
stat = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
pathlib.Path(sys.argv[2]).write_text(json.dumps({'pid': pid, 'starttime': stat[19], 'host': socket.gethostname()}))
PY
set +e
wait "$train_pid"
rc=$?
train_pid=""
exit "$rc"
