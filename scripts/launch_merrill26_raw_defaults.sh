#!/usr/bin/env bash
# X26c runs the upstream bash pipeline, not the reconciled training loop.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
if [[ "${1:-}" != --worker ]]; then
  exec "$PYTHON_BIN" -m scripts.watch_merrill26_raw_defaults --launch
fi
arm="$2"; task="$3"; card="$4"; name="${arm}_${task}_seed0"
case "$arm:$task:$card" in rnn:integer:4|delta:integer:5) ;; *) exit 64 ;; esac
"$PYTHON_BIN" -c 'from experiments.expressivity_tasks import run_merrill26_raw_defaults as r; r.configure(); r.base.verify_registration()'
repo="$PWD"
root="$repo/experiments/expressivity_tasks/results_stage26c_raw_defaults"
out="$root/$name"; receipt="$root/receipts/$name"
venv="/tmp/x26c-$arm-venv"
script=rnn; [[ "$arm" != delta ]] || script=deltanet
source_dir="$repo/experiments/expressivity_tasks/results_stage26_replication/sources/LinearRNN/imm_unbound"
train_pid=""
finish() {
  rc=$?; trap - EXIT INT TERM; set +e
  if [[ -n "$train_pid" ]]; then
    pkill -TERM -P "$train_pid" 2>/dev/null
    kill "$train_pid" 2>/dev/null; wait "$train_pid" 2>/dev/null
  fi
  if declare -F gpu_lease_release >/dev/null; then gpu_lease_release; fi
  if [[ -n "${GPU_LEASE_HB_PID:-}" ]]; then kill "$GPU_LEASE_HB_PID" 2>/dev/null; fi
  "$PYTHON_BIN" -m scripts.watch_merrill26_raw_defaults --terminal "$name" --exit-code "$rc"
}
trap finish EXIT
trap 'exit 143' INT TERM
export GPU_LEASE_VISIBLE="$card"
lease_code="$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"; eval "$lease_code"
trap finish EXIT
trap 'exit 143' INT TERM
cp "${GPU_LEASE_DIR:-$repo/.wg/gpu_leases}/$CUDA_VISIBLE_DEVICES" "$receipt/lease.txt"
mkdir "$out"
# env -i prevents unrelated experiments' hyperparameter overrides from leaking in.
cmd=(env -i HOME="$HOME" USER="$USER" PATH="/usr/bin:/bin" 
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
  VENV_DIR="$venv" TMP_BASE="/tmp/x26c-$arm-pipeline" WORKDIR="$out"
  TRAIN_PY="$source_dir/train_$script.py"
  DATA_DIR="$repo/experiments/expressivity_tasks/results_stage26b_native_control/datasets"
  SAVE_PATH="$out/checkpoint.pt" bash "$source_dir/$arm.sh")
printf '%q ' "${cmd[@]}" > "$receipt/command.txt"; printf '\n' >> "$receipt/command.txt"
(cd "$out"; exec "${cmd[@]}") > "$receipt/console.txt" 2>&1 &
train_pid=$!
# Dependencies are installed by their launcher before Command: is printed.
while kill -0 "$train_pid" 2>/dev/null; do
  if grep -q '^Command:' "$receipt/console.txt"; then
    "$venv/bin/python" -m experiments.expressivity_tasks.run_merrill26_raw_defaults \
      --environment "$receipt/environment_runtime.json"
    "$PYTHON_BIN" - "$train_pid" "$receipt/trainer.json" "$root/receipts/environment_pre_$arm.json" "$receipt/environment_runtime.json" <<'PY'
import json, pathlib, sys, time
from scripts.watch_merrill26_separators import identity
before, after = [json.loads(pathlib.Path(p).read_text()) for p in sys.argv[3:5]]
for key in ('python', 'torch', 'cuda', 'numpy', 'flash-linear-attention', 'triton'):
    if before[key] != after[key]:
        raise RuntimeError(f'environment changed since smoke: {key}')
parent = int(sys.argv[1])
children = pathlib.Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
trainers = [int(p) for p in children if b'train_' in pathlib.Path(f'/proc/{p}/cmdline').read_bytes()]
if len(trainers) != 1:
    raise RuntimeError(f'expected exactly one shipped trainer: {trainers}')
pathlib.Path(sys.argv[2]).write_text(json.dumps({**identity(trainers[0]), 'observed_at': time.time()}))
PY
    break
  fi
  sleep 1
done
set +e
wait "$train_pid"; rc=$?; train_pid=""
set -e
[[ "$rc" == 0 ]] || exit "$rc"
"$venv/bin/python" -m experiments.expressivity_tasks.run_merrill26_raw_defaults --arm "$arm" \
  > "$receipt/metrics_console.txt" 2>&1
