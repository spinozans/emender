#!/usr/bin/env bash
# X3: exclusive local leases; up to three small training processes per card.
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${OUT:-$PWD/experiments/expressivity_tasks/results_invertible_stage3_8m}"
PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export OUT PYTHON_BIN OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
case "${1:?choose smoke, full, or card}" in
  smoke)
    mkdir -p "$OUT/receipts"
    eval "$(bash scripts/gpu_lease.sh acquire 1 --wait)"
    for arm in e97 gdn2; do
      receipt="$OUT/receipts/smoke_$arm"
      mkdir "$receipt"
      printf 'gpu=%s\nhost=%s\n' "$CUDA_VISIBLE_DEVICES" "$(hostname)" > "$receipt/lease.txt"
      nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
      cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.run_invertible_stage3_8m --arm "$arm" --p 61 --drivers 128 --seed 0 --lr 0.001 --hard-cap 250 --out "$OUT/smoke_$arm")
      printf '%q ' "${cmd[@]}" > "$receipt/command.txt"; echo >> "$receipt/command.txt"
      set +e
      "${cmd[@]}" > "$receipt/console.txt" 2>&1
      rc=$?
      set -e
      printf 'exit_code=%s\nend=%s\n' "$rc" "$(date -Is)" > "$receipt/exit.txt"
      [[ "$rc" == 0 ]] || exit "$rc"
    done
    ;;
  full)
    test -f "$OUT/amendment4.json"
    test -f "$OUT/receipts/prelaunch.json"
    test -f "$OUT/smoke_e97/result.json"; test -f "$OUT/smoke_gdn2/result.json"
    mkdir "$OUT/receipts/full_launch"
    cp experiments/expressivity_tasks/run_invertible_stage3_8m.py "$OUT/receipts/runner_at_launch.py"
    cp "$0" "$OUT/receipts/launcher_at_launch.sh"
    "$PYTHON_BIN" -c 'import json,os; from pathlib import Path; from experiments.expressivity_tasks.run_invertible_stage3_8m import matrix; p=Path(os.environ["OUT"]); (p/"receipts/matrix.json").write_text(json.dumps(matrix(),indent=2)+"\n"); [(p/"receipts"/r["name"]).mkdir() for r in matrix()]'
    pids=()
    for lane in 0 1 2 3 4 5 6 7; do
      bash "$0" card "$lane" > "$OUT/receipts/full_launch/card_$lane.txt" 2>&1 & pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do if ! wait "$pid"; then failed=1; fi; done
    printf 'exit_code=%s\nend=%s\n' "$failed" "$(date -Is)" > "$OUT/receipts/full_launch/exit.txt"
    exit "$failed"
    ;;
  card)
    lane="$2"
    lease_code="$(bash scripts/gpu_lease.sh acquire 1 --wait)"
    eval "$lease_code"
    printf '%s\n' "host=$(hostname)" "pid=$$" "gpu=$CUDA_VISIBLE_DEVICES" "start=$(date -Is)" > "$OUT/receipts/full_launch/lane_${lane}_lease.txt"
    # Eight cards * three slots. 18 primaries launch before queued diagnostics.
    "$PYTHON_BIN" - "$lane" <<'PY'
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from experiments.expressivity_tasks.run_invertible_stage3_8m import digest, matrix, write_json
out = Path(os.environ['OUT'])
lane = int(sys.argv[1])
rows = matrix()
# Round-robin across cards, ensuring the first18 slots are primary runs.
jobs = rows[lane::8]
def launch(row):
    name = row['name']
    receipt = out / 'receipts' / name
    cmd = [os.environ['PYTHON_BIN'], '-m', 'experiments.expressivity_tasks.run_invertible_stage3_8m']
    for k in ('arm', 'p', 'drivers', 'seed', 'lr'):
        cmd += ['--'+k, str(row[k])]
    cmd += ['--out', str(out / name)]
    (receipt / 'command.txt').write_text(shlex.join(cmd)+'\n')
    write_json(receipt / 'lease.json', {'host': os.uname().nodename, 'gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'lane': lane, 'card_holder_pid': os.getppid(), 'started_at': time.time(), 'runner_sha256': digest('experiments/expressivity_tasks/run_invertible_stage3_8m.py')})
    with (receipt / 'gpu_before.txt').open('w') as f:
        subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'], stdout=f, check=True)
    with (receipt / 'console.txt').open('w') as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    (receipt / 'exit.txt').write_text(f'exit_code={result.returncode}\nend_epoch={time.time()}\n')
    with (receipt / 'gpu_after.txt').open('w') as f:
        subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu', '--format=csv'], stdout=f, check=True)
    return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    codes = list(pool.map(launch, jobs))
sys.exit(int(any(codes)))
PY
    ;;
  *) exit 2;;
esac
