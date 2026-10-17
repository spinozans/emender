#!/usr/bin/env bash
# X5c: six detached sessions, independent exclusive leases and terminal receipts.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OUT="${OUT:-$PWD/experiments/expressivity_tasks/results_invertible_stage5c_growth_8m}"
export PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ "${1:-}" == --worker ]]; then
  arm="$2"; seed="$3"; name="${arm}_seed${seed}"
  receipt="$OUT/receipts/$name"
  train_pid=""
  finish() {
    rc=$?
    trap - EXIT INT TERM
    set +e
    if [[ -n "$train_pid" ]]; then
      kill "$train_pid" 2>/dev/null
      wait "$train_pid" 2>/dev/null
    fi
    if declare -F gpu_lease_release >/dev/null; then gpu_lease_release; fi
    if [[ -n "${GPU_LEASE_HB_PID:-}" ]]; then kill "$GPU_LEASE_HB_PID" 2>/dev/null; fi
    "$PYTHON_BIN" - "$OUT" "$name" "$rc" <<'PY'
import hashlib, json, pathlib, sys, time
root, name, rc = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
result = root / name / 'result.json'
if rc == 0 and not result.exists():
    rc = 1
row = {'name': name, 'exit_code': rc, 'finished_at': time.time(), 'result': str(result)}
if result.exists():
    row['result_sha256'] = hashlib.sha256(result.read_bytes()).hexdigest()
    row['stop_reason'] = json.loads(result.read_text())['stop_reason']
receipt = root / 'receipts' / name / 'terminal.json'
receipt.with_suffix('.tmp').write_text(json.dumps(row, indent=2) + '\n')
receipt.with_suffix('.tmp').replace(receipt)
marker = root / 'markers' / (name + ('.DONE' if rc == 0 else '.FAIL'))
marker.with_suffix('.tmp').write_text(json.dumps(row) + '\n')
marker.with_suffix('.tmp').replace(marker)
PY
  }
  trap finish EXIT
  trap 'exit 143' INT TERM
  starttime="$(awk '{s=$0; sub(/.*\) /,"",s); split(s,a," "); print a[20]}' /proc/$$/stat)"
  printf '{"pid":%s,"starttime":"%s","host":"%s"}\n' "$$" "$starttime" "$(hostname)" > "$receipt/worker.json"
  lease_code="$(bash scripts/gpu_lease.sh acquire 1 --wait --timeout 7200)"
  eval "$lease_code"
  # The broker's EXIT trap is extended with our terminal receipt and child cleanup.
  trap finish EXIT
  trap 'exit 143' INT TERM
  cp "${GPU_LEASE_DIR:-$PWD/.wg/gpu_leases}/$CUDA_VISIBLE_DEVICES" "$receipt/lease.txt"
  cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.run_invertible_stage5b_growth_8m --arm "$arm" --seed "$seed" --out "$OUT/$name")
  { printf 'CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"; printf '%q ' "${cmd[@]}"; printf '\n'; } | sed 's/ *$//' > "$receipt/command.txt"
  "${cmd[@]}" > "$receipt/console.txt" 2>&1 &
  train_pid=$!
  starttime="$(awk '{s=$0; sub(/.*\) /,"",s); split(s,a," "); print a[20]}' /proc/$train_pid/stat)"
  printf '{"pid":%s,"starttime":"%s","host":"%s"}\n' "$train_pid" "$starttime" "$(hostname)" > "$receipt/trainer.json"
  set +e
  wait "$train_pid"
  rc=$?
  train_pid=""
  exit "$rc"
fi

test -f "$OUT/registration.json"
test -f "$OUT/receipts/tests_prelaunch.txt"
mkdir "$OUT/receipts/full_launch" "$OUT/markers"
receipt="$OUT/receipts/full_launch"
for path in experiments/expressivity_tasks/run_invertible_stage{5b_growth_8m,5_curriculum_8m,3_8m,2}.py experiments/expressivity_tasks/tasks/invertible_quadratic.py ndm/models/hybrid_ladder.py scripts/gpu_lease.sh scripts/watch_invertible_stage5b_growth.py "$0"; do
  cp "$path" "$receipt/$(basename "$path")"
done
git rev-parse HEAD > "$receipt/base_commit.txt"
git diff --binary > "$receipt/tracked_diff_at_launch.patch"
"$PYTHON_BIN" -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in m.distributions())))' > "$receipt/packages.txt"
df -B1 . > "$receipt/disk_before.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
for seed in 0 1 2; do
  for arm in e97 gdn2; do
    name="${arm}_seed${seed}"
    mkdir "$OUT/receipts/$name"
    nohup setsid bash "$0" --worker "$arm" "$seed" > "$OUT/receipts/$name/worker_console.txt" 2>&1 < /dev/null &
    printf '%s\n' "$!" > "$OUT/receipts/$name/detached_pid.txt"
  done
done
printf 'detached_launch=%s\nhost=%s\n' "$(date -Is)" "$(hostname)" > "$receipt/launched.txt"
echo "Six detached runs launched; terminal markers: $OUT/markers/{e97,gdn2}_seed{0,1,2}.{DONE,FAIL}"
