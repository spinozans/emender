#!/usr/bin/env bash
# X9b: one detached session, exclusive card-2 lease and terminal receipts.
set -euo pipefail
cd "$(dirname "$0")/../.."
export OUT="${OUT:-$PWD/experiments/expressivity_tasks/results_invertible_stage9b_sf_anneal_8m}"
export PYTHON_BIN="${PYTHON_BIN:-$PWD/.venv/bin/python}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ "${1:-}" == --worker ]]; then
  arm="$2"; precision="$3"; seed=0; name="${arm}_${precision}_seed${seed}"
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
  [[ "$arm" == e97 && "$precision" == bf16 ]]
  export GPU_LEASE_VISIBLE=2
  lease_code="$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"
  eval "$lease_code"
  # The broker's EXIT trap is extended with our terminal receipt and child cleanup.
  trap finish EXIT
  trap 'exit 143' INT TERM
  cp "${GPU_LEASE_DIR:-$PWD/.wg/gpu_leases}/$CUDA_VISIBLE_DEVICES" "$receipt/lease.txt"
  cmd=("$PYTHON_BIN" -m experiments.expressivity_tasks.run_invertible_stage9b_sf_anneal_8m --precision "$precision" --arm "$arm" --seed "$seed" --out "$OUT/$name")
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
"$PYTHON_BIN" - "$OUT" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
reg = json.loads((root / 'registration.json').read_text())
import subprocess
registration_path = str((root / 'registration.json').relative_to(pathlib.Path.cwd()))
assert subprocess.check_output(['git', 'show', 'HEAD:' + registration_path]) == (root / 'registration.json').read_bytes()
assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == h for p, h in reg['sources'].items())
# Operational sources and preflight evidence must already be registered in HEAD.
for p, h in reg['sources'].items():
    assert hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + p])).hexdigest() == h
for p, h in reg['preflight_receipts'].items():
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h
    relative = str((root / p).relative_to(pathlib.Path.cwd()))
    assert hashlib.sha256(subprocess.check_output(['git', 'show', 'HEAD:' + relative])).hexdigest() == h
assert reg['runs'] == ['e97_bf16_seed0']
assert reg['optimizer'] == 'AdamWScheduleFree' and reg['warmup'] == 5000
assert reg['betas'] == [.9, .95] and reg['internal_warmup'] == 0
for p, h in reg['installed_sources'].items():
    assert hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() == h
assert (root / 'receipts/tests_exit_code.txt').read_text().strip() == '0'
PY
export GPU_LEASE_VISIBLE=2
mkdir "$OUT/receipts/full_launch" "$OUT/markers"
receipt="$OUT/receipts/full_launch"
for path in experiments/expressivity_tasks/run_invertible_stage{9b_sf_anneal_8m,7_deepfirst_8m,6_truecur_8m,5_curriculum_8m,3_8m,2}.py experiments/expressivity_tasks/eval_invertible_stage6_positions.py experiments/expressivity_tasks/tasks/invertible_quadratic.py ndm/models/{hybrid_ladder,e88_fla_hybrid,fla_gated_delta}.py ndm/recurrent_precision.py ndm/triton/e88_triton_{forward,backward,optimized}.py scripts/gpu_lease.sh scripts/watch_invertible_stage9b_sf_anneal.py "$0"; do
  cp "$path" "$receipt/$(basename "$path")"
done
"$PYTHON_BIN" - "$receipt" <<'PY'
import importlib.metadata, inspect, pathlib, schedulefree, shutil, sys
out = pathlib.Path(sys.argv[1])
shutil.copy2(inspect.getfile(schedulefree.AdamWScheduleFree), out / 'installed_adamw_schedulefree.py')
dist = importlib.metadata.distribution('schedulefree')
license_path = next(p for p in dist.files if p.name == 'LICENSE')
shutil.copy2(dist.locate_file(license_path), out / 'ScheduleFree_LICENSE')
PY
git rev-parse HEAD > "$receipt/base_commit.txt"
git diff --binary > "$receipt/tracked_diff_at_launch.patch"
"$PYTHON_BIN" -c 'import importlib.metadata as m; print("\n".join(sorted(d.metadata["Name"] + "==" + d.version for d in m.distributions())))' > "$receipt/packages.txt"
df -B1 . > "$receipt/disk_before.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv > "$receipt/gpu_before.txt"
for spec in 'e97 bf16'; do
  read -r arm precision <<< "$spec"
  name="${arm}_${precision}_seed0"
  mkdir "$OUT/receipts/$name"
  nohup setsid bash "$0" --worker "$arm" "$precision" > "$OUT/receipts/$name/worker_console.txt" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "$OUT/receipts/$name/detached_pid.txt"
done
printf 'detached_launch=%s\nhost=%s\n' "$(date -Is)" "$(hostname)" > "$receipt/launched.txt"
echo "One detached X9b worker launched on fixed GPU E97=2; verify baseline and identities."
