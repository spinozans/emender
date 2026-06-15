#!/usr/bin/env bash
# Launch a long-running command in a detached session, with any GPU lease owned
# by that detached process rather than by the caller's shell.

set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/launch_detached_run.sh --name NAME --gpus N --logdir DIR [--] CMD...
  scripts/launch_detached_run.sh --self-test

Options:
  --name NAME     Logical run name written to launch_manifest.json.
  --gpus N       Number of GPUs to lease inside the detached session. Use 0 to skip leasing.
  --logdir DIR   Durable output directory for run.log, run.pid, and launch_manifest.json.
  --             End option parsing; remaining arguments are the command to exec.
  --self-test    Run the non-GPU parent-exit survival test and clean up afterwards.
EOF
}

die() {
  echo "launch_detached_run.sh: ERROR: $*" >&2
  exit 1
}

script_dir() {
  local src="${BASH_SOURCE[0]}"
  while [ -h "$src" ]; do
    local dir
    dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
    src="$(readlink "$src")"
    [[ $src != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd
}

self_test() {
  local self repo tmpdir child_out pid first_count second_count manifest_name manifest_gpus manifest_pid
  self="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/$(basename "${BASH_SOURCE[0]}")"
  repo="$(cd -P "$(dirname "$self")/.." >/dev/null 2>&1 && pwd)"
  tmpdir="$(mktemp -d /tmp/launch_detached_run.selftest.XXXXXX)"
  child_out="$tmpdir/launcher.out"

  cleanup_self_test() {
    if [ -f "$tmpdir/run.pid" ]; then
      local test_pid
      test_pid="$(cat "$tmpdir/run.pid" 2>/dev/null || true)"
      if [ -n "$test_pid" ] && kill -0 "$test_pid" 2>/dev/null; then
        kill "$test_pid" 2>/dev/null || true
        sleep 1
        kill -9 "$test_pid" 2>/dev/null || true
      fi
    fi
    rm -rf "$tmpdir"
  }
  trap cleanup_self_test EXIT

  echo "self-test logdir: $tmpdir"
  (
    cd "$repo"
    "$self" --name self-test --gpus 0 --logdir "$tmpdir" -- \
      bash -c 'for i in $(seq 1 90); do echo tick $i; sleep 1; done'
  ) >"$child_out" 2>&1

  pid="$(tr -d '[:space:]' <"$child_out")"
  case "$pid" in ''|*[!0-9]*) die "self-test launcher did not print a PID; output: $(cat "$child_out")";; esac
  echo "launcher exited after printing pid: $pid"

  [ -f "$tmpdir/run.pid" ] || die "self-test missing run.pid"
  [ "$(cat "$tmpdir/run.pid")" = "$pid" ] || die "self-test run.pid does not match printed PID"

  sleep 3
  kill -0 "$pid" 2>/dev/null || die "self-test detached pid $pid is not alive after launching shell exited"
  [ -f "$tmpdir/run.log" ] || die "self-test missing run.log"
  first_count="$(grep -c '^tick ' "$tmpdir/run.log" || true)"
  echo "tick count after parent exit: $first_count"
  [ "$first_count" -gt 0 ] || die "self-test run.log has no ticks"

  sleep 3
  kill -0 "$pid" 2>/dev/null || die "self-test detached pid $pid died before growth check"
  second_count="$(grep -c '^tick ' "$tmpdir/run.log" || true)"
  echo "tick count after growth wait: $second_count"
  [ "$second_count" -gt "$first_count" ] || die "self-test run.log did not grow"

  read -r manifest_name manifest_gpus manifest_pid < <(
    python3 - "$tmpdir/launch_manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as fh:
    data = json.load(fh)

required = {
    "name",
    "cmd",
    "gpus_requested",
    "leased_gpu_ids",
    "logfile",
    "pidfile",
    "start_unix",
    "pid",
}
missing = sorted(required - set(data))
if missing:
    raise SystemExit(f"missing manifest keys: {missing}")
if data["name"] != "self-test":
    raise SystemExit(f"unexpected name: {data['name']!r}")
if data["gpus_requested"] != 0:
    raise SystemExit(f"unexpected gpus_requested: {data['gpus_requested']!r}")
if data["leased_gpu_ids"] != "":
    raise SystemExit(f"--gpus 0 should not set leased_gpu_ids: {data['leased_gpu_ids']!r}")
print(data["name"], data["gpus_requested"], data["pid"])
PY
  )
  [ "$manifest_pid" = "$pid" ] || die "self-test manifest PID $manifest_pid does not match $pid"
  echo "manifest ok: name=$manifest_name gpus_requested=$manifest_gpus pid=$manifest_pid"

  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  trap - EXIT
  cleanup_self_test
  echo "self-test passed"
}

if [ "${1:-}" = "--self-test" ]; then
  shift
  [ "$#" -eq 0 ] || die "--self-test does not accept extra arguments"
  self_test
  exit 0
fi

name=""
gpus=""
logdir=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --name)
      [ "$#" -ge 2 ] || die "--name requires a value"
      name="$2"
      shift 2
      ;;
    --gpus)
      [ "$#" -ge 2 ] || die "--gpus requires a value"
      gpus="$2"
      shift 2
      ;;
    --logdir)
      [ "$#" -ge 2 ] || die "--logdir requires a value"
      logdir="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      break
      ;;
  esac
done

[ -n "$name" ] || die "--name is required"
[ -n "$gpus" ] || die "--gpus is required"
[ -n "$logdir" ] || die "--logdir is required"
case "$gpus" in ''|*[!0-9]*) die "--gpus must be a non-negative integer";; esac
[ "$#" -gt 0 ] || die "command is required"

mkdir -p "$logdir"

logdir="$(cd -P "$logdir" >/dev/null 2>&1 && pwd)"
logfile="$logdir/run.log"
pidfile="$logdir/run.pid"
manifest="$logdir/launch_manifest.json"
lease_script="$(script_dir)/gpu_lease.sh"
start_unix="$(date +%s)"

[ -x "$lease_script" ] || die "GPU lease broker is not executable: $lease_script"

cmd_json="$(
  python3 - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
)"

export LAUNCH_DETACHED_NAME="$name"
export LAUNCH_DETACHED_GPUS="$gpus"
export LAUNCH_DETACHED_LOGFILE="$logfile"
export LAUNCH_DETACHED_PIDFILE="$pidfile"
export LAUNCH_DETACHED_MANIFEST="$manifest"
export LAUNCH_DETACHED_LEASE_SCRIPT="$lease_script"
export LAUNCH_DETACHED_START_UNIX="$start_unix"
export LAUNCH_DETACHED_CMD_JSON="$cmd_json"

setsid bash -c '
set -euo pipefail
trap "" HUP

{
  export LAUNCH_DETACHED_PID="$$"

  if [ "$LAUNCH_DETACHED_GPUS" -gt 0 ]; then
    eval "$("$LAUNCH_DETACHED_LEASE_SCRIPT" "$LAUNCH_DETACHED_GPUS")"
  fi

  python3 - <<'"'"'PY'"'"'
import json
import os

manifest = {
    "name": os.environ["LAUNCH_DETACHED_NAME"],
    "cmd": json.loads(os.environ["LAUNCH_DETACHED_CMD_JSON"]),
    "gpus_requested": int(os.environ["LAUNCH_DETACHED_GPUS"]),
    "leased_gpu_ids": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "logfile": os.environ["LAUNCH_DETACHED_LOGFILE"],
    "pidfile": os.environ["LAUNCH_DETACHED_PIDFILE"],
    "start_unix": int(os.environ["LAUNCH_DETACHED_START_UNIX"]),
    "pid": int(os.environ["LAUNCH_DETACHED_PID"]),
}
tmp = os.environ["LAUNCH_DETACHED_MANIFEST"] + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(manifest, fh, indent=2, sort_keys=True)
    fh.write("\n")
os.replace(tmp, os.environ["LAUNCH_DETACHED_MANIFEST"])
PY

  exec nohup "$@"
} >"$LAUNCH_DETACHED_LOGFILE" 2>&1
' launch_detached_run "$@" &

pid="$!"
printf '%s\n' "$pid" >"$pidfile"
printf '%s\n' "$pid"
