#!/usr/bin/env bash
# One GPU phase = one process: clean environment, checked lease, patient CUDA
# probe, bounded runtime, unconditional receipt. Sweeps compose phases by
# invoking this script once per phase; no state leaks between phases because
# each phase is a fresh process with an explicitly constructed environment.
#
# Usage:
#   run_e97_gpu_phase.sh --name NAME --gpus N [--cwd DIR]
#     [--env KEY=VALUE ...] [--skip-file PATH] [--timeout SECONDS]
#     [--probe-attempts N] [--probe-interval SECONDS] [--probe none]
#     --receipt-dir DIR -- COMMAND...
#
# Exit codes: command's exit status; 64 usage; 66 skip-file receipt written;
# 69 lease unavailable; 70 CUDA probe exhausted. The receipt is written on
# every path, atomically, before exit.
set -uo pipefail
umask 077

NAME=""; GPUS=""; CWD="$PWD"; SKIP_FILE=""; TIMEOUT=7200
PROBE_ATTEMPTS=40; PROBE_INTERVAL=30; PROBE_MODE="cuda"; RECEIPT_DIR=""
ENVS=(); COMMAND=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME=$2; shift 2;;
    --gpus) GPUS=$2; shift 2;;
    --cwd) CWD=$2; shift 2;;
    --env) ENVS+=("$2"); shift 2;;
    --skip-file) SKIP_FILE=$2; shift 2;;
    --timeout) TIMEOUT=$2; shift 2;;
    --probe-attempts) PROBE_ATTEMPTS=$2; shift 2;;
    --probe-interval) PROBE_INTERVAL=$2; shift 2;;
    --probe) PROBE_MODE=$2; shift 2;;
    --receipt-dir) RECEIPT_DIR=$2; shift 2;;
    --) shift; COMMAND=("$@"); break;;
    *) echo "unknown argument: $1" >&2; exit 64;;
  esac
done
[[ -n "$NAME" && -n "$GPUS" && -n "$RECEIPT_DIR" && ${#COMMAND[@]} -gt 0 ]] || { echo "name, gpus, receipt-dir and command required" >&2; exit 64; }
[[ "$GPUS" =~ ^[0-9]+$ ]] || exit 64
[[ -d "$CWD" ]] || { echo "cwd does not exist: $CWD" >&2; exit 64; }
mkdir -p "$RECEIPT_DIR"; chmod 700 "$RECEIPT_DIR"

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LEASE="$SCRIPT_DIR/gpu_lease.sh"
STARTED=$(date -u +%Y%m%dT%H%M%SZ)
write_receipt() {
  # $1=status $2=detail ; atomic replace
  local tmp="$RECEIPT_DIR/.${NAME}.receipt.tmp"
  printf '{"schema":"emender-e97-gpu-phase-receipt-v1","phase":"%s","gpus":%s,"started_utc":"%s","ended_utc":"%s","status":"%s","detail":"%s"}\n' \
    "$NAME" "$GPUS" "$STARTED" "$(date -u +%Y%m%dT%H%M%SZ)" "$1" "$2" > "$tmp"
  mv "$tmp" "$RECEIPT_DIR/$NAME.receipt.json"
}

if [[ -n "$SKIP_FILE" && -e "$SKIP_FILE" ]]; then
  write_receipt "skipped" "skip-file present: $SKIP_FILE"
  echo "PHASE_SKIP $NAME"; exit 66
fi

LEASE_RC=0
CVD_VALUE=""
if (( GPUS > 0 )); then
  LEASE_OUT=$(bash "$LEASE" acquire "$GPUS" --no-wait) || LEASE_RC=$?
  if (( LEASE_RC != 0 )); then write_receipt "lease-failed" "no lease available"; echo "PHASE_LEASE_FAIL $NAME" >&2; exit 69; fi
  printf "%s\n" "$LEASE_OUT" | grep -v "^gpu_lease:" > /tmp/.e97-lease-eval.$$
  # shellcheck disable=SC1090
  source /tmp/.e97-lease-eval.$$ ; rm -f /tmp/.e97-lease-eval.$$
  CVD_VALUE="$CUDA_VISIBLE_DEVICES"
  [[ "$CVD_VALUE" =~ ^[0-9]+(,[0-9]+)*$ ]] || { write_receipt "lease-malformed" "CUDA_VISIBLE_DEVICES invalid: $CVD_VALUE"; echo "PHASE_LEASE_MALFORMED $NAME" >&2; exit 69; }
fi

probe_failed=0
if [[ "$GPUS" -gt 0 && "$PROBE_MODE" == "cuda" ]]; then
  EXPECTED=$(python3 -c "import os;print(len([x for x in os.environ.get('CUDA_VISIBLE_DEVICES','').split(',') if x.strip()]))")
  attempt=0; probe_ok=0
  while (( attempt < PROBE_ATTEMPTS )); do
    if python3 - <<EOF
import sys,torch
if not torch.cuda.is_available(): sys.exit(1)
if torch.cuda.device_count()!=$EXPECTED: sys.exit(1)
x=torch.zeros(1,device='cuda');torch.cuda.synchronize();del x
EOF
    then probe_ok=1; break; fi
    attempt=$((attempt+1)); echo "PHASE_CUDA_WAIT $NAME attempt=$attempt" >&2; sleep "$PROBE_INTERVAL"
  done
  if (( probe_ok == 0 )); then probe_failed=1; fi
fi

RC=0
if (( probe_failed == 0 )); then
  ENV_ARGS=(env -i PATH="$PATH" HOME="${HOME:-}" USER="${USER:-}" LOGNAME="${LOGNAME:-}" \
    LANG="${LANG:-en_US.UTF-8}" TMPDIR="${TMPDIR:-/tmp}" \
    GPU_LEASE_DIR="${GPU_LEASE_DIR:-}")
  [[ -n "$CVD_VALUE" ]] && ENV_ARGS+=("CUDA_VISIBLE_DEVICES=$CVD_VALUE")
  for kv in "${ENVS[@]}"; do ENV_ARGS+=("$kv"); done
  timeout --kill-after=30 "$TIMEOUT" "${ENV_ARGS[@]}" bash -c 'cd "$1"; shift; exec "$@"' _ "$CWD" "${COMMAND[@]}" || RC=$?
else
  RC=70
fi

if (( GPUS > 0 )); then
  for pid in ${GPU_LEASE_HB_PIDS:-}; do kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; done
  GPU_LEASE_DIR="${GPU_LEASE_DIR:-}" GPU_LEASE_PID=$$ "$LEASE" release "$GPU_LEASE_HELD" >/dev/null 2>&1 || true
fi

case "$RC" in
  0) write_receipt "complete" "phase command succeeded";;
  70) write_receipt "probe-failed" "CUDA never settled";;
  124|137) write_receipt "timeout" "phase exceeded --timeout";;
  *) write_receipt "failed" "phase command exit $RC";;
esac
echo "PHASE_DONE $NAME rc=$RC"
exit "$RC"
