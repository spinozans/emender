#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="/mnt/nvme1n1/erikg/ref_emender_mlp"
RUN_LOG="$OUT_ROOT/run.log"
PID_FILE="$OUT_ROOT/run.pid"

mkdir -p "$OUT_ROOT"

setsid env ROOT="$ROOT" bash -c '
set -euo pipefail
cd "$ROOT"
eval "$(scripts/gpu_lease.sh 1)"
echo "[gpu] leased CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[launch] holder_pid=$$ started_at=$(date -Is)"
exec python scripts/ref_emender_mlp_run.py "$@"
' bash "$@" >"$RUN_LOG" 2>&1 &

pid=$!
printf "%s\n" "$pid" >"$PID_FILE"
echo "PID=$pid log=$RUN_LOG pidfile=$PID_FILE"
