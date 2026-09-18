#!/usr/bin/env bash
# serve_e97_interactive.sh — single-GPU interactive OpenAI-compatible server for
# the promoted E97 4B Pi-native agent checkpoint, for operator use in Pi.
#
# Usage:
#   scripts/serve_e97_interactive.sh [start|stop|status]
#
# start (default): verify the checkpoint SHA-256 pin, materialize a clean
#   read-only snapshot of the pinned serving commit (the tree that produced
#   the passing v6-u96 dual gate — the live checkout's stricter uncommitted
#   attestation layer deliberately rejects the dirty tree), acquire 1 GPU via
#   the lease broker, launch scripts/serve_e97_agent_openai.py with the
#   canonical Pi-core system prompt, wait for /health, print the Pi launch
#   command, then stay in the foreground as supervisor (run inside
#   tmux/screen). On exit or `stop`, the server stops and the lease releases.
#
# Environment overrides:
#   E97_SERVE_PORT              (default 8797 — the canonical emender-local port)
#   E97_SERVE_SOURCE_COMMIT     (default: HEAD of /home/erikg/emender)
#   E97_SERVE_CHECKPOINT        (default: promoted v6-u96 checkpoint)
#   E97_SERVE_CHECKPOINT_SHA256 (default: d8146498… pin)
#   E97_SERVE_ARGS_JSON         (default: 100B base args.json)
#   E97_SERVE_LOG_DIR           (default /tmp/e97-interactive-serve)
#   E97_SERVE_MAX_OUTPUT_TOKENS (default 512)
#   E97_SERVE_MAX_SESSIONS      (default 4)
#
# The lease broker only grants physically idle GPUs, so while the wide-arc
# training holds all eight GPUs this launcher waits safely instead of colliding.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$ROOT"
PY=${EMENDER_PYTHON:-$ROOT/.venv/bin/python}
SOURCE_COMMIT=${E97_SERVE_SOURCE_COMMIT:-$(git rev-parse HEAD)}
CHECKPOINT=${E97_SERVE_CHECKPOINT:-/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair6-arc-segment3-training-v1/checkpoints/checkpoint_agent_sft_u000032_loss_0.5059.pt}
CHECKPOINT_SHA256=${E97_SERVE_CHECKPOINT_SHA256:-d81464982c3ebc0d72769a87e079068bf535d6ca2010185dc1bb03ce264b8f5b}
ARGS_JSON=${E97_SERVE_ARGS_JSON:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json}
PORT=${E97_SERVE_PORT:-8797}
LOG_DIR=${E97_SERVE_LOG_DIR:-/tmp/e97-interactive-serve}
PID_FILE=$LOG_DIR/supervisor.pid

cmd=${1:-start}
case "$cmd" in
  stop)
    if [[ -r "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      pid=$(cat "$PID_FILE"); kill -TERM "$pid"
      echo "stopping supervisor $pid (server stops and GPU lease releases on its exit)"
    else
      echo "not running (no live supervisor at $PID_FILE)" >&2; exit 1
    fi
    exit 0 ;;
  status)
    if [[ -r "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "serving e97-dense-agent on 127.0.0.1:${PORT}"
      else
        echo "supervisor alive but server not ready yet; see $LOG_DIR/server.log"
      fi
    else
      echo "not running"
    fi
    exit 0 ;;
  start) ;;
  *) echo "usage: $0 [start|stop|status]" >&2; exit 64 ;;
esac

[[ -r "$CHECKPOINT" && -r "$ARGS_JSON" ]] || { echo "checkpoint or args.json unreadable" >&2; exit 66; }
actual=$(sha256sum "$CHECKPOINT" | awk '{print $1}')
if [[ "$actual" != "$CHECKPOINT_SHA256" ]]; then
  echo "checkpoint SHA-256 mismatch: $actual != $CHECKPOINT_SHA256" >&2; exit 65
fi
git cat-file -e "${SOURCE_COMMIT}^{commit}" || { echo "unknown source commit $SOURCE_COMMIT" >&2; exit 66; }
mkdir -m 700 -p "$LOG_DIR"
SNAPSHOT=$LOG_DIR/snapshot-$SOURCE_COMMIT
if [[ ! -x "$SNAPSHOT/scripts/serve_e97_agent_openai.py" ]]; then
  rm -rf "$SNAPSHOT"; mkdir -m 700 -p "$SNAPSHOT"
  git archive "$SOURCE_COMMIT" | tar -xf - -C "$SNAPSHOT"
  [[ -f "$SNAPSHOT/scripts/serve_e97_agent_openai.py" ]] || { echo "serve script missing in snapshot" >&2; exit 66; }
  chmod -R u-w "$SNAPSHOT"
fi
printf '%s  %s\n' "$CHECKPOINT_SHA256  $CHECKPOINT" > "$LOG_DIR/checkpoint.sha256"
printf '%s\n' "$SOURCE_COMMIT" > "$LOG_DIR/source-commit.txt"
if [[ -r "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "already running (supervisor pid $(cat "$PID_FILE"))" >&2; exit 1
fi

eval "$(scripts/gpu_lease.sh acquire 1 --wait)"
IFS=, read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
export CUDA_VISIBLE_DEVICES=${GPUS[0]} LOCAL_RANK=0
export PYTHONPATH="$SNAPSHOT"
export PI_OFFLINE=1 TIKTOKEN_CACHE_DIR=${TIKTOKEN_CACHE_DIR:-/tmp/data-gym-cache} OMP_NUM_THREADS=4
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX=/tmp/e97-interactive-serve-triton-$$

cleanup() {
  trap - EXIT INT TERM
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  for pid in ${GPU_LEASE_HB_PIDS:-}; do
    kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true
  done
  gpu_lease_release 2>/dev/null || true
  unset GPU_LEASE_HELD 2>/dev/null || true
  rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

"$SNAPSHOT/scripts/numa_local_rank_exec.py" -- "$PY" "$SNAPSHOT/scripts/serve_e97_agent_openai.py" \
  --checkpoint "$CHECKPOINT" --args-json "$ARGS_JSON" \
  --host 127.0.0.1 --port "$PORT" \
  --model-id e97-dense-agent --weight-mode saved --ingest-mode tokenwise \
  --pi-core-canonical-system --trace-generated-errors \
  --max-output-tokens "$MAX_OUTPUT_TOKENS" --max-sessions "$MAX_SESSIONS" \
  > "$LOG_DIR/server.out" 2> "$LOG_DIR/server.log" &
SERVER_PID=$!
printf '%s\n' "$$" > "$PID_FILE"

ready=0
for _ in $(seq 1 240); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then ready=1; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited during startup; see $LOG_DIR/server.log" >&2; exit 70
  fi
  sleep 2
done
if [[ $ready != 1 ]]; then
  echo "server not ready after 480s; see $LOG_DIR/server.log" >&2; exit 70
fi

echo "E97_INTERACTIVE_SERVER_READY port=$PORT gpu=$CUDA_VISIBLE_DEVICES supervisor=$$ snapshot=$SNAPSHOT"
echo "smoke:  curl -s http://127.0.0.1:${PORT}/v1/models"
echo "chat:   curl -s http://127.0.0.1:${PORT}/v1/chat/completions -H 'Content-Type: application/json' \\"
echo "          -d '{\"model\":\"e97-dense-agent\",\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory.\"}]}'"
echo "pi:     pi --provider emender-local --model e97-dense-agent -e $ROOT/configs/pi/e97-core-tools.ts --no-builtin-tools --no-skills --no-context-files"
echo "(checkpoint SHA-256 verified by this launcher; canonical Pi-core system prompt pinned server-side; runbook: docs/operations/e97-4b-interactive-hosting.md)"
wait "$SERVER_PID"
