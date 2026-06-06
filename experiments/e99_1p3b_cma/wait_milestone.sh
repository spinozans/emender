#!/usr/bin/env bash
# Block until a meaningful milestone in the E99 search, then exit (re-invokes the agent).
# Args: OUT_DIR  TIMEOUT_SEC  MIN_DONES
set -u
OUT="$1"; TIMEOUT="${2:-6000}"; MIN_DONES="${3:-1}"
start=$(date +%s)
while true; do
  ndone=$(ls -d "$OUT"/eval_*/.done 2>/dev/null | wc -l | tr -d ' ')
  if [ "$ndone" -ge "$MIN_DONES" ]; then echo "MILESTONE: $ndone .done files"; exit 0; fi
  if [ -f "$OUT/stop_reason.json" ]; then echo "MILESTONE: stop_reason.json"; exit 0; fi
  if [ -f "$OUT/search.log" ] && grep -qE "Traceback|OutOfMemoryError|raise RuntimeError" "$OUT/search.log" 2>/dev/null; then
    echo "MILESTONE: error in search.log"; exit 0; fi
  now=$(date +%s)
  if [ $((now - start)) -ge "$TIMEOUT" ]; then echo "MILESTONE: timeout ${TIMEOUT}s"; exit 0; fi
  sleep 60
done
