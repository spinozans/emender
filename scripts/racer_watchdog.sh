#!/usr/bin/env bash
# Racer hang watchdog: if a single-GPU racer's training step stops advancing while
# its process is still alive (the silent CUDA-wedge failure mode), kill it fast and
# log a LOUD alert — so an 8-hour silent hang can never happen again.
set -uo pipefail
WLOG=/mnt/nvme1n1/erikg/race/watchdog.log
INTERVAL=${INTERVAL:-600}      # check every 10 min
STALL_CHECKS=${STALL_CHECKS:-2} # kill after 2 consecutive no-advance checks (~20 min)

declare -A last_step
declare -A stall

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" >> "$WLOG"; }

log "watchdog started (interval=${INTERVAL}s stall=${STALL_CHECKS})"
while true; do
  for r in emender gdn2; do
    dir=/mnt/nvme1n1/erikg/race/$r
    pidf=$dir/run.pid
    [ -f "$pidf" ] || continue
    pid=$(cat "$pidf" 2>/dev/null)
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then continue; fi   # not running -> nothing to watch
    log=$(ls -S "$dir"/run.log "$dir"/train.log 2>/dev/null | head -1)
    step=$(grep -hoE 'step +[0-9]+ \| loss' "$log" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    [ -n "$step" ] || continue
    prev=${last_step[$r]:-}
    if [ -n "$prev" ] && [ "$step" = "$prev" ]; then
      stall[$r]=$(( ${stall[$r]:-0} + 1 ))
      log "WARN $r step stuck at $step (stall ${stall[$r]}/${STALL_CHECKS}, pid $pid)"
      if [ "${stall[$r]}" -ge "$STALL_CHECKS" ]; then
        log "ALERT $r HUNG at step $step — killing pid $pid (silent CUDA wedge). Re-equalize needed."
        kill -KILL "$pid" 2>/dev/null
        stall[$r]=0
      fi
    else
      stall[$r]=0
    fi
    last_step[$r]=$step
  done
  sleep "$INTERVAL"
done
