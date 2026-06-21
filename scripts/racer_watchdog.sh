#!/usr/bin/env bash
# Racer hang watchdog: if a training run's step stops advancing while its process
# is still alive (the silent CUDA-wedge failure mode), kill it fast and log a LOUD
# alert — so an 8-hour silent hang can never happen again.
#
# Watches one or more run dirs. Each run dir must contain run.pid (the leader PID
# to monitor/kill) and a run.log or train.log that grows as training advances.
#
# Coverage is env-configurable (backward compatible):
#   RACER_DIRS  whitespace- or comma-separated absolute run dirs to watch.
#               Default = the two single-GPU racers (race/emender, race/gdn2).
#   WLOG        watchdog log path (default /mnt/nvme1n1/erikg/race/watchdog.log).
#   INTERVAL    seconds between checks (default 600 = 10 min).
#   STALL_CHECKS consecutive no-advance checks before kill (default 2 = ~20 min).
#
# Example (register the 8-GPU DiLoCo run alongside the default racers):
#   RACER_DIRS=/mnt/nvme1n1/erikg/diloco_8gpu/emender \
#     nohup bash scripts/racer_watchdog.sh >/dev/null 2>&1 &
set -uo pipefail
WLOG=${WLOG:-/mnt/nvme1n1/erikg/race/watchdog.log}
INTERVAL=${INTERVAL:-600}      # check every 10 min
STALL_CHECKS=${STALL_CHECKS:-2} # kill after 2 consecutive no-advance checks (~20 min)

# Run dirs to watch. Default preserves the original hardcoded behavior so the
# existing race watchdog is unchanged; override with RACER_DIRS for other runs.
DEFAULT_DIRS="/mnt/nvme1n1/erikg/race/emender /mnt/nvme1n1/erikg/race/gdn2"
RACER_DIRS=${RACER_DIRS:-$DEFAULT_DIRS}
RACER_DIRS=${RACER_DIRS//,/ }   # allow comma-separated input

declare -A last_step
declare -A stall

ts() { date -u +%FT%TZ; }
log() { echo "[$(ts)] $*" >> "$WLOG"; }

mkdir -p "$(dirname "$WLOG")"
log "watchdog started (interval=${INTERVAL}s stall=${STALL_CHECKS}) dirs=[${RACER_DIRS}]"
while true; do
  for dir in $RACER_DIRS; do
    r=$(basename "$dir")
    pidf=$dir/run.pid
    [ -f "$pidf" ] || continue
    pid=$(cat "$pidf" 2>/dev/null)
    [ -n "$pid" ] || continue
    if ! kill -0 "$pid" 2>/dev/null; then continue; fi   # not running -> nothing to watch
    rlog=$(ls -S "$dir"/run.log "$dir"/train.log 2>/dev/null | head -1)
    # Universal liveness = does the log keep GROWING. A precise step (when present) makes the
    # alert informative; but the stall signal itself is log byte-size, so an init-wedge that
    # never emits a first step (fused-kernel autotune deadlock under box load) is ALSO caught.
    step=$(grep -hoE 'step +[0-9]+ \| loss' "$rlog" 2>/dev/null | tail -1 | grep -oE '[0-9]+' | head -1)
    size=$(stat -c %s "$rlog" 2>/dev/null || echo 0)
    token="${size}@${step:-init}"
    prev=${last_step[$dir]:-}
    if [ -n "$prev" ] && [ "$token" = "$prev" ]; then
      stall[$dir]=$(( ${stall[$dir]:-0} + 1 ))
      log "WARN $r no log progress (at step ${step:-INIT}, ${size}B) (stall ${stall[$dir]}/${STALL_CHECKS}, pid $pid)"
      if [ "${stall[$dir]}" -ge "$STALL_CHECKS" ]; then
        log "ALERT $r HUNG at step ${step:-INIT} (log frozen ${size}B) — killing pid $pid (silent CUDA wedge). Re-equalize needed."
        kill -KILL "$pid" 2>/dev/null
        stall[$dir]=0
      fi
    else
      stall[$dir]=0
    fi
    last_step[$dir]=$token
  done
  sleep "$INTERVAL"
done
