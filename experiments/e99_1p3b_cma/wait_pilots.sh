#!/usr/bin/env bash
set -u
PD=/home/erikg/ndm/.wg-worktrees/agent-1119/experiments/e99_1p3b_cma/pilot_results
start=$(date +%s)
while true; do
  n=0
  for d in p0 p1 p2; do [ -f "$PD/$d/pilot_results.json" ] && n=$((n+1)); done
  if [ "$n" -ge 3 ]; then echo "PILOTS DONE: $n/3"; exit 0; fi
  if ! pgrep -f "pilot.py --configs" >/dev/null 2>&1; then echo "NO PILOT PROCS (n=$n/3 results)"; exit 0; fi
  now=$(date +%s); [ $((now-start)) -ge 4200 ] && { echo "timeout"; exit 0; }
  sleep 45
done
