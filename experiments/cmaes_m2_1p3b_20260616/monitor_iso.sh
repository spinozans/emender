#!/usr/bin/env bash
cd /home/erikg/ndm/.wg-worktrees/agent-1514
ROOT="experiments/cmaes_m2_1p3b_20260616"
PAT="iso_geometry_R_control.py"
MAXWAIT="${MAXWAIT:-2400}"
elapsed=0; reason="timeout"
while [ "$elapsed" -lt "$MAXWAIT" ]; do
  if ! pgrep -f "$PAT" >/dev/null 2>&1; then reason="control_exited"; break; fi
  sleep 60; elapsed=$((elapsed+60))
done
echo "=== ISO-CONTROL MONITOR WAKE ($(date -u +%H:%M:%SZ)) reason=$reason elapsed=${elapsed}s ==="
echo "--- control alive? ---"; pgrep -af "$PAT" | grep -v pgrep | head -1 || echo "NO CONTROL PROCESS"
echo "--- completed result.json ---"; find "$ROOT/iso_geometry_R_control" -name result.json 2>/dev/null | wc -l
echo "--- summary (if done) ---"; [ -f "$ROOT/iso_geometry_R_control/summary.json" ] && cat "$ROOT/iso_geometry_R_control/summary.json" || echo "no summary yet"
echo "--- tail iso_control.log ---"; tail -20 "$ROOT/iso_control.log"
