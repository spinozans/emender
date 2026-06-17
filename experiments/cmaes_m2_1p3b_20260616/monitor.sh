#!/usr/bin/env bash
# Monitor the e97-m2 CMA-ES search. Exits when the search process is gone
# (complete) OR after MAXWAIT seconds (periodic health re-invoke). Prints a
# status snapshot on exit. Not committed — operational helper for this run.
cd /home/erikg/ndm/.wg-worktrees/agent-1514
ROOT="experiments/cmaes_m2_1p3b_20260616"
GENS="$ROOT/e97-m2/e97-m2_20260616_135253/generations.jsonl"
MAXWAIT="${MAXWAIT:-3600}"
PAT="cmaes_search_v2.py --model e97-m2"
elapsed=0
done_reason="timeout"
while [ "$elapsed" -lt "$MAXWAIT" ]; do
  if ! pgrep -f "$PAT" >/dev/null 2>&1; then done_reason="search_exited"; break; fi
  sleep 60; elapsed=$((elapsed+60))
done
echo "=== MONITOR WAKE ($(date -u +%H:%M:%SZ)) reason=$done_reason elapsed=${elapsed}s ==="
echo "--- driver alive? ---"; pgrep -af "$PAT" | grep -v pgrep | head -1 || echo "NO DRIVER PROCESS"
echo "--- generations.jsonl (count + last) ---"
if [ -f "$GENS" ]; then wc -l < "$GENS"; tail -1 "$GENS"; else echo "no generations.jsonl yet"; fi
echo "--- completed evals (.done) ---"; find "$ROOT/e97-m2" -name '*.done' 2>/dev/null | wc -l
echo "--- tail search.log ---"; tail -8 "$ROOT/search.log"
