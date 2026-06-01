#!/usr/bin/env bash
# Wait for GPU-0 headroom, then run the resumable readback. Retry on contention
# OOM (other agents share GPU 0). Exits 0 only when ALL THREE read back SANE.
set -u
cd /home/erikg/ndm/.wg-worktrees/agent-764
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NEED_MIB=14000
MAX_TRIES=120          # ~ up to a few hours of waiting/retrying
for try in $(seq 1 $MAX_TRIES); do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
  echo "[runner] try=$try gpu0_free=${free}MiB need>=${NEED_MIB}MiB"
  if [ "${free:-0}" -ge "$NEED_MIB" ]; then
    echo "[runner] headroom OK -> launching readback"
    python3 scripts/hf_v03_republish_readback.py 2>&1 | grep -aviE "warning|deprecated|Fetching|Downloading|%\|"
    # check sanity from the result JSON
    sane=$(python3 -c "import json;d=json.load(open('scripts/hf_v03_republish_readback_result.json'));print(d.get('_all_sane'))" 2>/dev/null)
    echo "[runner] _all_sane=$sane"
    if [ "$sane" = "True" ]; then
      echo "[runner] ALL READBACK SANE — done"
      exit 0
    fi
    echo "[runner] not all sane yet (likely partial OOM) — will wait and retry"
  fi
  sleep 60
done
echo "[runner] gave up after $MAX_TRIES tries"
exit 1
