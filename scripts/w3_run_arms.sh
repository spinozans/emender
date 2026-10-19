#!/usr/bin/env bash
# One bounded serial pair on one leased card; no restart or score-dependent retry.
set -euo pipefail
umask 077
cd /home/erikg/emender
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export GPU_LEASE_VISIBLE=6,7
RECEIPTS=$PWD/receipts/w3-matcheddose
RUN=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/w3-matcheddose
# Fail before acquiring a card if X9b final/branch/preregistration are absent.
test -f "$RECEIPTS/fixed-branch.json"
test -f "$RECEIPTS/preregistration.json"
mkdir "$RUN/arms"
.venv/bin/python - "$RUN/arms/launcher.json" "$$" <<'PY'
import json,socket,sys
from pathlib import Path
pid=int(sys.argv[2]); stat=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
Path(sys.argv[1]).write_text(json.dumps(dict(pid=pid,starttime=int(stat[19]),host=socket.gethostname()))+'\n')
PY
eval "$(bash scripts/gpu_lease.sh acquire 1 --wait --timeout 7200)"
printf '{"world_size_per_arm":1,"arms":"serial","physical_gpu":"%s","automatic_retry":false}\n' "$CUDA_VISIBLE_DEVICES" > "$RUN/arms/topology.json"
for arm in A B; do
  set +e
  timeout --signal=TERM --kill-after=30 10800 .venv/bin/python -m scripts.w3_train_arm \
    --arm "$arm" --branch "$RECEIPTS/fixed-branch.json" \
    --preflight "$RECEIPTS/preflight.json" --freeze "$RECEIPTS/runtime-freeze-v2.json" \
    --preregistration "$RECEIPTS/preregistration.json" --viability "$RECEIPTS/adamw-viability.json" \
    --output "$RUN/arms/$arm" > "$RUN/arms/$arm-console.txt" 2>&1
  rc=$?
  set -e
  printf '{"arm":"%s","exit_code":%d,"automatic_retry":false}\n' "$arm" "$rc" > "$RUN/arms/$arm-terminal.json"
  if ((rc != 0)); then touch "$RUN/arms/FAIL"; exit "$rc"; fi
done
touch "$RUN/arms/DONE"
