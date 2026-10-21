#!/usr/bin/env bash
# Detached W4 world1 pipeline; one lease, one fixed trajectory, one exploratory read.
set -euo pipefail
umask 077
cd /home/erikg/emender
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export GPU_LEASE_VISIBLE=2,3,4,5,6,7
RUN=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/w4-chillwash
REG=$PWD/receipts/w4-chillwash/preregistration.json
PY=$PWD/.venv/bin/python
"$PY" -m scripts.w4_chillwash verify --registration "$REG"
mkdir "$RUN"
finish() {
  rc=$?
  trap - EXIT
  "$PY" - "$RUN" "$rc" <<'PY'
import json,sys
from pathlib import Path
from datetime import datetime,timezone
root=Path(sys.argv[1]);rc=int(sys.argv[2])
with (root/'terminal.json').open('x') as f:
 json.dump(dict(exit_code=rc,automatic_retry=False,finished_at=datetime.now(timezone.utc).isoformat()),f)
(root/('DONE' if rc==0 else 'STOPPED')).touch(exist_ok=False)
PY
  if declare -F gpu_lease_release >/dev/null; then gpu_lease_release; fi
  exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
# Broker's own trap is replaced below so terminal evidence and lease release both run.
eval "$(bash scripts/gpu_lease.sh acquire 1 --no-wait)"
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
case "$CUDA_VISIBLE_DEVICES" in 2|3|4|5|6|7) ;; *) exit 1 ;; esac
"$PY" - "$RUN" "$$" <<'PY'
import json,os,shutil,socket,sys
from pathlib import Path
from datetime import datetime,timezone
from scripts.w3_disk_preflight import MIN_ROOT_FREE_BYTES
root=Path(sys.argv[1]);pid=int(sys.argv[2]);fields=Path(f'/proc/{pid}/stat').read_text().rsplit(') ',1)[1].split()
identity=dict(pid=pid,ppid=int(fields[1]),starttime=fields[19],host=socket.gethostname(),physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],world_size=1,launched_at=datetime.now(timezone.utc).isoformat())
with (root/'launcher.json').open('x') as f:json.dump(identity,f,indent=2)
if identity['ppid']!=1:raise ValueError('launcher must be detached with PPID=1')
free=shutil.disk_usage('/').free
with (root/'capacity.json').open('x') as f:json.dump(dict(free_bytes=free,minimum_free_bytes=MIN_ROOT_FREE_BYTES,passed=free>=MIN_ROOT_FREE_BYTES),f)
if free<MIN_ROOT_FREE_BYTES:raise ValueError('root volume below W3 safety floor')
PY
timeout --signal=TERM --kill-after=30 10800 "$PY" -m scripts.w4_train_chill \
  --registration "$REG" --output "$RUN/training" > "$RUN/training-console.txt" 2>&1
"$PY" -m scripts.w4_chillwash prepare --registration "$REG"
read -r PANEL PI_BIN PI_INV IMAGE IMAGE_SHA < <("$PY" - <<'PY'
import json
s=json.load(open('receipts/w3-matcheddose/preflight.json'))['instruments']['stage_b']
print(s['panel'],s['pi_bin'],s['pi_inventory'],s['cli_image'],s['cli_image_sha256'])
PY
)
read -r CKPT CKPT_SHA < <("$PY" - "$RUN/reads/read-identity.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));print(r['checkpoint'],r['sha256'])
PY
)
"$PY" -m scripts.w4_stage_b_chill freeze --panel "$PANEL" \
  --checkpoint "$CKPT" --checkpoint-sha256 "$CKPT_SHA" --pi-bin "$PI_BIN" \
  --pi-inventory "$PI_INV" --cli-image "$IMAGE" --cli-image-sha256 "$IMAGE_SHA" \
  --output "$RUN/reads/stage-b-plan" > "$RUN/reads/stage-b-freeze.txt" 2>&1
PLAN=$RUN/reads/stage-b-plan/plan-private.json
PLAN_SHA=$(sha256sum "$PLAN" | cut -d' ' -f1)
timeout --signal=TERM --kill-after=30 10800 "$PY" -m scripts.w4_stage_b_chill run \
  --plan "$PLAN" --plan-sha "$PLAN_SHA" --output "$RUN/reads/stage-b" \
  > "$RUN/reads/stage-b-console.txt" 2>&1
timeout --signal=TERM --kill-after=30 1800 "$PY" -m scripts.w4_chillwash docnll \
  --registration "$REG" > "$RUN/reads/docnll-console.txt" 2>&1
"$PY" -m scripts.w4_chillwash summarize --registration "$REG"
