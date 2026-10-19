#!/usr/bin/env bash
# Frozen eight-rank execution instrument; do not run until all eight cards are free.
set -euo pipefail
umask 077
cd /home/erikg/emender
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
RECEIPTS=$PWD/receipts/w3-matcheddose
RUN=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/w3-matcheddose
PY=$PWD/.venv/bin/python
test -f "$RUN/arms/DONE"
# Fail without reading any gate if the original eight-GPU instrument cannot run.
unset CUDA_VISIBLE_DEVICES GPU_LEASE_VISIBLE
eval "$(bash scripts/gpu_lease.sh acquire 8 --no-wait)"
ALL_CARDS=$CUDA_VISIBLE_DEVICES
FIRST_CARD=${ALL_CARDS%%,*}
"$PY" -m scripts.w3_gate_read prepare --preflight "$RECEIPTS/preflight.json"
read -r PANEL PI_BIN PI_INV IMAGE IMAGE_SHA < <("$PY" - "$RECEIPTS/preflight.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))['instruments']['stage_b']
print(s['panel'],s['pi_bin'],s['pi_inventory'],s['cli_image'],s['cli_image_sha256'])
PY
)
for name in parent A B; do
  READ=$RUN/reads/$name
  read -r CKPT CKPT_SHA < <("$PY" - "$READ/read-identity.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1])); print(r['checkpoint'],r['sha256'])
PY
)
  for repetition in 1 2 3; do
    CUDA_VISIBLE_DEVICES=$FIRST_CARD "$PY" scripts/eval_e97_pi_native_stage_b.py freeze \
      --panel "$PANEL" --checkpoint "$CKPT" --checkpoint-sha256 "$CKPT_SHA" \
      --pi-bin "$PI_BIN" --pi-inventory "$PI_INV" \
      --cli-image "$IMAGE" --cli-image-sha256 "$IMAGE_SHA" \
      --output "$READ/stage-b-plan-$repetition" > "$READ/stage-b-freeze-$repetition.txt" 2>&1
    PLAN=$READ/stage-b-plan-$repetition/plan-private.json
    PLAN_SHA=$(sha256sum "$PLAN" | cut -d' ' -f1)
    CUDA_VISIBLE_DEVICES=$FIRST_CARD timeout --kill-after=30 10800 "$PY" scripts/eval_e97_pi_native_stage_b.py run \
      --plan "$PLAN" --plan-sha "$PLAN_SHA" --output "$READ/stage-b-$repetition" \
      > "$READ/stage-b-console-$repetition.txt" 2>&1
  done
  PANEL_PATH=$READ/execution-panel.json
  PANEL_SHA=$(sha256sum "$PANEL_PATH" | cut -d' ' -f1)
  mkdir "$READ/preflight" "$READ/execution"
  timeout --kill-after=15 900 env CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_e97_native_execution.py smoke \
    --panel "$PANEL_PATH" --panel-sha "$PANEL_SHA" --output "$READ/preflight" > "$READ/smoke.txt" 2>&1
  export CUDA_VISIBLE_DEVICES=$ALL_CARDS PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
  export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX=$READ/triton
  timeout --kill-after=30 5400 "$PY" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc_per_node=8 --max-restarts=0 \
    scripts/numa_local_rank_exec.py scripts/eval_e97_native_execution.py run \
    --panel "$PANEL_PATH" --panel-sha "$PANEL_SHA" --output "$READ/execution" > "$READ/execution.txt" 2>&1
  CUDA_VISIBLE_DEVICES='' "$PY" scripts/eval_e97_native_execution.py aggregate \
    --panel "$PANEL_PATH" --panel-sha "$PANEL_SHA" --output "$READ/execution" > "$READ/aggregate.txt" 2>&1
done
# Exact historical saved-x document-NLL instrument, trajectory only.
"$PY" - "$RECEIPTS/runtime-freeze-v2.json" "$RUN" "$FIRST_CARD" <<'PY'
import json,subprocess,sys
from pathlib import Path
freeze=json.load(open(sys.argv[1])); run=Path(sys.argv[2]); d=freeze['docnll']
old=json.load(open(d['bindings'][0]['path'])); substrate=d['baseline']
cmd=[sys.executable,str(Path(sys.argv[2]).parent/'scripts/score_docnll_gpu.py'),
 '--panel',old['panel'],'--panel-sha256',old['panel_sha256'],
 '--args-json',old['args_json'],'--args-sha256',old['args_sha256'],
 '--model','substrate',substrate['checkpoint'],substrate['sha256']]
for name in ('parent','A','B'):
 r=json.load(open(run/'reads'/name/'read-identity.json'))
 cmd+=['--model',name,r['checkpoint'],r['sha256']]
cmd+=['--baseline','substrate','--candidate','B','--output',str(run/'reads/docnll')]
import os
os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[3]
with (run/'reads/docnll-console.txt').open('x') as log:
 subprocess.run(['timeout','--kill-after=30','1800',*cmd],check=True,stdout=log,stderr=subprocess.STDOUT)
PY
CUDA_VISIBLE_DEVICES='' "$PY" -m scripts.w3_gate_read verdict --preflight "$RECEIPTS/preflight.json"
