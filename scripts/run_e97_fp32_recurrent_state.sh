#!/usr/bin/env bash
# Frozen, no-update qualification: kernel gradients, then the original full57 gate.
set -euo pipefail
umask 077
RUN=${1:?new output root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CONTROL"
: "${FROZEN_SOURCE_INVENTORY:?required source inventory outside the export}"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ ! -e "$RUN" ]]; mkdir -m 700 "$RUN"
sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$RUN/source-before.log"
# Before adopting a lease, audit on every exit, including a failed CPU preflight.
audit_exit() {
  local rc=$?
  trap - EXIT
  if ! sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$RUN/source-after.log"; then rc=91; fi
  if declare -F gpu_lease_release >/dev/null; then gpu_lease_release; fi
  exit "$rc"
}
trap audit_exit EXIT
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_recurrent_precision.py tests/test_e97_fp32_state_layer_cuda.py > "$RUN/cpu-tests.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.qualify_e97_native_rl_logprobs freeze --recurrent-state-precision fp32 --output "$RUN/assay"
RECIPE_SHA=$(sha256sum "$RUN/assay/recipe-private.json" | cut -d' ' -f1)
[[ $RECIPE_SHA == 82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9 ]]
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.qualify_e97_native_rl_logprobs freeze --output "$RUN/legacy"
LEGACY_SHA=$(sha256sum "$RUN/legacy/recipe-private.json" | cut -d' ' -f1)
[[ $LEGACY_SHA == ff5ccb768f08d096a40d1c105b80800d43117dcec4ff6048b644315a9c48217f ]]
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$LEASE"
# Adoption installs gpu_lease_release as its EXIT handler. Compose that same
# function with the source audit rather than discarding lease cleanup.
declare -F gpu_lease_release >/dev/null
trap audit_exit EXIT
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/kernel-triton"
timeout --kill-after=30 900 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py -m pytest -p no:cacheprovider -q -k cuda -o junit_family=xunit1 \
 tests/test_e97_recurrent_precision.py tests/test_e97_fp32_state_layer_cuda.py --junitxml="$RUN/kernel-tests.xml" 2>&1 | tee "$RUN/kernel-tests.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$RUN/kernel-tests.xml" <<'PY'
import sys,xml.etree.ElementTree as ET
cases=ET.parse(sys.argv[1]).findall('.//testcase')
assert len(cases)==4 and all(not any(c.find(tag) is not None for tag in ('skipped','failure','error')) for c in cases), 'four executed CUDA cases required'
PY
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/legacy-triton"
timeout --kill-after=30 1200 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/qualify_e97_native_rl_logprobs.py run \
 --recipe "$RUN/legacy/recipe-private.json" --recipe-sha "$LEGACY_SHA" --output "$RUN/legacy" 2>&1 | tee "$RUN/legacy/console.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$RUN" <<'PY'
import hashlib,json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);reference=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-rl-logprob-qualification-v1/measurements-private.json')
assert hashlib.sha256(reference.read_bytes()).hexdigest()=='4c9ef306d20a718db6e12a46c0d4dcb178695134e79c671a0f6267aebca1e753'
a=json.loads(reference.read_text())['records'];b=json.loads((root/'legacy/measurements-private.json').read_text())['records']
s=json.loads((root/'legacy/summary.json').read_text());assert len(a)==len(b)==57
errors=[]
for x,y in zip(a,b):
    assert (x['id'],x['turn'],x['recorded'])==(y['id'],y['turn'],y['recorded'])
    assert len(x['teacher'])==len(y['teacher'])==len(x['recorded'])
    errors.extend(abs(u-v) for u,v in zip(x['teacher'],y['teacher']))
assert len(errors)==2711 and all(math.isfinite(x) for x in errors)
checks=dict(historical_teacher=max(errors)<=.0001,actor=s['checks']['cached_replay'],finite=s['checks']['finite'],parameters=s['checks']['parameters_unchanged'],negative_control=not s['probability_path_passed'])
(root/'baseline-binding.json').write_text(json.dumps(dict(checks=checks,teacher_reference_max=max(errors)),sort_keys=True,indent=2)+'\n')
assert all(checks.values()),'legacy baseline did not bind; do not interpret an intervention'
PY
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/assay-triton"
timeout --kill-after=30 1200 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/qualify_e97_native_rl_logprobs.py run \
 --recipe "$RUN/assay/recipe-private.json" --recipe-sha "$RECIPE_SHA" --output "$RUN/assay" 2>&1 | tee "$RUN/assay/console.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1]);a=json.loads((root/'legacy/summary.json').read_text());b=json.loads((root/'assay/summary.json').read_text())
checks=dict(same_weights=a['parameter_sha256']==b['parameter_sha256'],coverage=b['turns']==57 and b['tokens']==2711,fp32=b['recurrent_state_precision']=='fp32',probability_path=b['probability_path_passed'])
(root/'qualification-gate.json').write_text(json.dumps(dict(checks=checks,passed=all(checks.values()),optimizer_updates=0),sort_keys=True,indent=2)+'\n')
assert all(checks.values()),'FP32 state did not pass the original full57 probability gate'
PY
