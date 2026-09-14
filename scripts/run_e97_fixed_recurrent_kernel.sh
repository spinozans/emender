#!/usr/bin/env bash
# Two fresh-process current-policy checks; original behavior gates remain separate.
set -euo pipefail
umask 077
RUN=${1:?new output root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CONTROL"
: "${FROZEN_SOURCE_INVENTORY:?required source inventory outside export}"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ ! -e "$RUN" ]]; mkdir -m 700 "$RUN"
audit_exit() {
 local rc=$?; trap - EXIT
 if ! sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$RUN/source-after.log"; then rc=91; fi
 if declare -F gpu_lease_release >/dev/null; then gpu_lease_release; fi
 exit "$rc"
}
trap audit_exit EXIT
sha256sum -c "$FROZEN_SOURCE_INVENTORY" > "$RUN/source-before.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_recurrent_precision.py tests/test_e97_fp32_state_layer_cuda.py tests/test_e97_fixed_recurrent_kernel_audit.py > "$RUN/cpu-tests.log" 2>&1
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.qualify_e97_native_rl_logprobs freeze --recurrent-state-precision fp32 --output "$RUN/authority"
RECIPE_SHA=$(sha256sum "$RUN/authority/recipe-private.json" | cut -d' ' -f1)
[[ $RECIPE_SHA == 82f53a7100423780fbab31f12f217c259928ae9a1802c608a43b6c54ef5247a9 ]]
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$RUN" <<'PY'
import json,sys
from pathlib import Path
from ndm.recurrent_precision import FIXED_RECURRENT_KERNEL
p=Path(sys.argv[1])/'plan.json'
with p.open('x') as f:
 json.dump(dict(schema='e97-fixed-recurrent-kernel-v1',kernel=FIXED_RECURRENT_KERNEL,workers=2,
  kernel_cases=10,kernel_seconds=900,worker_seconds=1200,outer_seconds=3900,teardown_seconds=30,
  repeat_requirement='exact actor/teacher probabilities and CE means across fresh workers',
  reference='Current pinned-policy forced-token scores; historical behavior is retained separately',
  optimizer_updates=0,training_eligible=False,automatic_expansion=False),f,sort_keys=True,indent=2)
 f.write('\n')
PY
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$LEASE"
declare -F gpu_lease_release >/dev/null; trap audit_exit EXIT
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/kernel-triton"
timeout --kill-after=30 900 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py -m pytest -p no:cacheprovider -q -k cuda -o junit_family=xunit1 \
 tests/test_e97_recurrent_precision.py tests/test_e97_fp32_state_layer_cuda.py --junitxml="$RUN/kernel-tests.xml" 2>&1 | tee "$RUN/kernel-tests.log"
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" - "$RUN/kernel-tests.xml" <<'PY'
import sys,xml.etree.ElementTree as ET
cases=ET.parse(sys.argv[1]).findall('.//testcase')
assert len(cases)==10 and all(not any(c.find(t) is not None for t in ('skipped','failure','error')) for c in cases), 'ten executed CUDA cases required'
PY
for i in 1 2; do
 mkdir -m 700 "$RUN/worker-$i"
 export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/worker-$i-triton"
 timeout --kill-after=30 1200 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
  scripts/numa_local_rank_exec.py scripts/qualify_e97_native_rl_logprobs.py run \
  --recipe "$RUN/authority/recipe-private.json" --recipe-sha "$RECIPE_SHA" --output "$RUN/worker-$i" 2>&1 | tee "$RUN/worker-$i/console.log"
done
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m scripts.audit_e97_fixed_recurrent_kernel "$RUN"
