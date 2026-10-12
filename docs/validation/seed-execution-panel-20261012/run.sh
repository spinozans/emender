#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/seed-execution-panel-20261012
PY=$PWD/.venv/bin/python
export PYTHONPATH=$PWD PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX=$P/triton
[[ -f "$P/GPUS_RELEASED" && -f "$P/smoke-r2/smoke-summary.json" ]]
LEASE_CODE=$(scripts/gpu_lease.sh acquire 8 --no-wait)
eval "$LEASE_CODE"
sha=$(sha256sum "$P/panel.json" | cut -d' ' -f1)
date +%s > "$P/run-start-unix.txt"
set +e
"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 scripts/numa_local_rank_exec.py scripts/eval_e97_native_execution.py run --panel "$P/panel.json" --panel-sha "$sha" --output "$P"
rc=$?
set -e
date +%s > "$P/run-end-unix.txt"
echo "$rc" > "$P/run-exit-code.txt"
if [[ "$rc" != 0 ]]; then
  echo "8-rank evaluation failed: exit $rc; see run.log" > "$P/GPUS_FAILED"
  exit "$rc"
fi
"$PY" -m scripts.eval_e97_native_execution aggregate --panel "$P/panel.json" --panel-sha "$sha" --output "$P" > "$P/aggregate.log" 2>&1
