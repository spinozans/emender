#!/usr/bin/env bash
# One complete matrix reproduction using its unchanged original numerical source.
set -euo pipefail
umask 077
RUN=${1:?new output root required}
SOURCE_CONTROL=/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-activation-alignment-v2-control
read -r COMMIT < "$SOURCE_CONTROL/commit.txt"
[[ $COMMIT == f58a7bee340e11a5c765e964fc5585bd6b0d3582 ]]
test ! -e "$RUN"
cd "$SOURCE_CONTROL/worktree"
sha256sum --check --quiet "$SOURCE_CONTROL/source.sha256"
trap 'status=$?; trap - EXIT; cd "$SOURCE_CONTROL/worktree"; sha256sum --check --quiet "$SOURCE_CONTROL/source.sha256" || exit 1; exit "$status"' EXIT
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$SOURCE_CONTROL/worktree" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" -m pytest -q tests/test_e97_activation_alignment.py tests/test_e97_head_precision_probe.py tests/test_e97_outcome_rl_candidate.py tests/test_e97_native_execution.py
CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" scripts/diagnose_e97_activation_alignment.py freeze --output "$RUN"
RECIPE_SHA=$(sha256sum "$RUN/recipe-private.json" | cut -d' ' -f1)
[[ $RECIPE_SHA == 1aebe6302d42032ca5f9a0af3d63cc5f36d7f401b509801d1a483952cb23d9d5 ]]
cmp "$RUN/recipe-private.json" /mnt/nvme2n1/erikg/e97_systematic_posttraining/native-activation-alignment-v2/recipe-private.json
printf 'MATRIX_REPRODUCTION source=%s recipe=%s optimizer_updates=0\n' "$COMMIT" "$RECIPE_SHA"
export GPU_LEASE_DIR=/home/erikg/emender/.wg/gpu_leases
LEASE=$(bash scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$LEASE"
# Lease adoption installs its own EXIT trap; compose release with source audit.
trap 'status=$?; trap - EXIT; gpu_lease_release; cd "$SOURCE_CONTROL/worktree"; sha256sum --check --quiet "$SOURCE_CONTROL/source.sha256" || exit 1; exit "$status"' EXIT
export NUMA_LOCAL_RANK_TRITON_CACHE_PREFIX="$RUN/triton"
timeout --kill-after=30 3600 "$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 --max-restarts=0 \
 scripts/numa_local_rank_exec.py scripts/diagnose_e97_activation_alignment.py run \
 --recipe "$RUN/recipe-private.json" --recipe-sha "$RECIPE_SHA" --output "$RUN" 2>&1 | tee "$RUN/console.log"
