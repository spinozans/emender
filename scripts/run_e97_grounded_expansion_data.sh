#!/usr/bin/env bash
set -euo pipefail
umask 077
OUT=${1:?new data root required}
EXTRA=()
if [[ -n ${2:-} ]]; then EXTRA=(--completed-from "$2"); fi
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$CONTROL"
: "${FROZEN_SOURCE_INVENTORY:?frozen inventory required}"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export PYTHONPATH="$CONTROL" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ ! -e "$OUT" ]]
[[ $(sha256sum configs/pi/e97-grounded-expansion-v1.json | cut -d' ' -f1) == 12a6d13088dfdcfd1484e413ad13a402bc2ce99dfdb64651829f3c5775e5987f ]]
"$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_grounded_curriculum.py tests/test_e97_grounded_resume.py tests/test_e97_native_onpolicy_canary.py tests/test_e97_grounding_correction.py tests/test_e97_open_swe_native_runtime_protocol.py tests/test_e97_open_swe_native_sft.py
sha256sum -c "$FROZEN_SOURCE_INVENTORY" >/dev/null
timeout --kill-after=30 1800 "$PYTHON_BIN" -m scripts.build_e97_grounded_expansion --recipe configs/pi/e97-grounded-expansion-v1.json --output "$OUT" "${EXTRA[@]}"
