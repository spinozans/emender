#!/usr/bin/env bash
set -euo pipefail
umask 077
OUT=${1:?frozen preparation root required}
CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
: "${FROZEN_SOURCE_INVENTORY:?immutable source inventory required}"
PYTHON_BIN=${PYTHON_BIN:-/home/erikg/emender/.venv/bin/python}
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 PYTHONPATH="$CONTROL"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$CONTROL"
[[ ! -e "$OUT/native-summary.json" && ! -e "$OUT/native-receipts-private.jsonl" ]]
"$PYTHON_BIN" -m pytest -p no:cacheprovider -q tests/test_e97_grounded_curriculum.py tests/test_e97_grounded_expansion_gate.py tests/test_e97_grounded_resume.py tests/test_e97_native_onpolicy_canary.py tests/test_e97_open_swe_native_runtime_protocol.py
sha256sum -c "$FROZEN_SOURCE_INVENTORY" >/dev/null
timeout --kill-after=30 600 "$PYTHON_BIN" -m scripts.prepare_e97_representation_bridge native --output "$OUT" --plan-sha 4fe5dfe085f207d078a382fb9c727f0bfe8ab6188dd7bdff4a36e7d258d60166
