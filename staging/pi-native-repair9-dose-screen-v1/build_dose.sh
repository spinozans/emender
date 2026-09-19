#!/usr/bin/env bash
# Build one dose prep end to end: preparation -> packs -> pack validation -> prep audit.
# Usage: build_dose.sh <tag d25|d50|d75> <authored-max> <bridge-max>
set -euo pipefail
TAG=$1; AMAX=$2; RMAX=$3
H=/mnt/nvme2n1/erikg/e97_systematic_posttraining
W=$H/pi-native-repair9-dose-screen-v1
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
PREP=$H/pi-native-repair9-full-preparation-v3-$TAG
[[ ! -e "$PREP" ]] || { echo "refusing to overwrite $PREP" >&2; exit 1; }
bash "$W/build_prep.sh" "$PREP" "$AMAX" "$RMAX" "$W/prep-$TAG.log"
AUTH_SHA=$(sha256sum "$PREP/manifest.json" | cut -d' ' -f1)
"$PY" "$REPO/scripts/build_e97_sft_packs.py" \
  --authority-root "$PREP" --output-root "$PREP/packs" --context-size 65536 \
  --authority-manifest-sha256 "$AUTH_SHA" \
  --boundary-aware --sampler-mode epoch-permutation --diagnostic-cpu-system-gate \
  2>&1 | tee "$W/packs-$TAG.log"
PACK_SHA=$(sha256sum "$PREP/packs/manifest.json" | cut -d' ' -f1)
"$PY" "$REPO/scripts/validate_e97_sft_packs.py" \
  --authority-root "$PREP" --pack-root "$PREP/packs" \
  --authority-manifest-sha256 "$AUTH_SHA" --pack-manifest-sha256 "$PACK_SHA" \
  --diagnostic-cpu-system-gate 2>&1 | tee "$W/validate-packs-$TAG.log"
"$PY" "$REPO/scripts/audit_e97_extension_preparation_v1.py" \
  --preparation "$PREP" --manifest-sha256 "$AUTH_SHA" \
  --output "$PREP/audit.json" 2>&1 | tee "$W/prep-audit-$TAG.log"
echo "DOSE_PREP_DONE $TAG auth_sha=$AUTH_SHA pack_sha=$PACK_SHA"
