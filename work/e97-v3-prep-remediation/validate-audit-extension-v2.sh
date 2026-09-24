#!/usr/bin/env bash
set -euo pipefail
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
E=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-extension-preparation-v2
AUTH=b6dc70b7b704719a4412373e3afb8ef74887b1d52bb24c2f971a08568d3ba262
PACK=b55097e6cfbbfc7c2fa40dc31bcf486f6832fa73623f97a51cb6f13ff8e8e629
"$PY" "$REPO/scripts/validate_e97_sft_packs.py" \
  --authority-root "$E" --pack-root "$E/packs" \
  --authority-manifest-sha256 "$AUTH" --pack-manifest-sha256 "$PACK" \
  --diagnostic-cpu-system-gate 2>"$E/packs/validation.stderr.log" \
  | grep -v '^\[pin-autotune\]' | "$PY" -m json.tool > "$E/packs/validation.json"
"$PY" - "$E/packs/validation.json" <<'EOF'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='pass', r
print('EXT_V2_PACKS_VALIDATION pass', r['packs_validated'], r['records_validated'])
EOF
"$PY" "$REPO/scripts/audit_e97_extension_preparation_v1.py" \
  --preparation "$E" --manifest-sha256 "$AUTH" --output "$E/audit.json"
"$PY" "$REPO/scripts/screen_e97_extension_probe_keys.py" \
  --preparation "$E" --packs "$E/packs/manifest.json" --start 1403761
