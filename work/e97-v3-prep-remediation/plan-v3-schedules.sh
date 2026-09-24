#!/usr/bin/env bash
set -euo pipefail
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-full-preparation-v3
AUTH=3c4f0b38ba107cabd4d0505ceb789bf2bba81ee2e11890421b7a163a96877f9b
PACK=5a4baffbb0a3ac4e2b6531d2610bb4ae26e9ec0f790813326de42d90d03175a0
KEYS="1401011 1401042 1401043 1401046 1401052 1401056 1401057 1401074"
i=0
for K in $KEYS; do
  i=$((i+1))
  OUT=$P/schedule-segment${i}-proposal.json
  "$PY" "$REPO/scripts/plan_e97_pi_native_training_schedule.py" \
    --authority "$P" --packs "$P/packs" --authority-sha256 "$AUTH" --pack-sha256 "$PACK" \
    --sampler-key "$K" --world-size 8 --context-size 65536 --steps 128 --output "$OUT"
  "$PY" "$REPO/scripts/verify_e97_repair_schedule_window_rule.py" --schedule "$OUT"
done
cp "$P/schedule-segment1-proposal.json" "$P/schedule-proposal.json"
"$PY" "$REPO/scripts/verify_e97_repair_schedule_window_rule.py" --schedule "$P/schedule-proposal.json"
