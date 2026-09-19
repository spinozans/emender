#!/usr/bin/env bash
# Dose-screen probe launcher for the d25 prep — baked constants (STAGED; NOT EXECUTED).
# usage: run-probe.sh <admission-dir> [args-json]
set -euo pipefail
W=/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-v1
ADMISSION=${1:?usage: run-probe.sh <admission-dir> [args-json]}
ARGS_JSON=${2:-/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json}
exec bash "$W/run-probe.sh" \
  --dose d25 \
  --prep /mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-full-preparation-v3-d25 \
  --key 1402002 \
  --admission "$ADMISSION" \
  --args-json "$ARGS_JSON" \
  --output /mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-probe-d25-v1
