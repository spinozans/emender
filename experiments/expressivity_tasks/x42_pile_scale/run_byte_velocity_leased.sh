#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
source scripts/gpu_lease.sh
lease_exports="$(scripts/gpu_lease.sh acquire 1 --wait)"
eval "$lease_exports"
case "$1:$GPU_LEASE_HELD" in transformer:2|gdn2:3|pure:4) ;; *) exit 91;; esac
mkdir -p "${3}_launch"
cp "$GPU_LEASE_DIR/$GPU_LEASE_HELD" "${3}_launch/lease_initial.txt"
set +e
/home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x42_pile_scale.byte_velocity train --arm "$1" --policy "$2" --output "$3"
status=$?
printf '{"exit_code":%d}\n' "$status" > "${3}_launch/terminal.tmp"
mv "${3}_launch/terminal.tmp" "${3}_launch/terminal.json"
exit "$status"
