#!/usr/bin/env bash
# Managed continuation of both smoke checkpoints, sequentially; no restart.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
package=experiments/expressivity_tasks/x34_lm_pilot
# Full runs must have a committed, unchanged registration and completed smokes.
git ls-files --error-unmatch "$package/RECEIPTS.md" >/dev/null
git diff --quiet HEAD -- "$package/RECEIPTS.md"
for arm in base_lm tape_lm; do
    [[ -f "$package/runs/pilot_${arm}/latest.pt" ]]
    [[ -f "$package/receipts/smoke_${arm}.json" ]]
done
for arm in base_lm tape_lm; do
    "$package/run_leased.sh" train --arm "$arm" --tag pilot --steps 24415
    "$package/run_leased.sh" probe --arm "$arm" --tag pilot
done
