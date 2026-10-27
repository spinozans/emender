#!/usr/bin/env bash
# Sequential, resumable attended-session smoke; never launches full arms.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
runner=experiments/expressivity_tasks/x34_lm_pilot/run_leased.sh
"$runner" train --arm base_lm --tag pilot --steps 64
"$runner" train --arm base_lm --tag pilot --steps 512
"$runner" train --arm tape_lm --tag pilot --steps 512
"$runner" probe --arm base_lm --tag pilot --probe-smoke
"$runner" probe --arm tape_lm --tag pilot --probe-smoke
for arm in base_lm tape_lm; do
    cp "experiments/expressivity_tasks/x34_lm_pilot/receipts/pilot_${arm}.json" \
       "experiments/expressivity_tasks/x34_lm_pilot/receipts/smoke_${arm}.json"
    cp "experiments/expressivity_tasks/x34_lm_pilot/receipts/probe_pilot_${arm}.json" \
       "experiments/expressivity_tasks/x34_lm_pilot/receipts/probe_smoke_${arm}.json"
done
PYTHONDONTWRITEBYTECODE=1 /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x34_lm_pilot.summarize
