#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
receipt=experiments/expressivity_tasks/x40_emender_lm/receipts/part5
runner=experiments/expressivity_tasks/x40_emender_lm/run_leased.sh
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.scale_probe --gate-only --output "$receipt/scale_gate_rebased" > "$receipt/scale_gate_rebased.log" 2>&1
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_pilot --label pilot_baseline > "$receipt/pilot_baseline.log" 2>&1
X40_BATCHED_READS=1 "$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_gate > "$receipt/reads_parity.log" 2>&1
X40_BATCHED_READS=1 "$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_pilot --label pilot_reads > "$receipt/pilot_reads.log" 2>&1
