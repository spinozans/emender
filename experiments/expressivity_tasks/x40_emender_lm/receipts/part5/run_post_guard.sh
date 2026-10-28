#!/usr/bin/env bash
set -euo pipefail
cd /home/erikg/emender
receipt=experiments/expressivity_tasks/x40_emender_lm/receipts/part5
runner=experiments/expressivity_tasks/x40_emender_lm/run_leased.sh
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_STABLE_MAPPING=1 X40_FUSED_INIT=1
mv "$receipt/pilot_B128_bounded" "$receipt/pilot_B128_bounded_pre_guard"
mv "$receipt/curve_bounded" "$receipt/curve_bounded_pre_guard"
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_gate --receipt fused_parity_after_guard.json > "$receipt/fused_parity_after_guard.log" 2>&1
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.part5_pilot --label pilot_B128_bounded --batch 128 --bounded-loss --gate fused_parity_after_guard.json > "$receipt/pilot_B128_bounded_after_guard.log" 2>&1
"$runner" -m experiments.expressivity_tasks.x40_emender_lm.train_slice --bounded-loss --steps 64 --output "$receipt/curve_bounded" > "$receipt/curve_bounded_after_guard.log" 2>&1
