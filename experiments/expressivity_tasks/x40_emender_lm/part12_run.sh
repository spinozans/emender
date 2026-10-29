#!/usr/bin/env bash
# Attended sequential driver. No dispatcher, service, workers, or .pt files.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_LONG_POOL=1 X40_FUSED_INIT=1 X40_STABLE_MAPPING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
base=experiments/expressivity_tasks/x40_emender_lm
run() {
    local label="$1"
    shift
    local code=0
    "$base/run_leased.sh" "$@" > "$base/receipts/part12/$label.log" 2>&1 || code=$?
    printf '%s\t%s\t%s\n' "$label" "$code" "$*" >> "$base/receipts/part12/commands.tsv"
    return "$code"
}
# The baseline T2048 run is executed separately first.
if [[ "${1:-}" != "--measure-only" ]]; then
run curve_candidate_T2048 -m experiments.expressivity_tasks.x40_emender_lm.part12_curve --arm candidate --context 2048
run parity_fused_forward_random -m experiments.expressivity_tasks.x40_emender_lm.part12_parity --fused-forward
run parity_split_random -m experiments.expressivity_tasks.x40_emender_lm.part12_parity --split
fi
for context in 512 2048; do
    for arm in pure transformer; do
        run "profile_${arm}_T${context}" -m experiments.expressivity_tasks.x40_emender_lm.part12_measure profile --context "$context" --arm "$arm"
        run "sustained_${arm}_T${context}" -m experiments.expressivity_tasks.x40_emender_lm.part12_measure sustained --context "$context" --arm "$arm"
    done
    run "fused_forward_T${context}" -m experiments.expressivity_tasks.x40_emender_lm.part12_measure fused-forward --context "$context"
    run "split_T${context}" -m experiments.expressivity_tasks.x40_emender_lm.part12_measure split --context "$context"
done
run pilot -m experiments.expressivity_tasks.x40_emender_lm.part12_pilot
