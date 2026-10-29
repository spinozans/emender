#!/usr/bin/env bash
# Single attended sequential experiment driver. No service or workers.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
export X40_BATCHED_READS=1 X40_FRESH_POOL=1 X40_FUSED_INIT=1 X40_STABLE_MAPPING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
base=experiments/expressivity_tasks/x40_emender_lm
: > "$base/receipts/part10/commands.tsv"
run() {
    local name="$1"
    shift
    "$base/run_leased.sh" "$@" > "$base/receipts/part10/$name.log" 2>&1
    local code=$?
    printf '%s\t%s\t%s\n' "$name" "$code" "$*" >> "$base/receipts/part10/commands.tsv"
}
for context in 512 2048; do
    for arm in pure transformer; do
        run "profile_${arm}_T${context}" -m experiments.expressivity_tasks.x40_emender_lm.part10_profile --context "$context" --arm "$arm"
    done
done
run read_probe -m experiments.expressivity_tasks.x40_emender_lm.part10_read_probe
run cache_probe -m experiments.expressivity_tasks.x40_emender_lm.part10_cache_probe
for arm in fp32 bf16; do
    run "curve_$arm" -m experiments.expressivity_tasks.x40_emender_lm.part10_curve --arm "$arm"
done
for context in 512 2048; do
    for arm in pure transformer; do
        run "sustained_${arm}_T${context}" -m experiments.expressivity_tasks.x40_emender_lm.part10_sustained --scale 1.3B --arm "$arm" --batch 64 --context "$context" --steps 20
    done
done
run pilot -m experiments.expressivity_tasks.x40_emender_lm.part10_pilot
run rate_probe -m experiments.expressivity_tasks.x40_emender_lm.part10_rate_probe
run store_probe -m experiments.expressivity_tasks.x40_emender_lm.part10_store_probe
run batch_read_probe -m experiments.expressivity_tasks.x40_emender_lm.part10_batch_read_probe
for context in 512 2048; do
    run "reverse_clock_T${context}" -m experiments.expressivity_tasks.x40_emender_lm.part10_reverse_clock --context "$context" --arm pure
done
