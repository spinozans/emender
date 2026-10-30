#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x40_emender_lm/receipts/part16
export GPU_LEASE_VISIBLE=2,3,4
lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait) || exit 1
eval "$lease"
cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$r/lease_phase3.txt"
export CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/home/erikg/emender OMP_NUM_THREADS=1
for tool in memcheck racecheck initcheck; do
 timeout 180 /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --error-exitcode 86 /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x40_emender_lm/receipts/part15/isolated.py original > "$r/${tool}_isolated.txt" 2>&1
 printf '%s_isolated %s\n' "$tool" "$?" >> "$r/status.txt"
done
for i in 6 7 8 9 10; do
 timeout 180 /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x40_emender_lm/receipts/part15/single.py > "$r/repeat_$i.txt" 2>&1
 printf 'repeat_%s %s\n' "$i" "$?" >> "$r/status.txt"
done
timeout 180 /home/erikg/emender/.venv/bin/python "$r/probe.py" isolated > "$r/isolated100_plain.txt" 2>&1
printf 'isolated100_plain %s\n' "$?" >> "$r/status.txt"
for rep in 2 3; do
 for stable in 0 1; do
  for pad in 0 512 65536 1048576 16777216; do
   PART16_ADDRESSES=1 PART16_STABLE=$stable PART16_PADDING=$pad timeout 180 /home/erikg/emender/.venv/bin/python "$r/probe.py" full > "$r/layout_${stable}_${pad}_r${rep}.txt" 2>&1
   printf 'layout_%s_%s_r%s %s\n' "$stable" "$pad" "$rep" "$?" >> "$r/status.txt"
  done
 done
done
