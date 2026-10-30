#!/usr/bin/env bash
set -uo pipefail
cd /home/erikg/emender
r=experiments/expressivity_tasks/x40_emender_lm/receipts/part16
export GPU_LEASE_VISIBLE=2,3,4
lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait) || exit 1
eval "$lease"
cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$r/lease_phase2.txt"
export CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=/home/erikg/emender OMP_NUM_THREADS=1
probe="$r/probe.py"
for tool in memcheck racecheck initcheck; do
 timeout 600 /usr/local/cuda/bin/compute-sanitizer --tool "$tool" --kernel-name kns=scan_kernel --launch-skip 9 --launch-count 1 --error-exitcode 86 /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x40_emender_lm/receipts/part15/single.py > "$r/${tool}_scan10.txt" 2>&1
 printf '%s_scan10 %s\n' "$tool" "$?" >> "$r/status.txt"
done
for i in 1 2 3 4 5; do
 timeout 180 /home/erikg/emender/.venv/bin/python experiments/expressivity_tasks/x40_emender_lm/receipts/part15/single.py > "$r/repeat_$i.txt" 2>&1
 printf 'repeat_%s %s\n' "$i" "$?" >> "$r/status.txt"
done
PART16_ADDRESSES=1 timeout 600 /home/erikg/emender/.venv/bin/python "$probe" isolated > "$r/isolated100.txt" 2>&1
printf 'isolated100 %s\n' "$?" >> "$r/status.txt"
for stable in 0 1; do
 for pad in 0 512 65536 1048576 16777216; do
  PART16_ADDRESSES=1 PART16_STABLE=$stable PART16_PADDING=$pad timeout 180 /home/erikg/emender/.venv/bin/python "$probe" full > "$r/layout_${stable}_${pad}.txt" 2>&1
  printf 'layout_%s_%s %s\n' "$stable" "$pad" "$?" >> "$r/status.txt"
 done
done
