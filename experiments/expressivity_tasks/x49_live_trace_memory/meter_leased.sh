#!/usr/bin/env bash
# One short-lived parent launches this owner, then exits. Never acquire from a training child.
set -euo pipefail
umask 077
cd "$(git rev-parse --show-toplevel)"
d=output/x49-live/synthetic-meter-1; py=$PWD/.venv/bin/python; m=experiments.expressivity_tasks.x49_live_trace_memory.meter
[[ $(hostname) == lambda01 && -z ${GPU_LEASE_HELD:-} && -d $d && ! -e $d/initial && ! -e $d/released ]]
for _ in {1..50}; do grep -q '^PPid:[[:space:]]*1$' /proc/$$/status && break; sleep .1; done
grep -q '^PPid:[[:space:]]*1$' /proc/$$/status
start=$(awk '{sub(/.*\) /, ""); print $20}' /proc/$$/stat)
CUDA_VISIBLE_DEVICES='' "$py" -c 'import sys,json; a=json.load(open(sys.argv[1])); assert (a["pid"],a["start"])==(int(sys.argv[2]),sys.argv[3])' "$d/launch.json" "$$" "$start"
grep -E '^(Pid|PPid):' /proc/$$/status > "$d/detachment"; CUDA_VISIBLE_DEVICES='' "$py" -m "$m" check
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=4 GPU_LEASE_DIR="$PWD/.wg/gpu_leases" PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
unset CUDA_VISIBLE_DEVICES
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"; file=$GPU_LEASE_DIR/$GPU_LEASE_HELD
finish(){ rc=$?; set +e; cp "$file" "$d/last" || rc=1; gpu_lease_release || rc=1; if grep -qx "pid=$$" "$file" 2>/dev/null; then rc=1; fi; printf 'released_epoch=%s\nrc=%s\n' "$(date +%s)" "$rc" > "$d/released.tmp"; mv "$d/released.tmp" "$d/released"; exit "$rc"; }; trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES == "$GPU_LEASE_HELD" ]]; cp "$file" "$d/initial.tmp"; mv "$d/initial.tmp" "$d/initial"; grep -qx "pid=$$" "$d/initial"; grep -qx "starttime=$start" "$d/initial"; grep -qx 'adopted=1' "$d/initial"
limit=$(( $(awk -F= '$1=="created"{print $2}' "$d/initial") + 900 - $(date +%s) - 25 )); [[ $limit -gt 0 ]]; timeout -k 4 "$limit" "$py" -m "$m" inside
for _ in {1..30}; do [[ -s $d/watch-ack ]] && break; sleep .2; done
[[ -s $d/watch-ack && -s $d/complete.json ]]; cp "$file" "$d/after"; [[ $(awk -F= '$1=="heartbeat"{print $2}' "$d/after") -gt $(awk -F= '$1=="heartbeat"{print $2}' "$d/initial") ]]
