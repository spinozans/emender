#!/usr/bin/env bash
# One scheduled job only. The detached outer driver, not this worker, verifies unit advances.
set -euo pipefail
umask 077
cd "$(git rev-parse --show-toplevel)"
j=${1:?job}; cap=${2:?cap}; [[ $cap =~ ^[0-9]+$ ]]; py=$PWD/.venv/bin/python
if [[ $j == pc-fit-probe ]]; then
  d=output/x49-live/paired-correction-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.paired_correction_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == et-short-cal ]]; then
  d=output/x49-live/error-transfer-panel-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.error_transfer_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == eh-fit-probe ]]; then
  d=output/x49-live/error-history-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.error_history_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == rb-fit-probe ]]; then
  d=output/x49-live/residual-byte-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.residual_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == dr-fit-probe ]]; then
  d=output/x49-live/decode-record-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.decode_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == jt-short-cal ]]; then
  d=output/x49-live/joint-transfer-panel-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.transfer_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == ls-fit-probe ]]; then
  d=output/x49-live/literal-surprise-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.literal_mark_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == v-fit-probe ]]; then
  d=output/x49-live/alternative-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.alternative_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == uc-final ]]; then
  d=output/x49-live/confirm-panel-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.confirm_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == u-fit-probe ]]; then
  d=output/x49-live/surprise-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.surprise_run; export OPENBLAS_NUM_THREADS=1
elif [[ $j == q-probe ]]; then
  d=output/x49-live/query-probe-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.query_run
elif [[ $j =~ ^(w[BPL]-490[123]|w-final)$ ]]; then
  d=output/x49-live/write-panel-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.write_run
elif [[ $j =~ ^(R-490[123]|r-final)$ ]]; then
  d=output/x49-live/relative-panel-1/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.relative_run
else
  [[ $j =~ ^([ABFC]-490[123]|cal-(AB|FC)|final)$ ]]; d=output/x49-live/natural-panel-v2/$j; m=experiments.expressivity_tasks.x49_live_trace_memory.panel
fi
[[ $(hostname) == lambda01 && -z ${GPU_LEASE_HELD:-} && -d $d && ! -e $d/initial && ! -e $d/released ]]
for _ in {1..50}; do grep -q '^PPid:[[:space:]]*1$' /proc/$$/status && break; sleep .1; done
grep -q '^PPid:[[:space:]]*1$' /proc/$$/status; start=$(awk '{sub(/.*\) /, ""); print $20}' /proc/$$/stat)
CUDA_VISIBLE_DEVICES='' "$py" -c 'import sys,json; a=json.load(open(sys.argv[1])); sys.exit(0 if (a["pid"],a["start"])==(int(sys.argv[2]),sys.argv[3]) else 1)' "$d/launch.json" "$$" "$start"
grep -E '^(Pid|PPid):' /proc/$$/status > "$d/detachment"; approved=$(CUDA_VISIBLE_DEVICES='' "$py" -m "$m" prelease --job "$j"); [[ $approved == "$cap" ]]
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=4 GPU_LEASE_DIR="$PWD/.wg/gpu_leases" PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
unset CUDA_VISIBLE_DEVICES
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"; file=$GPU_LEASE_DIR/$GPU_LEASE_HELD
finish(){ rc=$?; set +e; cp "$file" "$d/last" || rc=1; gpu_lease_release || rc=1; if grep -qx "pid=$$" "$file" 2>/dev/null; then rc=1; fi; printf 'released_epoch=%s\nend_monotonic_ns=%s\nowner_cpu_s=%s\nrc=%s\n' "$(date +%s)" "$("$py" -c 'import time; print(time.monotonic_ns())')" "$("$py" -c 'import os,sys; a=open(f"/proc/{int(sys.argv[1])}/stat").read().rsplit(") ",1)[1].split(); print(sum(map(int,a[11:15]))/os.sysconf("SC_CLK_TCK"))' "$$")" "$rc" > "$d/released.tmp"; mv "$d/released.tmp" "$d/released"; exit "$rc"; }; trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES == "$GPU_LEASE_HELD" ]]; cp "$file" "$d/initial.tmp"; mv "$d/initial.tmp" "$d/initial"; grep -qx "pid=$$" "$d/initial"; grep -qx "starttime=$start" "$d/initial"; grep -qx 'adopted=1' "$d/initial"
limit=$(( $(awk -F= '$1=="created"{print $2}' "$d/initial") + cap - $(date +%s) - 25 )); [[ $limit -gt 0 ]]; timeout -k 4 "$limit" "$py" -m "$m" inside --job "$j"
[[ -s $d/watch-ack && -s $d/complete.json ]]; cp "$file" "$d/after"; [[ $(awk -F= '$1=="heartbeat"{print $2}' "$d/after") -gt $(awk -F= '$1=="heartbeat"{print $2}' "$d/initial") ]]
