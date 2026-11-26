#!/usr/bin/env bash
# Called by device_gate.spawn, never as a standalone substitute for admission.
set -euo pipefail; umask 077
cd "$(git rev-parse --show-toplevel)"; py=$PWD/.venv/bin/python; mod=experiments.expressivity_tasks.retrieved_span_effects.corpus_kernel
d=output/retrieved-span-effects/triton-window-1/corpus-kernel-meter
[[ $(hostname) == lambda01 && -z ${GPU_LEASE_HELD:-} && -d $d && ! -e $d/initial && ! -e $d/released ]]
for _ in {1..100}; do [[ -f $d/owner.json ]] && grep -q '^PPid:[[:space:]]*1$' /proc/$$/status && break; sleep .1; done
grep -q '^PPid:[[:space:]]*1$' /proc/$$/status; start=$(awk '{sub(/.*\) /, ""); print $20}' /proc/$$/stat)
read -r cap deadline < <(CUDA_VISIBLE_DEVICES='' "$py" -m "$mod.device_gate" prelease "$$" "$start")
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 GPU_LEASE_DIR="$PWD/.wg/gpu_leases" CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset CUDA_VISIBLE_DEVICES TRITON_INTERPRET PYTHONOPTIMIZE
scripts/gpu_lease.sh status >"$d/preflight-broker"; df -Pk "$PWD" >"$d/preflight-disk"
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"; file=$GPU_LEASE_DIR/$GPU_LEASE_HELD
finish(){ rc=$?; trap - EXIT; set +e; cp "$file" "$d/last" || rc=1; gpu_lease_release || rc=1; if grep -qx "pid=$$" "$file" 2>/dev/null; then rc=1; fi; printf 'released_epoch=%s\nrc=%s\n' "$(date +%s)" "$rc" >"$d/released.tmp"; mv "$d/released.tmp" "$d/released"; exit "$rc"; }; trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES == "$GPU_LEASE_HELD" ]]; cp "$file" "$d/initial.tmp"; mv "$d/initial.tmp" "$d/initial"; grep -qx "pid=$$" "$d/initial"; grep -qx "starttime=$start" "$d/initial"; grep -qx 'adopted=1' "$d/initial"
created=$(awk -F= '$1=="created"{print $2}' "$d/initial"); limit=$(( (created+cap < deadline ? created+cap : deadline) - $(date +%s) - 15 )); [[ $limit -gt 0 ]]
timeout -k 4 "$limit" "$py" -m "$mod.device_meter" "$d"
[[ -s $d/watch-ack && -s $d/complete.json ]]; cp "$file" "$d/after"; [[ $(awk -F= '$1=="heartbeat"{print $2}' "$d/after") -gt $(awk -F= '$1=="heartbeat"{print $2}' "$d/initial") ]]
