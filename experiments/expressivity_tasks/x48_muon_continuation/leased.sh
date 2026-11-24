#!/usr/bin/env bash
# One arm, inclusive X48 cap. Called only under the registered detached launcher.
set -euo pipefail
umask 077
cd "$(git rev-parse --show-toplevel)"
root=output/x48-muon; arm=${1:?arm}; d=$root/$arm; py=/home/erikg/emender/.venv/bin/python; m=experiments.expressivity_tasks.x48_muon_continuation
[[ $arm == direct || $arm == program ]]; [[ -d $d && ! -e $d/initial && ! -e $d/released && -z ${GPU_LEASE_HELD:-} && $(hostname) == lambda01 ]]
for _ in {1..50}; do [[ $(ps -o ppid= -p $$ | tr -d ' ') == 1 ]] && break; sleep .1; done
grep -q '^PPid:[[:space:]]*1$' /proc/$$/status; grep -E '^(Pid|PPid):' /proc/$$/status > "$d/detachment"
CUDA_VISIBLE_DEVICES='' "$py" -m "$m.lease" check
if [[ $arm == program ]]; then grep -qx 'rc=0' "$root/direct/released"; [[ -s $root/direct/watch-ack && -s $root/direct/charge.json ]]; fi
export GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=4 GPU_LEASE_DIR="$PWD/.wg/gpu_leases" PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUBLAS_WORKSPACE_CONFIG=:4096:8
unset CUDA_VISIBLE_DEVICES
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"; file=$GPU_LEASE_DIR/$GPU_LEASE_HELD
finish(){ rc=$?; set +e; cp "$file" "$d/last"; gpu_lease_release; if grep -qx "pid=$$" "$file" 2>/dev/null; then rc=1; fi; printf 'released_epoch=%s\nrc=%s\n' "$(date +%s)" "$rc" > "$d/released.tmp"; mv "$d/released.tmp" "$d/released"; exit "$rc"; }; trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES == "$GPU_LEASE_HELD" ]]; echo $$ > "$d/owner-pid"; cp "$file" "$d/initial.tmp"; mv "$d/initial.tmp" "$d/initial"
grep -qx "pid=$$" "$d/initial"; grep -qx "starttime=$(awk '{sub(/.*\) /, ""); print $20}' /proc/$$/stat)" "$d/initial"
limit=$(( $(awk -F= '$1=="created"{print $2}' "$d/initial") + 10800 - $(date +%s) - 25 )); [[ $limit -gt 0 ]]
timeout -k 4 "$limit" "$py" -m "$m.train" "$arm"
for _ in {1..40}; do [[ -s $d/watch-ack ]] && break; sleep .2; done
[[ -s $d/watch-ack && -s $d/complete.json ]]; cp "$file" "$d/after"
[[ $(awk -F= '$1=="heartbeat"{print $2}' "$d/after") -gt $(awk -F= '$1=="heartbeat"{print $2}' "$d/initial") ]]
