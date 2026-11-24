#!/usr/bin/env bash
# Detached short natural-LM training/eval lease. Watcher does not own the lease.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
stage=$1; run=$2; dir=$3; D=experiments/expressivity_tasks/x47_xsa_attention
[[ $stage == train || $stage == confirm ]]; [[ -z ${GPU_LEASE_HELD:-} ]]
for i in {1..40}; do [[ $(ps -o ppid= -p $$ | tr -d ' ') == 1 ]] && break; sleep .1; done
grep -q '^PPid:[[:space:]]*1$' "/proc/$$/status"; grep -E '^(Pid|PPid):' "/proc/$$/status" > "$dir/detachment"
[[ -f output/w3/train/receipt.json && -f output/w3/dev/receipt.json ]]; git diff --quiet HEAD -- "$D"/w3* "$D/W3_PROTOCOL.md" "$D/test_w3.py"
used=0
for old in output/w3/lease*/initial; do [[ -f $old ]] || continue; other=${old%/initial}; [[ -f $other/released ]]; created=$(awk -F= '$1=="created"{print $2}' "$old"); rel=$(awk -F= '$1=="released_epoch"{print $2}' "$other/released"); used=$((used+rel-created+1)); done
limit=$((1400-used-12)); [[ $limit -gt 20 ]]; if ((limit>1200)); then limit=1200; fi
export OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"; file=${GPU_LEASE_DIR:-.wg/gpu_leases}/$GPU_LEASE_HELD
finish(){ rc=$?; set +e; cp "$file" "$dir/last"; gpu_lease_release; if grep -qx "pid=$$" "$file" 2>/dev/null; then rc=1; fi; printf 'released_epoch=%s\nrc=%d\n' "$(date +%s)" "$rc" > "$dir/released"; exit "$rc"; }; trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES == "$GPU_LEASE_HELD" ]]; cp "$file" "$dir/initial"; grep -qx "pid=$$" "$dir/initial"; grep -qx "starttime=$(awk '{sub(/.*\) /, ""); print $20}' /proc/$$/stat)" "$dir/initial"
git rev-parse HEAD > "$dir/ref"
timeout -k 3 "$limit" /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x47_xsa_attention.w3_train "$stage" "$run"
cp "$file" "$dir/after"; [[ $(awk -F= '$1=="heartbeat"{print $2}' "$dir/after") -gt $(awk -F= '$1=="heartbeat"{print $2}' "$dir/initial") ]]
