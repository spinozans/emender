#!/usr/bin/env bash
# Short, bounded, detached P1 training-path profile. No held-out stage.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
root=output/memory-program-p1; dir=$root/attempt${1:?attempt required}; D=experiments/expressivity_tasks/x47_xsa_attention
[[ $1 =~ ^[1-9][0-9]*$ && -d $dir && ! -e $dir/initial && -z ${GPU_LEASE_HELD:-} && $(hostname) == lambda01 ]]
for i in {1..50}; do [[ $(ps -o ppid= -p $$ | tr -d ' ') == 1 ]] && break; sleep .1; done
grep -q '^PPid:[[:space:]]*1$' /proc/$$/status; grep -E '^(Pid|PPid):' /proc/$$/status > "$dir/detachment"
git diff --quiet HEAD -- ndm/memory_program "$D" tests/test_memory_program*.py; git rev-parse HEAD > "$dir/ref"
cmp "$dir/ref" "$root/gated-ref"; grep -q '41 passed' "$root/qualification.txt"; if grep -Eq 'failed|skipped' "$root/qualification.txt"; then exit 1; fi
used=0
for old in "$root"/attempt*/initial; do [[ -f $old ]] || continue; parent=${old%/initial}; [[ -f $parent/released ]]; created=$(awk -F= '$1=="created"{print $2}' "$old"); released=$(awk -F= '$1=="released_epoch"{print $2}' "$parent/released"); [[ $released -ge $created ]]; used=$((used+released-created+1)); done
limit=$((200-used-12)); [[ $limit -gt 15 ]]
export OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=2 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait); eval "$lease"; file=${GPU_LEASE_DIR:-.wg/gpu_leases}/$GPU_LEASE_HELD
finish(){ rc=$?; set +e; cp "$file" "$dir/last"; gpu_lease_release; if grep -qx "pid=$$" "$file" 2>/dev/null; then rc=1; fi; printf 'released_epoch=%s\nrc=%s\n' "$(date +%s)" "$rc" > "$dir/released.tmp"; mv "$dir/released.tmp" "$dir/released"; exit "$rc"; }; trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES == "$GPU_LEASE_HELD" ]]; cp "$file" "$dir/initial"
grep -qx "pid=$$" "$dir/initial"; grep -qx "starttime=$(awk '{sub(/.*\) /, ""); print $20}' /proc/$$/stat)" "$dir/initial"
timeout -k 3 "$limit" /home/erikg/emender/.venv/bin/python -m experiments.expressivity_tasks.x47_xsa_attention.memory_program_profile "$dir"
cp "$file" "$dir/after"; [[ $(awk -F= '$1=="heartbeat"{print $2}' "$dir/after") -gt $(awk -F= '$1=="heartbeat"{print $2}' "$dir/initial") ]]
