#!/usr/bin/env bash
# One registered, short inference qualification; no training, commits, or background trainer.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
cpu=${1:?passing CPU output}; out=${2:?new attempt directory}
D=experiments/expressivity_tasks/x47_xsa_attention
case ${3:-w1} in
    w1) module=workbench_run ;;
    w2) module=w2_run; git diff --quiet HEAD -- "$D"/w2_*.py "$D/test_w2.py" "$D/W2_PROTOCOL.md" ;;
    *) echo 'expected w1 or w2' >&2; exit 2 ;;
esac
[[ -f $cpu/cpu.json && -f $cpu/initial.pt && -z ${GPU_LEASE_HELD:-} ]]
git diff --quiet HEAD -- "$D"/workbench*.py "$D/test_workbench.py" "$D/W1_PROTOCOL.md" "$D/workbench_leased.sh"
mkdir "$out"
export OMP_NUM_THREADS=4 GPU_LEASE_VISIBLE=0,1 GPU_LEASE_HB_INTERVAL=1 PYTHONUNBUFFERED=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
lease=$(scripts/gpu_lease.sh acquire 1 --no-wait)
eval "$lease"
file=${GPU_LEASE_DIR:-.wg/gpu_leases}/$GPU_LEASE_HELD
finish() {
    rc=$?; set +e
    cp -- "$file" "$out/last"
    gpu_lease_release
    if grep -qx "pid=$$" "$file" 2>/dev/null; then echo 'owner still holds lease' >&2; rc=1; fi
    printf 'released_epoch=%s\nrc=%d\n' "$(date +%s)" "$rc" > "$out/released"
    exit "$rc"
}
trap finish EXIT
[[ $GPU_LEASE_HELD =~ ^[01]$ && $CUDA_VISIBLE_DEVICES = "$GPU_LEASE_HELD" ]]
cp -- "$file" "$out/initial"; grep -qx "pid=$$" "$out/initial"
start=$(awk '{sub(/.*\) /, ""); print $20}' "/proc/$$/stat")
grep -qx "starttime=$start" "$out/initial"
git rev-parse HEAD > "$out/ref"
timeout -k 3 162 /home/erikg/emender/.venv/bin/python -m "experiments.expressivity_tasks.x47_xsa_attention.$module" gpu "$out/run" --cpu-from "$cpu" 2>&1 | tee "$out/log"
cp -- "$file" "$out/after-bench"
[[ $(awk -F= '$1=="heartbeat" {print $2}' "$out/after-bench") -gt $(awk -F= '$1=="heartbeat" {print $2}' "$out/initial") ]]
