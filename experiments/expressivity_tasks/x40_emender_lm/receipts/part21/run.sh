#!/usr/bin/env bash
# Parse the whole body before execution; never edit a running runner.
main() {
    set -uo pipefail
    cd /home/erikg/emender || return 1
    local root=experiments/expressivity_tasks/x40_emender_lm/receipts/part21
    local name=$1 mode=$2 seconds=$3
    shift 3
    [[ $name =~ ^[a-zA-Z0-9_-]+$ && $seconds =~ ^[1-9][0-9]*$ ]] || return 2
    local run
    run=$(mktemp -d "$root/${name}.XXXXXXXX") || return 1
    printf '%s\n' "$run"
    # Append on a unique reserved path, including setup and error diagnostics.
    exec >> "$run/console.txt" 2>&1
    printf '%q ' "$@" > "$run/command.txt"
    printf '\n' >> "$run/command.txt"
    sha256sum "$root/run.sh" "$root/candidate.py" > "$run/sources.txt"
    if [[ $mode != host ]]; then
        export GPU_LEASE_VISIBLE=2,3,4
        bash scripts/gpu_lease.sh status > "$run/broker_before.txt"
        local lease
        lease=$(bash scripts/gpu_lease.sh acquire 1 --no-wait) || { printf '1\n' > "$run/status.txt"; return 1; }
        eval "$lease"
        printf '%s\n' "$lease" > "$run/lease_exports.txt"
        cp ".wg/gpu_leases/$CUDA_VISIBLE_DEVICES" "$run/lease.txt"
        export PYTHONPATH=/home/erikg/emender CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1
    fi
    local rc
    timeout --signal=TERM --kill-after=10 "$seconds" "$@"
    rc=$?
    printf '%s\n' "$rc" > "$run/status.txt"
    sha256sum "$run/console.txt" > "$run/console.sha256"
    return "$rc"
}
main "$@"
