#!/usr/bin/env bash
# Supervisor for the 8-GPU I=8 DiLoCo emender arm: keep it alive until a
# wall-clock DEADLINE, auto-resuming from the latest checkpoint whenever the run
# exits. Built after an external SIGTERM preempted the run at hour ~20.6 on the
# shared box (healthy at the time, loss 2.96) — the hang-watchdog only catches
# FROZEN runs, not DEATHS, so a dead arm sat idle. Resume-safety is PROVEN
# (task roll-emender-8-gpu: seam loss-continuous, full SF state preserved), so
# auto-resume is sound.
#
# This shell LEASES the GPUs and never exec()s away, so the lease trap + heartbeat
# keeper live for the whole supervised window (also fixes the launch_detached_run
# exec-drops-the-trap fragility). torchrun runs as a normal child; on exit we
# resume from the newest merge-aligned checkpoint and relaunch until DEADLINE.
#
# Env:
#   DEADLINE_UNIX  (required) stop launching once `date +%s` >= this.
#   LOGDIR         run dir (default /mnt/nvme1n1/erikg/diloco_8gpu/emender).
#   MAX_FAST_FAILS consecutive <FAST_FAIL_SECS launches before aborting (default 5).
#   FAST_FAIL_SECS a launch shorter than this counts as a fast-fail (default 180).
set -uo pipefail

REPO_ROOT="$(cd -P "$(dirname "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)"
LOGDIR="${LOGDIR:-/mnt/nvme1n1/erikg/diloco_8gpu/emender}"
OUTPUT="$LOGDIR/runs"
RUNLOG="$LOGDIR/run.log"
PIDFILE="$LOGDIR/run.pid"
SLOG="$LOGDIR/supervisor.log"
LEASE="$REPO_ROOT/scripts/gpu_lease.sh"
: "${DEADLINE_UNIX:?DEADLINE_UNIX is required}"
MAX_FAST_FAILS="${MAX_FAST_FAILS:-5}"
FAST_FAIL_SECS="${FAST_FAIL_SECS:-180}"

mkdir -p "$OUTPUT"
log() { echo "[$(date -u +%FT%TZ)] $*" >> "$SLOG"; }

# Lease 8 GPUs for the supervisor's lifetime (auto-release on EXIT/INT/TERM via
# the trap the broker installs in THIS shell; heartbeat keeper runs in background).
eval "$("$LEASE" 8)"
log "supervisor up pid=$$ leased=[${CUDA_VISIBLE_DEVICES:-NONE}] deadline=$(date -u -d "@$DEADLINE_UNIX" +%FT%TZ)"

build_cmd() {
  local resume="$1"
  CMD=(env NCCL_P2P_DISABLE=1 TORCH_NCCL_ENABLE_MONITORING=0
       TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
       torchrun --standalone --nproc_per_node=8 "$REPO_ROOT/train.py")
  [ -n "$resume" ] && CMD+=(--resume "$resume")
  CMD+=(--level E97 --dim 1792 --n_heads 216 --n_state 32 --depth 11 --expansion 1.0
        --use_gate 1 --gate_activation silu --mlp_ratio 2.2623 --mlp_multiple 64
        --use_triton 1 --optimizer schedulefree --lr 0.001007 --bf16
        --batch_size 4 --chunk_size 2048
        --data /home/erikg/elman/data/pile.txt --tokenizer p50k_base
        --diloco --diloco_k 250 --diloco_outer_lr 1.0 --diloco_outer_beta 0.0
        --steps 100000000 --save_every 500 --keep_checkpoints 20 --log_every 25
        --output "$OUTPUT")
}

fast_fails=0
while [ "$(date +%s)" -lt "$DEADLINE_UNIX" ]; do
  # Newest merge-aligned checkpoint across all timestamped run subdirs.
  CKPT="$(ls -t "$OUTPUT"/*/checkpoint_step_*.pt 2>/dev/null | head -1)"
  build_cmd "$CKPT"
  log "launching torchrun (resume=${CKPT:-NONE})"
  t0="$(date +%s)"
  "${CMD[@]}" >>"$RUNLOG" 2>&1 &
  child="$!"
  echo "$child" > "$PIDFILE"
  log "torchrun child pid=$child"
  wait "$child"; rc=$?
  dt=$(( $(date +%s) - t0 ))
  log "torchrun exited rc=$rc after ${dt}s"

  [ "$(date +%s)" -ge "$DEADLINE_UNIX" ] && break

  if [ "$dt" -lt "$FAST_FAIL_SECS" ]; then
    fast_fails=$(( fast_fails + 1 ))
    log "FAST-FAIL ${fast_fails}/${MAX_FAST_FAILS} (ran ${dt}s < ${FAST_FAIL_SECS}s)"
    if [ "$fast_fails" -ge "$MAX_FAST_FAILS" ]; then
      log "ABORT: ${MAX_FAST_FAILS} consecutive fast-fails — stopping supervisor (investigate; ckpts are safe)."
      break
    fi
    sleep 30
  else
    fast_fails=0
    sleep 10   # let CUDA contexts tear down before relaunch
  fi
done

log "supervisor exiting (deadline reached or aborted); lease auto-releases on shell exit"
