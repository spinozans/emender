#!/usr/bin/env bash
# gpu_lease.sh — REAL lock-based GPU lease broker for workgraph fan-out.
#
# Multiple agents/tasks share one 8-GPU box with NO central allocator. Without
# coordination they self-pick CUDA_VISIBLE_DEVICES and collide. This broker
# hands out *exclusive* GPU leases atomically using a single flock-serialized
# critical section. A GPU is grantable only if it is BOTH:
#   (1) physically idle per nvidia-smi (util < THRESH_UTIL% AND mem < THRESH_MEM MiB), and
#   (2) not already held by a live lease.
#
# Lease files live at $GPU_LEASE_DIR/<gpu_id> (default <repo>/.wg/gpu_leases),
# which is shared across all worktrees because .wg is a shared symlink. Each
# lease records PID, the PID's /proc starttime (PID-reuse guard), the holder's
# hostname, a creation timestamp and a heartbeat timestamp. A lease is STALE
# (and silently reclaimable) when its PID is dead (starttime-verified) OR its
# heartbeat is older than GPU_LEASE_TTL seconds (default 900 = 15 min). The
# heartbeat-TTL gate reclaims even a still-alive-but-hung holder (heartbeat
# stopped) by design — the nvidia-smi idle gate then prevents re-granting a GPU
# that is still physically busy, so a reaped-but-working GPU is not handed out.
#
# SCOPE & LIMITATIONS (honest):
#   * SINGLE HOST. Lease files are keyed by bare GPU index, so the lease dir
#     must not be shared across machines (two hosts both have a GPU "0"). On
#     this box .wg is a local symlink shared only across worktrees — correct.
#   * Broker-vs-broker exclusion is SOUND (flock + lease-before-unlock + reap
#     under lock). Protection against NON-broker GPU users (a hand-launched
#     `python train.py`, another tool) is BEST-EFFORT: it relies on the
#     instantaneous nvidia-smi idle snapshot, which is racy against a process
#     that is mid-startup. For hard cross-tool exclusivity, route ALL GPU jobs
#     through this broker (or use nvidia exclusive-process compute mode).
#
# Usage (the one-liner agents put at task start):
#   eval "$(scripts/gpu_lease.sh acquire 2)"   # or: eval "$(scripts/gpu_lease.sh 2)"
#     -> exports CUDA_VISIBLE_DEVICES with N exclusive GPUs, installs an
#        EXIT/INT/TERM trap that auto-releases, and starts a heartbeat keeper.
#
# Subcommands:
#   acquire N [--wait|--no-wait] [--timeout S]   reserve N GPUs (default: wait)
#   release [id,...]                              release my GPUs (default: all mine)
#   heartbeat [id,...]                            refresh heartbeat on my leases
#   status                                        human-readable lease + idle table
#   list-free                                     print currently-grantable GPU ids
#   reap                                          reclaim stale leases now (maintenance)
#
# Env knobs:
#   GPU_LEASE_DIR        lease directory (default <repo>/.wg/gpu_leases)
#   GPU_LEASE_TTL        heartbeat staleness TTL in seconds (default 900)
#   GPU_LEASE_THRESH_UTIL  max %util to count a GPU idle (default 10)
#   GPU_LEASE_THRESH_MEM   max MiB used to count a GPU idle (default 256)
#   GPU_LEASE_VISIBLE    comma list restricting the universe of GPU ids the
#                        broker may consider (a real allowlist; default = all
#                        GPUs reported by nvidia-smi)
#   GPU_LEASE_POLL       seconds between retries while waiting (default 5)
#   GPU_LEASE_HB_INTERVAL heartbeat keeper interval seconds (default 60)
#
# This file is safe to `source` (defines functions, runs nothing) and to
# execute (dispatches the subcommand). NO MOCKS: real flock, real nvidia-smi.

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration / paths
# --------------------------------------------------------------------------
_gpu_lease_repo_root() {
  # Resolve repo root from this script's location so the shared .wg symlink is
  # found regardless of the caller's CWD or which worktree we are in.
  local src="${BASH_SOURCE[0]}"
  while [ -h "$src" ]; do
    local dir; dir="$(cd -P "$(dirname "$src")" >/dev/null 2>&1 && pwd)"
    src="$(readlink "$src")"; [[ $src != /* ]] && src="$dir/$src"
  done
  cd -P "$(dirname "$src")/.." >/dev/null 2>&1 && pwd
}

GPU_LEASE_DIR="${GPU_LEASE_DIR:-$(_gpu_lease_repo_root)/.wg/gpu_leases}"
GPU_LEASE_TTL="${GPU_LEASE_TTL:-900}"
GPU_LEASE_THRESH_UTIL="${GPU_LEASE_THRESH_UTIL:-10}"
# Default mem ceiling for "idle": a truly free GPU sits at a few MiB; a CUDA
# context (warming-up external job) is typically >=300 MiB, so 256 catches an
# external process that has merely created a context but not yet ramped util.
GPU_LEASE_THRESH_MEM="${GPU_LEASE_THRESH_MEM:-256}"
GPU_LEASE_POLL="${GPU_LEASE_POLL:-5}"
GPU_LEASE_HB_INTERVAL="${GPU_LEASE_HB_INTERVAL:-60}"
GPU_LEASE_VISIBLE="${GPU_LEASE_VISIBLE:-}"
# Grace seconds during which a freshly-created UNADOPTED lease (the eval
# one-liner path) is held live regardless of PID, giving the holder shell time
# to run the emitted `_adopt` and rebind the lease to its own $$. Without this,
# the short-lived acquire process's PID is already dead by the time a concurrent
# acquirer scans, and the lease would be wrongly reclaimed (causing overlap).
GPU_LEASE_ADOPT_GRACE="${GPU_LEASE_ADOPT_GRACE:-10}"
_GPU_LEASE_LOCK="$GPU_LEASE_DIR/.lock"
_GPU_LEASE_HOST="$(hostname)"

_gpu_lease_log() { echo "gpu_lease: $*" >&2; }
_gpu_lease_die() { _gpu_lease_log "ERROR: $*"; return 1; }

_gpu_lease_ensure_dir() {
  mkdir -p "$GPU_LEASE_DIR"
  [ -e "$_GPU_LEASE_LOCK" ] || : > "$_GPU_LEASE_LOCK"
}

# starttime (field 22 of /proc/<pid>/stat) uniquely identifies a process
# instance together with its PID, guarding against PID reuse.
_gpu_lease_starttime() {
  local pid="$1"
  [ -r "/proc/$pid/stat" ] || { echo ""; return 0; }
  # /proc/<pid>/stat is `pid (comm) state ...`; comm is wrapped in parens by the
  # kernel and may itself contain ')' and spaces. Strip GREEDILY up to the LAST
  # ") " so a comm like "(foo) bar" cannot shift the field index. After the
  # strip, fields begin at `state` (orig field 3), so starttime (orig field 22)
  # is a[20].
  awk '{ s=$0; sub(/.*\) /, "", s); split(s, a, " "); print a[20] }' "/proc/$pid/stat"
}

# Is the process behind a lease still the SAME live process? PID alive AND
# starttime matches what the lease recorded.
_gpu_lease_pid_alive() {
  local pid="$1" want_start="$2"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local now_start; now_start="$(_gpu_lease_starttime "$pid")"
  # If we cannot read starttime (different user / race) fall back to kill -0.
  [ -z "$want_start" ] && return 0
  [ -z "$now_start" ] && return 0
  [ "$now_start" = "$want_start" ]
}

# --------------------------------------------------------------------------
# Lease file format: newline-separated key=value
#   pid=<pid>
#   starttime=<proc starttime ticks>
#   host=<hostname>
#   created=<epoch>
#   heartbeat=<epoch>
#   gpu=<id>
# --------------------------------------------------------------------------
_gpu_lease_read_field() {
  local file="$1" key="$2"
  [ -f "$file" ] || { echo ""; return 0; }
  # Strip the leading "key=" and print the remainder verbatim (values may
  # themselves contain '='). Using sub() avoids the OFS-rebuild that an
  # assignment like `$1=""` would introduce (which prepends a stray space).
  awk -v k="$key" 'index($0, k"=")==1 { sub(/^[^=]*=/, ""); print; exit }' "$file"
}

# True if the lease file is held by a live owner (not stale).
_gpu_lease_is_live() {
  local file="$1"
  [ -f "$file" ] || return 1
  local pid start hb now host created adopted
  pid="$(_gpu_lease_read_field "$file" pid)"
  start="$(_gpu_lease_read_field "$file" starttime)"
  hb="$(_gpu_lease_read_field "$file" heartbeat)"
  host="$(_gpu_lease_read_field "$file" host)"
  created="$(_gpu_lease_read_field "$file" created)"
  adopted="$(_gpu_lease_read_field "$file" adopted)"
  now="$(date +%s)"
  # Adoption grace: an unadopted lease (eval one-liner path, where the acquire
  # process's PID is already gone) is held live for a short window after
  # creation so the holder shell can run `_adopt` and rebind to its own $$.
  # Prevents a concurrent acquirer from reclaiming it in that window.
  if [ "$adopted" = "0" ] && [ -n "$created" ] && [ "$host" = "$_GPU_LEASE_HOST" ]; then
    local cage=$((now - created))
    [ "$cage" -lt 0 ] && cage=0
    [ "$cage" -lt "$GPU_LEASE_ADOPT_GRACE" ] && return 0
  fi
  # Heartbeat TTL gate (applies regardless of host). Clamp a negative age to 0:
  # a backward clock step (NTP, suspend/resume, cross-host skew) must not pin a
  # lease "infinitely fresh" — treat the heartbeat as just-seen and let a real
  # forward TTL elapse after the clock settles (conservative: never mass-reap
  # live leases on a backward jump).
  if [ -n "$hb" ]; then
    local age=$((now - hb))
    [ "$age" -lt 0 ] && age=0
    [ "$age" -gt "$GPU_LEASE_TTL" ] && return 1
  fi
  # PID-liveness is only meaningful on the same host.
  if [ "$host" = "$_GPU_LEASE_HOST" ]; then
    _gpu_lease_pid_alive "$pid" "$start" || return 1
  fi
  return 0
}

# Echo the set of GPU ids that nvidia-smi reports as physically idle, filtered
# by GPU_LEASE_VISIBLE if set. One id per line.
_gpu_lease_idle_gpus() {
  local visible_filter=""
  if [ -n "$GPU_LEASE_VISIBLE" ]; then
    visible_filter=",${GPU_LEASE_VISIBLE},"
  fi
  # index, util.gpu [%], memory.used [MiB]
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
             --format=csv,noheader,nounits 2>/dev/null \
  | while IFS=',' read -r idx util mem; do
      idx="${idx// /}"; util="${util// /}"; mem="${mem// /}"
      [ -n "$idx" ] || continue
      if [ -n "$visible_filter" ]; then
        case "$visible_filter" in *",$idx,"*) ;; *) continue;; esac
      fi
      # nvidia-smi can emit non-integer tokens (e.g. "[N/A]" for util on some
      # MIG/driver states). Treat any empty/non-integer field as NOT idle (fail
      # safe) rather than letting `[ -lt ]` error.
      case "$idx" in ''|*[!0-9]*) continue;; esac
      case "$util" in ''|*[!0-9]*) continue;; esac
      case "$mem" in ''|*[!0-9]*) continue;; esac
      # integer compare; util/mem are integers with nounits
      if [ "$util" -lt "$GPU_LEASE_THRESH_UTIL" ] && [ "$mem" -lt "$GPU_LEASE_THRESH_MEM" ]; then
        echo "$idx"
      fi
    done
}

# Reclaim stale lease files (dead PID or expired heartbeat). MUST be called
# under the lock.
_gpu_lease_reap_locked() {
  local f
  shopt -s nullglob
  for f in "$GPU_LEASE_DIR"/*; do
    [ "$f" = "$_GPU_LEASE_LOCK" ] && continue
    case "$f" in *.lock) continue;; esac
    [ -f "$f" ] || continue
    if ! _gpu_lease_is_live "$f"; then
      rm -f "$f"
    fi
  done
  shopt -u nullglob
}

# Write a lease file for one GPU id (caller-supplied pid). Under the lock.
# adopted=1 means the recorded pid is a confirmed long-lived holder (explicit
# GPU_LEASE_PID/--pid, or after the eval one-liner's _adopt). adopted=0 means
# provisional — protected only by the creation grace window until adopted.
_gpu_lease_write() {
  local id="$1" pid="$2" adopted="${3:-1}" now start
  now="$(date +%s)"
  start="$(_gpu_lease_starttime "$pid")"
  local tmp="$GPU_LEASE_DIR/.tmp.$id.$$"
  {
    echo "pid=$pid"
    echo "starttime=$start"
    echo "host=$_GPU_LEASE_HOST"
    echo "created=$now"
    echo "heartbeat=$now"
    echo "adopted=$adopted"
    echo "gpu=$id"
  } > "$tmp"
  mv -f "$tmp" "$GPU_LEASE_DIR/$id"
}

# --------------------------------------------------------------------------
# One atomic acquisition attempt. Prints granted ids (space separated) to
# stdout and returns 0 on success; returns 1 (no output) if not enough GPUs
# are free right now. Entire body runs under flock.
# --------------------------------------------------------------------------
_gpu_lease_try_acquire() {
  local n="$1" pid="$2" adopted="${3:-1}"
  (
    flock -x 200
    _gpu_lease_reap_locked
    # Build set of currently-leased (live) ids.
    declare -A leased=()
    shopt -s nullglob
    local f id
    for f in "$GPU_LEASE_DIR"/*; do
      case "$f" in *.lock) continue;; esac
      [ -f "$f" ] || continue
      id="$(basename "$f")"
      leased["$id"]=1
    done
    shopt -u nullglob
    # Snapshot idle GPUs into a variable via command substitution (which fully
    # reaps the nvidia-smi child before continuing) rather than a `< <(...)`
    # process substitution. The latter would let the producer child inherit and
    # hold lock fd 200 open after an early `break`, leaking the flock until that
    # child exits — serializing the whole fleet behind a lingering/stalled
    # nvidia-smi whenever N < (#idle GPUs), the common case.
    local idle_snapshot; idle_snapshot="$(_gpu_lease_idle_gpus)"
    # Candidate = idle AND not leased.
    local granted=() count=0
    while read -r id; do
      [ -n "$id" ] || continue
      [ -n "${leased[$id]:-}" ] && continue
      granted+=("$id")
      count=$((count + 1))
      [ "$count" -ge "$n" ] && break
    done <<< "$idle_snapshot"
    if [ "$count" -lt "$n" ]; then
      exit 1
    fi
    for id in "${granted[@]}"; do
      _gpu_lease_write "$id" "$pid" "$adopted"
    done
    # Emit granted ids on fd 1 (inside subshell -> parent captures).
    echo "${granted[*]}"
  ) 200>"$_GPU_LEASE_LOCK"
}

# --------------------------------------------------------------------------
# Public: acquire. Emits shell code on stdout for `eval`.
# --------------------------------------------------------------------------
gpu_lease_acquire() {
  local n="${1:-1}"; shift || true
  # adopted=1 when the caller gives an EXPLICIT, trusted holder pid (GPU_LEASE_PID
  # env or --pid): the lease is born confirmed. adopted=0 when we fall back to
  # $PPID for the eval one-liner — that pid is the short-lived acquire process,
  # so the lease is provisional and the emitted code re-binds it to the holder
  # shell's $$ via `_adopt` within the grace window.
  local wait_mode=1 timeout=0 caller_pid adopted
  if [ -n "${GPU_LEASE_PID:-}" ]; then caller_pid="$GPU_LEASE_PID"; adopted=1; else caller_pid="$PPID"; adopted=0; fi
  while [ $# -gt 0 ]; do
    case "$1" in
      --wait) wait_mode=1;;
      --no-wait) wait_mode=0;;
      --timeout) [ $# -ge 2 ] || { _gpu_lease_die "--timeout needs a value"; return 1; }; shift; timeout="$1";;
      --pid)     [ $# -ge 2 ] || { _gpu_lease_die "--pid needs a value"; return 1; };     shift; caller_pid="$1"; adopted=1;;
      *) _gpu_lease_die "unknown acquire arg: $1"; return 1;;
    esac
    shift
  done
  case "$n" in (*[!0-9]*|'') _gpu_lease_die "N must be a positive integer, got '$n'"; return 1;; esac
  [ "$n" -ge 1 ] || { _gpu_lease_die "N must be >= 1"; return 1; }

  _gpu_lease_ensure_dir

  # Sanity: can we ever satisfy N? (visible universe size)
  local universe
  universe="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -c .)" || universe=0
  if [ -n "$GPU_LEASE_VISIBLE" ]; then
    universe="$(echo "$GPU_LEASE_VISIBLE" | tr ',' '\n' | grep -c .)"
  fi
  if [ "$universe" -lt "$n" ]; then
    _gpu_lease_die "requested $n GPUs but only $universe exist in the visible universe — impossible"
    return 1
  fi

  local start_ts granted
  start_ts="$(date +%s)"
  while true; do
    if granted="$(_gpu_lease_try_acquire "$n" "$caller_pid" "$adopted")"; then
      # Success: emit eval-able shell.
      local csv; csv="$(echo "$granted" | tr ' ' ',')"
      _gpu_lease_emit_eval "$csv" "$caller_pid"
      _gpu_lease_log "granted GPUs: $csv (pid $caller_pid)"
      return 0
    fi
    if [ "$wait_mode" -eq 0 ]; then
      _gpu_lease_die "could not acquire $n GPUs right now (--no-wait)"
      return 1
    fi
    if [ "$timeout" -gt 0 ] && [ $(( $(date +%s) - start_ts )) -ge "$timeout" ]; then
      _gpu_lease_die "timed out after ${timeout}s waiting for $n GPUs"
      return 1
    fi
    _gpu_lease_log "waiting for $n free GPUs (queued)... retry in ${GPU_LEASE_POLL}s"
    sleep "$GPU_LEASE_POLL"
  done
}

# Emit the shell code the caller evals. CRITICAL: the first thing it does is
# `adopt` — rebind the provisional lease to the EVALING shell's own $$ (which is
# the real long-lived holder). This makes correctness independent of how $PPID
# resolved inside the command substitution (redirections/pipes can make $PPID a
# transient subshell that is already dead). The heartbeat keeper and release
# trap are then tied to $$.
_gpu_lease_emit_eval() {
  local csv="$1" pid="$2"
  local script_path; script_path="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  cat <<EOF
# Rebind this lease to THIS shell ($$) so liveness/release track the real holder.
GPU_LEASE_DIR='$GPU_LEASE_DIR' '$script_path' adopt "$csv" "\$\$" >/dev/null 2>&1 || true
# ACCUMULATE into the shell-global held set so a SECOND acquire in the same shell
# does not clobber the first lease's release/heartbeat. Trap installed once and
# releases the FULL accumulated set on exit.
if [ -n "\${GPU_LEASE_HELD:-}" ]; then GPU_LEASE_HELD="\${GPU_LEASE_HELD},$csv"; else GPU_LEASE_HELD="$csv"; fi
export GPU_LEASE_HELD
export CUDA_VISIBLE_DEVICES="\$GPU_LEASE_HELD"
gpu_lease_release() { [ -n "\${GPU_LEASE_HELD:-}" ] && GPU_LEASE_DIR='$GPU_LEASE_DIR' GPU_LEASE_PID="\$\$" '$script_path' release "\$GPU_LEASE_HELD" >/dev/null 2>&1; return 0; }
if [ -z "\${_GPU_LEASE_TRAP_SET:-}" ]; then trap gpu_lease_release EXIT INT TERM; export _GPU_LEASE_TRAP_SET=1; fi
# One background heartbeat keeper per acquire, refreshing its own ids, living as
# long as THIS holder shell (\$\$) is alive.
( while kill -0 \$\$ 2>/dev/null; do
    GPU_LEASE_DIR='$GPU_LEASE_DIR' GPU_LEASE_PID="\$\$" '$script_path' heartbeat "$csv" >/dev/null 2>&1 || true
    sleep $GPU_LEASE_HB_INTERVAL
  done ) >/dev/null 2>&1 &
export GPU_LEASE_HB_PID=\$!
export GPU_LEASE_HB_PIDS="\${GPU_LEASE_HB_PIDS:-} \$!"
disown \$GPU_LEASE_HB_PID 2>/dev/null || true
EOF
}

# --------------------------------------------------------------------------
# Public (internal): adopt. Rebind the given leases to a confirmed holder pid
# and mark them adopted=1. Used by the emitted eval one-liner to bind a
# provisional lease to the real holder shell's $$.
# --------------------------------------------------------------------------
gpu_lease_adopt() {
  _gpu_lease_ensure_dir
  local ids_csv="${1:-}" newpid="${2:-}"
  [ -n "$ids_csv" ] && [ -n "$newpid" ] || return 0
  (
    flock -x 200
    local IFS=',' id
    for id in $ids_csv; do
      id="${id// /}"; [ -n "$id" ] || continue
      # Only adopt a lease that still exists (within its grace window it is ours).
      [ -f "$GPU_LEASE_DIR/$id" ] || continue
      _gpu_lease_write "$id" "$newpid" 1
    done
  ) 200>"$_GPU_LEASE_LOCK"
}

# --------------------------------------------------------------------------
# Public: release. Default = all GPUs leased by this caller (PID match);
# optional explicit id list.
# --------------------------------------------------------------------------
gpu_lease_release() {
  _gpu_lease_ensure_dir
  local ids_csv="${1:-}" caller_pid="${GPU_LEASE_PID:-$PPID}"
  (
    flock -x 200
    local f id pid host
    if [ -n "$ids_csv" ]; then
      local IFS=','
      for id in $ids_csv; do
        id="${id// /}"; [ -n "$id" ] || continue
        f="$GPU_LEASE_DIR/$id"
        [ -f "$f" ] || continue
        pid="$(_gpu_lease_read_field "$f" pid)"
        host="$(_gpu_lease_read_field "$f" host)"
        # Only release a lease we actually own (pid+host), so `release <id>`
        # cannot free another agent's GPU. GPU_LEASE_FORCE=1 overrides (operator
        # maintenance). The auto-release trap passes its own ids+pid, so it is
        # unaffected.
        if [ "${GPU_LEASE_FORCE:-0}" = "1" ] || { [ "$pid" = "$caller_pid" ] && [ "$host" = "$_GPU_LEASE_HOST" ]; }; then
          rm -f "$f"
        else
          _gpu_lease_log "refusing to release GPU $id: owned by pid=$pid host=$host (not $caller_pid@$_GPU_LEASE_HOST); set GPU_LEASE_FORCE=1 to override"
        fi
      done
    else
      shopt -s nullglob
      for f in "$GPU_LEASE_DIR"/*; do
        case "$f" in *.lock) continue;; esac
        [ -f "$f" ] || continue
        pid="$(_gpu_lease_read_field "$f" pid)"
        if [ "$pid" = "$caller_pid" ]; then rm -f "$f"; fi
      done
      shopt -u nullglob
    fi
  ) 200>"$_GPU_LEASE_LOCK"
}

# Public: heartbeat. Refresh heartbeat timestamp on the given (or all-mine) leases.
gpu_lease_heartbeat() {
  _gpu_lease_ensure_dir
  local ids_csv="${1:-}" caller_pid="${GPU_LEASE_PID:-$PPID}" now
  now="$(date +%s)"
  (
    flock -x 200
    local f id pid tmp
    _gpu_lease_touch_one() {
      local file="$1"
      [ -f "$file" ] || return 0
      tmp="$file.hb.$$"
      awk -F= -v now="$now" 'BEGIN{OFS="="} $1=="heartbeat"{print "heartbeat", now; next} {print}' "$file" > "$tmp" && mv -f "$tmp" "$file"
    }
    if [ -n "$ids_csv" ]; then
      local IFS=','
      for id in $ids_csv; do
        id="${id// /}"; [ -n "$id" ] || continue
        _gpu_lease_touch_one "$GPU_LEASE_DIR/$id"
      done
    else
      shopt -s nullglob
      for f in "$GPU_LEASE_DIR"/*; do
        case "$f" in *.lock) continue;; esac
        pid="$(_gpu_lease_read_field "$f" pid)"
        [ "$pid" = "$caller_pid" ] && _gpu_lease_touch_one "$f"
      done
      shopt -u nullglob
    fi
  ) 200>"$_GPU_LEASE_LOCK"
}

# Public: reap stale leases now.
gpu_lease_reap() {
  _gpu_lease_ensure_dir
  ( flock -x 200; _gpu_lease_reap_locked ) 200>"$_GPU_LEASE_LOCK"
}

# Public: list currently grantable (idle AND unleased) GPU ids, csv on stdout.
gpu_lease_list_free() {
  _gpu_lease_ensure_dir
  ( flock -x 200
    _gpu_lease_reap_locked
    declare -A leased=()
    shopt -s nullglob
    local f id
    for f in "$GPU_LEASE_DIR"/*; do
      case "$f" in *.lock) continue;; esac
      [ -f "$f" ] || continue
      leased["$(basename "$f")"]=1
    done
    shopt -u nullglob
    # Command substitution (not `< <(...)`) so no producer child lingers holding
    # lock fd 200 — see the note in _gpu_lease_try_acquire.
    local idle_snapshot; idle_snapshot="$(_gpu_lease_idle_gpus)"
    local out=()
    while read -r id; do
      [ -n "$id" ] || continue
      [ -n "${leased[$id]:-}" ] && continue
      out+=("$id")
    done <<< "$idle_snapshot"
    ( IFS=','; echo "${out[*]}" )
  ) 200>"$_GPU_LEASE_LOCK"
}

# Public: human-readable status table.
gpu_lease_status() {
  _gpu_lease_ensure_dir
  echo "== GPU physical state (nvidia-smi) =="
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
    || echo "(nvidia-smi unavailable)"
  echo
  echo "== Active leases ($GPU_LEASE_DIR, TTL=${GPU_LEASE_TTL}s) =="
  ( flock -x 200
    _gpu_lease_reap_locked
    shopt -s nullglob
    local f any=0 id pid host created hb now
    now="$(date +%s)"
    for f in "$GPU_LEASE_DIR"/*; do
      case "$f" in *.lock) continue;; esac
      [ -f "$f" ] || continue
      any=1
      id="$(basename "$f")"
      pid="$(_gpu_lease_read_field "$f" pid)"
      host="$(_gpu_lease_read_field "$f" host)"
      created="$(_gpu_lease_read_field "$f" created)"
      hb="$(_gpu_lease_read_field "$f" heartbeat)"
      echo "GPU $id  pid=$pid host=$host  age=$((now-created))s  hb_age=$((now-hb))s"
    done
    [ "$any" -eq 0 ] && echo "(none)"
    shopt -u nullglob
  ) 200>"$_GPU_LEASE_LOCK"
}

# --------------------------------------------------------------------------
# Dispatch (only when executed, not when sourced)
# --------------------------------------------------------------------------
_gpu_lease_main() {
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    acquire)      gpu_lease_acquire "$@";;
    adopt)        gpu_lease_adopt "$@";;
    release)      gpu_lease_release "$@";;
    heartbeat)    gpu_lease_heartbeat "$@";;
    reap)         gpu_lease_reap "$@";;
    list-free)    gpu_lease_list_free "$@";;
    status)       gpu_lease_status "$@";;
    ''|-h|--help|help)
      sed -n '2,55p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      ;;
    *)
      # Bare number => acquire that many (convenience for the eval one-liner).
      case "$cmd" in
        *[!0-9]*) _gpu_lease_die "unknown subcommand: $cmd"; return 1;;
        *)        gpu_lease_acquire "$cmd" "$@";;
      esac
      ;;
  esac
}

# Detect execution vs sourcing.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  _gpu_lease_main "$@"
fi
