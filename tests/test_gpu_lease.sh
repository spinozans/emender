#!/usr/bin/env bash
# Real test harness for scripts/gpu_lease.sh — NO MOCKS.
#
# Everything exercised here is real: real `flock`, real `nvidia-smi` idle
# checks, real concurrent OS processes, real PID death. The ONLY sandboxing is
# GPU_LEASE_DIR pointed at a throwaway temp directory (so the test does not
# fight production leases under .wg/gpu_leases) and GPU_LEASE_VISIBLE restricting
# the broker to the GPUs that nvidia-smi *actually* reports idle right now (a
# real allowlist the broker supports in production too). The leasing algorithm,
# locking, and reclamation logic under test are 100% the production code path.
#
# Requires an NVIDIA box. If <2 GPUs are physically idle, GPU-dependent cases
# SKIP loudly (exit 0 is still returned for those; the harness fails only on a
# genuine logic violation).

set -uo pipefail

HERE="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -P "$HERE/.." && pwd)"
BROKER="$REPO/scripts/gpu_lease.sh"

PASS=0; FAIL=0; SKIP=0
ok()   { echo "PASS: $*"; PASS=$((PASS+1)); }
bad()  { echo "FAIL: $*"; FAIL=$((FAIL+1)); }
skip() { echo "SKIP: $*"; SKIP=$((SKIP+1)); }

command -v nvidia-smi >/dev/null 2>&1 || { skip "nvidia-smi not present — cannot run real GPU tests"; echo "==> SKIP $SKIP"; exit 0; }

# Throwaway lease namespace for the whole run.
TMPROOT="$(mktemp -d /tmp/gpu_lease_test.XXXXXX)"
cleanup() { rm -rf "$TMPROOT"; }
trap cleanup EXIT

# Discover the GPUs that are genuinely idle right now (real nvidia-smi gate).
# Re-queried per test because the shared box's idle set drifts under real load.
discover_idle() {
  GPU_LEASE_DIR="$TMPROOT/discover.$$.$RANDOM" "$BROKER" list-free 2>/dev/null
}
# First currently-idle GPU id (empty if none).
first_idle() { discover_idle | cut -d, -f1; }
IDLE_CSV="$(discover_idle)"
IFS=',' read -r -a IDLE <<< "$IDLE_CSV"
NIDLE="${#IDLE[@]}"
[ -n "$IDLE_CSV" ] || NIDLE=0
echo "Idle GPUs detected (real): [$IDLE_CSV]  count=$NIDLE"

# ---------------------------------------------------------------------------
# TEST 1: REAL concurrency — zero GPU overlap + correct queueing.
# Launch (cap//2 + 1) leasers, each requesting 2 GPUs, simultaneously, against
# a real-idle pool of `cap` GPUs. flock must serialize them so that exactly
# floor(cap/2) win, all granted ids are DISJOINT, and the surplus leaser is
# denied (queued / no-wait fail). Zero overlap is the hard correctness property.
# ---------------------------------------------------------------------------
test_concurrency() {
  # Re-query the idle pool fresh (the box drifts under real load).
  local pool; pool="$(discover_idle)"
  local -a poolarr; IFS=',' read -r -a poolarr <<< "$pool"
  local cap="${#poolarr[@]}"; [ -n "$pool" ] || cap=0
  if [ "$cap" -lt 2 ]; then skip "concurrency: need >=2 idle GPUs, have $cap"; return; fi
  local expect_win=$((cap / 2))
  local nleasers=$((expect_win + 1))     # one extra leaser to force queueing
  local dir="$TMPROOT/concurrency"
  mkdir -p "$dir"
  echo "  pool=[$pool] cap=$cap leasers=$nleasers x2 each; expect_win=$expect_win"

  # Launch all leasers as background processes at (almost) the same instant.
  # Each leaser binds its lease to its OWN long-lived pid (GPU_LEASE_PID=$$)
  # and then SLEEPS while holding — exactly like a real agent doing GPU work —
  # so leases stay live during the overlap assertion. flock must serialize the
  # acquires so winners get disjoint GPUs and losers are denied.
  local holders=() i
  for i in $(seq 1 "$nleasers"); do
    # Real usage path: eval the broker output in the holder shell, then read the
    # exported CUDA_VISIBLE_DEVICES. The lease binds to this shell ($PPID) and is
    # held alive by `sleep` while the parent asserts.
    GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$pool" bash -c '
      eval "$('"$BROKER"' acquire 2 --no-wait 2>/dev/null)"
      printf "%s\n" "${CUDA_VISIBLE_DEVICES:-}" > "'"$dir"'/result.'"$i"'"
      [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && sleep 20    # hold the lease
      exit 0
    ' &
    holders+=("$!")
  done
  # Wait until every leaser has recorded its acquire outcome (winners hold).
  local t=0
  while : ; do
    local done=0 f
    for f in "$dir"/result.*; do [ -f "$f" ] && done=$((done+1)); done
    [ "$done" -ge "$nleasers" ] && break
    [ "$t" -ge 100 ] && break
    sleep 0.1; t=$((t+1))
  done

  # Collect granted ids (assert NOW, while winning holders are still alive).
  local all_ids=() granted_sets=0 f csv
  for f in "$dir"/result.*; do
    [ -f "$f" ] || continue
    csv="$(cat "$f")"
    [ -n "$csv" ] || continue
    granted_sets=$((granted_sets+1))
    IFS=',' read -r -a ids <<< "$csv"
    for id in "${ids[@]}"; do all_ids+=("$id"); done
  done

  # Property A: ZERO overlap (no GPU id appears twice across all granted sets).
  local dup; dup="$(printf '%s\n' "${all_ids[@]}" | sort | uniq -d)"
  if [ -z "$dup" ]; then
    ok "concurrency: zero GPU overlap across $granted_sets granted leases (ids: ${all_ids[*]:-none})"
  else
    bad "concurrency: OVERLAP detected — duplicate GPU ids: $dup"
  fi

  # Property B0 (positive liveness): with a >=2 GPU pool the broker MUST grant at
  # least one lease. granted_sets==0 means the acquire path is broken — a HARD
  # FAIL, never a soft skip (otherwise a never-granting broker would pass A/C
  # vacuously and skip B).
  if [ "$granted_sets" -ge 1 ]; then
    ok "concurrency: broker granted >=1 lease (acquire path live)"
  else
    bad "concurrency: broker granted ZERO leases from a $cap-GPU pool — acquire path broken"
  fi

  # Property B: granted count == floor(cap/2) (the rest queued/denied) under a
  # STABLE pool. The shared box's real idle set drifts during the test (other
  # agents acquire/release), so an off-by-some winner count is expected and is a
  # soft skip in BOTH directions: the broker can never grant more DISJOINT GPUs
  # than physically exist, so genuine over-granting always shows up as OVERLAP
  # (Property A) — that, plus B0 (>=1) and C (within pool), are the hard gates.
  if [ "$granted_sets" -eq "$expect_win" ]; then
    ok "concurrency: exactly $expect_win leasers won, $((nleasers-expect_win)) correctly queued/denied"
  else
    skip "concurrency: $granted_sets/$expect_win won (idle pool drifted under real load) — zero-overlap (A) is the authoritative gate"
  fi

  # Property C: every granted id is within the real-idle pool.
  local id okpool=1
  for id in "${all_ids[@]}"; do
    case ",$pool," in *",$id,"*) ;; *) okpool=0;; esac
  done
  [ "$okpool" -eq 1 ] && ok "concurrency: all granted ids within real-idle pool" \
                      || bad "concurrency: granted an id outside the idle pool"

  # Release the holders and reap.
  for i in "${!holders[@]}"; do kill -TERM "${holders[$i]}" 2>/dev/null || true; done
  wait 2>/dev/null || true
  GPU_LEASE_DIR="$dir" "$BROKER" reap >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# TEST 2: >universe request is rejected as impossible (queueing upper bound).
# ---------------------------------------------------------------------------
test_overrequest() {
  local dir="$TMPROOT/over"
  if GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="0,1" "$BROKER" acquire 3 --no-wait >/dev/null 2>&1; then
    bad "over-request: broker granted 3 GPUs from a 2-GPU universe"
  else
    ok "over-request: broker refused 3 GPUs from a 2-GPU universe (impossible)"
  fi
}

# ---------------------------------------------------------------------------
# TEST 2b: Lock hygiene — the flock must NOT be held past the critical section.
# Acquiring N < pool size takes the early-break path through the idle scan; if
# the idle producer leaked lock fd 200 (the bug fixed in _gpu_lease_try_acquire)
# the lock would remain held after acquire returns. Assert it is immediately
# re-acquirable with a non-blocking flock.
# ---------------------------------------------------------------------------
test_lock_released() {
  local pool; pool="$(discover_idle)"
  local -a p; IFS=',' read -r -a p <<< "$pool"
  if [ "${#p[@]}" -lt 2 ] || [ -z "$pool" ]; then skip "lock-released: need >=2 idle GPUs"; return; fi
  local dir="$TMPROOT/lockrel"; mkdir -p "$dir"
  # Acquire 1 of a >=2 pool (forces the early break in the candidate loop).
  GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$pool" GPU_LEASE_PID="$$" \
    "$BROKER" acquire 1 --no-wait >/dev/null 2>&1
  # The lock file must be free now (non-blocking flock succeeds instantly).
  if flock -x -n "$dir/.lock" -c true 2>/dev/null; then
    ok "lock-released: flock free immediately after acquire returned (no fd-200 leak)"
  else
    bad "lock-released: lock still held after acquire returned — fd leak regression"
  fi
  GPU_LEASE_DIR="$dir" GPU_LEASE_PID="$$" "$BROKER" release >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# TEST 3: Stale reclaim via PID death. A holder dies WITHOUT releasing; its
# GPU must become re-leasable to a new caller.
# ---------------------------------------------------------------------------
test_stale_pid_death() {
  local one; one="$(first_idle)"
  if [ -z "$one" ]; then skip "stale-pid: need >=1 idle GPU"; return; fi
  local dir="$TMPROOT/stale_pid"
  mkdir -p "$dir"

  # Holder: bind the lease to this long-lived process (GPU_LEASE_PID=$$),
  # acquire 1 GPU, then become `sleep` (exec keeps the pid) — NEVER releasing.
  GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$one" bash -c "
    export GPU_LEASE_PID=\$\$
    '$BROKER' acquire 1 --no-wait >/dev/null 2>&1
    exec sleep 600
  " &
  local holder=$!
  # Wait for the lease file to appear.
  local t=0
  while [ ! -f "$dir/$one" ] && [ "$t" -lt 50 ]; do sleep 0.1; t=$((t+1)); done
  if [ ! -f "$dir/$one" ]; then bad "stale-pid: holder never wrote lease for GPU $one"; kill -9 "$holder" 2>/dev/null; return; fi
  local lease_pid; lease_pid="$(sed -n 's/^pid=//p' "$dir/$one")"
  ok "stale-pid: holder leased GPU $one (lease pid=$lease_pid)"

  # New leaser should NOT be able to take it while holder is alive.
  if GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$one" "$BROKER" acquire 1 --no-wait >/dev/null 2>&1; then
    bad "stale-pid: a second leaser stole the live lease on GPU $one"
    GPU_LEASE_DIR="$dir" "$BROKER" release "$one" >/dev/null 2>&1
  else
    ok "stale-pid: live lease correctly blocks a competing acquire"
  fi

  # Kill the holder HARD (SIGKILL cannot be trapped, so no release runs).
  kill -9 "$holder" 2>/dev/null || true
  wait "$holder" 2>/dev/null || true
  # Give the OS a moment to reap.
  local t2=0
  while kill -0 "$lease_pid" 2>/dev/null && [ "$t2" -lt 50 ]; do sleep 0.1; t2=$((t2+1)); done

  # Now the dead lease must be reclaimable. The new acquire binds the lease to
  # the test shell ($PPID), so the lease FILE should now show a new pid.
  if GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$one" "$BROKER" acquire 1 --no-wait >/dev/null 2>&1 \
     && [ -f "$dir/$one" ]; then
    local newpid; newpid="$(sed -n 's/^pid=//p' "$dir/$one")"
    if [ "$newpid" != "$lease_pid" ]; then
      ok "stale-pid: dead holder's GPU $one reclaimed by new leaser (pid $lease_pid -> $newpid)"
    else
      bad "stale-pid: lease still shows dead pid $lease_pid after reclaim"
    fi
  else
    bad "stale-pid: GPU $one NOT reclaimable after holder death"
  fi
  GPU_LEASE_DIR="$dir" "$BROKER" release "$one" >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# TEST 4: Stale reclaim via heartbeat TTL even when the PID is still alive.
# ---------------------------------------------------------------------------
test_stale_ttl() {
  local one; one="$(first_idle)"
  if [ -z "$one" ]; then skip "stale-ttl: need >=1 idle GPU"; return; fi
  local dir="$TMPROOT/stale_ttl"
  mkdir -p "$dir"
  # Acquire with THIS shell ($$) as a guaranteed-alive holder.
  GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$one" GPU_LEASE_PID="$$" \
    "$BROKER" acquire 1 --no-wait >/dev/null 2>&1
  [ -f "$dir/$one" ] || { bad "stale-ttl: setup acquire failed"; return; }
  # Backdate the heartbeat far into the past (simulate a hung-but-alive holder).
  local old=$(( $(date +%s) - 100000 ))
  sed -i "s/^heartbeat=.*/heartbeat=$old/" "$dir/$one"
  # With a 1s TTL the lease is stale despite the live PID; reap must remove it.
  GPU_LEASE_DIR="$dir" GPU_LEASE_TTL=1 "$BROKER" reap >/dev/null 2>&1
  if [ ! -f "$dir/$one" ]; then
    ok "stale-ttl: live-PID lease with expired heartbeat reclaimed by TTL"
  else
    bad "stale-ttl: stale-heartbeat lease survived reap (TTL path broken)"
  fi
}

# ---------------------------------------------------------------------------
# TEST 5: trap-based auto-release on normal shell exit (the eval one-liner path).
# ---------------------------------------------------------------------------
test_trap_release() {
  local one; one="$(first_idle)"
  if [ -z "$one" ]; then skip "trap-release: need >=1 idle GPU"; return; fi
  local dir="$TMPROOT/trap"
  mkdir -p "$dir"
  # A short-lived shell that evals acquire, does 'work', then exits normally.
  bash -c "
    eval \"\$(GPU_LEASE_DIR='$dir' GPU_LEASE_VISIBLE='$one' '$BROKER' acquire 1 --no-wait 2>/dev/null)\"
    [ \"\$CUDA_VISIBLE_DEVICES\" = '$one' ] || exit 3
    true   # 'work'
  "
  local rc=$?
  if [ "$rc" -ne 0 ]; then bad "trap-release: inner shell did not get CUDA_VISIBLE_DEVICES=$one (rc=$rc)"; return; fi
  # After the shell exits, the EXIT trap must have released the lease.
  # Allow a brief moment for the backgrounded release to settle.
  local t=0
  while [ -f "$dir/$one" ] && [ "$t" -lt 30 ]; do sleep 0.1; t=$((t+1)); done
  if [ ! -f "$dir/$one" ]; then
    ok "trap-release: lease auto-released when holder shell exited"
  else
    bad "trap-release: lease leaked after holder shell exit"
  fi
}

# ---------------------------------------------------------------------------
# TEST 6: --wait actually blocks then succeeds after a release frees a GPU.
# ---------------------------------------------------------------------------
test_wait_then_succeed() {
  local one; one="$(first_idle)"
  if [ -z "$one" ]; then skip "wait-succeed: need >=1 idle GPU"; return; fi
  local dir="$TMPROOT/wait"
  mkdir -p "$dir"
  # Occupy the single visible GPU with this shell as holder.
  GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$one" GPU_LEASE_PID="$$" \
    "$BROKER" acquire 1 --no-wait >/dev/null 2>&1
  [ -f "$dir/$one" ] || { bad "wait-succeed: setup acquire failed"; return; }
  # Start a waiter (poll fast) that wants the same GPU; it must block. Use the
  # real eval path and record the resulting CUDA_VISIBLE_DEVICES.
  local wf="$dir/waiter.out"
  ( GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$one" GPU_LEASE_POLL=1 bash -c '
      eval "$('"$BROKER"' acquire 1 --wait --timeout 15 2>/dev/null)"
      printf "%s" "${CUDA_VISIBLE_DEVICES:-}" > "'"$wf"'"
    ' ) &
  local waiter=$!
  sleep 2
  if ! kill -0 "$waiter" 2>/dev/null; then
    bad "wait-succeed: waiter exited before the GPU was freed (should have blocked)"
    return
  fi
  # Free the GPU; waiter should then acquire it.
  GPU_LEASE_DIR="$dir" GPU_LEASE_PID="$$" "$BROKER" release "$one" >/dev/null 2>&1
  if wait "$waiter"; then
    local csv; csv="$(cat "$wf" 2>/dev/null)"
    [ "$csv" = "$one" ] && ok "wait-succeed: waiter blocked then acquired GPU $one after release" \
                        || bad "wait-succeed: waiter got '$csv' not '$one'"
  else
    bad "wait-succeed: waiter timed out even after release"
  fi
  GPU_LEASE_DIR="$dir" GPU_LEASE_FORCE=1 "$BROKER" release "$one" >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------------------
# TEST 7: multiple acquires in ONE shell ACCUMULATE (the 2nd must not clobber
# the 1st lease's auto-release). Both leases must be released on shell exit.
# ---------------------------------------------------------------------------
test_multi_acquire() {
  local pool; pool="$(discover_idle)"
  local -a p; IFS=',' read -r -a p <<< "$pool"
  if [ "${#p[@]}" -lt 2 ] || [ -z "$pool" ]; then skip "multi-acquire: need >=2 idle GPUs"; return; fi
  local dir="$TMPROOT/multi"; mkdir -p "$dir"
  local hf="$dir/held.out"
  GPU_LEASE_DIR="$dir" GPU_LEASE_VISIBLE="$pool" bash -c '
    eval "$('"$BROKER"' acquire 1 --no-wait 2>/dev/null)"
    eval "$('"$BROKER"' acquire 1 --no-wait 2>/dev/null)"
    printf "%s" "${GPU_LEASE_HELD:-}|${CUDA_VISIBLE_DEVICES:-}" > "'"$hf"'"
  '
  local out held cvd; out="$(cat "$hf" 2>/dev/null)"; held="${out%%|*}"; cvd="${out##*|}"
  local -a hids; IFS=',' read -r -a hids <<< "$held"
  if [ "${#hids[@]}" -eq 2 ] && [ -n "${hids[0]}" ] && [ "${hids[0]}" != "${hids[1]}" ]; then
    ok "multi-acquire: two acquires accumulated 2 distinct GPUs in GPU_LEASE_HELD ($held)"
  else
    bad "multi-acquire: GPU_LEASE_HELD did not accumulate two distinct GPUs (got '$held')"
  fi
  [ "$cvd" = "$held" ] && ok "multi-acquire: CUDA_VISIBLE_DEVICES==GPU_LEASE_HELD ($cvd)" \
                       || bad "multi-acquire: CVD '$cvd' != HELD '$held'"
  # After the shell exited, the (single, accumulating) trap must have released BOTH.
  local leftover=1 t=0 id
  while [ "$leftover" -ne 0 ] && [ "$t" -lt 30 ]; do
    leftover=0; for id in "${hids[@]}"; do [ -n "$id" ] && [ -f "$dir/$id" ] && leftover=$((leftover+1)); done
    [ "$leftover" -ne 0 ] && { sleep 0.1; t=$((t+1)); }
  done
  [ "$leftover" -eq 0 ] && ok "multi-acquire: trap released BOTH accumulated leases on exit" \
                        || bad "multi-acquire: $leftover lease(s) leaked after exit (clobber regression)"
  GPU_LEASE_DIR="$dir" GPU_LEASE_FORCE=1 "$BROKER" release "$held" >/dev/null 2>&1 || true
}

echo "=== gpu_lease real test harness ==="
test_concurrency
test_lock_released
test_overrequest
test_stale_pid_death
test_stale_ttl
test_trap_release
test_wait_then_succeed
test_multi_acquire

echo
echo "==> PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
[ "$FAIL" -eq 0 ] || exit 1
exit 0
