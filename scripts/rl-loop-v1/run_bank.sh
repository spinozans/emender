#!/usr/bin/env bash
# e97-rl-loop-v1 TRANCHE 3: launch the 8-GPU DiLoCo-synced bank on the full
# (admitted) first-party gym and run the first sustained multi-lane session.
#
#   preflight (CPU): pinned-validator binding + unit validation
#   -> lane init: all 8 lanes seeded from the loop's current checkpoint
#   -> coordinator (CPU): K=32 DiLoCo sync, stale-claim sweep, pool rounds
#   -> 8 lane processes: claim -> serve -> attempt -> grade -> teacher-correct
#      -> receipts -> train -> re-serve, yield-when-idle GPU leases
#   -> SESSION_MINUTES bounded session -> STOP -> final report
#      (SESSION_MINUTES=0 => INDEFINITE: the monitor loop NEVER touches STOP;
#       clean stop is `touch $BANK/STOP` — lanes exit at cycle boundaries,
#       the coordinator at its next poll)
#
# Shared box (lambda01, co-user shuoc): lanes hold GPU leases ONLY around
# cycles with claimed work and release at every cycle boundary.
set -euo pipefail
umask 077

W=/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
export PYTHONPATH=$REPO:$W/scripts PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export GPU_LEASE_DIR=$REPO/.wg/gpu_leases

BANK=${BANK:-$W/bank}
LAKE=/mnt/nvme2n1/erikg/task_lake/e97-firstparty-cpu-phase-bc-v1-admitted
ARGS_JSON=/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json
LANES=${LANES:-8}
# ---- task F4 parallel collection (operator pre-approval 2026-10-08): with
#      COLLECTOR_LANES=N>0 the bank runs the one learner lane (lane 0,
#      LANES=1) plus N collector-only lanes (lanes 1..N). Collectors claim
#      pool work and form/chain receipts (teacher-corrected and
#      on-policy-success alike) but never train, probe, or adopt a merge
#      (COLLECTOR_ONLY=1 in rl_bank_lane.py). Their lane-local streams feed
#      the learner's block channel through the streams union
#      (BLOCK_STREAMS_ROOT defaults to $BANK/lanes). Unset (default 0) is
#      byte-identical current behavior. The coordinator watches ONLY the
#      learner lane — with COLLECTOR_LANES>0, LANES is pinned to 1 so the
#      --lanes "$LANES" below binds exactly 1, and collectors never enter
#      the K-merge geometry or the per-lane reports.
COLLECTOR_LANES=${COLLECTOR_LANES:-0}
if [ "$COLLECTOR_LANES" -gt 0 ] && [ "$LANES" -ne 1 ]; then
  echo "run_bank.sh: COLLECTOR_LANES requires the single-learner bank (LANES=1)" >&2
  exit 1
fi
K=${K:-32}
SESSION_MINUTES=${SESSION_MINUTES:-75}
if [ "$SESSION_MINUTES" -gt 0 ]; then
  MAX_SECONDS=$((SESSION_MINUTES * 60))
else
  # indefinite operation (operator directive): bounded only by STOP + a
  # 30-day R14 hard bound so nothing can wedge forever
  MAX_SECONDS=$((30 * 86400))
fi

log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$BANK/launch.log"; }

mkdir -p "$BANK"

log "[preflight] pinned validator + CPU unit validation"
$PY "$W/scripts/rl_bank_unit_test.py" --scratch "$BANK/unit-test" \
    > "$BANK/unit-test.log" 2>&1
tail -2 "$BANK/unit-test.log"
# NOTE: this literal tracks rl_bank_unit_test.py's current total (52 at
# authoring time; 101 after the 2026-10-07 audit fixes added 49 regressions;
# 138 after task F4 added 18 parallel-collection regressions;
# 162 after task F5 added 24 PG-grid regressions;
# 184 after F7 added 22 bounded-correction/config regressions).
grep -q '"passed": 184' "$BANK/unit-test.log" || {
  log "[preflight] UNIT VALIDATION FAILED"; exit 1; }
log "[preflight] unit validation PASSED (184/184)"

rm -f "$BANK/STOP"

log "[init] seed $LANES learner + $COLLECTOR_LANES collector lanes from the loop's current checkpoint"
read -r SEED_CKPT SEED_SHA < <($PY - <<'PYEOF'
import json
s = json.load(open("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/state/current_checkpoint.json"))
print(s["checkpoint_path"], s["checkpoint_sha256"])
PYEOF
)
for LANE in $(seq 0 $((LANES + COLLECTOR_LANES - 1))); do
  $PY "$W/scripts/rl_bank_lane.py" init --bank "$BANK" --lane "$LANE" \
      --checkpoint "$SEED_CKPT" --checkpoint-sha256 "$SEED_SHA" \
      --note "bank session seed (proven v1 loop current checkpoint)" \
      >> "$BANK/launch.log" 2>&1
done
log "[init] seeded from $SEED_CKPT ($SEED_SHA)"

log "[launch] coordinator (K=$K) + $LANES learner + $COLLECTOR_LANES collector"
log "[launch] lanes; session ${SESSION_MINUTES}m"
nohup $PY "$W/scripts/rl_bank_coordinator.py" --bank "$BANK" \
    --lanes "$LANES" --k "$K" --lake "$LAKE" \
    --max-seconds $((MAX_SECONDS + 600)) \
    > "$BANK/coordinator.log" 2>&1 &
COORD_PID=$!
for LANE in $(seq 0 $((LANES + COLLECTOR_LANES - 1))); do
  # task F4: lanes at or above $LANES are collector-only — they collect and
  # chain receipts; training, probing, and merge adoption are disabled by
  # COLLECTOR_ONLY=1 inside rl_bank_lane.py (explicit early-out, not a
  # threshold hack); lineage stays at the init seed.
  LANE_COLLECTOR_ENV=()
  if [ "$LANE" -ge "$LANES" ]; then
    LANE_COLLECTOR_ENV=(env COLLECTOR_ONLY=1)
  fi
  # v2 RL-regime ruling (operator, 2026-09-27): LANES_ON_POLICY_GRADIENT is a
  # comma list of lane numbers that run the policy-gradient train step
  # (e.g. "0,1,2,3,4,5,6,7" for the full-RL bank); unlisted lanes keep the
  # default receipts-SFT channel.  Default: EMPTY (no lane changes).
  LANE_TRAIN_ARGS=()
  if [ -n "${LANES_ON_POLICY_GRADIENT:-}" ] && \
     grep -qw "$LANE" <<<"${LANES_ON_POLICY_GRADIENT//,/ }"; then
    LANE_TRAIN_ARGS=(--lane-train-step policy-gradient)
  fi
  # ---- v3 corpus-replay anchor (operator ruling 2026-09-29: RL shaping on
  #      the corpus checkpoint; the gym is a shaping layer on a
  #      corpus-dominant diet, never the whole meal - the merge-5 fix):
  #      every ANCHOR_PERIOD-th cycle the lane trains on a pack from the
  #      frozen admitted corpus (ANCHOR_* env; empty ANCHOR_PERIOD = off).
  LANE_ANCHOR_ARGS=()
  if [ -n "${ANCHOR_PERIOD:-}" ] && [ "${ANCHOR_PERIOD}" -gt 0 ]; then
    LANE_ANCHOR_ARGS=(--anchor-period "${ANCHOR_PERIOD}"
                      --anchor-authority-root "${ANCHOR_AUTHORITY_ROOT}"
                      --anchor-authority-sha256 "${ANCHOR_AUTHORITY_SHA256}"
                      --anchor-pack-root "${ANCHOR_PACK_ROOT}"
                      --anchor-pack-sha256 "${ANCHOR_PACK_SHA256}"
                      --anchor-keys "${ANCHOR_KEYS}")
  fi
  nohup "${LANE_COLLECTOR_ENV[@]}" $PY "$W/scripts/rl_bank_lane.py" run --bank "$BANK" --lane "$LANE" \
      --args-json "$ARGS_JSON" --teacher live --max-tasks 4 \
      --max-seconds $MAX_SECONDS "${LANE_TRAIN_ARGS[@]}" \
      "${LANE_ANCHOR_ARGS[@]}" \
      > "$BANK/lane-$(printf '%02d' "$LANE").log" 2>&1 &
done
log "[launch] coordinator pid=$COORD_PID; lanes launched"

monitor_once() {
  SUMMARY=$($PY - "$BANK/report/session-metrics.json" <<'PYEOF' 2>/dev/null || echo "(report not yet written)"
import json, sys
report = json.load(open(sys.argv[1]))
print(json.dumps({
    "lanes_live": report["lanes_live"],
    "receipts_by_kind": report["receipts_by_kind"],
    "pool": {k: report["pool"][k] for k in ("pending", "claims_live", "done")},
    "updates_total": sum(l.get("updates_total", 0) for l in report["lanes"]),
    "merge_events": len(report["merge_events"]),
    "rounds": len(report["pass_rate_trajectory"]),
}, sort_keys=True))
PYEOF
)
  log "MONITOR $SUMMARY"
}

if [ "$SESSION_MINUTES" -gt 0 ]; then
  DEADLINE=$(( $(date +%s) + SESSION_MINUTES * 60 ))
  while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    sleep 60
    monitor_once
  done
  log "[stop] session window elapsed; touching STOP"
  touch "$BANK/STOP"
else
  log "[launch] INDEFINITE session (operator directive): the bank runs until" \
      "$BANK/STOP is touched; clean stop: touch $BANK/STOP"
  while [ ! -f "$BANK/STOP" ]; do
    sleep 60
    monitor_once
  done
  log "[stop] STOP observed; waiting for graceful exit"
fi
sleep 30
# lanes/coordinator exit at their boundaries; final report is theirs
log "[stop] waiting for graceful exit (bounded 1800s — lanes finish their" \
    "in-flight cycles at the STOP boundary; no state is lost)"
for _ in $(seq 1 180); do
  if ! pgrep -f "rl_bank_lane.py run" >/dev/null && \
     ! pgrep -f "rl_bank_coordinator.py" >/dev/null; then
    log "[stop] all bank processes exited cleanly"
    break
  fi
  sleep 10
done
pgrep -f "rl_bank_lane.py run" >/dev/null && log "[stop] WARNING: lane processes still alive"
pgrep -f "rl_bank_coordinator.py" >/dev/null && log "[stop] WARNING: coordinator still alive"

log "[report] final session metrics"
$PY - "$BANK" "$LANES" <<'PYEOF'
import json, sys, time
sys.path.insert(0, "/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-rl-loop-v1/scripts")
from rl_bank import bank_paths
from rl_bank_coordinator import assemble_session_metrics
bank = bank_paths(sys.argv[1])
report = assemble_session_metrics(bank, int(sys.argv[2]), started_unix=time.time())
from rl_bank import atomic_write_json
atomic_write_json(bank["report"] / "session-metrics-final.json", report)
print(json.dumps({
    "lanes_live": report["lanes_live"],
    "receipts_by_kind": report["receipts_by_kind"],
    "tasks_completed_per_lane": report["tasks_completed_per_lane"],
    "teacher_calls_total": report["teacher_calls_total"],
    "step_losses": report["step_losses"],
    "merge_events": report["merge_events"],
    "pass_rate_trajectory": report["pass_rate_trajectory"],
}, sort_keys=True, indent=1))
PYEOF
log "[report] RUN_BANK session complete (bank/report/session-metrics-final.json)"
