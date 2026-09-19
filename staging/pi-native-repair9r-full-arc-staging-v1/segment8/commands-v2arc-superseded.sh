#!/usr/bin/env bash
# ============================================================================
# repair-v9r full-arc SEGMENT 8 of 8 — STAGED COMMAND SHEET (NOT EXECUTED).
#
# The v9-full arc stopped fail-closed at seg1 (raw-OpenHands protocol capture);
# this arc restarts on the translated + replay-verified OH collection (prep v2).
# Every step is fail-closed: a failing step stops the sequence; NO automatic
# retries; NO threshold changes; evaluation protocol unchanged.
#
# Segment sampler key 1400081 (screened order); segment schedule schedule-segment8-proposal.json.
# Juncture at u1024 (Stage-B panel + conversation NLL probe; full frozen dual
# gates at u256/u512/u896/u1024 (THIS SEGMENT ENDS AT THE u1024 FULL FROZEN DUAL GATE)); fail-closed stopping armed.
#
# *** ADMISSION (step 4) REQUIRES THE OPERATOR'S EXPLICIT SIGN-OFF for the new
# *** data authority: the authorization statement is typed by the operator.
# ============================================================================
set -euo pipefail
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
PREP=$P/pi-native-repair9-full-preparation-v2
ST=$P/pi-native-repair9r-full-arc-staging-v1
PROP=$REPO/configs/pi/e97-pi-native-repair9r-full-arc-segment8-training-proposal-v1.json
ADM=$P/pi-native-repair9r-full-arc-segment8-training-admission-v1
RUN=$P/pi-native-repair9r-full-arc-segment8-training-v1
SCHED=$PREP/schedule-segment8-proposal.json
KEY=1400081
SRC_COMMIT=d77dcc462e8b25c0b362b196347ddc9f50ffe683
ARGS=/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json

# ---- STEP 0: resolve the parent (fail-closed) segment 7's final audited checkpoint; the freeze-time writer and the generic auditor both verify the chained parent.
PREV=$P/pi-native-repair9r-full-arc-segment7-training-v1
PARENT=$(readlink -f "$PREV/checkpoints/latest.pt")
PARENT_SHA=$(sha256sum "$PARENT" | cut -d' ' -f1)
"$PY" - "$PREV" "$PARENT" "$PARENT_SHA" <<'EOF'
import json,sys
prev,parent,sha=sys.argv[1:4]
audit=json.load(open(prev+"/audit.json"))
if audit.get("status")!="passed-training-not-promoted" or audit.get("checkpoint",{}).get("sha256")!=sha:
    raise SystemExit("prior segment training audit does not verify parent "+parent)
print("PARENT_AUDITED",sha)
EOF
# ---- STEP 1: freeze the segment proposal (writes $PROP)
"$PY" "$ST/freeze_segment_proposal.py" --segment-index 8 \
  --parent-checkpoint "$PARENT" --parent-sha256 "$PARENT_SHA" \
  --output "$PROP"

# ---- STEP 2: audit the proposal (MUST PASS; deterministic generic auditor)
cd "$REPO"
"$PY" scripts/audit_e97_pi_native_repair_generic.py --proposal "$PROP" \
  --schema emender-e97-pi-native-repair9r-full-arc-segment8-proposal-v1 \
  --schedule "$SCHED" \
  --output "$ST/proposal-audits/segment8-proposal-audit.json"

# ---- STEP 3: scoped commit + push of the frozen proposal
git add "$PROP"
git commit -m "Freeze repair-v9r full-arc segment 8 proposal"
git push origin main

# ---- STEP 4: ADMISSION — OPERATOR SIGN-OFF REQUIRED (new data authority).
# The operator types: I authorize the exact 128 update proposal.
PROPOSAL_SHA=$(sha256sum "$PROP" | cut -d' ' -f1)
"$PY" "$REPO/scripts/admit_e97_pi_native_training_proposal.py" --proposal "$PROP" \
  --proposal-sha256 "$PROPOSAL_SHA" --preparation "$PREP" --output "$ADM" \
  --authorization-statement "I authorize the exact 128 update proposal."

# ---- STEP 5: admitted schedule + identity check against the frozen plan
AUTH_SHA=$(sha256sum "$ADM/manifest.json" | cut -d' ' -f1)
PACK_SHA=$(sha256sum "$ADM/packs/manifest.json" | cut -d' ' -f1)
"$PY" "$REPO/scripts/plan_e97_pi_native_training_schedule.py" --authority "$ADM" \
  --packs "$ADM/packs" --authority-sha256 "$AUTH_SHA" --pack-sha256 "$PACK_SHA" \
  --sampler-key $KEY --steps 128 --output "$ADM/expected-schedule.json" \
  --admission "$ADM/admission.json"
"$PY" - "$SCHED" "$ADM/expected-schedule.json" <<'EOF'
import json,sys
frozen=json.load(open(sys.argv[1]));admitted=json.load(open(sys.argv[2]))
assert frozen['steps']==admitted['steps'],'admitted schedule deviates from the frozen plan'
assert frozen['source_target_totals']==admitted['source_target_totals']
assert frozen['source_unique_records']==admitted['source_unique_records']
print('ADMITTED_SCHEDULE_MATCHES_FROZEN_PLAN')
EOF

# ---- STEP 6: instantiate the training run dir + launch (GPU lease; 128 updates)
bash "$ST/templates/instantiate-run-dir.sh" --segment 8 \
  --parent "$PARENT" --parent-sha256 "$PARENT_SHA" --admission "$ADM" \
  --proposal "$PROP" --args-json "$ARGS" --source-commit "$SRC_COMMIT"
bash "$RUN/run.sh"

# ---- STEP 7: training audit (writes $RUN/audit.json — REQUIRED for the next
#      segment's chained-parent verification; status must be
#      'passed-training-not-promoted')
"$PY" "$ST/templates/write-recipe.py" --run "$RUN" --admission "$ADM" \
  --proposal "$PROP" --parent "$PARENT" --source-commit "$SRC_COMMIT"
"$PY" "$REPO/scripts/audit_e97_pi_native_training_run.py" --run "$RUN" \
  --admission "$ADM/admission.json" --output "$RUN/audit.json"

# ---- STEP 8: juncture evaluation at u1024 (Stage-B panel + conversation NLL
#      probe; regenerate panel.json per checkpoint first — v9-full defect class;
#      re-baseline the conversation-retention NLL panel at the reasoning
#      cohort's first entry). Then run the FULL FROZEN DUAL GATE (uu1024) before any continuation.
bash "$ST/templates/instantiate-juncture-dir.sh" --segment 8 --updates 1024
# regenerate $P/pi-native-repair9r-full-arc-juncture-evals/seg8-u1024/panel.json
# for the exact audited checkpoint, then:
# bash $P/pi-native-repair9r-full-arc-juncture-evals/seg8-u1024/stage-juncture.sh

# ---- STEP 9: decision point — frozen juncture policy:
#      full frozen dual gate at u256/u512/u896/u1024; fail-closed stopping if retention is
#      decisively breached; no automatic retries; no threshold changes.
#      If continuing: run segment no further (arc complete at u1024) command sheet (chained to this
#      segment's audited checkpoint). If this is a gate juncture, run the full
#      frozen dual-gate evaluation before any continuation.
