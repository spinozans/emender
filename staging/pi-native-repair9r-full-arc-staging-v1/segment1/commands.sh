#!/usr/bin/env bash
# ============================================================================
# repair-v9r V3 RESTART arc — SEGMENT 1 of 8 — STAGED COMMAND SHEET (NOT EXECUTED).
#
# The v10 (v2-prep) arc stopped fail-closed at u256 (execution 0/96; the scrub
# over-correction removed the execution dialect's training data). This arc
# restarts from the bridge parent on the v3 preparation: translated +
# replay-verified OH collection PLUS the grounded-authored and
# representation-bridge cohorts RESTORED IN FULL; every other cohort and
# budget identical to the v2 prep. Every step is fail-closed: a failing step
# stops the sequence; NO automatic retries; NO threshold changes; evaluation
# protocol unchanged.
#
# Segment sampler key 1401011 (screened order, fresh range 1401002..); segment
# schedule schedule-proposal.json. Juncture at u128 (Stage-B panel + conversation NLL probe;
# full frozen dual gates at u256/u512/u896/u1024); fail-closed
# stopping armed.
#
# *** ADMISSION (step 4) REQUIRES THE OPERATOR'S EXPLICIT SIGN-OFF for the new
# *** data authority: the authorization statement is typed by the operator.
# ============================================================================
set -euo pipefail
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
PREP=$P/pi-native-repair9-full-preparation-v3
ST=$P/pi-native-repair9r-full-arc-staging-v1
PROP=$REPO/configs/pi/e97-pi-native-repair9r-full-arc-v3-segment1-training-proposal-v1.json
ADM=$P/pi-native-repair9r-full-arc-v3-segment1-training-admission-v1
RUN=$P/pi-native-repair9r-full-arc-v3-segment1-training-v1
SCHED=$PREP/schedule-proposal.json
KEY=1401011
ARC_KEYS="1401011,1401042,1401043,1401046,1401052,1401056,1401057,1401074"
SRC_COMMIT=d77dcc462e8b25c0b362b196347ddc9f50ffe683
ARGS=/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json

# ---- STEP 0: resolve the parent (fail-closed) bridge parent, full retention headroom (v10 restarts from the bridge parent).
PARENT=$P/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt
PARENT_SHA=9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa
# ---- STEP 1: freeze the segment proposal (writes $PROP)
"$PY" "$ST/freeze_segment_proposal.py" --segment-index 1 \
  --keys "$ARC_KEYS" \
  --output "$PROP"

# ---- STEP 2: audit the proposal (MUST PASS; deterministic generic auditor)
cd "$REPO"
"$PY" scripts/audit_e97_pi_native_repair_generic.py --proposal "$PROP" \
  --schema emender-e97-pi-native-repair9r-full-arc-v3-segment1-proposal-v1 \
  --schedule "$SCHED" \
  --output "$ST/proposal-audits/segment1-v3-proposal-audit.json"

# ---- STEP 3: scoped commit + push of the frozen proposal
git add "$PROP"
git commit -m "Freeze repair-v9r v3-restart arc segment 1 proposal (scrub-reversal prep)"
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
bash "$ST/templates/instantiate-run-dir.sh" --segment 1 --arc-suffix -v3 \
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

# ---- STEP 8: juncture evaluation at u128 (Stage-B panel + conversation NLL
#      probe; regenerate panel.json per checkpoint first — v9-full defect class;
#      re-baseline the conversation-retention NLL panel at the reasoning
#      cohort's first entry). 
bash "$ST/templates/instantiate-juncture-dir.sh" --segment 1 --updates 128 --arc-suffix -v3
# regenerate $P/pi-native-repair9r-full-arc-v3-juncture-evals/seg1-u128/panel.json
# for the exact audited checkpoint, then:
# bash $P/pi-native-repair9r-full-arc-v3-juncture-evals/seg1-u128/stage-juncture.sh

# ---- STEP 9: decision point — frozen juncture policy:
#      full frozen dual gate at u256/u512/u896/u1024; fail-closed stopping if retention is
#      decisively breached; no automatic retries; no threshold changes.
#      If continuing: run segment 2 command sheet (chained to this
#      segment's audited checkpoint). If this is a gate juncture, run the full
#      frozen dual-gate evaluation before any continuation.
