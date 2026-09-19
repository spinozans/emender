#!/usr/bin/env bash
# ============================================================================
# E97 DOSE SCREEN — D25 PROBE — STAGED COMMAND SHEET (NOT EXECUTED).
#
# Dose axis: the v2 prep (restored pools scrubbed to zero) gave Stage-B
# 9-10/14 + execution 0/96; the v3 prep (both restored at FULL pool) gave
# execution 59-60/96 + Stage-B 4-5/14. Both extremes fail the frozen gate
# (Stage-B >= 12/14 valid + >= 10/14 correct; execution >= 64/96). This probe
# trains 32 updates from the bridge parent on the d25 prep
# (pi-native-repair9-full-preparation-v3-d25: identical to v3 EXCEPT the restored
# pools are seeded first-N subsamples — grounded-authored 646 of 2,583 records (seed 882843), representation-bridge 556 of 2,223 records (seed 882844)) and measures
# Stage-B (14-case standard panel) + the 24-case execution slice.
#
# Every step is fail-closed; NO automatic retries; NO threshold changes; the
# frozen dual gates are unchanged. STAGE/ADMISSION/TRAIN/EVAL are executed by
# the parent orchestrator with the operator's authorization — never by the
# staging lane that wrote this sheet.
# ============================================================================
set -euo pipefail
REPO=/home/erikg/emender
PY=$REPO/.venv/bin/python
P=/mnt/nvme2n1/erikg/e97_systematic_posttraining
W=$P/pi-native-repair9-dose-screen-v1
PREP=$P/pi-native-repair9-full-preparation-v3-d25
KEY=1402002
PROP=$REPO/configs/pi/e97-pi-native-repair9-dose-d25-probe-proposal-v1.json
ADM=/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-probe-d25-admission-v1
PROBE=/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-probe-d25-v1
ARGS_JSON=/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json
SRC_COMMIT=d77dcc462e8b25c0b362b196347ddc9f50ffe683

# Precondition: the prep identity this sheet binds (fail closed if moved).
[[ "$(sha256sum "$PREP/manifest.json" | cut -d' ' -f1)" == "640ba1796da8e045d00e4222cf9368833b35b13805de7bccfe4b3467c2fdd8ce" ]]
[[ "$(sha256sum "$PREP/packs/manifest.json" | cut -d' ' -f1)" == "56582a3e284e6be6f87c58349fcb258221b1668cd57768b2719d093fe12e1e43" ]]
[[ "$(sha256sum "$PREP/schedule-$KEY-probe.json" | cut -d' ' -f1)" == "458f8bec65722ce635c3864b1178fc3af5fee0c0c4750ab60cb49882b383704a" ]]

# ---- STEP 1: freeze the d25 probe proposal (deterministic bytes; the
#      validated DRAFT sha is asserted fail-closed).
"$PY" "$W/freeze_dose_probe_proposal.py" --dose d25 --preparation "$PREP" \
  --schedule "$PREP/schedule-$KEY-probe.json" --key $KEY --output "$PROP"
[[ "$(sha256sum "$PROP" | cut -d' ' -f1)" == "232e6c041c5393d5640e5a81c9cd5eb012fd9f436655825f5562cf21112972df" ]]

# ---- STEP 2: audit the proposal (MUST print REPAIR_PROPOSAL_AUDIT; the
#      receipt must be the validated deterministic receipt).
cd "$REPO"
"$PY" scripts/audit_e97_pi_native_repair_generic.py --proposal "$PROP" \
  --schema emender-e97-pi-native-repair9-dose-probe-proposal-v1 \
  --schedule "$PREP/schedule-$KEY-probe.json" \
  --output "$W/proposal-audits/d25-probe-audit-final.json"
[[ "$(sha256sum "$W/proposal-audits/d25-probe-audit-final.json" | cut -d' ' -f1)" == "ba0e61888df0d3d78dedf9c4639a0802c9efa5f5fcc85017a5a9274a3c714f23" ]]

# ---- STEP 3: scoped commit + push of the frozen proposal (NEVER git add -A).
git add "$PROP"
git commit -m "Freeze repair-v9r dose-screen d25 probe proposal (32 updates, not authorized)"
git push origin main

# ---- STEP 4: ADMISSION — OPERATOR SIGN-OFF REQUIRED (new data authority).
# The operator types: I authorize the exact 32 update proposal.
PROP_SHA=$(sha256sum "$PROP" | cut -d' ' -f1)
"$PY" "$REPO/scripts/admit_e97_pi_native_training_proposal.py" --proposal "$PROP" \
  --proposal-sha256 "$PROP_SHA" --preparation "$PREP" --output "$ADM" \
  --authorization-statement "I authorize the exact 32 update proposal."

# ---- STEP 5: instantiate the probe run dir (worktree at $SRC_COMMIT,
#      expected-schedule check, run-train.sh + eval runners). NOT executed here.
bash "$W/d25/run-probe.sh" "$ADM"

# ---- STEP 6: execute the probe (GPU leases; 32 updates -> Stage-B -> slice).
bash "$PROBE/run-train.sh"
bash "$PROBE/run-stageb.sh"
bash "$PROBE/regen-execution-slice-panel.sh"
bash "$PROBE/run-execution-slice.sh"

# ---- STEP 7: readings (after ALL THREE probes; see $W/read-dose-results.py).
# "$PY" "$W/read-dose-results.py"
echo "D25_PROBE_SHEET_DONE — readings via $W/read-dose-results.py once all probes are evaluated"
