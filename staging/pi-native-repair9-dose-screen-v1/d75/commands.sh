#!/usr/bin/env bash
# ============================================================================
# E97 DOSE SCREEN — D75 PROBE — STAGED COMMAND SHEET (NOT EXECUTED).
#
# Dose axis: the v2 prep (restored pools scrubbed to zero) gave Stage-B
# 9-10/14 + execution 0/96; the v3 prep (both restored at FULL pool) gave
# execution 59-60/96 + Stage-B 4-5/14. Both extremes fail the frozen gate
# (Stage-B >= 12/14 valid + >= 10/14 correct; execution >= 64/96). This probe
# trains 32 updates from the bridge parent on the d75 prep
# (pi-native-repair9-full-preparation-v3-d75: identical to v3 EXCEPT the restored
# pools are seeded first-N subsamples — grounded-authored 1937 of 2,583 records (seed 882843), representation-bridge 1667 of 2,223 records (seed 882844)) and measures
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
PREP=$P/pi-native-repair9-full-preparation-v3-d75
KEY=1402001
PROP=$REPO/configs/pi/e97-pi-native-repair9-dose-d75-probe-proposal-v1.json
ADM=/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-probe-d75-admission-v1
PROBE=/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-probe-d75-v1
ARGS_JSON=/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json
SRC_COMMIT=d77dcc462e8b25c0b362b196347ddc9f50ffe683

# Precondition: the prep identity this sheet binds (fail closed if moved).
[[ "$(sha256sum "$PREP/manifest.json" | cut -d' ' -f1)" == "65ecd1c702baaa549c7a557dae87871a64ed808563deafc2f244e4004ab7495d" ]]
[[ "$(sha256sum "$PREP/packs/manifest.json" | cut -d' ' -f1)" == "1263aa9c00e4942328d7c6bb0a01f008cd9a1e11bfdfc07ecc9fed1ab04af125" ]]
[[ "$(sha256sum "$PREP/schedule-$KEY-probe.json" | cut -d' ' -f1)" == "8bf7dacd4a373db82b9a5318fc09c65d1cb795da7395aec5efc71fcbc15e0c03" ]]

# ---- STEP 1: freeze the d75 probe proposal (deterministic bytes; the
#      validated DRAFT sha is asserted fail-closed).
"$PY" "$W/freeze_dose_probe_proposal.py" --dose d75 --preparation "$PREP" \
  --schedule "$PREP/schedule-$KEY-probe.json" --key $KEY --output "$PROP"
[[ "$(sha256sum "$PROP" | cut -d' ' -f1)" == "420467233388203752a2e30d22ce455f4478a58e8474cff48e8afe27e4194384" ]]

# ---- STEP 2: audit the proposal (MUST print REPAIR_PROPOSAL_AUDIT; the
#      receipt must be the validated deterministic receipt).
cd "$REPO"
"$PY" scripts/audit_e97_pi_native_repair_generic.py --proposal "$PROP" \
  --schema emender-e97-pi-native-repair9-dose-probe-proposal-v1 \
  --schedule "$PREP/schedule-$KEY-probe.json" \
  --output "$W/proposal-audits/d75-probe-audit-final.json"
[[ "$(sha256sum "$W/proposal-audits/d75-probe-audit-final.json" | cut -d' ' -f1)" == "09eb6cc70820e1e99ae501e3f47700907c88a74255b6eca73d2d5937f8d86e1f" ]]

# ---- STEP 3: scoped commit + push of the frozen proposal (NEVER git add -A).
git add "$PROP"
git commit -m "Freeze repair-v9r dose-screen d75 probe proposal (32 updates, not authorized)"
git push origin main

# ---- STEP 4: ADMISSION — OPERATOR SIGN-OFF REQUIRED (new data authority).
# The operator types: I authorize the exact 32 update proposal.
PROP_SHA=$(sha256sum "$PROP" | cut -d' ' -f1)
"$PY" "$REPO/scripts/admit_e97_pi_native_training_proposal.py" --proposal "$PROP" \
  --proposal-sha256 "$PROP_SHA" --preparation "$PREP" --output "$ADM" \
  --authorization-statement "I authorize the exact 32 update proposal."

# ---- STEP 5: instantiate the probe run dir (worktree at $SRC_COMMIT,
#      expected-schedule check, run-train.sh + eval runners). NOT executed here.
bash "$W/d75/run-probe.sh" "$ADM"

# ---- STEP 6: execute the probe (GPU leases; 32 updates -> Stage-B -> slice).
bash "$PROBE/run-train.sh"
bash "$PROBE/run-stageb.sh"
bash "$PROBE/regen-execution-slice-panel.sh"
bash "$PROBE/run-execution-slice.sh"

# ---- STEP 7: readings (after ALL THREE probes; see $W/read-dose-results.py).
# "$PY" "$W/read-dose-results.py"
echo "D75_PROBE_SHEET_DONE — readings via $W/read-dose-results.py once all probes are evaluated"
