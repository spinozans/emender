#!/usr/bin/env python3
"""Emit per-segment staged command sheets for the repair-v9r V3 RESTART arc
(segments 1..8). STAGED; NOT EXECUTED.

Regenerated for the v3 preparation (scrub-reversal remediation): the v10 arc
stopped fail-closed at u256 because the v2 prep's scrub over-correction
removed the execution dialect's training data; the v3 prep restores the
grounded-authored and representation-bridge cohorts in full and keeps the
translated + replay-verified OH collection. Fresh screened keys (the v2-arc
keys are invalid for the new manifest sha). The v2-arc sheets are retained as
segment{N}/commands-v2arc-superseded.sh.
"""
import argparse
from pathlib import Path

ST = Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9r-full-arc-staging-v1')

TEMPLATE = r'''#!/usr/bin/env bash
# ============================================================================
# repair-v9r V3 RESTART arc — SEGMENT {N} of 8 — STAGED COMMAND SHEET (NOT EXECUTED).
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
# Segment sampler key {KEY} (screened order, fresh range 1401002..); segment
# schedule {SCHED}. Juncture at u{U} (Stage-B panel + conversation NLL probe;
# full frozen dual gates at u256/u512/u896/u1024{GATE_LINE}); fail-closed
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
PROP=$REPO/configs/pi/e97-pi-native-repair9r-full-arc-v3-segment{N}-training-proposal-v1.json
ADM=$P/pi-native-repair9r-full-arc-v3-segment{N}-training-admission-v1
RUN=$P/pi-native-repair9r-full-arc-v3-segment{N}-training-v1
SCHED=$PREP/{SCHED}
KEY={KEY}
ARC_KEYS="{KEYS}"
SRC_COMMIT=d77dcc462e8b25c0b362b196347ddc9f50ffe683
ARGS=/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json

# ---- STEP 0: resolve the parent (fail-closed) {PARENT_COMMENT}
{PARENT_BLOCK}
# ---- STEP 1: freeze the segment proposal (writes $PROP)
"$PY" "$ST/freeze_segment_proposal.py" --segment-index {N}{PARENT_ARGS} \
  --keys "$ARC_KEYS" \
  --output "$PROP"

# ---- STEP 2: audit the proposal (MUST PASS; deterministic generic auditor)
cd "$REPO"
"$PY" scripts/audit_e97_pi_native_repair_generic.py --proposal "$PROP" \
  --schema emender-e97-pi-native-repair9r-full-arc-v3-segment{N}-proposal-v1 \
  --schedule "$SCHED" \
  --output "$ST/proposal-audits/segment{N}-v3-proposal-audit.json"

# ---- STEP 3: scoped commit + push of the frozen proposal
git add "$PROP"
git commit -m "Freeze repair-v9r v3-restart arc segment {N} proposal (scrub-reversal prep)"
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
bash "$ST/templates/instantiate-run-dir.sh" --segment {N} --arc-suffix -v3 \
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

# ---- STEP 8: juncture evaluation at u{U} (Stage-B panel + conversation NLL
#      probe; regenerate panel.json per checkpoint first — v9-full defect class;
#      re-baseline the conversation-retention NLL panel at the reasoning
#      cohort's first entry). {JUNCTURE_NOTE}
bash "$ST/templates/instantiate-juncture-dir.sh" --segment {N} --updates {U} --arc-suffix -v3
# regenerate $P/pi-native-repair9r-full-arc-v3-juncture-evals/seg{N}-u{U}/panel.json
# for the exact audited checkpoint, then:
# bash $P/pi-native-repair9r-full-arc-v3-juncture-evals/seg{N}-u{U}/stage-juncture.sh

# ---- STEP 9: decision point — frozen juncture policy:
#      full frozen dual gate at {GATES}; fail-closed stopping if retention is
#      decisively breached; no automatic retries; no threshold changes.
#      If continuing: run segment {NEXT} command sheet (chained to this
#      segment's audited checkpoint). If this is a gate juncture, run the full
#      frozen dual-gate evaluation before any continuation.
'''

BRIDGE_PARENT = '''PARENT=$P/representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt
PARENT_SHA=9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa'''
CHAINED_PARENT = '''PREV=$P/pi-native-repair9r-full-arc-v3-segment__PREV__-training-v1
PARENT=$(readlink -f "$PREV/checkpoints/latest.pt")
PARENT_SHA=$(sha256sum "$PARENT" | cut -d' ' -f1)
"$PY" - "$PREV" "$PARENT" "$PARENT_SHA" <<'EOF'
import json,sys
prev,parent,sha=sys.argv[1:4]
audit=json.load(open(prev+"/audit.json"))
if audit.get("status")!="passed-training-not-promoted" or audit.get("checkpoint",{}).get("sha256")!=sha:
    raise SystemExit("prior segment training audit does not verify parent "+parent)
print("PARENT_AUDITED",sha)
EOF'''

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, default=ST)
    ap.add_argument('--keys', default='1401011,1401042,1401043,1401046,1401052,1401056,1401057,1401074',
                    help='comma-separated screened arc keys in segment order')
    a = ap.parse_args()
    keys = [int(k) for k in a.keys.split(',') if k.strip()]
    if len(keys) != 8:
        raise SystemExit('exactly 8 keys required')
    gates = {2: 'u256', 4: 'u512', 7: 'u896', 8: 'u1024'}
    for n in range(1, 9):
        sched = 'schedule-proposal.json' if n == 1 else f'schedule-segment{n}-proposal.json'
        u = 128 * n
        gate_line = f' (THIS SEGMENT ENDS AT THE {gates[n]} FULL FROZEN DUAL GATE)' if n in gates else ''
        juncture_note = ('' if n not in gates else
                         'Then run the FULL FROZEN DUAL GATE (u%s) before any continuation.' % gates[n])
        if n == 1:
            parent_comment = 'bridge parent, full retention headroom (v10 restarts from the bridge parent).'
            parent_block = BRIDGE_PARENT
            parent_args = ''
        else:
            parent_comment = (f'segment {n-1}\'s final audited checkpoint; the freeze-time writer '
                              f'and the generic auditor both verify the chained parent.')
            parent_block = CHAINED_PARENT.replace('__PREV__', str(n - 1))
            parent_args = ' \\\n  --parent-checkpoint "$PARENT" --parent-sha256 "$PARENT_SHA"'
        text = TEMPLATE.format(N=n, KEY=keys[n - 1], KEYS=','.join(str(k) for k in keys), SCHED=sched, U=u,
                               GATE_LINE=gate_line, PARENT_COMMENT=parent_comment, PARENT_BLOCK=parent_block,
                               PARENT_ARGS=parent_args, JUNCTURE_NOTE=juncture_note,
                               NEXT=n + 1 if n < 8 else 'no further (arc complete at u1024)',
                               GATES='u256/u512/u896/u1024')
        d = a.output_root / f'segment{n}'
        d.mkdir(parents=True, exist_ok=True)
        (d / 'commands.sh').write_text(text)
        print('wrote', d / 'commands.sh')

if __name__ == '__main__':
    main()
