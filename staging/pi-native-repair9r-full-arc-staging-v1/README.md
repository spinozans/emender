# repair-v9r full-arc staging (v1) — NOT EXECUTED

Staging directory for the v9r full-arc restart (v9-full stopped fail-closed at
segment 1 on raw-OpenHands protocol capture). Everything here is staged for the
operator's sign-off; **nothing in this directory has been executed**, and no
admission or training has occurred. The new data authority (the translated +
replay-verified OpenHands collection inside the v2 preparation) requires
explicit operator authorization before any admission step runs.

## Contents

- `freeze_segment_proposal.py` — fail-closed freeze-time proposal writer for
  segments 1..8. Segment 1 binds the bridge parent; segments 2..8 accept the
  prior segment's real audited checkpoint and verify its training audit
  (status `passed-training-not-promoted`, exact sha) before writing anything.
  Every bound identity is re-hashed from disk at freeze time.
- `segment{1..8}/commands.sh` — the exact per-segment command sheets
  (freeze -> audit -> commit+push -> ADMISSION [operator sign-off] ->
  admitted-schedule regeneration + frozen-plan identity check -> launch ->
  training audit -> juncture evaluation -> decision point). Not executed.
  Segment 1 steps 1-3 are already complete (see its header): proposal sha256
  `063b3f7567ce5306791f2ec3c745a7b732691ef442ee4eb577bc77bb0647e351`,
  audit receipt sha256
  `dc96fe255ef26b4b77a3dbeed5c4338037bc496dafa9bec4f6b6c4a2fa5e16bf`
  (`proposal-audits/segment1-proposal-audit.json`), commit `2358816f` (pushed).
- `proposal-audits/` — generic-auditor receipts. Segment 1 filled; segments
  2..8 are written at their freeze points (chained-parent discipline, matching
  the v5/v7/v8/v9 precedent: a segment proposal is frozen and audited only
  when every identity it binds — including its parent checkpoint — is real).
- `templates/instantiate-run-dir.sh` — renders the per-segment training run
  directory (detached worktree at the pinned trainer commit
  `d77dcc462e8b25c0b362b196347ddc9f50ffe683`, source closure manifest from
  `templates/source-files.txt` (the v9-full 607-file closure), input manifest,
  and the launcher `run.sh` modeled on the v9-full segment-1 launcher with the
  qualified chunk-2048/16384 recipe).
- `templates/write-recipe.py` — post-training recipe.json writer; every field
  is re-derived and re-hashed from run artifacts.
- `templates/instantiate-juncture-dir.sh` — renders the per-juncture eval
  directory (Stage-B panel + conversation NLL probe) with a fail-closed
  panel-binding guard against the recorded v9-full defect (a copied probe panel
  silently evaluated the wrong checkpoint; panels must be regenerated per
  checkpoint).
- `segment1-proposal-draft-from-prior-worker.json` — the prior worker's untracked
  draft, retained for forensics. Its field set was complete against the generic
  auditor except the missing `proposed_behavioral_gate`; the frozen segment-1
  proposal adds the frozen gate dict plus extended provenance/overlap-audit
  identity bindings and cohort bindings.

## Operator sign-off sequence (per segment)

1. Approve/execute `segment{N}/commands.sh` step by step; no automatic retries.
2. Step 4 (admission) is the authorization gate: the operator types
   `I authorize the exact 128 update proposal.` and the admission script
   re-verifies the proposal sha, preparation identity, and non-trainable state
   before flipping eligibility.
3. Before each probe run, regenerate `panel.json` for the exact audited
   checkpoint (the staged `run-probe.sh` refuses un-bound panels).
4. T2 caveat: the conversation-retention NLL panel needs re-baselining when
   the reasoning cohort enters training (segment 1 of this arc).

## Standing constraints honored

- No threshold changes; frozen dual gates unchanged
  (Stage-B `valid first frame >= 12/14 and correct first action >= 10/14`;
  execution `>= 64/96`; composition `>= 4/16`; retention gates).
- No GPU touched by this staging; all work here is CPU-side file preparation.
- The generic auditor `scripts/audit_e97_pi_native_repair_generic.py` is
  authoritative and unmodified; proposal writers were fixed instead.
- Scoped git commits only (never `git add -A`).

## v3 restart regeneration (2026-09-19) — scrub-reversal remediation

The v10 (v2-prep) arc stopped fail-closed at u256 (execution 0/96; the scrub
over-correction removed the execution dialect's training data). The staging
machinery has been regenerated for the **v3 preparation**
(`pi-native-repair9-full-preparation-v3`, manifest sha256
`3c4f0b38ba107cabd4d0505ceb789bf2bba81ee2e11890421b7a163a96877f9b`):
grounded-authored (2,583 records) and representation-bridge (2,223 records)
cohorts restored in full; translated+replay-verified OH collection and every
other cohort/budget identical to the v2 prep. The chain restarts from the
bridge parent. Still STAGED, NOT EXECUTED.

- `freeze_segment_proposal.py` — rewritten for the v3 restart arc (bridge +
  authored authority bindings, restored-cohort cohort bindings, remediation
  evidence, `--keys` parameterized). The v2-arc writer is retained as
  `freeze_segment_proposal_v2arc-superseded.py`.
- `segment{1..8}/commands.sh` — regenerated for the v3 prep and the fresh
  screened keys (1401011, 1401042, 1401043, 1401046, 1401052, 1401056,
  1401057, 1401074; fresh range 1401002.. — the v2-arc keys are invalid for
  the new manifest sha). The v2-arc sheets are retained as
  `segment{N}/commands-v2arc-superseded.sh`.
- Template fixes (defect class found in review):
  1. `templates/instantiate-juncture-dir.sh` — the stage-juncture checkpoint
     glob now zero-pads the update number to six digits
     (`checkpoint_agent_sft_u000128_*.pt`); the old `u128*` glob matched
     nothing.
  2. `templates/instantiate-run-dir.sh` — every input binding (parent,
     admission, proposal, args json) is absolutized via `readlink -f` before
     being written to `input.sha256`, so `sha256sum --check` resolves them
     from the worktree cwd.
  3. Both templates take `--arc-suffix` (default `-v3`; `''` reproduces the
     legacy v10 dir names) so the restart arc's run/juncture dirs do not
     collide with the retained v10 evidence dirs.
- A full dry-run of the staged segment-1 freeze + generic audit was executed
  against an uncommitted DRAFT proposal (work dir, not configs/pi, not
  committed): freeze writer sha
  `4ca083afe0e5244812bbe37fc6502cbafba4a904b2ab1595f741cccbdd44db3b`, audit
  PASS receipt sha
  `30e6d5a4739949a5d3ff2106b741292fa1e23f9ad98af52d0a4fcb13a457a94c`.
  The real freeze/audit/commit remains the parent orchestrator's step with
  operator authorization.
