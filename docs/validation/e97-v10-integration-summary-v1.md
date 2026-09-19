# E97 v10 integration summary — repair-v9r full-arc restart on the translated OH authority (v1)

Status: staged for operator sign-off. Segment 1's proposal is frozen, audited,
committed and pushed. Admissions and launches are STAGED ONLY and require the
operator's explicit authorization for the new data authority (the translated +
replay-verified OpenHands collection). Nothing has been admitted or trained.
All work was CPU-only; the interactive-model GPU was untouched.

## 1. Preparation identity (the data authority being signed off)

- Preparation root:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-full-preparation-v2`
- Prep manifest sha256
  `24919e73ec2dfaf55ab09cfd5d8393ab25ae2e784de7247fe92c23fdc016d0e8`;
  packs manifest sha256
  `15813bf29d41068693494ea09e99e5bc15d13d7898f29a8066088dcb7bfca311`
  (`diagnostic_system_gate: cpu-system-gate`).
- Counts: **429,775 records / 324,349,514 tokens / 150,363,228 assistant
  target tokens**; 5,641 boundary-aware train packs; tokenizer `p50k_base`;
  schema `emender-e97-tulu3-masked-sft-v1`; `training_eligible: false`,
  `packing_authorized: false`, `optimizer_updates_authorized: 0`.
- Parent: representation-bridge checkpoint
  `checkpoint_agent_sft_u000032_loss_0.5302.pt`
  (sha256 `9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa`).
- Purpose (manifest, verbatim): translated replay-verified OpenHands rehearsal
  plus Pi-native curriculum, cohort-interleaved; representation-bridge and
  grounded-authored cohorts dropped per the OH scrub report.
- Provenance + overlap-audit coverage, bound and verified on disk (honest
  coverage statement; nothing invented):
  - The prep manifest binds the selected-curriculum provenance triple
    `selected_authority_sha256 = 8cf83db8d608f6af74bbd2cbe60206ef435fb3003ab2bb63f4eadab17eb05562`,
    `selected_selection_audit_sha256 = de977726f0758206ee9defe9c3687d2485e3ba2b80eb9ef9ceafb065c5f6646b`,
    `selected_overlap_audit_sha256 = 9c104d73b104f4d29a83ca1472b66ef2df681e020a2b9688eb4f87c6ef814a78`
    = `pi-native-curriculum-2000-selected-v1` (candidate-authority manifest,
    selection-audit.json, overlap-audit.json; all three verified byte-exact on
    disk). This is the same whole-prep coverage every prior repair prep bound
    (identical values in the repair8-preparation-v1 manifest). **No prior prep
    ever ran a whole-prep protected-overlap audit**; per-collection audits are
    the existing coverage pattern, and that is what exists and is bound here.
  - Per-collection coverage:
    - reasoning-rehearsal (T2 pilot): collection audit
      `f62c86efb264d47eb88669132e5e8f17f5cb64b4335d20675c7d4b5805471580`
      (50/50 verified) and protected-overlap audit
      `d8ed9720c9d55721a35420b43e1a753ea3d0034b1c35851f7f5a792363033ae6`
      (zero significant entity collisions; fixed-panel trivial-scalar
      reporting under the exact-and-significant-entity-v1 policy).
    - openhands-translated-rehearsal (the new data authority): coverage =
      machine replay verification (below) + the OH-vocabulary scrub (below) +
      public SWE-rebench provenance (nvidia/Open-SWE-Traces derivative). There
      is **no protected-panel entity overlap audit for it** — no such audit
      ever existed for any prior OH cohort either; this is the coverage gap the
      operator signs off, stated plainly.
    - The translated-OH candidate authority manifest
      (`a179c140b4e8efa77a8163574bb78d4fe34dea91adfeb0439e8339d5002be1fb`),
      scrub report (`4af96b8fa60bfa4dd06dbede7ad47b20a0803c5ff24645dcc8ee0e01562248a8`),
      translate report
      (`b6b702a078307f6ddbcf2b77fed373c19610c9a39c171ab05649d214d6296cf1`),
      all ten cohort-authority manifests, and the active tool surface
      (`55421905438806223414d96a4e64f1a6ae64772afdb1fbc67fbbfb77e8d5d908`)
      are byte-verified in the frozen segment-1 proposal's `identity_bindings`
      (the generic auditor enforces every one of them).
  - The failed v9-full preparation (`a05156ad...`) and its run artifacts are
    retained untouched as evidence (operator ruling; nothing was deleted).
- Disk note: nvme2 is near full; no caches were deleted and no prep authority
  was moved during this task. The 39G repo-cache in the translation work dir
  (1,122 bare clones) remains in place (move-only if space requires).

## 2. Cohort table (with corrected scrub + replay counts)

Prep-level source counts (records / assistant target tokens / share of the
150.36M target pool). Scrub dispositions follow the operator ruling: markers
count **OH-specific vocabulary only** (`str_replace_editor`, `execute_bash`,
`Think:`, OH-only tool names) in **supervised assistant targets**;
`Analysis:`/`Commentary:`/`Action:`/`Arguments:` are the Pi codec itself and
are NOT contamination. Only confirmed supervised-target OH-vocabulary hits were
dropped. Kept/dropped counts are documented below.

| Cohort | Records | Target tokens | Share | Provenance | Scrub disposition (supervised-target OH-vocab) |
|---|---:|---:|---:|---|---|
| conversation-rehearsal | 56,835 | 60,000,000 | 39.9% | e97-4b-smoltalk2-admitted-v1 (seed 613117, 60M budget consumed) | KEPT — 0.0 (150 sampled) |
| instruction-rehearsal | 248,021 | 45,340,400 | 30.2% | e97-4b-pi-instruction-mix-v2 | KEPT — 0.0 (120 sampled) |
| openhands-translated-rehearsal | 2,413 | 25,998,219 | 17.3% | e97-oh-pi-native-translation-v1 candidate-authority (seed 920001, 26M budget; 2,413 of the 5,087 sealed records; 2,413 distinct instance ids) | REPLACES the raw OH cohort (raw fulltraj measured 100% raw-protocol capture) — see replay funnel below |
| cumulative-recovery-rehearsal | 60,173 | 9,167,950 | 6.1% | e97-4b-pi-cumulative-recovery-mix-v1 | KEPT — 0.0 (120 sampled) |
| compositional-rehearsal | 41,080 | 7,335,698 | 4.9% | e97-4b-pi-compositional-retention-mix-v1 | KEPT — 0.0 (120 sampled) |
| live-aligned-rehearsal | 19,810 | 2,376,849 | 1.6% | e97-4b-pi-live-aligned-all-assistant-v1 | KEPT — 0.0 (120 sampled) |
| pi-native-curriculum | 823 | 78,768 | 0.05% | pi-native-curriculum-2000-selected-v1 under the family filter | KEPT — 0.0 (150 sampled) |
| pointerchase-rehearsal | 240 | 34,096 | 0.02% | pi-native-pointerchase-collection-v1 | KEPT — 0.0 |
| longcopy-rehearsal | 180 | 7,224 | <0.01% | pi-native-longcopy-collection-v1 | KEPT — 0.0 |
| loopbreak-rehearsal | 150 | 8,171 | <0.01% | pi-native-extracterror-collection-v1 | KEPT — 0.0 |
| reasoning-rehearsal (T2 pilot; minor cohort) | 50 | 15,853 | 0.01% | pi-native-reasoning-rehearsal-collection-v1 (50/50 machine-verified observe-then-quote; 230 real Pi tool calls; 30 authentic tool errors; 331,929 tokens) | KEPT — new cohort, enters with this prep |

Dropped cohorts (per the scrub report, sampling-confirmed OH-only tool
vocabulary in supervised assistant targets):

- **grounded-authored-rehearsal — DROPPED**: 80% of 150 sampled supervised
  targets carried OH-only tool vocabulary; pool 2,583 records. No retained
  workspaces or base commits, so replay verification was impossible (fail-closed
  drop). This cohort had trained the promoted v6-u96 parent lineage.
- **representation-bridge-rehearsal — DROPPED**: 57% of 150 sampled; pool
  2,223 records; same fail-closed reasoning (also in the promoted v6-u96
  lineage).
- **openhands-execution-rehearsal (raw fulltraj) — REPLACED**: 100% of 150
  sampled (54/60 in the v9-full stop forensics carried the full
  Analysis/Commentary/Think scaffold in the raw protocol). The translated +
  replay-verified collection below replaces it.

Scrub method note (for the record): markers were `Action: str_replace_editor`
or `Action: execute_bash` in supervised text — i.e. OH-only tool names, exactly
the ruling's definition; the surrounding `Action:`/`Arguments:` scaffolding is
the Pi codec. Fractions are sampling-based (150/120-record samples per cohort),
not exhaustive; the two dropped cohorts are quantified at 80% and 57%.

T1 translation + replay funnel (all machine-verified, zero GPUs; commits
78904dd7 + 9099b92c):

- 5,969 raw trajectories (5,969/5,969 base_commit coverage)
- → 5,940 translated (29 dropped: 26 invalid_view_range, 3 malformed_create_args)
- → **5,181 replay-PASSED** / 759 fail-closed drops
  (read_missing_file 332, read_content_mismatch 141, final_state_mismatch 125,
  patch_apply_failed 31, edit_missing_file 25, edit_not_unique 20,
  read_is_directory 20, write_existing_file 19, dir_listing_mismatch 16,
  exception 9, checkout_failed 8, clone_failed 6, read_offset_error 4,
  read_binary 3)
- → **5,087-record sealed collection** (104 excluded over-context), accepted
  as-is per the operator ruling: **231,652,231 tokens / 57,814,225 supervised
  target tokens**; byte-exact final-state oracle; oracle patches never entered
  the payload (`oracle_metadata_copied: false`).
- The prep's OH cohort consumed 25,998,219 of its 26,000,000 target budget
  (2,413 records, distinct instance ids) from that sealed 5,087-record pool.

## 3. The 8 screened keys (window-coverage rule)

Training-data-only screen over keys 1400000.. under the window-coverage rule:
major cohorts (>=2% of targets: conversation, instruction, translated-OH,
cumulative, compositional, live-aligned) must appear in **every** update; minor
cohorts every **32** consecutive updates. 8 keys admitted (screened order;
one key per 128-update segment):

| Segment | Key | Schedule file sha256 | Scheduled input tokens | Scheduled assistant targets | Unique packs | Unique records |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1400005 | b877e14e2d9c6f6466fe9b5270f792c3b8c7a593d5b7573c3695e6dde10a9dfc | 58,353,360 | 26,893,064 | 1,024 | 76,541 |
| 2 | 1400010 | b5216ca3922dc1711de3ae30848673927dc13e2124373fd4ce984954a296c070 | 58,607,542 | 27,239,757 | 1,024 | 78,469 |
| 3 | 1400023 | be7843c462a714f834b4190e81e3e65bae6ddc2baa6c8b1c25e3432d115ee376 | 59,145,327 | 27,308,035 | 1,024 | 77,536 |
| 4 | 1400027 | cb5723d4a5a1939c6f0ec1364b5c2634bf16bed20334686d98488f6a7c9eaedf | 59,139,175 | 27,320,719 | 1,024 | 77,456 |
| 5 | 1400050 | 94b4b0b06bc3cc90d9ba760b203b5c87db24c5aad60c81d2a6913658eb9979cb | 58,817,675 | 27,277,490 | 1,024 | 77,706 |
| 6 | 1400057 | d171ffc7d18b8b93e20d33787a389fb163fe44946e0e9c87f426102c23d0adba | 58,976,881 | 27,146,686 | 1,024 | 77,599 |
| 7 | 1400080 | 3cd1b801376d5615df27026d72269aa05a4a14d13a281a0643b5248da4c547d9 | 58,891,535 | 27,301,483 | 1,024 | 77,620 |
| 8 | 1400081 | 85bebf4c7ad79eb916c2a0627bebc67a807125e06df5b952978c8eaa799779f0 | 58,839,381 | 27,042,307 | 1,024 | 76,061 |

All eight segment schedules were verified: 128 steps each, no gaps, cohort
sets identical to the prep authority, `planning-only-not-authorized`, and the
generic auditor independently verified the window-coverage floor rule for
segment 1 (the audit enforces it; segments 2..8 are re-verified by the same
auditor at their freeze points).

## 4. Segment proposals + audits

Chained-parent discipline (matching the v5/v7/v8/v9 precedent and the
auditor's fail-closed design, commit 306ef0f8): a segment proposal is frozen
and audited only when every identity it binds is real. Segment 1 binds the
bridge parent and is frozen now; segments 2..8 chain to the prior segment's
final audited checkpoint and are frozen + audited at their segment boundaries
by the staged freeze-time writer — their proposal/audit sha256 values are
produced at those freeze points, not now. No audit.json was fabricated and no
placeholder parent was bound.

- **Segment 1 proposal (frozen, committed, pushed)**:
  `configs/pi/e97-pi-native-repair9r-full-arc-segment1-training-proposal-v1.json`
  - sha256 `063b3f7567ce5306791f2ec3c745a7b732691ef442ee4eb577bc77bb0647e351`
  - schema `emender-e97-pi-native-repair9r-full-arc-segment1-proposal-v1`;
    status `frozen-proposal-not-authorized`; 128 updates; lr 1e-05;
    data world size 8; context 65,536; key 1400005; bridge parent; the frozen
    behavioral gate (unchanged thresholds, incl. Stage-B
    `valid first frame >= 12/14 and correct first action >= 10/14` and
    execution `>= 64/96`); 24 verified identity bindings incl. the
    provenance/overlap-audit coverage files and cohort bindings for the
    translated-OH, reasoning, and conversation cohorts; trainer source commit
    `d77dcc462e8b25c0b362b196347ddc9f50ffe683` (verified in git log; the
    `7a84257b` reference in older committed proposals is the historical
    pointer-chase commit and was NOT used here).
- **Segment 1 proposal audit (PASSED)**: generic auditor receipt
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9r-full-arc-staging-v1/proposal-audits/segment1-proposal-audit.json`
  - receipt sha256 `dc96fe255ef26b4b77a3dbeed5c4338037bc496dafa9bec4f6b6c4a2fa5e16bf`
  - status `qualified-proposal-not-authorized`; checker sha256
    `22c8b73c51af8dca86d340c16c3d736da9015f82ee8443cc80e92fe0cd19f3a1`.
- **Segments 2..8**: staged freeze-time machinery (below). Proposal + audit
  shas will be recorded here at each freeze point as the arc progresses.

## 5. Staged admissions and launchers (NOT EXECUTED)

Staging root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9r-full-arc-staging-v1/`

- `segment{1..8}/commands.sh` — exact per-segment command sheets. Each sheet
  is fail-closed and ordered:
  1. resolve/verify the parent (segment 1: bridge; segment N: prior segment's
     audited checkpoint, verified against its `audit.json`);
  2. freeze the proposal (`freeze_segment_proposal.py` — every binding
     re-hashed from disk);
  3. audit it (`scripts/audit_e97_pi_native_repair_generic.py --schedule
     <prep>/schedule-segment{N}-proposal.json` — MUST print
     `REPAIR_PROPOSAL_AUDIT`);
  4. scoped commit + push;
  5. **ADMISSION — OPERATOR SIGN-OFF REQUIRED** (new data authority):
     `scripts/admit_e97_pi_native_training_proposal.py --proposal
     configs/pi/e97-pi-native-repair9r-full-arc-segment{N}-training-proposal-v1.json
     --proposal-sha256 <sha> --preparation <prep> --output
     pi-native-repair9r-full-arc-segment{N}-training-admission-v1
     --authorization-statement "I authorize the exact 128 update proposal."`
     (the operator types the statement);
  6. regenerate the admitted schedule with
     `scripts/plan_e97_pi_native_training_schedule.py ... --admission
     <admission>/admission.json --output <admission>/expected-schedule.json`
     and fail-closed verify it matches the frozen plan (steps, cohort totals,
     unique records) — the admit script's nonce search preserves the epoch-zero
     affine, so the admitted pack sequence is exactly the screened one;
  7. instantiate the run dir
     (`templates/instantiate-run-dir.sh`: detached worktree at
     `d77dcc462e8b25c0b362b196347ddc9f50ffe683`, source/input sha manifests,
     launcher rendered from the v9-full segment-1 launcher with the qualified
     chunk-2048/16384 recipe at ~72 s/update, GPU lease, EXIT-trap audits) and
     execute `run.sh` (128 updates);
  8. write the recipe from run artifacts (`templates/write-recipe.py`) and run
     `scripts/audit_e97_pi_native_training_run.py --output <run>/audit.json`
     (status `passed-training-not-promoted`; required by the next segment's
     chained-parent verification);
  9. juncture evaluation at u{128*N} (below);
  10. decision point: continue to segment N+1 or stop fail-closed. Full frozen
      dual gates at u256/u512/u896/u1024 (segments 2, 4, 7, 8) run before any
      continuation past those boundaries.
- `templates/instantiate-juncture-dir.sh` — renders the per-juncture eval
  directory (Stage-B panel + conversation NLL probe) with a fail-closed
  panel-binding guard: `run-probe.sh` refuses any `panel.json` whose primary
  models are not bound to the exact audited checkpoint sha. Panels must be
  regenerated per checkpoint (the recorded v9-full defect: a copied panel
  silently evaluated v8-u96).
- `templates/write-recipe.py`, `templates/source-files.txt` (the v9-full
  607-file trainer source closure), `make_command_sheets.py` (the generator for
  the per-segment sheets), `README.md`, and
  `segment1-proposal-draft-from-prior-worker.json` (prior worker's draft,
  retained for forensics).

Nothing above was executed: no admission directory exists, no GPU was touched,
no `git add -A` was used (scoped commits only).

## 6. Unchanged evaluation protocol

- **Every 128-update juncture (u128, u256, ..., u1024):** cheap-tier Stage-B
  panel AND conversation-retention NLL probe (plus the conversational-behavior
  panel recorded in the frozen proposal's evaluation caveats per the operator's
  first unscripted-session findings).
- **Full frozen dual gates at u256, u512, u896, u1024** (and at any
  detector-flagged checkpoint): Stage-B gate
  `valid first frame >= 12/14 and correct first action >= 10/14`; execution
  `>= 64/96` with `openhands_fresh >= 30/32`, `openhands_prior_fresh = 16/16`,
  `openhands_prior_regression >= 10/16`, `openhands_transfer >= 5/16`;
  `composition >= 4/16`; both saved-x/train-y retention gates.
- **Fail-closed stopping** if retention is decisively breached; no automatic
  retries; no threshold changes; no checkpoint promotion without a separate
  gate pass.
- **T2 caveat (carried from the reasoning cohort spec):** the
  conversation-retention NLL panel needs **re-baselining when the reasoning
  cohort enters training** — i.e. at segment 1 of this arc, before any
  juncture comparison against the v6/v8 conversation windows is read as a
  retention signal. Thinking-trained emission may shift the panel independent
  of retention.

## 7. Sign-off checklist for the operator

1. New data authority: the translated + replay-verified OH collection (5,087
   sealed records; the 5,087-record authority accepted as-is per ruling) and
   its place in the v2 prep (17.3% of targets) — including the stated
   coverage gap (no protected-panel entity overlap audit exists for it).
2. Cohort dispositions: grounded-authored and representation-bridge dropped
   (80%/57% sampled supervised-target OH-only vocabulary); raw OH cohort
   replaced; all other cohorts kept (0.0).
3. Segment-1 proposal + audit (shas above); the staged 8-segment freeze /
   audit / admit / launch / juncture sequence; the chained-parent freeze-time
   discipline for segments 2..8.
4. Unchanged evaluation protocol and the T2 conversation-NLL re-baselining
   caveat.
5. Admission authorization statement (typed by the operator at step 4 of the
   chosen segment's command sheet).
