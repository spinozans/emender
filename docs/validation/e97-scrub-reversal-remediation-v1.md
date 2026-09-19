# E97 scrub-reversal remediation — v3 prep + extension-prep correction (v1)

Status: **STAGED, NOT ADMITTED**. This lane implements the remediation for the
v10 (repair-v9r, v2-prep) fail-closed stop at u256, diagnosed authoritatively in
`docs/validation/e97-pi-native-tool-copy-progress.md` (section "v10 (repair-v9r)
u256 dual gate: FAIL-CLOSED — scrub over-correction"): the v2 prep's scrub
ruling treated OH vocabulary in supervised targets as contamination and dropped
the grounded-authored and representation-bridge cohorts, but those records are
the execution dialect's REQUIRED training data (the execution panel runs in an
OpenHands-compatible runtime whose tool surface is
`str_replace_editor`/`execute_bash`; the model must be bidialectal). The
remediation — v3 = v2 with the scrub REVERSED, the translated+replay-verified OH
collection retained, everything else identical, chain restarted from the bridge
parent — is applied to both the full-arc prep (v3) and the extension prep
(extension-preparation-v2). Everything here requires operator sign-off as a new
data authority before any freeze/admission/launch runs. All work was CPU-only;
the interactive/arc GPUs were never touched.

## 1. WORK 1 — v3 prep identity (pi-native-repair9-full-preparation-v3)

- Preparation root:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-full-preparation-v3`
- Prep manifest sha256
  `3c4f0b38ba107cabd4d0505ceb789bf2bba81ee2e11890421b7a163a96877f9b`
- Counts: **434,581 records / 339,454,089 tokens / 152,137,979 assistant
  target tokens** — exactly the v2 prep (429,775 / 324,349,514 / 150,363,228)
  plus the two restored cohorts; **+4,806 records vs v2**.
- Schema `emender-e97-tulu3-masked-sft-v1`; tokenizer `p50k_base`;
  `training_eligible: false`, `packing_authorized: false`,
  `optimizer_updates_authorized: 0`. Same 64K whole-record packing plan
  (boundary-aware packs v2, epoch-permutation sampler, interleaved emission,
  same budgets for every kept cohort — every cohort's records/targets are
  byte-identical to v2's).
- Packs manifest sha256
  `5a4baffbb0a3ac4e2b6531d2610bb4ae26e9ec0f790813326de42d90d03175a0`
  (5,870 train packs, **zero oversize exclusions** — the restored pools'
  longest records are 65,426 (bridge) and 64,791 (authored) tokens, both
  inside the 64K pack window).
- Packs validation **PASSED** (`scripts/validate_e97_sft_packs.py`,
  receipt `packs/validation.json` sha256
  `8bff6df9f91f2c22388a88ca7111cf2f81378a5d4af758f58ee3c23ab35d0051`:
  5,870 packs / 434,581 records, per-pack target sums re-verified).
- Full prep audit **PASSED** (`scripts/audit_e97_extension_preparation_v1.py`
  — the generic prep auditor; extended this lane to re-hash the bridge
  cohort binding): receipt `audit.json` sha256
  `e41f97a8f627f48f9ea8ced7c3e3774ef55cee41015fc1f5b0b732d278b89cbe`,
  status `qualified-preparation-not-admitted`; machine-verifies manifest +
  payload identities, index accounting, per-record mask sums for all 434,581
  records, cohort-table reconstruction, and re-hashes all 12 cohort-authority
  bindings from disk.
- Parent (v2 precedent, restart point): representation-bridge checkpoint
  `checkpoint_agent_sft_u000032_loss_0.5302.pt` sha256
  `9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa`.

### 1.1 Restored counts vs the documented drop counts

The scrub report (`e97-oh-pi-native-translation-v1/scrub-report.json`, sha
`4af96b8fa60bfa4dd06dbede7ad47b20a0803c5ff24645dcc8ee0e01562248a8`) is
cohort-level and sampling-based: **no per-record drop lists exist**. The
documented drop counts are therefore the cohort pools, dropped whole:

| Cohort | Documented pool (scrub report) | Sampled OH-vocab fraction | Restored in v3 | Match |
|---|---:|---:|---:|---|
| grounded-authored-rehearsal | 2,583 | 80% | **2,583 (full pool, 926,091 targets)** | exact |
| representation-bridge-rehearsal | 2,223 | 57% | **2,223 (full pool, 848,660 targets)** | exact |

- Sources: grounded-expansion-v1-data-r2/authority (manifest sha
  `08752c53e2199533252c77821ec9696012b19dc09ea0ff1508ef6676a6cd2140`,
  training-eligible) and representation-bridge-v1-data/authority (manifest sha
  `a21dba6f80e58e87ea838a98018b40650decebed3cf4b0f98dacdb750520469d`,
  training-eligible) — the promoted v6-u96 lineage's cohorts.
- The authored cohort enters at seed 882843 with a 1,000,000-target budget,
  which consumes the entire 926,091-target pool (all 2,583 records; no
  budget-limited slice). The bridge cohort is consumed whole, as in every
  prior prep.
- Honest note: the task's rough expectation was "+2,000-2,600 records vs v2".
  That range matches **no** faithful reading of the documented drop: the
  drops were cohort-level (whole pools), so "restore ALL ... records" yields
  +4,806 (2,583 + 2,223). The estimate is superseded by the exact restoration
  above; an alternative reading (restoring only the v9-full prep's historical
  400K-target authored slice of 1,153 records → +3,376) would have left 1,430
  authored records still dropped and was rejected as a partial undo. The
  operator signs off the actual counts recorded here.
- The OH cohort stays exactly as v2 (translated+replay-verified collection,
  2,413 records / 25,998,219 of the 26,000,000-target budget, seed 920001,
  from the 5,087-record sealed collection); the raw OH cohort stays replaced.

### 1.2 Cohort table (records / assistant target tokens / share of 152.14M)

| Cohort | Records | Target tokens | Share | Disposition vs v2 |
|---|---:|---:|---:|---|
| conversation-rehearsal | 56,835 | 60,000,000 | 39.4% | identical |
| instruction-rehearsal | 248,021 | 45,340,400 | 29.8% | identical |
| openhands-translated-rehearsal | 2,413 | 25,998,219 | 17.1% | identical (replaces raw OH) |
| cumulative-recovery-rehearsal | 60,173 | 9,167,950 | 6.0% | identical |
| compositional-rehearsal | 41,080 | 7,335,698 | 4.8% | identical |
| live-aligned-rehearsal | 19,810 | 2,376,849 | 1.6% | identical |
| **grounded-authored-rehearsal** | **2,583** | **926,091** | **0.61%** | **RESTORED (full pool)** |
| **representation-bridge-rehearsal** | **2,223** | **848,660** | **0.56%** | **RESTORED (full pool)** |
| pi-native-curriculum | 823 | 78,768 | 0.05% | identical |
| reasoning-rehearsal (T2 pilot) | 50 | 15,853 | 0.01% | identical |
| pointerchase-rehearsal | 240 | 34,096 | 0.02% | identical |
| longcopy-rehearsal | 180 | 7,224 | <0.01% | identical |
| loopbreak-rehearsal | 150 | 8,171 | 0.01% | identical |

Provenance/overlap-audit coverage is unchanged from the v10 integration
summary's honest statement: the selected-curriculum provenance triple is bound
in the prep manifest; per-collection audits (reasoning, hybrid) bound; the
translated OH cohort's coverage is replay verification + scrub + public
SWE-rebench provenance; the restored cohorts are the previously-qualified
training-eligible authorities of the promoted v6-u96 lineage (historical
receipts: representation-bridge and grounded-expansion audits), with no new
protected-panel overlap audit run for the restoration.

## 2. Fresh key screen (window-coverage) and 8 segment schedules

The manifest sha changed, so keys were re-screened from scratch (fresh range
1401002..) with `scripts/screen_e97_repair_v3_arc_keys.py` (fast in-process
replication of the planner's affine pack permutation) under the auditor's exact
rule — majors (>=2% of prep targets: conversation, instruction,
openhands-translated, cumulative, compositional) in EVERY update; minors
(incl. the restored cohorts) in every aligned 32-update window. 8 keys admitted
(73 keys tried), then each was re-verified by generating the real 128-update
schedule with `scripts/plan_e97_pi_native_training_schedule.py` and verifying
the window rule directly on the planner output
(`scripts/verify_e97_repair_schedule_window_rule.py`): **all 8 PASS**.

| Segment | Key | Schedule file sha256 | Scheduled input tokens | Scheduled targets | Unique packs | Unique records |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1401011 | `843013a2dd19b4ce9db17f6ea3711836ef17733f53ba948d263004483679ffbc` | 59,205,679 | 26,719,734 | 1,024 | 77,468 |
| 2 | 1401042 | `6522cd86208fbcccf4188724be4b63315c75e9c6619f628735c07cb9958bdf5f` | 59,072,401 | 26,381,987 | 1,024 | 76,230 |
| 3 | 1401043 | `6782c1a3bf53fac76807fb7f35d10a39f7e0afb88bcc6ffafe6b50160be5cad2` | 59,151,596 | 26,343,236 | 1,024 | 74,914 |
| 4 | 1401046 | `bed12645ab87e7b948159b7864135a16b9c610a8b7fad0b4afe7ee382ff0d493` | 59,200,244 | 26,387,467 | 1,024 | 73,644 |
| 5 | 1401052 | `e12b591a8a60cdb77b5ca2827360b8e31cc6a2357930c52b13d391123510816d` | 59,524,616 | 26,867,075 | 1,024 | 76,146 |
| 6 | 1401056 | `d9f8e85395f64f6fa4dfca45351bdc8ba16a12f99769d5c0e20d50e96320c1bc` | 59,351,990 | 26,190,250 | 1,024 | 73,203 |
| 7 | 1401057 | `4c62b7b8a1fd3275ed9bb7f7083c1355d4b583a4c6690cfb1195c3128e754281` | 59,196,097 | 25,962,307 | 1,024 | 72,448 |
| 8 | 1401074 | `5aef869c498db1bacd6627ab5b0112a23941109b98653c9e2f2f65977db28fad` | 59,143,528 | 26,169,491 | 1,024 | 73,713 |

`schedule-proposal.json` (segment 1's schedule, sha
`843013a2dd19b4ce9db17f6ea3711836ef17733f53ba948d263004483679ffbc`) and
`schedule-segment{2..8}-proposal.json` live in the prep root; all are
`planning-only-not-authorized`.

## 3. STAGED segment-1 freeze command (NOT EXECUTED)

The staged per-segment command sheets live at
`pi-native-repair9r-full-arc-staging-v1/segment{1..8}/commands.sh` (v3
regeneration; the v10-era sheets are retained as
`commands-v2arc-superseded.sh`). The parent orchestrator executes the chain
(freeze -> audit -> admit -> launch) with the operator's authorization; **no
proposal was frozen or committed by this lane**. The staged segment-1 freeze
command is:

```
.venv/bin/python pi-native-repair9r-full-arc-staging-v1/freeze_segment_proposal.py \
  --segment-index 1 \
  --keys 1401011,1401042,1401043,1401046,1401052,1401056,1401057,1401074 \
  --output configs/pi/e97-pi-native-repair9r-full-arc-v3-segment1-training-proposal-v1.json
```

Machine validation of the staged chain (without freezing/committing): the
writer was dry-run to an **uncommitted DRAFT** proposal in the work dir (not
configs/pi, never committed) — all 27 identity bindings re-hashed from disk —
and the generic auditor PASSED on it:

- DRAFT proposal sha256
  `4ca083afe0e5244812bbe37fc6502cbafba4a904b2ab1595f741ccfbdd44db3b`
  (deterministic: the parent's real freeze reproduces these bytes);
- DRAFT audit receipt sha256
  `30e6d5a4739949a5d3ff2106b741292fa1e23f9ad98af52d0a4fcb13a457a94c`
  (`REPAIR_PROPOSAL_AUDIT`, status qualified-proposal-not-authorized).

## 4. Staging-template fixes (committed, scoped)

Commit `14930e21` (repo mirror of the live staging root under
`staging/pi-native-repair9r-full-arc-staging-v1/`):

1. **Juncture glob zero-padding** —
   `templates/instantiate-juncture-dir.sh`'s stage-juncture resolver now uses
   `printf '%06d'` (`checkpoint_agent_sft_u000128_*.pt`); the old `u128*`
   glob matched nothing (checkpoints are zero-padded to six digits).
2. **Absolute input.sha256 bindings** —
   `templates/instantiate-run-dir.sh` absolutizes the parent/admission/
   proposal/args-json paths (`readlink -f`) before writing `input.sha256`,
   so `sha256sum --check` resolves them from the worktree cwd (a relative
   proposal path was unresolvable there).
3. **Regenerated per-segment command sheets for the v3 prep** —
   `templates/make_command_sheets.py` rewritten (v3 prep, fresh keys,
   `--arc-suffix -v3` run/admission/juncture dir names so the restart arc
   cannot collide with the retained v10 evidence dirs); the freeze writer
   (`freeze_segment_proposal.py`) rewritten for the v3 restart (restored-cohort
   bindings + remediation evidence; `--keys` parameterized). v10-era files
   retained as `*-v2arc-superseded`.

Supporting machinery commits: `637bd2b7` (preparer restoration-aware purpose +
`rehearsal_rehearsal` manifest binding; prep auditor covers the bridge
cohort), `773f6941` (v3 key-screen + window-rule verifier scripts; fixed the
probe screen's `sys.path` so it runs as `scripts/…` directly).

## 5. WORK 2 — extension prep correction (e97-extension-preparation-v2)

The extension prep `e97-extension-preparation-v1` (manifest
`57b7b71b8c6e9b685fdf775d2d569d001dc39c2fdd2af74632951f62172149e4`, 987,986
records / 1,069,134,639 tokens / 740,110,362 targets) inherited the scrub
defect (its cohort list used v2's kept cohorts). The SAME reversal is applied
at its full-pool-strength design; v1 is retained untouched as evidence.

- Preparation root:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-extension-preparation-v2`
- Prep manifest sha256
  `b6dc70b7b704719a4412373e3afb8ef74887b1d52bb24c2f971a08568d3ba262`
- Counts: **992,792 records / 1,084,239,214 tokens / 741,885,113 assistant
  target tokens** — v1 plus the restored grounded-authored (2,583 /
  926,091) and representation-bridge (2,223 / 848,660) cohorts; 14 cohorts;
  everything else identical (SmolTalk2 full strength 614,335 /
  644,276,669; OH-with-thinking enriched authority 2,804 / 31,428,617;
  hybrid conversations 320 / 40,067; instruction/compositional/cumulative/
  live-aligned/curriculum/pointerchase/longcopy/loopbreak/reasoning unchanged).
- Packs manifest sha256
  `b55097e6cfbbfc7c2fa40dc31bcf486f6832fa73623f97a51cb6f13ff8e8e629`
  (17,583 train packs, zero oversize); packs validation **PASSED**
  (`packs/validation.json` sha256
  `a3e37e9ab734db8e52ed7cc495dcd512b4a4c14c4bebaa8b8a91f1e8a292921e`).
- Full prep audit **PASSED**: `audit.json` sha256
  `c03777aee5ca4674771c608e5a11327d9859b778d8ae0398268cabb1bb388870`,
  status `qualified-preparation-not-admitted` (per-record mask sums over all
  992,792 records; 13 cohort-authority bindings re-hashed, including the two
  restored cohorts).
- Probe key **re-screened fresh** (the v1 key 1403772 binds the superseded
  manifest's pack geometry):
  `scripts/screen_e97_extension_probe_keys.py` found key **1408370** after
  4,610 keys (from 1403761), re-verified with the real planner: 32-update
  probe schedule
  `lr-screen-post-v10-v2/schedule-1408370-probe.json` sha256
  `0c8d88cf888960a419c60fb8c4a5d83a0d0f63f9c5d63fa850c0790bc731dcda`
  (15,451,451 input tokens / 10,581,059 targets; window-coverage verified for
  all 32 updates with the restored cohorts present as minors).
- LR-screen harness **re-pinned** as
  `e97-extension-prep-v1/lr-screen-post-v10-v2/` (corrected-prep root/manifest/
  packs identities, key 1408370, restored-cohort bindings in the probe
  proposal freezer, updated measurement plan). The v1 harness is retained
  untouched as evidence. **STAGED, NOT RUN.**

## 6. Disk usage notes

- nvme2: ~684G free (96% used) after this lane — the v3 prep (~1.8G + ~2G
  packs) and extension-preparation-v2 (~5.7G + ~6.8G packs) were added;
  nothing was deleted, moved, or cached; the v2/v10 evidence dirs are intact.
- The 39G repo-cache in the T1 translation work dir remains in place
  (move-only if space requires).
- The superseded extension v1 prep and r1 build are retained untouched.

## 7. Operator sign-off checklist

1. New data authority: the v3 prep (restored cohorts at the exact pool counts
   above; translated+replay-verified OH retained; all else identical to v2).
2. The +4,806-record restoration vs the "+2,000-2,600" rough estimate
   (section 1.1) — the documented drops were cohort-level pools, so the full
   restoration exceeds the estimate; the partial-slice alternative was
   rejected as not "ALL records".
3. Fresh keys + schedules (section 2); the staged segment-1 freeze command
   (section 3) and the v3 command sheets.
4. The extension correction identity + re-screened probe key (section 5).
5. Admission authorization statements remain typed by the operator at the
   command sheets' step 4; nothing in this lane froze, admitted, trained, or
   promoted anything.
