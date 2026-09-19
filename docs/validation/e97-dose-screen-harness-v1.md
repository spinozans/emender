# E97 dose-screen harness (v1) — restored-pool dose probes, STAGED

Status: **STAGED, NOT EXECUTED**. The v10/v3 gate verdicts bracket an
unexplored dose axis: the v2 prep (grounded-authored + representation-bridge
restored pools scrubbed to zero) reached Stage-B 9-10/14 valid but execution
0/96; the v3 prep (both restored at FULL pool) reached execution 59-60/96 but
Stage-B 4-5/14. Both extremes fail the frozen gate (Stage-B >= 12/14 valid +
>= 10/14 correct; execution >= 64/96). This lane builds THREE dose-variant
preps (25/50/75% of the restored pools), one fresh-screened 32-update probe
schedule each, the probe launchers, and the probe evaluation set — everything
STAGED for the parent orchestrator, which executes the GPU probes itself.
All work here was CPU-only; no GPU state was touched; no admission, training,
or evaluation was executed; no threshold changed.

Work dir (harness + logs): `/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-v1/`
(repo mirror of the staged machinery: `staging/pi-native-repair9-dose-screen-v1/`).

## 1. Pipeline validation: v3 reproduced byte-exact

The dose preps must be "v3 except the dose". The preparer
(`scripts/prepare_e97_pi_native_repair_training.py`) was extended with
additive, default-off knobs (`--authored-max-records`,
`--rehearsal-max-records`, `--rehearsal-subsample-seed`); the reconstructed
v3 command was then re-run to a scratch dir with the knobs off:

- preparation manifest sha256 **3c4f0b38ba107cabd4d0505ceb789bf2bba81ee2e11890421b7a163a96877f9b**
  — byte-identical to the recorded v3 prep manifest; counts 434,581 /
  339,454,089 / 152,137,979;
- packs manifest sha256 **5a4baffbb0a3ac4e2b6531d2610bb4ae26e9ec0f790813326de42d90d03175a0**
  — byte-identical to the recorded v3 packs manifest (5,870 train packs).

So the dose-prep command differs from v3 by exactly the dose knobs, and the
current tree (including its uncommitted masked-SFT hardening pass — which the
v3 packs themselves already carry) is the tree that reproduces v3. Scratch
payloads were then deleted (manifests retained as receipts:
`v3-repro-manifest.json`, `v3-repro-packs-manifest.json` in the work dir).

## 2. The three dose preps

Roots: `.../pi-native-repair9-full-preparation-v3-d{25,50,75}`. Identical to
v3 in every cohort, budget, seed, family filter, parent (bridge checkpoint
`9b78628d...`), interleave, and pack plan — except the two restored pools are
seeded first-N subsamples (same method across variants: fixed seeded shuffle,
first-N; nested by construction and machine-verified on the built preps by
`source_record_id` set inclusion: d25 ⊂ d50 ⊂ d75 ⊂ v3 for both cohorts):

- grounded-authored: seed **882843** (the slice shuffle; the dose subsets are
  prefixes of the full-pool selection order), pool 2,583 records;
- representation-bridge: seed **882844**, pool 2,223 records.

| | d25 | d50 | d75 | (v3) |
|---|---:|---:|---:|---:|
| authored records (of 2,583) | 646 | 1,292 | 1,937 | 2,583 |
| bridge records (of 2,223) | 556 | 1,112 | 1,667 | 2,223 |
| prep manifest sha256 | `640ba1796da8e045d00e4222cf9368833b35b13805de7bccfe4b3467c2fdd8ce` | `07d8b956fc2ef7fae8c8f6c2dc0b4073685f9dce4e73325ed8ae82301b675991` | `65ecd1c702baaa549c7a557dae87871a64ed808563deafc2f244e4004ab7495d` | `3c4f0b38...` |
| packs manifest sha256 | `56582a3e284e6be6f87c58349fcb258221b1668cd57768b2719d093fe12e1e43` | `83cf1e8ad9a03c02ac179d943236272d6adfd434416b75d159bdb98524f77d5c` | `1263aa9c00e4942328d7c6bb0a01f008cd9a1e11bfdfc07ecc9fed1ab04af125` | `5a4baffb...` |
| records / tokens / targets | 430,977 / 328,096,197 / 150,826,647 | 432,179 / 331,813,728 / 151,254,858 | 433,379 / 335,748,456 / 151,712,014 | 434,581 / 339,454,089 / 152,137,979 |
| authored targets (of 926,091) | 222,691 | 453,992 | 711,836 | 926,091 |
| bridge targets (of 848,660) | 240,728 | 437,638 | 636,950 | 848,660 |
| train packs (validation PASS) | 5,703 | 5,744 | 5,824 | 5,870 |
| pack-validation receipt | `a341c85f4d8a57b0b1b1261583fb0dfb28d9717f8e60b7bd818b491863af1d17` | `c00593fc75dd227fba6e9feed53adb223b85b05a751d96cfad37a8a3f1742e43` | `f811dc63ba9ee068ee3c33423f93077dc3297586bb16b39d556a60570598247d` | `8bff6df9...` |
| prep-audit receipt (PASS, `qualified-preparation-not-admitted`) | `1f0b194b51af1e4f8bb7d7ccd6e2f5ced9dc135f56de8ad9898901ada33a3c17` | `77682abb9fffb9eda070e729d46841fd5d7d06660e7a1141ee3ee57ebbd1c6af` | `11d4aa3f2473ca547916e237ee2ab9e899588f904ffb7f72af0b8a4fa4105396` | `e41f97a8...` |

Every non-restored cohort is count- and target-identical to v3
(machine-diffed: only the two restored cohorts differ). The prep audit
(`scripts/audit_e97_extension_preparation_v1.py`) re-verified per-record mask
sums over every record, index accounting, cohort-table reconstruction, the
selected-curriculum provenance triple, and all 12 cohort-authority bindings
re-hashed from disk. The dose manifests record the subsample under
`authored_rehearsal`/`rehearsal_rehearsal` (`subsample: seeded-shuffle-first-N`,
seeds, caps, full-pool counts) and the manifest purpose states the reduced dose
explicitly (never claims full restoration at partial dose). Disk: ~1.7G per
prep; nvme2 has ~576G free after this lane.

## 3. Fresh probe keys + 32-update schedules

Keys screened fresh from 1402000 (`scripts/screen_e97_extension_probe_keys.py`,
window-coverage floor rule: majors >= 2% every update, minors at least once per
32-update window; 32-update probe geometry, world size 8), then re-verified
with the REAL planner (`scripts/plan_e97_pi_native_training_schedule.py`,
`planning-only-not-authorized`) and the window-rule verifier
(`scripts/verify_e97_repair_schedule_window_rule.py`, `--expected-steps 32`):

| dose | key | keys tried | schedule sha256 | scheduled tokens / targets | unique packs |
|---|---:|---:|---|---|---:|
| d25 | 1402002 | 3 | `458f8bec65722ce635c3864b1178fc3af5fee0c0c4750ab60cb49882b383704a` | 14,537,992 / 6,728,233 | 256 |
| d50 | 1402000 | 1 | `09200b1db773cb8a344543fa30d19a84a4e33b0386136592c6dcff55c8cc7e9f` | 14,859,227 / 6,895,798 | 256 |
| d75 | 1402001 | 2 | `8bf7dacd4a373db82b9a5318fc09c65d1cb795da7395aec5efc71fcbc15e0c03` | 14,864,789 / 6,713,470 | 256 |

All three schedules: WINDOW_RULE_OK — both restored cohorts present in every
probe schedule. Schedules live at `$PREP/schedule-<key>-probe.json`.

## 4. Probe evaluation set

- **Stage-B**: the standard 14-case panel binding — `pi-native-baseline-v1-r3-panel/panel.json`,
  frozen per probe checkpoint via `scripts/eval_e97_pi_native_stage_b.py
  freeze --checkpoint ...` then `run` (the freeze/plan step takes the
  checkpoint, so the panel itself is fixed).
- **Execution slice**: a FIXED seeded 24-case subset of the frozen 96-case
  execution panel (the v3-u256 dual-gate panel, sha
  `cdcd3ac9e4bde2325f2dc28e082efd8e74a04cd03a29fe34d2663882cd59aa74`; case set
  byte-identical to the v10 u256 panel). The 96 cases partition into six
  id-prefix groups x four families (regression-old, regression-fresh, fresh,
  transfer, bridge-fresh, bridge-composition; lookup/sum/edit/recovery). The
  slice takes ONE case per (group, family) cell — 24 cases, every group at 4
  cases, every family at 6 — chosen by `random.Random(240977).choice` over
  id-sorted cell candidates (`execution-slice/slice-definition.json`; the 24
  ids are listed there and in the work-dir report). Slice panels are
  regenerated PER probe checkpoint by
  `execution-slice/freeze_execution_slice_panel.py` (fail-closed on source
  identity, membership, 6x4 balance, checkpoint identity; smoke-tested on the
  bridge checkpoint — 24 cases, balanced families, 4 models bound) and bind
  FOUR models per the dual-gate convention: dose-probe-y (train),
  dose-probe-x (saved), bridge-y-control, bridge-x-control (same-run window
  reference) — 96 episodes per probe run. The slice is a SCREEN: the frozen
  `>= 64/96` execution gate stays bound to the full 96-case panel.

## 5. Staged machinery (executed only by the orchestrator)

Work dir: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/pi-native-repair9-dose-screen-v1/`

- `freeze_dose_probe_proposal.py` — fail-closed probe-proposal writer
  (bridge parent, per-dose prep/schedule/key; full identity-binding discipline
  of the v3 segment writer, incl. the selected-curriculum provenance triple
  and the dose subsample recorded in `cohort_bindings`). **Machine-validated**:
  all three DRAFT freezes PASS the generic auditor
  (`scripts/audit_e97_pi_native_repair_generic.py`, schema
  `emender-e97-pi-native-repair9-dose-probe-proposal-v1`):
  - d25: proposal `232e6c041c5393d5640e5a81c9cd5eb012fd9f436655825f5562cf21112972df`, receipt `ba0e61888df0d3d78dedf9c4639a0802c9efa5f5fcc85017a5a9274a3c714f23`
  - d50: proposal `e170a2f892a2711b4aea6d935bb2158d8b095e28eaa7d26ec707cbe7fdb5514a`, receipt `d6c881ddff1294bd0c0f175c56f56a6e346dd80dbf0948d6049a7a0fdd702676`
  - d75: proposal `420467233388203752a3e30d22ce455f4478a58e8474cff48e8afe27e4194384`, receipt `09eb6cc70820e1e99ae501e3f47700907c88a74255b6eca73d2d5937f8d86e1f`
- `d{25,50,75}/commands.sh` — the exact per-probe command sheets:
  freeze (asserts the DRAFT proposal sha) -> generic audit (asserts the
  receipt sha) -> scoped commit+push -> ADMISSION (operator types
  `I authorize the exact 32 update proposal.`) -> probe-dir instantiation ->
  train u32 -> Stage-B -> slice panel regen -> execution slice.
- `run-probe.sh` (shared) + `d{25,50,75}/run-probe.sh` (baked constants, one
  per dose prep) — instantiate the probe dir: admission identity check,
  bridge parent identity, admitted-schedule equality with the frozen probe
  schedule, trainer worktree at `d77dcc462e8b25c0b362b196347ddc9f50ffe683`
  (v9r source closure), input/source sha256 manifests, and the runners:
  - `run-train.sh` — 32 updates from the bridge parent, `--steps 32
    --save-every 32 --keep-checkpoints 1`, chunk 2048/16384, lr 1e-5, every
    other per-step setting identical to the qualified arc recipe (the arc
    segment run.sh with only the step count changed);
  - `run-stageb.sh` — Stage-B on the u32 checkpoint (resolved via the atomic
    `checkpoints/latest.pt` pointer with a fail-closed
    `checkpoint_agent_sft_u000032_*.pt` basename check — the recorded
    juncture-resolver defect class), 1-GPU lease;
  - `regen-execution-slice-panel.sh` + `run-execution-slice.sh` — slice panel
    bound to the exact u32 checkpoint (binding guard refuses un-bound/stale
    panels — recorded v9-full defect class), 8-GPU lease
    `eval_e97_native_execution.py run` + `aggregate`.
  All rendered runners `bash -n` clean; all embedded python snippets compile.
- `read-dose-results.py` — post-probe readings table (Stage-B valid/correct,
  slice successes per model, bridge controls, bracketing evidence, unchanged
  gate thresholds).

Probe dirs (created by the orchestrator at instantiation):
`.../pi-native-repair9-dose-screen-probe-d{25,50,75}-v1`; admissions:
`.../pi-native-repair9-dose-screen-probe-d{25,50,75}-admission-v1`.

### Exact staged commands (per dose; identical structure, constants baked)

```
# 1-2. freeze + audit (proposal sha must equal the validated DRAFT sha)
.venv/bin/python $W/freeze_dose_probe_proposal.py --dose dXX \
  --preparation $PREP --schedule $PREP/schedule-<KEY>-probe.json --key <KEY> \
  --output configs/pi/e97-pi-native-repair9-dose-dXX-probe-proposal-v1.json
.venv/bin/python scripts/audit_e97_pi_native_repair_generic.py \
  --proposal configs/pi/e97-pi-native-repair9-dose-dXX-probe-proposal-v1.json \
  --schema emender-e97-pi-native-repair9-dose-probe-proposal-v1 \
  --schedule $PREP/schedule-<KEY>-probe.json \
  --output $W/proposal-audits/dXX-probe-audit-final.json
# 3. scoped commit + push (NEVER git add -A)
# 4. ADMISSION (operator sign-off — types the exact statement)
.venv/bin/python scripts/admit_e97_pi_native_training_proposal.py \
  --proposal configs/pi/e97-pi-native-repair9-dose-dXX-probe-proposal-v1.json \
  --proposal-sha256 <PROP_SHA> --preparation $PREP \
  --output $ADM --authorization-statement "I authorize the exact 32 update proposal."
# 5. instantiate + execute the probe (GPU leases; ordered)
bash $W/dXX/run-probe.sh "$ADM"
bash $PROBE/run-train.sh && bash $PROBE/run-stageb.sh && \
  bash $PROBE/regen-execution-slice-panel.sh && bash $PROBE/run-execution-slice.sh
# 6. readings (after all three probes)
.venv/bin/python $W/read-dose-results.py
```

## 6. Standing constraints honored

- CPUs only; the interactive/arc GPUs were never touched (probe runs are the
  orchestrator's).
- No threshold changes; frozen dual gates unchanged; the slice is a screen,
  not a gate; no promotion path in any staged artifact.
- Scoped commits only (preparer knobs + this report + the staged-machinery
  mirror); never `git add -A`.
- No admission, no training, no evaluation, no key reuse across manifests
  (fresh 1402000.. range; the v3 arc keys 1401011.. bind the v3 manifest and
  were NOT reused).
