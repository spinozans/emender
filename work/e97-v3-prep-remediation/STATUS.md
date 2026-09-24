# E97 v3 prep (scrub-reversal remediation) + extension-prep correction — worker status

Repo /home/erikg/emender; python .venv/bin/python; CPU-only (arc relaunch takes
all 8 GPUs — stayed off). NEVER `git add -A`. Everything STAGED, NOT ADMITTED.

## COMPLETE (2026-09-19). Final report in the task response; durable record:
docs/validation/e97-scrub-reversal-remediation-v1.md (committed + pushed).

## Delivered
- WORK 1 — v3 prep `pi-native-repair9-full-preparation-v3`
  - manifest 3c4f0b38ba107cabd4d0505ceb789bf2bba81ee2e11890421b7a163a96877f9b
    (434,581 rec / 339,454,089 tok / 152,137,979 targets; +4,806 vs v2 =
    grounded-authored 2,583 full pool + representation-bridge 2,223 full pool;
    all kept cohorts byte-identical counts/targets to v2).
  - packs 5a4baffbb0a3ac4e2b6531d2610bb4ae26e9ec0f790813326de42d90d03175a0
    (5,870 packs, 0 oversize); validation receipt 8bff6df9... PASS.
  - full prep audit receipt e41f97a8... PASS (qualified-preparation-not-admitted).
  - 8 fresh keys screened (range 1401002.., 73 tried): 1401011, 1401042,
    1401043, 1401046, 1401052, 1401056, 1401057, 1401074; 8 real-planner
    schedules generated + window-rule verified (all PASS; shas in the doc).
  - staged seg-1 freeze command dry-run validated (uncommitted DRAFT):
    proposal 4ca083af..., generic-audit PASS 30e6d5a4...; NOT frozen/committed.
- WORK 2 — extension correction `e97-extension-preparation-v2`
  - manifest b6dc70b7b704719a4412373e3afb8ef74887b1d52bb24c2f971a08568d3ba262
    (992,792 rec / 1,084,239,214 tok / 741,885,113 targets; v1 + both restored
    cohorts at full pool; all else identical).
  - packs b55097e6... (17,583, 0 oversize); validation a3e37e9a... PASS;
    audit c03777ae... PASS.
  - probe key re-screened: 1408370 (4,610 keys from 1403761); real-planner
    probe schedule 0c8d88cf... (32 steps), window-rule verified.
  - LR harness re-pinned: e97-extension-prep-v1/lr-screen-post-v10-v2/
    (v1 harness retained untouched).
- Template fixes (committed 14930e21 + 06876b05): (1) juncture resolver now
  uses checkpoints/latest.pt + segment-local u000128 basename check (old
  u{U}* glob matched nothing; zero-padding alone also fails for seg>=2 since
  names are segment-local) — verified against the executed v10 seg1-u128
  juncture (26fd9da1...); (2) instantiate-run-dir.sh absolutizes all
  input.sha256 bindings via readlink -f; (3) command sheets regenerated for
  the v3 prep (v2-arc sheets preserved as *-v2arc-superseded).
- Commits (pushed): 637bd2b7 preparer/auditor; 14930e21 staging mirror+fixes;
  773f6941 screen/verifier scripts; a473dd44 remediation report doc;
  06876b05 juncture resolver fix. Push also carried one pre-existing unpushed
  commit from another lane (58c02cc4).

## Notable findings
- The scrub report has NO per-record drop lists (cohort-level, sampling-based);
  the documented drop counts are the pools (2,583 / 2,223) — restored exactly.
- Task estimate "+2,000-2,600 records vs v2" matches no faithful reading;
  full restoration = +4,806. Alternative (v9-full 400K authored slice) = +3,376
  and would leave 1,430 records still dropped. Chose full restoration per
  "restore ALL ... records"; flagged for operator sign-off.
- Checkpoint naming: every 128-update run names its final checkpoint
  checkpoint_agent_sft_u000128_* (segment-local), so the juncture defect was
  deeper than the zero-padding diagnosis; fixed via latest.pt.

## Disk
- nvme2: 674G free (96%) after adding v3 prep (1.8G) + extension-v2 (5.5G);
  nothing deleted/moved; all v2/v10 evidence intact.
