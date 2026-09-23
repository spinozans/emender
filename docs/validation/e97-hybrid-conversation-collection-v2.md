# E97 hybrid-conversation collection v2 — final report (fourth session)

Task: scale the hybrid-conversation seam family to thousands of records with
the same verification bar as v1 (every episode machine-verified end-to-end
through real Pi; zero rejections or documented rejection handling), with
regression tests for the u-ymd clock-check bug class and the stride-sampling
coverage lesson that hid it.

Workspace:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-hybrid-conversation-collection-v2`
(`W`). All work CPU-only; no GPU state, no docker, no `git add -A`.

## 1. Regression-test receipt (the third session's unfinished work, completed)

The third session died mid-sentence after writing — but not committing — the
u-ymd clock-check fix and its regression tests. This session verified that
work against HEAD, corrected one wrong assertion (u-ymd date-question cases
split 107 observation-date / 18 observation-exact; `uymd[0]` is an exact
one), and extended the u-ymd regression to run end-to-end through the
auditor's verification path. Committed scoped as `5158f8b9`:

- `test_utc_ymd_span_is_date_strings` — the span is date strings
  (`observation date +/- one day`), never weekday names; the live UTC date
  is always a member.
- `test_u_ymd_case_clock_check_and_finish` — a u-ymd case exercised
  end-to-end: plan spec (bash `date -u '+%Y-%m-%d'`), live clock check
  (accepts the live date, rejects a stale one), dynamic finish resolution,
  and the independent auditor's `verify_case` + `verify_hybrid_v2_case`
  re-derivation, for all three groups that the bug rejected
  (date-question observation-date 107, date-question observation-exact 18,
  chat-into-tool-date seam 60).
- `test_pilot_sample_covers_every_case_kind` — the pilot-coverage rule:
  `pilot_sample` (first case of every `case_kind` = family x date_fmt x
  dynamic-finish pairs, then the stride sample) must cover all 31 kinds;
  the raw stride-100 sample covers only 19 — the gap that hid the bug.
- 29/29 tests pass across the six pi-native suites
  (`PYTHONPATH=/home/erikg/emender .venv/bin/python -m pytest
  tests/test_e97_hybrid_conversation_collection_v2.py
  tests/test_audit_e97_pi_native_curriculum.py
  tests/test_audit_e97_pi_native_curriculum_overlap.py
  tests/test_build_e97_pi_native_curriculum.py
  tests/test_select_e97_pi_native_curriculum.py
  tests/test_prepare_e97_pi_native_curriculum_training.py`).

## 2. What actually happened at full scale, and the fix applied

The third session started the Phase 3 collect in the background with the
PRE-FIX builder (the then-frozen plan pins builder sha `3a2d677e...`, whose
u-ymd span was seeded with `weekday_now()` — a weekday string, never a
date). That collect ran to completion: 5200 attempted / 5015 verified /
185 rejected. All 185 rejections were `clock date mismatch: 2026-09-22`,
exactly the two u-ymd groups (date-question 125/125, chat-into-tool-date
60/60) — the bug class at full scale, invisible to the stride-100 pilot.

This session then: committed the fix (`5158f8b9`), re-froze the plan under
the fixed builder (`W/plan-fixed/plan-private.json` sha256
`c146ba3b9e7e95d2e24a23fe4cbbf048ee7e2e27b07ed61b68b39ebbb2f64eba`,
source_commit `5158f8b9`; diff vs the previous freeze is only the builder
authority sha and source_commit, cases byte-identical; `W/plan` kept as
superseded evidence), preserved the pre-fix rejection record
(`W/collect/rejections-uymd-pre-fix.jsonl` sha256
`5cc7f20890515301934208d37ece7da4a683067ddb4d1682405fcfa1c9c0c18d`), and
re-collected the 185 cases under the fixed plan. The 5015 already-verified
episodes carried forward untouched (per-episode checkpointing; no
episode-private.json was ever deleted).

## 3. Collection manifest and verification stats (Phase 3 COMPLETE)

- **5200/5200 verified, 0 rejections, 0 transient failures** — the
  zero-rejection standard met at FULL plan scale across all 19 families
  (`W/collect/summary.json` sha256
  `54a4317598e3f494a08bd08679c549fa40b4f8ac43ab6c5ae5425698b9ccc804`).
- 911,713 assistant target tokens (175.3 per record), 30,752,966 tokens,
  5,563 native tool calls, 0 authentic tool errors, 0 repository-discovery
  records; global sequence dedup 5200/5200.
- Candidate authority manifest sha256
  `f88ff570d3cee472a691df018bd72db30710f434a63418c571d0d0d6a014141c`.
- Re-collect log: `W/collect-uymd-refix.log` (sha256
  `50c4b7ec9f43ce582cc88c3bcadc7210b5f8dfd86158d827b10b4f27dc7aad86`).

## 4. Phase 4 audits (all PASS, zero rejections to document)

1. **Full independent reconstruction audit PASS** (`W/audit.json` sha256
   `93c41cc9dbeb5107e4f63acf9348224b2949ca26bd25592487a9a5b24d6d6991`):
   5200/5200 machine-replayed from the recorded Pi transcripts; dynamic v2
   finish frames re-derived; token/mask re-encoded and matched; summary
   reconstructed; `training_eligible: false`.
2. **Protected-panel overlap PASS** (`W/overlap-audit.json` sha256
   `4b21d8ead45d45eceb298707efa13c75fe0d22330c2a6f59b5498adc6f94af5b`):
   zero significant entity collisions vs the three fixed protected panels
   and the Stage-A panel; 75 trivial sub-8-byte numeric fixture scalars
   reported under the exempt exact-and-significant-entity-v1 policy.
3. **In-family/global dedup PASS**: collect and audit both enforce global
   sequence dedup — 5200 distinct sequences for 5200 records.
4. **SmolTalk2 shingle exclusion PASS**
   (`W/smoltalk2-shingle-audit.json` sha256
   `969d5f67a4f5cabba656c4413d2f9465b6fc9bc8e7e03da223df0e62cf14f62d`):
   zero content collisions between the collection's 30,550,166 in-record
   width-40-token windows and the production-admitted SmolTalk2 corpus
   (`e97-4b-smoltalk2-admitted-v1`: 620,389 records / 801,365,124 tokens /
   777,171,237 windows), every hash hit verified byte-exact; new audit
   `scripts/audit_e97_hybrid_conversation_smoltalk2_shingles.py` (committed
   `7bd57581`).

## 5. Phase 5 deliverables

- Non-trainable masked-SFT render of the audited collection
  (`scripts/render_e97_hybrid_conversation_v2_sft.py`; `W/sft-render`
  manifest sha256
  `7bd0e15bd9d54995cbcd68b47c1d735c43a2d85a546e0fba39f5523604bcffef`),
  fail-closed bound to the plan, the reconstruction audit, and the overlap
  audit.
- 64K boundary-aware pack render: 488 packs / 5200 records / zero oversize
  exclusions (`W/packs-64k`, pack manifest sha256
  `a16df791e4e8995be830d8bc1b773c93077604f852975d5f3c23213fa9bccb06`).
- Independent pack validation PASS
  (`scripts/validate_e97_sft_packs.py`; `W/packs-64k/validation.json`
  sha256
  `c1d665bbcaa6b4c5ef0dd11d51df502d98412dd531dfb4d36d08cf6cacb493f8`).
- Cohort-spec entry:
  `docs/validation/e97-hybrid-conversation-cohort-spec-v2.md` —
  recommendation **~2-4% of prep assistant targets via x4 whole-record
  epochs** (2.78% of an extension-prep-v1-scale ~131.2M-target budget;
  x3-x6 epochs stay within the band), with the dosage caveat that
  date/time records repeat a narrow window of live observed dates.

## 6. Honest continuation point

The >=5,000-verified-records goal is MET (5200). The >=1.5M assistant-target
goal is NOT met and is UNREACHABLE under the frozen v2 plan: 5200 records
cap at ~912K targets. Reaching 1.5M requires ~3,400 additional verified
records — a new frozen plan/distribution, which is an operator scope
decision and was not taken unilaterally. Any future extension resumes
rather than restarts: the collect is checkpointed per episode, and a
re-freeze carrying the same case ids keeps all 5200 verified episodes
valid. Two further notes for the next session:

- Widen the date-observation window: the full collect ran inside a few
  live dates, so all date/time records share a handful of observed dates
  (benign at the recommended share; documented in the cohort spec).
- The `u-ymd` lesson is now structural: `pilot_sample` guarantees
  kind coverage, and the regression tests pin both the clock check and
  the coverage rule.

## 7. Commits (scoped; never `git add -A`)

- `5158f8b9` — u-ymd clock-check regression fix + tests; pilot kind
  coverage (builder + tests only).
- `7bd57581` — SmolTalk2 shingle-exclusion audit, non-trainable SFT render,
  v2 plan guard in the overlap auditor (scripts + tests).
- This report and the cohort spec (docs only).

Everything produced is a verified-candidate-not-admitted authority:
`training_eligible: false` throughout; no optimizer updates; no packing
authorization beyond the diagnostic render validated here.
