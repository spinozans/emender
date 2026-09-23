# Hybrid-Conversation Cohort Specification v2 (scaled seam collection)

Status: frozen cohort entry, staged for the next chat-agent-arc prep freeze.
Collection authority:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-hybrid-conversation-collection-v2`
(plan-private sha256
`c146ba3b9e7e95d2e24a23fe4cbbf048ee7e2e27b07ed61b68b39ebbb2f64eba`,
source commit `5158f8b9`, superseding the v1 320-record pilot entry at
`0.01%` of the extension-prep-v1 target budget).

## 1. What the cohort teaches

The scaled hybrid-conversation seam family: conversational openers that flow
into minimal tool use (chat-into-tool), clock/date grounding through real
`date` executions (all six observation formats incl. UTC `u-ymd`),
pure-chat mirrors that must answer without tools, small file-op
conversations, and multi-step tool-then-chat continuations. Every episode is
authored (no model generation), executed through real Pi 0.87.0 under the
frozen tool surface, and machine-verified end to end; the independent
auditor re-derives every emitted frame from the plan spec plus the recorded
observations.

## 2. Verified collection (5200/5200, zero rejections)

| item | value |
|---|---|
| records verified / attempted | 5200 / 5200 (0 rejected, 0 transient) |
| assistant target tokens | 911,713 (175.3 per record) |
| total tokens | 30,752,966 |
| native tool calls | 5,563 (authentic tool errors: 0) |
| families | 19 (greeting 650, tool-then-chat 850, date-question 750, chat-into-tool-read 300, chat-into-tool-date 300, smalltalk 330, bash-derive 330, read-fact 330, pure-chat 400, edit/write-verify 240, remaining seam families 720) |
| candidate authority manifest sha256 | `f88ff570d3cee472a691df018bd72db30710f434a63418c571d0d0d6a014141c` |
| collection summary sha256 | `54a4317598e3f494a08bd08679c549fa40b4f8ac43ab6c5ae5425698b9ccc804` |

## 3. Audit receipts (all fail-closed, all `training_eligible: false`)

- **Full independent reconstruction audit PASS** —
  `audit.json` sha256
  `93c41cc9dbeb5107e4f63acf9348224b2949ca26bd25592487a9a5b24d6d6991`:
  5200/5200 records machine-replayed from the recorded Pi transcripts,
  dynamic v2 finish frames (observation-exact and observation-date, all
  formats) re-derived, token/mask re-encoded and matched against the
  authority payload, global sequence dedup 5200/5200, summary
  reconstruction; status `qualified-candidates-not-admitted`.
- **Protected-panel overlap audit PASS** —
  `overlap-audit.json` sha256
  `4b21d8ead45d45eceb298707efa13c75fe0d22330c2a6f59b5498adc6f94af5b`:
  zero significant entity collisions against the three fixed protected
  panels and the Stage-A panel (`07cc1d58...`); 75 trivial sub-8-byte
  numeric fixture scalars reported (exempt policy).
- **SmolTalk2 shingle exclusion PASS** —
  `smoltalk2-shingle-audit.json` sha256
  `969d5f67a4f5cabba656c4413d2f9465b6fc9bc8e7e03da223df0e62cf14f62d`
  (`scripts/audit_e97_hybrid_conversation_smoltalk2_shingles.py`):
  zero content collisions between the collection (30,550,166 in-record
  width-40-token windows) and the production-admitted SmolTalk2 corpus
  `e97-4b-smoltalk2-admitted-v1` (620,389 records / 801,365,124 tokens /
  777,171,237 windows), every hash hit verified byte-exact; the cohort adds
  new signal, it does not duplicate conversation-rehearsal content.

## 4. Non-trainable SFT render and 64K pack validation

- Render (`scripts/render_e97_hybrid_conversation_v2_sft.py`,
  `sft-render/manifest.json` sha256
  `7bd0e15bd9d54995cbcd68b47c1d735c43a2d85a546e0fba39f5523604bcffef`):
  byte-exact masked-SFT copy of the audited candidate authority, metadata
  rows tagged `source=hybrid-conversation-rehearsal-v2`; fail-closed bound
  to the plan, the reconstruction audit and the overlap audit.
- 64K boundary-aware packs: 488 packs / 5200 records / zero oversize
  exclusions; pack manifest sha256
  `a16df791e4e8995be830d8bc1b773c93077604f852975d5f3c23213fa9bccb06`.
- Independent pack validation PASS
  (`scripts/validate_e97_sft_packs.py`, per-record mask sums verified):
  `packs-64k/validation.json` sha256
  `c1d665bbcaa6b4c5ef0dd11d51df502d98412dd531dfb4d36d08cf6cacb493f8`.

## 5. Recommended entry: ~2-4% of assistant targets, x4 whole-record epochs

The v1 pilot entry was 0.01% of the extension-prep-v1 budget — a
presence-only dose. The verified v2 pool supports a real seam dose:

- **Share recommendation: ~2-4% of prep assistant-target tokens** (the
  operator's band for this seam), achieved with **x4 whole-record
  repetition epochs** of the full 5200-record pool (never in-record
  repetition), matching the prep precedent for seam families (x4).
- Math at extension-prep-v1 scale (~131.2M target budget): x4 epochs =
  3,646,852 targets = **2.78%** (x3 = 2.09%, x5 = 3.47%, x6 = 4.17% —
  stay within 3-5 epochs for the band).
- Dosage caveat: date/time records are grounded in real observations from a
  narrow collection window (a handful of live dates); repetition epochs
  multiply those same observed dates. That is inherent to clock-grounded
  verified data and acceptable at this share, but a future collection
  round should widen the observation window (see the continuation point).
- In-collection span repetition is expected and verified benign:
  whole-record sequence dedup is the collection guarantee (5200/5200
  distinct), while shared templates grounded by the same runtime date
  repeat sub-record spans (30.55M windows collapse to 1.12M distinct).

## 6. Continuation point (honest gap)

The v2 plan caps at 5200 records ≈ 912K assistant targets. The collection
goal of ≥1.5M assistant targets requires ~3,400 additional verified records,
i.e. a NEW frozen plan/distribution (operator scope decision, not taken
here). Everything already verified stays valid under any extension: the
collect is checkpointed per episode and a future plan re-freeze carrying the
same case ids resumes rather than restarts.
