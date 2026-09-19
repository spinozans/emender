# E97 extension-prep-v1 — staged segments-9+ preparation authority (v1)

Status: STAGED, NOT ADMITTED. This preparation is the continuation authority for
the extension arc (segments 9+ after the v10 repair-v9r arc's gates). The
extension gets its own proposals and admissions after v10, with operator
sign-off; nothing here trains, packs for training, or promotes anything. All
work was CPU-only; the v10 training arc's GPUs were never touched.

## 1. Preparation identity

- Preparation root:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-extension-preparation-v1`
- Prep manifest sha256
  `115fb75eedb8512da14c94290095649b879ac64056b2619bfac9820238836ada`
- Counts: **987,986 records / 1,069,134,639 tokens / 740,110,362 assistant
  target tokens**; schema `emender-e97-tulu3-masked-sft-v1`; tokenizer
  `p50k_base`; `training_eligible: false`, `packing_authorized: false`,
  `optimizer_updates_authorized: 0`.
- Packs: boundary-aware whole-record 64K packs
  (`emender-e97-sft-boundary-aware-packs-v2`), packs manifest sha256
  `7958b35129f18cc10423cc4cdcce7827c8ab0fae1bac0ec4d6420c0691b80898`;
  17,351 train packs, zero oversize records, `diagnostic_system_gate:
  cpu-system-gate`. Independent validation PASSED
  (`scripts/validate_e97_sft_packs.py` receipt `packs/validation.json`:
  17,351 packs / 987,986 records; per-pack target sums re-verified from the
  token/mask payloads; status `pass`).
- Full prep audit PASSED:
  `scripts/audit_e97_extension_preparation_v1.py` receipt
  `audit.json` sha256
  `1be90d4d0dba81bf8af4aa1e189e4e78e168228fb061ed1bad78cab2d78c3838`,
  status `qualified-preparation-not-admitted`. The auditor machine-verifies:
  manifest + payload identities (bytes + sha256 per output), index
  accounting (contiguous offsets, split==0, n/target sums), per-record mask
  sums for all 987,986 records, cohort-table reconstruction from the
  metadata + index, every cohort-authority binding re-hashed from disk, the
  selected-curriculum provenance triple, and the parent checkpoint identity.
- Parent binding (v2 precedent): representation-bridge checkpoint
  `checkpoint_agent_sft_u000032_loss_0.5302.pt` sha256
  `9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa`.
  Extension segment proposals rebind their actual chained parents at their
  freeze points (v10 terminal audited checkpoint onward).
- Sizing note (honest): full pool strength over the named cohort list sums to
  ~1.07B tokens / 740.1M supervised targets. The earlier "~1.8-2.2B tokens"
  working estimate is not reachable from these sources without repetition or
  additional unnamed data; no additional source was invented. At the v10 arc's
  measured ~58.5M scheduled input tokens per 128-update segment, this prep
  supports ~18 such segments before repetition.

## 2. Cohort table (records / tokens / assistant targets / share)

| Cohort | Records | Tokens | Target tokens | Share | Provenance |
|---|---:|---:|---:|---:|---|
| conversation-rehearsal | 614,335 | 793,658,451 | 644,276,669 | 87.0% | e97-4b-smoltalk2-admitted-v1 FULL train split (seed 613117, 700M target budget) |
| instruction-rehearsal | 248,021 | 89,558,610 | 45,340,400 | 6.1% | e97-4b-pi-instruction-mix-v2 (seed 913001, full eligible train split) |
| openhands-translated-rehearsal | 2,804 | 125,065,925 | 31,428,617 | 4.2% | OH-with-thinking enriched authority (seed 920001, 58M budget; all 2,804 distinct instance ids under the reader's instance dedup) |
| cumulative-recovery-rehearsal | 60,173 | 23,340,721 | 9,167,950 | 1.2% | e97-4b-pi-cumulative-recovery-mix-v1 (seed 913003, full eligible train split) |
| compositional-rehearsal | 41,080 | 15,210,608 | 7,335,698 | 1.0% | e97-4b-pi-compositional-retention-mix-v1 (seed 913002, full eligible train split) |
| live-aligned-rehearsal | 19,810 | 5,517,358 | 2,376,849 | 0.3% | e97-4b-pi-live-aligned-all-assistant-v1 (seed 913004, full eligible train split) |
| pi-native-curriculum | 823 | 8,324,928 | 78,768 | 0.01% | pi-native-curriculum-2000-selected-v1 under the v2 family filter |
| hybrid-conversation-rehearsal | 320 | 1,869,583 | 40,067 | 0.01% | pi-native-hybrid-conversation-collection-v1 (NEW; audited below) |
| pointerchase-rehearsal | 240 | 1,497,400 | 34,096 | <0.01% | pi-native-pointerchase-collection-v1 |
| longcopy-rehearsal | 180 | 3,837,931 | 7,224 | <0.01% | pi-native-longcopy-collection-v1 |
| loopbreak-rehearsal | 150 | 921,195 | 8,171 | <0.01% | pi-native-extracterror-collection-v1 |
| reasoning-rehearsal | 50 | 331,929 | 15,853 | <0.01% | pi-native-reasoning-rehearsal-collection-v1 (T2 pilot) |
| **total** | **987,986** | **1,069,134,639** | **740,110,362** | 100% | 12 cohorts |

Cohort dispositions follow the v10 integration summary's corrected scrub:
grounded-authored-rehearsal and representation-bridge-rehearsal stay DROPPED
(80%/57% sampled supervised-target OH-only vocabulary, fail-closed); the raw
OH cohort stays REPLACED by the translated + replay-verified authority (here
thinking-merged). Whole records are emitted in the deterministic
weighted-fair-by-token-share interleave, so greedy 64K boundary packing yields
cohort-mixed packs.

## 3. Hybrid-conversation collection (verified-candidate-not-admitted)

Collection: 320 cases (240 authored curriculum + 80 validated teacher cases)
executed teacher -> real-Pi -> verified, 0 rejections, 320 verified records /
239 native tool calls / 40,067 assistant target tokens / 1,869,583 tokens;
plan sha256
`aa64cfbd2ec0ecbaab134d3f167932eb98459e76beab967270529fac9340586c`
(source commit 54db4ffc).

- **Reconstruction audit PASSED**
  (`scripts/audit_e97_pi_native_curriculum.py`):
  `pi-native-hybrid-conversation-collection-v1/audit.json` sha256
  `64fa533a19c3a611cc1edf9f3e697ba0b2d8a6bebb3e1438e436c65363ec80e5` —
  320/320 records machine-replayed from the recorded Pi transcripts,
  episode reconstruction, token/mask re-derivation, per-record metadata,
  hybrid answer/observation/pure-chat oracles and analysis-cap/grounding
  verification; status `qualified-candidates-not-admitted`;
  `training_eligible: false`.
- **Protected-panel overlap audit PASSED**
  (`scripts/audit_e97_hybrid_conversation_overlap.py`):
  `overlap-audit.json` sha256
  `b13beba0eae016172a7296e0c2186aba7e0574a4ae6fbb5055c5ba706b895b2d` —
  zero significant entity collisions against the three fixed protected panels
  and the Stage-A panel (07cc1d58...); 23 trivial sub-8-byte numeric fixture
  scalars reported under the exact-and-significant-entity-v1 policy
  (ledger precedent: trivial alpha/1 content); status `pass`.

The collection is therefore a verified-candidate-not-admitted authority, bound
into the prep through its candidate-authority manifest sha256
`7e5fec2a67b1e7ba45bcbcc866d86cb133bb0c454cdabdb274978eb42ebc0b1d` (bound in
the prep manifest's `extra4_rehearsal` entry and enforced by the prep audit).

## 4. OH-with-thinking merge summary

The OH cohort is the T1 translated + replay-verified OpenHands collection
(5,969 raw -> 5,940 translated -> 5,181 replay-PASSED -> 5,087 sealed
records; 231,652,231 tokens / 57,814,225 supervised targets) with the
recorded model thinking merged into the Analysis channel
(`scripts/merge_e97_oh_recorded_thinking.py`, commit d0aaa43b):

- 203,240/203,240 assistant turns aligned against think-export.jsonl, 0
  mismatches, 0 fills needed (the translator already carried the model's own
  thinking), 0 degenerate drops; payloads byte-identical to the sealed T1
  authority (`byte_identical_to_source: true`).
- 1,464 think pseudo-action thoughts carried in canonical Arguments;
  observe-then-quote coverage measured at 97,592/198,113 checkable turns.
- Enriched authority:
  `e97-extension-prep-v1/oh-thinking-enriched-authority` manifest sha256
  `f226f3b3e1a45352dd7a9f302c741890e479827182c2a27b6919ddf4344b8fd0`
  (provenance: merge report
  `cd766e19919a9a17dc5d783bbb797d1ac1ae6f2e79b9455a1b02e9101537672`,
  think-export
  `4c305827ea386767ce054ba9c301902ea140664e21fbc30c63f20c7348143e47`,
  source T1 manifest
  `a179c140b4e8efa77a8163574bb78d4fe34dea91adfeb0439e8339d5002be1fb`).
- The prep binds the merge provenance verbatim in its manifest
  (`openhands_rehearsal.source_provenance`) and consumed all 2,804 distinct
  SWE instance ids (31,428,617 target tokens) under the preparer's
  shortest-first instance dedup.

## 5. Post-v10 LR screen harness (STAGED; NOT RUN)

Location:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-extension-prep-v1/lr-screen-post-v10-v1/`

- `measurement-plan.md` — rates **1e-5 / 3e-5 / 1e-4**, exposure-matched
  32-update probes on THIS prep (same frozen schedule, sampler key 1403761,
  world 8, context 65,536), fresh optimizer state from the v10 arc's terminal
  audited checkpoint; measures frame validity (Stage-B valid first frame +
  correct first action), full Stage-B probe results, and conversation /
  development NLL drift against a re-baselined parent panel (T2 caveat: the
  thinking cohorts enter here, so the conversation-NLL panel must be
  re-baselined at the probe parent); read-out rules; no promotion, no retry.
- `schedule-1403761-probe.json` (sha256
  `bfe4eff6a6866ce07435562681ca5baac762842316d4a5e0f61631b09a2d9bff`) — the
  frozen 32-update schedule; key 1403761 was screened with
  `scripts/screen_e97_extension_probe_keys.py` (3,461 keys) and satisfies the
  window-coverage floor (majors in every update, minors at least once per
  window), independently re-verified with the real planner.
- `commands.sh` — fail-closed ordered sheet: per rate, freeze the 32-update
  probe proposal -> audit with
  `scripts/audit_e97_pi_native_repair_generic.py` (the same generic-auditor
  path as the v10 segment proposals) -> scoped commit/push -> ADMISSION with
  the operator's typed statement -> run the probe.
- `run-probe.sh` — per-rate launcher: admission/parent/schedule identity
  checks, 32-update training run on the qualified v9r recipe, Stage-B panel
  eval, and a checkpoint-bound NLL probe (refuses panels not regenerated for
  the exact probe checkpoint — the recorded v9-full defect class).
- Preconditions: v10 arc complete and audited; GPUs released; operator
  sign-off on the extension prep (new data authority). Nothing was executed.

## 6. Provenance and overlap-audit coverage (honest statement)

The prep inherits the same coverage pattern as every prior repair prep (v2
included): the selected-curriculum provenance triple is bound
(`8cf83db8d608f6af74bbd2cbe60206ef435fb3003ab2bb63f4eadab17eb05562` /
`de977726f0758206ee9defe9c3687d2485e3ba2b80eb9ef9ceafb065c5f6646b` /
`9c104d73b104f4d29a83ca1472b66ef2df681e020a2b9688eb4f87c6ef814a78`),
per-collection audits cover the machine-verified candidate cohorts:

- reasoning-rehearsal: collection audit `f62c86ef...` (50/50 verified) +
  protected-overlap audit `d8ed9720...` (zero significant collisions),
- hybrid-conversation-rehearsal: reconstruction audit `64fa533a...` (320/320)
  + protected-overlap audit `b13beba0...` (zero significant collisions),
- openhands-translated-rehearsal: machine replay verification (T1 funnel) +
  OH-vocabulary scrub + the thinking-merge alignment audit; **no
  protected-panel entity overlap audit exists for it** — no such audit ever
  existed for any prior OH cohort either; this is the coverage gap the
  operator signs off, stated plainly (unchanged from the v10 summary),
- the production-admitted conversation/instruction/mix pools (SmolTalk2
  full train split, instruction/compositional/cumulative/live-aligned) carry
  their own source-level audits and admission manifests; no whole-prep
  protected-overlap audit is claimed.

## 7. What is staged next (post-v10, operator sign-off required)

1. v10 arc gates pass and the arc terminates with an audited checkpoint.
2. The LR screen above measures 1e-5 / 3e-5 / 1e-4 on this prep (operator
   admission per probe).
3. The extension arc's segment proposals freeze against this prep (chained
   parents from the v10 terminal checkpoint), audit through
   `scripts/audit_e97_pi_native_repair_generic.py`, and admit only with the
   operator's typed authorization statement. The unchanged evaluation
   protocol (juncture detectors every 128 updates; full frozen dual gates at
   u256/u512/u896/u1024-equivalent boundaries; fail-closed stopping) applies,
   with the conversation-NLL panel re-baselined when the thinking cohorts
   first enter training.
