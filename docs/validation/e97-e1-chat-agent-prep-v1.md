# E97 E1 chat-agent arc prep v1 — the chat-agent arc's data authority, keys, probe panel, and staged launch package

**Status:** STAGED. The preparation, keys, schedules, probe panel, segment-1
proposal, and launch command sheets are complete and audited. Nothing is
admitted, trained, or executed; admission of any part requires the operator's
explicit sign-off. All work was CPU-only; the GPUs stayed idle and reserved.

Work dir: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-e1-chat-agent-prep-v1/`
(STATUS.md there is kept current). Commits: `80db924d` (machinery), `2e646660`
(frozen segment-1 proposal).

The E1 arc = v6's PASSING recipe proportions applied to fresh, verified pools at
few-hundred-million scale, plus the new chat-seam families, with the operator's
conversational acceptance tests as a standing probe. Chained from the bridge
parent `9b78628d...` (full retention headroom — the v6-passing chain), lr 1e-5,
4 × 128 updates (u512 arc end).

## 1. The E1 preparation (deliverable b)

`e97-e1-chat-agent-prep-v1/e1-preparation`: schema `emender-e97-tulu3-masked-sft-v1`,
status complete, `training_eligible:false`, `packing_authorized:false`,
`optimizer_updates_authorized:0`.

- **Identity:** manifest sha256 `07dea664375f89d9c81e1705f83532e41d3a108791e068579170affc9a2877c4`
- **Scale:** 414,763 records / 378,205,117 tokens / **131,230,771 supervised targets**
  (within the ~130-150M target window; the token total is ~11% above the ~340M
  estimate — see §6).
- **Packing:** 6,454 whole-record boundary-aware 64K packs (v2 schema, zero oversize
  exclusions), pack manifest `36b6a6b2f3000f555e749d2dd2bd9114e3ce28fa1e4d5e1e1c1eb563b3c24424`;
  independent pack validation PASS (6,454 packs / 414,763 records, per-record mask
  sums verified): `packs/validation.json`.
- **Generic prep audit PASS:** receipt `e1-preparation/audit.json` sha256
  `352d882bc08ad7f9f638f4e99f4ea1eb532135f1f0efc24e13fcd2c92bd90c6e`
  (manifest identity, payload identities, index accounting, per-record mask sums,
  cohort-table reconstruction, every cohort-authority binding re-hashed, the
  selected-curriculum provenance triple, parent checkpoint identity).
- **Emission:** weighted-fair-by-token-share interleave across all 14 cohorts,
  greedy next-fit packing; repetition is whole-record only (never in-record).

### Cohort table (exact shares and repetition epochs)

| cohort | records | targets | share | rep epochs | source |
|---|---:|---:|---:|---:|---|
| conversation-rehearsal | 37,336 | 39,000,000 | 29.72% | 1 | SmolTalk2 admitted v1, **fresh seed 613119**, 57,196 prior-draw identities excluded |
| cumulative-recovery-rehearsal | 180,519 | 27,503,850 | 20.96% | 3 | e97-4b-pi-cumulative-recovery-mix-v1 (full train pool) |
| compositional-rehearsal | 123,240 | 22,007,094 | 16.77% | 3 | e97-4b-pi-compositional-retention-mix-v1 (full train pool) |
| **verified-raw-oh** | 1,715 | 19,697,664 | **15.01%** | 1 (fresh draw) | NEW authority, see §2 |
| **tooltalk-rehearsal** | 10,236 | 10,817,522 | **8.24%** | 1 | NEW full-intake render, see §3 |
| live-aligned-rehearsal | 39,620 | 4,753,698 | 3.62% | 2 | e97-4b-pi-live-aligned-all-assistant-v1 (full train pool) |
| grounded-authored-rehearsal (restored) | 10,332 | 3,704,364 | 2.82% | **4** | full 2,583-record pool |
| representation-bridge-rehearsal (restored) | 8,892 | 3,394,640 | 2.59% | **4** | full 2,223-record pool |
| hybrid-conversation-rehearsal | 1,280 | 160,268 | 0.12% | 4 | 320-record verified collection (239 native calls) |
| pi-native-curriculum | 823 | 78,768 | 0.06% | 1 | family-filtered selection (unchanged) |
| reasoning-rehearsal | 200 | 63,412 | 0.05% | 4 | 50-record verified pilot |
| pointerchase-rehearsal | 240 | 34,096 | 0.03% | 1 | verified family |
| loopbreak-rehearsal (extracterror) | 150 | 8,171 | 0.01% | 1 | verified family |
| longcopy-rehearsal | 180 | 7,224 | 0.01% | 1 | verified family |

Restored pools combined: **7,099,004 targets = 5.41%** (≥5% required, 4 ≤ 5
whole-record epochs; presence saturates the slice at ≥1% per the dose screen).
**ZERO instruction-mix and ZERO translated-Pi-frame-OH** — neither convicted
cohort appears (verified in the freeze writer as a fail-closed check).

## 2. The verified-raw-OH cohort authority (deliverable a1)

`e97-e1-chat-agent-prep-v1/verified-raw-oh-authority`, manifest sha256
`fa6a47daf69a3671ca523966cee38774db576d8d50d06520094b853759ee22e4`.

- **Selection:** the 5,087 replay-verified OpenHands trajectories, in their
  NATIVE OH format (byte-exact `tokens.bin`/`loss_mask.bin` copies — NOT
  translated). The verified ids come from the translated collection's records
  (`e97-oh-pi-native-translation-v1/candidate-authority`, manifest
  `a179c140...`); the **id mapping** is `translated (instance_id, trajectory_id)
  → raw trajectory_identity "open-swe:<uuid>"`, cross-checked
  `instance_id == raw problem_key[1]`, 1:1 with no orphans on either side,
  documented record-by-record in `id-mapping.jsonl`.
- **Census:** 5,087 records / 58,533,278 targets / 254,441,721 tokens;
  4,704 train-split + 383 raw-validation-split (split flags preserved; the
  preparer skips validation rows as for the parent pool); 882 non-verified raw
  records excluded.
- **Prep draw:** seeded uniform draw (seed 914101) of 1,715 train records /
  19,697,664 targets = **15.01%** of prep targets (15-25% required; the v9
  capture came from UNVERIFIED full-pool records — this cohort is the
  replay-verified subset).

## 3. The tool-talk full-intake render (deliverable a2)

Raw downloads re-fetched from the pinned revisions (receipt
`tooltalk-intake/download-receipt.json`, per-file SHA-256 pins; raw files
deleted after rendering per the spike policy), licenses re-verified
(`license-receipt.json`: Toucan apache-2.0, Nemotron CC-BY-4.0, Tool-Reasoning
apache-2.0), rendered Pi-native by `scripts/render_e97_public_tooltalk.py`
(now emitting per-section payloads for masked supervision).

`tooltalk-authority`, manifest sha256
`fe0bfaffd8f00ab8011fc4a267c75686ac8caa04f4c862afeac3636d98294a9c` —
**10,236 records / 10,817,522 assistant targets / 34,238,152 tokens (8.24%)**,
five-line frames supervised, Protocol/System/User/ToolResult as context, the
spike's mechanical lint applied:

| source | rendered | dropped (lint) | selected | targets |
|---|---:|---:|---:|---:|
| Toucan SFT | 6,944 | 46 (32 invalid-frame, 14 oversized) | 6,898 | 6,417,735 |
| Nemotron-Agentic interactive_agent | 2,000 | 0 | 1,442 | 3,599,979 |
| Tool-Reasoning-31K | 3,200 | 1,298 (1,072 When2Call-tagged, 116 mid-orphans, 100 unmatched results, 10 invalid frames) | 1,896 | 799,808 |

Protected-panel overlap on the full intake: **PASS** (zero significant exact
scalars ≥8 bytes, zero significant normalized content overlaps; trivial
sub-8-byte scalars only — 55/66/150 per source, the spike's structural-noise
class).

## 4. Keys, schedules, window rule (deliverable c)

Training-data-only screen (fresh E1 range 1700001..) under the window-coverage
rule (majors ≥2% of targets in every update; minors every 32 updates), then the
real planner + the independent window-rule verifier:

| segment | key | schedule sha256 | input tokens | targets | window rule |
|---:|---:|---|---:|---:|---|
| 1 | 1770555 | `428bd0aac99d2ab37213323ece5daece95fdc1149957a56ac43bc15be910dc39` | 60,094,816 | 20,954,952 | OK |
| 2 | 1775117 | `69bfaa662564c90c85e4389f31c641c94a859ea660ab27ddd95ee2b3833609f9` | 59,979,480 | 20,342,936 | OK |
| 3 | 1787698 | `381c490c7d87a5c5073e3ab6bdc5dcb238ade8c8c47fa5e873fbf7eb553f7692` | 60,536,437 | 20,848,314 | OK |
| 4 | 1850767 | `5444adb41e90f4f0fbd885d33f4a128fbd87b5ee2c1f51000f284c3015f95eb2` | 59,951,722 | 20,806,540 | OK |

Majors: compositional, conversation, cumulative-recovery, grounded-authored,
live-aligned, representation-bridge, tooltalk, verified-raw-oh. Minors: hybrid,
longcopy, loopbreak, curriculum, pointerchase, reasoning. Admission rates:
50,000-88,000 keys screened per admit.

## 5. The chat-probe panel (deliverable d)

Frozen panel `configs/pi/e97-e1-chat-probe-panel-v1.json` (sha256
`21e958f1f926abc8ddc7be2428101b265852ec43ab6a4a6fbd86401a80b926ac`) +
evaluator `scripts/eval_e97_e1_chat_probe_panel.py` (7 unit tests in
`tests/test_eval_e97_e1_chat_probe_panel.py`; prompt construction verified
byte-identical to the canonical PiNativeEpisode codec).

The operator's three acceptance tests, deterministic, run per checkpoint on the
CPU llama.cpp port (greedy; the harness NEVER executes model-chosen commands —
ToolResults come from frozen fixtures or the harness's own `date`):

1. **Greeting** — reply in plain text with NO tool-call frame: a valid frame
   with `Action: finish` (tool==none), or a text-only response. Verification
   mirrors the hybrid-collection machinery (`parse_turn` + `semantic_turn` +
   canonical round-trip recorded).
2. **Date** — must call `bash date`; the harness feeds the real `date` output;
   the finish message must state the weekday computed at eval runtime from the
   real clock (deterministic per run).
3. **Two-tool mini-task** (hybrid-bash-derive pattern) — `read` the ledger
   fixture, `bash`-compute the operands, `finish` with the observed sum.

**v6 negative control (the probe discriminates):** the promoted v6-u96
(`d8146498...`, port q8_0 GGUF `da3b0d17...`) scores **0/3** —
`chat-probe-v6-negative-control/summary.json` sha256
`692d625c43f7900e87209822aad63e5c1efd13c783d78f93d49f9b1fb434a49a`. All three
cases fail with raw-OpenHands dialect capture: undeclared
`str_replace_editor` frames with hallucinated SWE-bench workspace paths on a
plain greeting, a date question, and the two-tool task. This is the same
emission pathology that poisoned v9 segment 1 — visible now on chat cases, so
the panel discriminates exactly as required.

## 6. Sizing note (approved by the operator)

Targets 131,230,771 sit inside the ~130-150M window. Tokens 378,205,117 carry
the share-floor overshoot the operator accepted: the arc was sized "a few
hundred million tokens, a few hours", and the measured pool ratios (verified-
raw-OH 4.32 tok/target, restored pools 7.14-9.77, tool-talk 3.16, conversation
1.23, fillers 2.07-2.55) make the explicit share floors jointly infeasible
inside 340M tokens; the share constraints were honored and the target window
held. The seam-families constraint ("~2-3%") is a TOKEN share (operator-
confirmed): at x4 whole-record epochs the reasoning pilot + hybrid collection
hold 8.81M tokens = 2.3% of prep tokens, matching the restored-pool light-
repetition precedent — as a target share it is arithmetically impossible. The
full 320-record hybrid collection enters (the operator's "239 recs" was the
audit's native_calls figure).

## 7. Segment-1 proposal + staging (deliverable e)

- **Frozen + audited + committed:** `configs/pi/e97-e1-chat-agent-segment1-proposal-v1.json`,
  schema `emender-e97-e1-chat-agent-segment1-proposal-v1`, sha256
  `2b87b95cb958fd417a4e21266e2145b1a6f32cb6119a3790a85f121791115cec`;
  generic-auditor receipt
  `61f252259557b3741f7c7e2e1e7344c6f8c081725067e467dfb4fd519c330dd9`
  (`qualified-proposal-not-authorized`); commit `2e646660`. 128 updates,
  bridge parent, lr 1e-5, key 1770555, window-coverage stratification, frozen
  dual gates unchanged, 40+ disk-verified identity bindings, the convicted
  cohorts fail-closed excluded, the chat-probe panel bound.
- **Staged (NOT executed):** `e1-arc-staging-v1/` — freeze-time chained-parent
  writer for segments 2-4, per-segment command sheets (segment 1's freeze/audit/
  commit steps marked DONE), run-dir/juncture/recipe templates (juncture =
  Stage-B + conversation NLL probe + chat-probe panel, with the per-checkpoint
  panel-binding and per-checkpoint GGUF-conversion guards).

## 8. Exact launch commands (for the orchestrator, after the operator's sign-off)

Segment 1 (admission onward; from `e1-arc-staging-v1/segment1/commands.sh`):

```bash
# STEP 4 — ADMISSION (operator types: I authorize the exact 128 update proposal.)
/home/erikg/emender/.venv/bin/python scripts/admit_e97_pi_native_training_proposal.py \
  --proposal configs/pi/e97-e1-chat-agent-segment1-proposal-v1.json \
  --proposal-sha256 2b87b95cb958fd417a4e21266e2145b1a6f32cb6119a3790a85f121791115cec \
  --preparation /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-e1-chat-agent-prep-v1/e1-preparation \
  --output /mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-e1-chat-agent-segment1-training-admission-v1 \
  --authorization-statement "I authorize the exact 128 update proposal."
# STEP 5 — admitted schedule + identity check; STEP 6 — instantiate run dir +
# bash $RUN/run.sh; STEP 7 — training audit; STEP 8 — juncture (Stage-B +
# conversation NLL probe + chat-probe panel); STEP 9 — frozen decision point.
# Segments 2-4: run their command sheets at their freeze points (chained parents).
```

## 9. Residual risks (stated plainly)

- **Raw-OH capture risk (accepted by the operator with this exact framing):**
  verified-raw-OH at 15% re-tests the v9 failure mode, and the differences vs
  v9 are what make it a measured risk rather than a repeat: **verification**
  (the replay-proven 5,087-record subset, not the unverified full pool),
  **dose** (the 15% floor vs v9's 17.1%), and **context** (a Pi-native-majority
  diet plus chat-probe/Stage-B junctures that catch capture fail-closed at
  u128, hours into the arc — not after it).
- **Filler repetition:** cumulative/compositional at 3 epochs (121K distinct
  records, 38% of targets) approaches v5's memorization exposure per record;
  the conversation NLL probe + chat-probe panel at every juncture are the
  detectors, fail-closed.
- **Chat-probe panel has no passing baseline** yet (only the 0/3 v6 negative
  control); the first E1 juncture establishes it.
- **Token overage** (~11% above estimate) documented in §6.

## E1 arc terminal record (fail-closed stop at u384)

- Segment 1 (u128): trained+audited (5ee1e7d3), retention 1.522 best-in-class, Stage-B 7/14+3/14 (transient), chat 1/3 (greeting PASS — first in programme history).
- Segment 2 (u256): trained+audited (853a3b95), full dual gate: Stage-B 10/14+7/14, execution 61-62/96 (bridge control 67), retention 1.515, chat 1/3 (greeting PASS; two-tool learned the bash step — failure class upgraded from missing-behavior to operand-fidelity).
- Segment 3 (u384): trained+audited (3bc3f1a5), juncture: Stage-B 6/14+4/14 (regression), retention 1.517 (held), chat 0/3 (REGRESSION — the previously-passing greeting now fails with undeclared_pi_action; phantom tool calls on plain-chat contexts).
- STOP: the frozen chat-regression trigger fired (0/3 below prior juncture). No promotion anywhere in the arc (gate floors never crossed). No further segments launched; no retries; thresholds unchanged.
- Diagnosis recorded: late-arc behavioral oscillation — the model became tool-eager in wrong contexts (emission-level dialect competition; retention untouched). u256 checkpoint 853a3b95 is the best measured E1 state and the designated E2 parent.
- Evidence: e97-e1-chat-agent-prep-v1/e1-juncture-evals/seg{1,2,3}-u128/, e97-e1-chat-agent-u256-dual-gate-v1/, audited runs e97-e1-chat-agent-segment{1,2,3}-training-v1/ (all passed-training-not-promoted).
