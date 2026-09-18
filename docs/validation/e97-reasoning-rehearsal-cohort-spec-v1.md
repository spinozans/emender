# Reasoning-Rehearsal Cohort Specification v1 (thinking traces for the Pi-native agent)

Status: frozen design under the integration plan
`docs/validation/e97-oh-translation-and-reasoning-integration-plan-v1.md` (T2).
Pilot: `pi-native-reasoning-rehearsal-collection-v1` (50 records, real-Pi
executed, deterministically verified, audited).

## 1. Codec verdict — the thinking channel already exists; no amendment needed

The plan asked whether bounded pre-action commentary breaks the Stage-B
"valid first frame" contract and, if so, to propose an amendment. **The
evidence resolves this differently: the Pi-native codec already defines the
thinking channel, and it is first-frame-valid.**

`scripts/e97_pi_native_codec.py` (`parse_turn`) requires every assistant turn
to be the strict five-line canonical frame:

```
Analysis: <JSON string — private reasoning_content>
Commentary: <JSON string — visible text or null>
Think: <bool>
Action: <tool name>
Arguments: <JSON object>
```

- `generate_turn` in the Stage-B/baseline path **requires the opening to be
  `Analysis: `** (`eval_e97_pi_native_baselines.py` rejects any other opening
  as `invalid_opening`), so an Analysis-led turn is not merely valid — it is
  the expected first frame.
- `validate_generated_turn` enforces the private-text cap (reasoning_content
  plus any `think` thought ≤ 2,048 tokens, ≤ 65,536 bytes: `analysis_cap`).
- Free text before `Action:` (outside the canonical frame) is invalid — the
  commentary channel is the `Commentary:` field, and the private channel is
  `Analysis:`. There is nothing to amend; the contract already has the two
  bounded channels.

**The actual gap is in the data, not the codec.** The pointerchase/loopbreak/
extracterror/longcopy collections authored every turn with
`reasoning_content=None` (`frame()` in
`build_e97_pi_native_curriculum.py` serialized `Analysis: null` on ~88–97% of
supervised turns in the promoted arc's prep — measured: repair6 prep,
bridge-rehearsal 100 null / 13 string, grounded 75/5, OH 26/0 per sampled
records). The promoted v6-u96 model therefore emits structurally valid frames
with an *empty* reasoning channel — it passed its gate while never thinking.
Separately, the v9-full OH cohort poisoned behavior with raw OpenHands
vocabulary (`str_replace_editor`, `execute_bash`) and bare two-line actions
that violate the five-line frame. Both defects are data defects.

## 2. Design — grounded, bounded, verified thinking

- **Channel:** `Analysis:` (private `reasoning_content`) carries the plan;
  `Commentary:` carries an optional short visible note. `Think:` stays a bool
  flag (the private `think` pseudo-tool remains available for RL later; not
  used in this cohort).
- **Grounding rule (the cohort-defining property):** an analysis is valid only
  if it *quotes or derives from actually observed values* — never invented.
  Enforcement is deterministic: every authored step carries
  `analysis_requires` (substrings that must appear, e.g. the observed port
  number, the computed sum); the collector rejects any turn whose analysis
  misses a required substring (`ungrounded analysis`), and the independent
  audit re-checks it (`verify_reasoning_case`) against the reconstructed
  episode.
- **Boundedness:** ≤ 2,048 tokens per analysis (codec cap); spec target
  1–3 sentences: name the observation, state the next concrete action, note
  the risk being avoided. No metaphysics, no restating the prompt.
- **Supervision rule:** analysis text is supervised **only** inside
  deterministically verified successful trajectories. Failure prefixes stay
  masked (existing `supervise_from` discipline) — the model never learns to
  imitate thinking that preceded an unverified outcome. Thinking text is never
  trusted on its own; the workspace snapshot, tool-result pairing, and final
  oracles remain the only success criteria.
- **Target family:** multi-step composition — the measured 4/16 axis. Chains
  of 4–6 tool turns where each turn's plan depends on the previous
  observation (config→derive→edit→verify; read→compute→write→register→read
  back; search-recover→compose). Composition plus error-recovery, with the
  recovery variant carrying a masked failure prefix.

## 3. Sources (in priority order)

1. **Authored deterministic rehearsal (this pilot):** the
   `build_e97_pi_native_curriculum.py --mix reasoning` family — scripted
   multi-step composition cases executed through real Pi
   (`NativePiToolBridge` + `serve_pi_native_tools`), grounded analysis authored
   per step and verified by `analysis_requires`, workspace snapshot oracles,
   and the full audit reconstruction. Scales deterministically; zero model
   generations; no GPU.
2. **Teacher scale-out (next, after arc restart):** DeepSeek proposes
   task specifications + per-step rationales via the existing
   teacher-pilot authority (`lunaroute/deepseek-4.1-flash`, validated task
   schema); each proposal executes through real Pi; only deterministically
   verified trajectories are admitted; rationales inherit the same
   `analysis_requires`-style grounding checks where the oracle allows.
3. **T1 think-export:** non-null OpenHands `think` actions salvaged by the
   translator (plan T1 step 2) as candidate commentary content — admitted only
   inside replay-verified translated trajectories.
4. **STaR self-sampling (follow-up):** sample k analyses from the served
   model on verified tasks, keep the ones whose trajectories verify, retrain.
   Requires the interactive server's GPU; deliberately deferred.

## 4. Rendering contract for the pilot

- System prompt: the collection's canonical `SYSTEM`
  (`Complete the task using the declared Pi-native tools...`) — unchanged; the
  five-line frame is the existing wire format, not a new prompt behavior.
- Every non-final supervised turn: non-null `Analysis` (grounded per
  `analysis_requires`) + short `Commentary` (some turns null Commentary to
  exercise the private-only style); `Think: null` flag; canonical Action/
  Arguments.
- Finish turns: plain `finish` frame (no analysis requirement) — the final
  answer stays the verified oracle string.
- Records are standard candidate-authority payloads (tokens/assistant_mask/
  index/metadata) produced by the shared `write_authority` path — directly
  consumable by the preparer as a cohort source.

## 5. Pilot result

Collection: `pi-native-reasoning-rehearsal-collection-v1` (plan sha
`b078ca3f3ff2d32b40723abdd84178ea3a3c802523f3ce0d960a7b85fc92a2f7`, frozen at
source commit `cc016d31`). **50/50 attempted, 50 verified, 0 rejected, 0
automatic retries** — 230 native Pi tool calls, 331,929 tokens, 15,853
assistant target tokens across three composition families
(20 config-derive with masked observe-the-failing-checker prefix, 20
cross-file compute-compose, 10 search-recovery-compose with masked direct-path
prefix). Independent reconstruction audit:
`PI_NATIVE_CURRICULUM_AUDIT 50 15853` (receipt `f62c86efb60de7c2df6329b043714000ea881c3c9d68adcda1c5f1eeb846ac4d`),
including mix-gated grounded-analysis verification (observe-then-quote:
each supervised analysis must contain its authored observed-value literals).
Protected-panel overlap audit PASS (receipt
`d8ed9720c9d55721a35420b43e1a753ea3d0034b1c35851f7f5a792363033ae6`):
zero entity collisions against the three fixed panels and the Stage-A panel;
13 trivial sub-8-byte numeric fixture scalars (e.g. operands `60`, `211`)
reported as non-entity structural noise per the ledger precedent.
Two rejected drafts were retained (`*-draft1*`, `*-draft2*`): draft1 authored
the observe-the-defect step inside the supervised region (violating the
post-prefix success oracle — fixed by masking it as the failure prefix);
draft2 froze against uncommitted source (fixed by committing first).
