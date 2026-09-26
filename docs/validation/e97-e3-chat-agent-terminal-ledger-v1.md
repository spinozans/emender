# E3 chat-agent arc — terminal ledger

**Schema:** emender-e3-chat-agent-terminal-ledger-v1
**Prep:** `e97-e3-chat-agent-prep-v1/e3-preparation-blend` (manifest `0a0e7945`, 295,666
records / 450.8M tokens / 127,145,988 targets / 18 cohorts; real-human-chat 27,136
records / 20.0M targets = 15.73% — the combined WildChat+LMSYS authority `4521b87d`
per admission `4984f89e`; keys 2300002/2300003/2300004/2300007; proposal v2
`7d4ac2b9`, committed `abf73407`).
**Parent:** E2-u512 `5ea4e078` (operator directive, E2 terminal addendum).

## The four segments

| Segment | Checkpoint | Updates | Notes |
|---|---|---|---|
| seg1 (u128) | `8e0d425c` | 128 | clean run after a correctly-denied launch during the screen |
| seg2 (u256) | `3770621c` | 128 | u256 full gate below |
| seg3 (u384) | `f309bebf` | 128 | first production segment on the qualified chunk-2048 config (patch commit `3cf6439e`; tiling-class per `docs/validation/e97-throughput-qualification-standard-v1.md`; measured 71.4→64.9 s/step) |
| seg4 (u512) | `34660c0c` | 128 | final segment; the u512 full gate below |

All four audits: `passed-training-not-promoted`.

## The full-gate and juncture cards (all legs, same frozen instruments as E2)

| Leg | u128 | u256 | u384 | u512 |
|---|---|---|---|---|
| Stage-B valid / correct | **12/14** / 6 | 11/14 / 6 | 11/14 / 6 | **8/14 / 3** |
| Execution (96-ep panel) | — | 63/96 | — | 64/96 (at floor; seg-controls 64) |
| Chat probe | 1/3 (two-tool PASS — first ever) | **2/3** (greeting+two-tool; best ever) | 1/3 | 1/3 (two-tool 4/4) |
| Retention | 1.526 | 1.520 | 1.517 | 1.524 |
| Doc-NLL (8-doc panel) | 2.560 | **2.5559** | 2.5605 | 2.5646 |

## Verdicts

- **u256 full gate: FAIL** — Stage-B 11/14 valid (floor 12) and 6/14 correct
  (floor 10); execution 63/96 (floor 64, one below).
- **u512 full gate: FAIL** — Stage-B 8/14 valid, 3/14 correct: a terminal
  collapse, deeper than any prior arc's terminal reading (E1 10/6, E2 10/6).
  Execution HELD at exactly 64/96; retention held (1.524); doc-NLL mildly
  eroded (+0.0079 vs baseline — 21x below the +0.167 drift class).
- **No promotion.** Third consecutive arc to fail the frozen gate; the SFT
  line's ceiling is diagnosed as structural, not dietary or parametric
  (sfparam screen: correct-first-action 6-8/14 everywhere on lr/warmup axes;
  per-case flips checkpoint-idiosyncratic).

## The dissociation (E3's scientific yield)

The u512 collapse is *specific*: long-horizon agentic execution (96-episode
suite, 64/96), multi-step tool composition (two-tool case passed ALL FOUR
junctures — the operand-copy repair is durable through 512 updates), retention
(second-best band ever) and the anchor all substantially held while the
short-frame first-move precision (Stage-B) collapsed. Late-SFT on this mixture
class trades opening-move crispness for everything else. The failing
competence is exactly the one the RL leg's reward channel shapes.

## Operator rulings during the arc (receipts in the juncture dirs)

- u384 chat-trigger override (greeting flip judged checkpoint-idiosyncratic
  oscillation, not regression) — `seg3-u384/chat-trigger-override-receipt.md`.
- u384 doc-NLL within-band receipt (+0.0039, E2's own band) —
  `seg3-u384/docnll-within-band-receipt.md`.

## Throughput program interlude (qualified/falsified levers, evidence-banked)

- **chunk-2048: QUALIFIED** (tiling-class, armB 1.13x) — shipped in seg3/seg4,
  promoted to canonical `3cf6439e`, production-verified ~9% end-to-end.
- CUDA graphs: parked after four distinct capture failures (autocast cache;
  static-output aliasing; checkpoint-recompute-inside-capture with RNG query;
  legacy-stream dependency) — forensics in the workspace.
- V-split: **DISQUALIFIED** — 0.51x (issue-bandwidth saturation at 60 programs)
  and 3e-4 loss divergence; the probe's original "gross combine bug" was
  exonerated by forensics (probe-metric miscalibration, 1-ulp class); the
  strengthened 5-phase validator is the template for kernel-tier probes.
- P=2 batching / smaller packs: recipe-level levers carried to the RL leg's
  proposal (machinery built, armE).

## Disposition

- **E3's product for the RL leg: the u256 checkpoint `3770621c`** — the best
  balanced card in programme history (chat 2/3, Stage-B 11/14, exec 63/96,
  retention 1.520, doc-NLL 2.5559). Published to HF as the honest mid-arc
  candidate. The RL loop prototype (`e97-rl-loop-v1`) has already trained its
  first cycle on it (checkpoint `1e059464`).
- The 5,181-environment gym, teacher transport, DiLoCo sync machinery, and the
  continuous per-GPU operating model proceed under the RL leg's proposal
  (`e97-rl-leg-proposal-v1/DRAFT.md`).
