# Completed Open-SWE source-native rebuild v1

Date: 2026-09-10. **Rebuild and complete source/mask validation passed.**
This is a real tokenized dataset, not another sizing audit. It is not yet an
admitted training recipe or an execution-qualified runtime.

## Published output

`/mnt/nvme2n1/erikg/sft/e97-4b-open-swe-source-native-fulltraj-v1`

Managed build `proc_2105` exited zero in **675 seconds**, including eight tests,
full validation, atomic no-replace publication, and final frozen-source checks.
The old dataset and raw authorities were not modified.

| Split | Complete trajectories | Input tokens | Assistant targets |
|---|---:|---:|---:|
| Train | **5,533** | 279,273,852 | **64,409,332** |
| Development | **436** | 21,960,568 | **4,983,961** |
| Total | **5,969** | **301,234,420** | **69,393,293** |

The retained set covers **3,159 distinct problems** and 243,158 assistant units.
Raw private text totals 24,173,075 tokens; public commentary totals 2,783,964 tokens.
These are constituent counts, not additive to serialized assistant targets.

## What is repaired

- Original execute_bash, str_replace_editor, think, and finish calls are retained.
  No bounded Pi read is substituted for a source full-file request; no source
  observation is attached to a different command. is_input and timeout arguments
  retain their original values.
- Paths, code payloads, observations/errors, commentary, reasoning, think thoughts,
  and final-message values retain their original content, whitespace and newlines.
- Public commentary is separate from private reasoning, not folded into it.
- Every retained record includes its complete source message history. No window
  restart, hidden truncation, or omitted preceding observation occurs.
- Original validation problems are reserved. Remaining problems use a frozen 5%
  hash assignment keyed by normalized repository plus original instance ID.
  **Zero problems occur in both rebuilt splits.** This does not erase earlier
  lineage exposure or create an independent final holdout.

The new profile is `e97-open-swe-source-native-v1`, not the deployed Pi grammar.
Context messages are reversible JSON under role headers. Assistant targets have
Analysis, Commentary, Think (source flag), Action, and Arguments fields. Source
transport IDs and original argument-JSON spelling are archived, not generated
supervision; decoded executable argument values are exact. Oracle patch metadata
is never included. The record separator is unsupervised.

## Validation and exclusions

The independent validator checked all **10,905 original source identities**,
including inclusion/exclusion coverage and original per-row payload hashes.
For every retained record it checked complete message/argument reconstruction,
canonical tokenization, byte-level assistant masks, indices, target accounting,
source schemas, split grouping, and original source hashes. No raw commands were
executed and no environment replay is implied.

Whole-trajectory exclusions, counted by first failing reason:

- 4,893: complete serialized trajectory exceeds 65,536 tokens.
- 36: private-analysis cap exceeded.
- 4: missing/nonstring think thought.
- 3: missing explicit final message.

No retained record was shortened to meet a cap. The earlier 86.3M train-target
number was a feasibility proposal using a different representation and old split;
**64.4M is the actual rebuilt training inventory**. Keeping full source fields and
using a lossless transport changes context costs; grouped development reservation
also changes the train allocation.

## Identities

- Frozen build inventory:
  `2b356bf199499e727bd7a85aa92e723bcf17f62d054b370d044ea8cdcbe4e6a5`
- Published manifest:
  `255b02f3b3bbdc85eb15a89d22e7ff2c4bf69f151c2dfb5941a22468711993f5`
- Validation receipt:
  `1529da7910ff23e9ab519a6ac88630d86f28ea19f740a7fcbf309cda187e8a75`

Outputs include `tokens.bin`, `loss_mask.bin`, `records.idx`, `records.jsonl`,
`source_messages.jsonl`, `exclusions.jsonl`, `manifest.json`, `validation.json`,
and the exact builder/codec/validator source snapshots. Publication is owner-only,
read-only, atomic and no-replace.

## Remaining boundary

`training_eligible:false` deliberately prevents accidental use under the old Pi
recipe. Next work is source-native runtime/training integration, remaining source
admission checks, and a frozen learning experiment—not another conversion redesign.
No sustained SFT, checkpoint promotion, or live Pi-interface change occurred.
