# e97-rl-loop-v1 — Receipt format specification

Normative for every training receipt this prototype writes. Governed by the
2026-07-25 **no-database amendment** of
[`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`](/home/erikg/emender/docs/RESILIENT_DILOCO_COMPUTE_POOL.md):
receipts are **durable evidence**, never a live coordination store; no loop
path may open SQLite; a compatibility `latest.json`-style pointer is never
authority.

## Storage model

- `receipts/stream.jsonl` — append-only line stream; line N binds line N-1 via
  `prev_receipt_sha256` (hash chain; the first line binds `null`).
- `receipts/cycle-NNNN/<receipt_sha256>.json` — immutable (mode 0400,
  no-replace publication) copy of each receipt.
- Every append fsyncs the stream file; every immutable copy fsyncs its
  directory.
- The stream is the training-side authority: an authority/pack build consumes
  only receipts present in a verified stream walk.

## Receipt schema (`emender-rl-loop-training-receipt-v1`)

Canonical JSON (sorted keys, compact separators, no NaN). Exact field set:

| Field | Type | Meaning |
|---|---|---|
| `schema` | const | `emender-rl-loop-training-receipt-v1` |
| `receipt_sha256` | hex64 | SHA-256 over the canonical JSON of the body with this field removed |
| `prev_receipt_sha256` | hex64 \| null | digest of the previous stream receipt (chain link) |
| `cycle` | int | loop cycle number (1-based) |
| `task_id` | str | queue task id |
| `task_sha256` | hex64 | digest of the canonical task body (authoring-bound) |
| `kind` | enum | `on-policy-success` \| `teacher-corrected` |
| `policy_checkpoint.path` / `.sha256` | str / hex64 | exact checkpoint the policy served for this cycle |
| `episode.episode_text` | str | full canonical Pi-native episode transcript (header + turns) |
| `episode.generations` | list | per assistant turn: `turn`, `token_ids`, `reason`, `prompt_sha256` (owner-side bridge generation records). For spliced teacher-corrected receipts this list walks the WHOLE transcript: the policy's retained prefix frames first (their own original token_ids), then the teacher's suffix frames in order |
| `episode.supervise_from` | int | first supervised generation index. `on-policy-success` receipts: 0 (all turns supervised). Spliced `teacher-corrected` receipts: the number of retained policy prefix frames — the policy prefix is MASKED CONTEXT (unsupervised) and ONLY the teacher suffix tokens are supervised targets (encode_candidate partial-target masking, the same machinery the canonical loopbreak/pointerchase/extracterror families use) |
| `episode.tokens` / `episode.targets` | int | verified token and assistant-target counts (re-encoded, must reproduce) |
| `grade` | object | the mechanical grade receipt (`emender-rl-loop-grade-receipt-v1`): `passed`, per-check `type`/`ok`/`detail`, `stage` (`policy` \| `teacher`), task binding |
| `teacher` | object \| null | live: `{mode:"live", model, api}` plus splice metadata (`continuation:"same-transcript-splice"`, `prefix_frames`, `dropped_trailing_finish`, `dropped_unresolved_call`); stub: `{mode:"fixture", note:"authored deterministic continuation stub (not GLM-5.3)"}`; null for on-policy successes |
| `attempt_link` | object \| null | for teacher-corrected receipts: the failed on-policy attempt's `attempt_episode_sha256` (immutable episode evidence), `attempt_grade_sha256`, `attempt_status`, `attempt_reason`, plus the splice boundary (`policy_generations`, `retained_prefix_frames`, `dropped_trailing_finish`, `dropped_unresolved_call`) |
| `encoder` | const | `p50k_base` |
| `created_unix` | number | wall time |

## Verified-only invariants (fail-closed)

1. A receipt may exist only if its mechanical grade `passed` is true — failed
   attempts and failed teacher continuations never become receipts (they remain
   immutable episode/grade evidence only).
2. `on-policy-success` receipts carry no teacher identity; `teacher-corrected`
   receipts must carry both a teacher identity and an `attempt_link`. A
   spliced teacher-corrected receipt must additionally re-encode with ONLY
   the teacher suffix supervised: `supervise_from` equals the number of
   retained policy prefix frames, and the masked target span is exactly the
   teacher frames' tokens (proved by scripts/rl_splice_test.py and enforced
   by the encode_candidate re-encode in stream verification).
3. Stream verification re-encodes each episode with the canonical curriculum
   encoder (`scripts/build_e97_pi_native_curriculum.encode_candidate`) and
   requires `tokens`/`targets`/supervised-unit counts to reproduce exactly.
4. Chain links must verify in order; any mutation breaks every downstream
   digest.
5. Receipts bind the exact policy checkpoint sha256; a stream spanning multiple
   cycles/documents carries its own checkpoint lineage per receipt.

## Related durable evidence (not receipts)

- `episodes/cycle-NNNN/<task>/{attempt,correction}/episode-private.json` —
  full owner-side episode record (native transcript, generations with token
  IDs, public history, terminal transport receipt).
- `episodes/.../grade.json` — the immutable mechanical grade for each stage.
- `packs/cycle-NNNN/build-summary.json`, `training/cycle-NNNN/log.jsonl` — the
  pack-build and train-step evidence (authority/pack manifest digests,
  step losses, checkpoint identity).
