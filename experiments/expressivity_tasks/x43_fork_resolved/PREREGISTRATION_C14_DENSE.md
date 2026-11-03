# C14 dense-only launch amendment

The operator explicitly authorizes the dense control while the UINT64_MAX feature
fix is pending. This waives feature-cache prerequisites for dense training only;
all raw-source, shape, budget, boundary and document-split requirements in
PREREGISTRATION_C14_BYTE.md remain binding. Both heads retain seed 3401,
batch 16, context 128, 2700 active seconds each and all-zero consultation.
No inference of consultation strata from missing features is permitted.

P1/P2/P3 remain the pre-existing directional predictions. The dense arm is the
paired reference, not evidence for any of those predictions by itself.
Before launch reconstruct ALL selected documents from exactly the source bound
by receipts/c14_byte_source.json. Exhaustion is a failed preflight: do not train
on the discovered prefix. Exclude token-truncated final documents explicitly.
Freeze every real within-document valid target, source-ordered independent
128-byte reset windows, and document-disjoint calibration/final-test halves.
The consultation arm must reuse these positions and reset contexts; causal
features may be attached later after full validation. Until then support/fork
and zero-target NLL strata are unavailable, never fabricated.

## Take 2: operator-authorized full-source binding

The confirmed natural source is the complete 30-shard EleutherAI v1 train dump.
Scan shards 00, 01, 02 in order and bind the minimal prefix that exhausts all
7,831,213 selected document receipts. Each consumed compressed shard must be
SHA256-bound in receipts/c14_dense/source_binding.json. Exhaustion of the selected
receipt stream proves that later shards are unnecessary; exhaustion of all three
without full coverage is a failed preflight and prohibits training. The earlier
single-shard discovery receipt is retained as historical evidence, not the current
source-set scope. Exclude the explicitly token-truncated final document.

All /mnt/nvme2n1 sources remain read-only. Materialized bytes, offsets, model
checkpoints and full per-target evaluation artifacts live outside git at
/home/erikg/c14_dense_bytes. Commit their hash manifests and aggregate results.
The immutable eval_windows.jsonl construction is shared with consultation.
Report calibration/final-test NLL and descriptive reset-position strata (positions
1–16 and 17–128 inside each independent window); these are not support/fork
strata. Model shapes, initialization, budgets, data order and optimizer stay as
registered. No features are admitted in this dense control.

## Take 3: operator-authorized zero-materialization source

The sole text source is the immutable, read-only
`/mnt/nvme2n1/erikg/pile.txt` (1,307,910,802,540 bytes, dated 2025-07-11).
This supersedes all instructions above to materialize text or read shards.
No text is copied, re-encoded, or written to either the mount or a new corpus.
Bind all 7,831,213 selected document hashes to direct offsets in this file.
The original pipeline deduplicated, excluded documents, and capped text at one
million characters: a covering source interval can therefore contain omitted
text. Only selected, hash-matched document slices enter training or evaluation;
never train on the intervening bytes or on a capped document's remaining suffix.
Previously hash-bound offset metadata may supply capped byte lengths, but every
text digest is checked anew against pile.txt. Exclude both token-truncated
receipts, retain document splits, and receipt first/middle/last boundaries.
Missing/out-of-order selected hashes prohibit training. The source inode, size,
and modification time must match the completed mapping at launch.

Use the registered model shapes, seed, optimizer, all-zero consultation, batch,
context, and 2700 active seconds per head without adjustment. Freeze and hash the
shared eval positions before inference. Keep model checkpoints and per-target
artifacts outside git. Support/fork strata remain unavailable; report actual
within-window position strata. This is a dense reference, not a consultation test.
