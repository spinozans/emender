# C14 byte pivot — registered; training blocked

The binding protocol is PREREGISTRATION_C14_BYTE.md. GPT-2 C14 proposals are
superseded. No training or held-out inference was performed. No GPU was allocated.
The boundary directive is retained identically in AGENTS.md and CLAUDE.md.

Natural Pile text was located in the read-only original Pile shard
`/mnt/nvme2n1/erikg/pile/EleutherAI_ThePile_v1/pile/train/00.jsonl.zst`.
The entire shard SHA256 is
`15202e860ccd6311a3ab862958a59df9a9c16200205e7897d956007546aa5c38`.
The first 4,096 selected capped natural documents match the pipeline's SHA256
receipts in strict order, scanning 4,849 original rows. This is a bounded source
attestation; the full uncopyrighted corpus is not yet reconstructed/attested.
Original Pile row numbers differ from the uncopyrighted stream's source indices.

176,677 within-document windows, containing 22,351,133 shifted byte targets,
were checked against the source bytes; zero windows cross a document boundary.
Four automated tests cover context and shifted-target bounds, short/empty docs,
UTF-8/NUL/byte-30 text, corruption/token-truncation rejection, train/eval split
isolation, and document-disjoint calibration/test assignment. Model construction
ran on CPU: Emender 2,133,700 and transformer 2,133,748 trainable parameters;
dense and consultation reuse identical head shapes. The registration binds the
native Wikipedia index's whole-file hash as well as its corpus MANIFEST.

Phase-1 work was consumed, not duplicated: coverage/content-strata/validation/
provenance at committed 6ff620c5 are bound by exact commit and digest. Sparse
long-context support and whitespace matches justify directional predictions
without a numeric gain extrapolation. The phase-1 audit found two UINT64_MAX
native branch counts and rejected the affected features. This is a scientific
query failure; passing receipt consistency is not healthy feature extraction.

Residual risks / launch blockers:

- Full selected natural-byte corpus and document offsets remain unattested;
  the discovered shard prefix cannot stand in for the complete corpus.
- Native branch-count failures and omitted byte-30 counts prevent accepting
  a complete 256-byte feature cache. No corrupted count is coerced to zero.
- Full causal train/eval cache, storage/transfer/extraction cost receipts and
  coverage-subset caches remain absent. No training, coverage-scaling or stacking
  outcome is claimed. Their arms and document-disjoint evaluation are registered.
- The first integration uses tier-1 2M heads. A larger shape requires a separate
  preregistered construction/count receipt, never an unrecorded scaling change.

Manual note: the runtime exposes no contact_supervisor tool (tool discovery
returned no matches). Coordination used committed phase-1 receipts as directed.
The initially attempted ledger tests failed during the phase-1 validator/ledger
transition; retain the attempt and use the final rerun for acceptance. Existing
unrelated working-tree changes and checkpoints are outside this change.
