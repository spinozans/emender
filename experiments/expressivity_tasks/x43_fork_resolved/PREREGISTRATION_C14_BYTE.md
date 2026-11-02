# C14 — byte-level pivot; registration before training

This operator-directed amendment supersedes the proposed GPT-2 C14 registration.
C14 and all subsequent training MUST NOT cross document boundaries. Inputs,
shifted targets, consultation context, and held-out windows belong to one Pile
document. No EOS joining, cross-document context, or hidden state carried between
windows. UTF-8 bytes have vocabulary 256; GPT-2 IDs are never model inputs.

## Natural source and document identity

Natural text is available read-only at
`/mnt/nvme2n1/erikg/pile/EleutherAI_ThePile_v1/pile/train/00.jsonl.zst`.
`receipts/c14_byte_source.json` binds the entire compressed shard SHA256 and the
pipeline's `x42_pile_scale/data/pile24/documents.jsonl`. The first 4,096 selected
documents match their capped-text SHA256 in strict receipt order (4,849 original
Pile rows scanned). They contain 22,351,133 shifted targets in 176,677 independent
windows. Original Pile and the uncopyrighted stream have different row numbers:
use hash matching, never assume the row numbers agree. The raw shard is a source
of matching natural documents, NOT authority to include excluded documents.

`c14_documents.py` reads natural JSONL text, applies the original 1M-character
cap, checks the document digest, and truncates each final window within that doc.
No model-context or target window spans two documents. It rejects token-budget
truncated documents until a separately registered raw byte boundary exists;
there is no inferred token-to-byte boundary. No normalization or BPE decoding.
Short tails remain real targets and must be batched by length, with state reset.
The same sampler is mandatory for train and eval. These receipts attest discovery
and boundary preflight, not a full Pile byte corpus or completed training.

Before launch materialize and hash-bind the full selected byte corpus and byte
document offsets in receipt order. Match all selected documents, preserve train/
valid document assignment, and exclude the token-truncated final documents with
explicit counts. Fail closed if any selected document is absent or out of order.
No arbitrary original-Pile replacement and no resampling to improve coverage.

## Models and arms

Start from C12's native byte-level `c11.Boosted`: Emender and transformer, 256-byte
vocabulary, embedding width 256, two layers, base target 2,000,000 parameters;
retain the tier-1 head construction and 260-to-256 input projection. Within each
head compare dense all-zero consultation against Wikipedia consultation. Same
initialization seed 3401, document order, batch 16, context 128, BF16, AdamW
lr=3e-4, weight decay .01, clip 1, and trainable parameter count (0.1% tolerance).
CPU construction in `receipts/c14_byte/model_shapes.json` fixes 2,133,700 Emender
and 2,133,748 transformer trainable parameters, identical within each head across
its two feature modes. No silent width change.
Budget 2700 active seconds per arm; report actual bytes/steps and end-to-end cost.
Each consultation training position uses independent keep=.5 all-input feature
dropout, no rescaling, including length/metadata. Device-off evaluation zeros ALL
features. Dense CE includes every real target, including zero-support targets;
no fork masks, target filtering, or hard routing. Never treat padding as text.

Wikipedia is the read-only native byte array at
`/mnt/nvme2n1/erikg/wf-finish-as-100gb-20261102T044713Z/corpus.sxi2`, bound to
`/mnt/nvme2n1/erikg/wiki/MANIFEST.json`. Use the rebuilt xsa, hash-bound in the
phase-1 provenance; do not modify sxgc or either mount. Features are 256 native
next-byte probabilities plus log1p outgoing count, entropy, fork indicator,
log1p context length (260 scalars). Maximal supported causal suffix cap 1024,
clipped at window reset. Empty context/support maps to all zeros. No Pile future
or target in lookup; no EOS alias. Byte 30 is a real vocabulary byte. The existing
phase-1 CLI geometry omits byte 30 and is NOT admitted as a complete feature
cache. Before launch require full-alphabet brute-force validation, explicit
Wikipedia separator/sentinel policy, complete train/valid cache hashes and
extraction/storage/transfer receipts. Register a length-delimited native API or
an exact fallback before accepting that cache. Do not fill a missing count with
zero and call it measured. No 40M cache or online throughput claim is made.

## Predictions, coverage scaling, stacking

Predictions are calibrated to phase-1 receipts committed in `6ff620c5`; exact
commits and hashes are recorded in `receipts/c14_byte/registration.json`.
The receipt audit passes while native query health fails (see below).
Training-interior measurements show support 36.59%/2.36%/.59%/.09% at
16/64/256/1024 bytes, median supported length 12 bytes, 0.5% centered 256-byte
span overlap. All five centered matches are whitespace; only 23/10,000 contexts
have nonwhitespace 256-byte support and 2/10,000 at 1024 bytes. These are
selected token-boundary diagnostics, NOT unbiased raw-byte or held-out coverage.
Do not claim corpus disjointness or extrapolate C12 gain numerically. The
coverage binding includes content strata and validation, not rates alone.

P1 directional: consultation lowers final-test byte NLL in each matched head;
predict larger gains in supported/agreement strata than zero-target strata,
without dropping targets. P2: with the registered dropout, device-off NLL stays
within .02 nats/byte of its matched dense control. P3: frozen consultation plus
C13-style calibration beats consultation alone and dense plus calibration.
These are hypotheses, not measured improvements; coverage is bound before
training. All signs and failures must be reported.

Freeze weights before coverage-scaling swaps. Pre-register Wikipedia coverage
subsets and identical geometry; hash-bind each subset/cache and its extraction
cost before inference. Report coverage changes and interface shifts separately;
never build a held-out/self-containing Pile index. Compare total, support/fork,
empty-support and nonempty-zero-target strata with their exact denominators.
Split valid DOCUMENTS in receipt order into calibration and final-test halves
before inference. No document appears in both halves, even through adjacent
windows. Fit C13's confidence/log1p-context/log1p-support gate on calibration
only; freeze it before final test. On complete validated byte distributions,
forced nonempty single-byte support may use the exact walk; empty support
abstains. Report dense, consultation, dense+calibration,
consultation+calibration, and all-input-zero rows on identical final targets.

No GPU/training launched by this registration. Launch only via broker on cards
2/4; card 7 forbidden, searches on 1/5/6 untouched. No .pt commits. Full raw
corpus/cache and green coverage binding remain launch prerequisites.

Phase-1 discovery while registering: `validation.json` reports two UINT64_MAX
branch-count failures (samples 2769, 9057), rejects those features, and explicitly
blocks training. Passing the receipt-consistency audit does not establish healthy
native queries. C14 must remain launch-blocked until this failure is resolved and
complete byte-feature validation is green; never coerce these counts to zero.
