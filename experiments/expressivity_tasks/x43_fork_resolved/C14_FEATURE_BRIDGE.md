# C14 feature bridge: Pile tokens, Wikipedia consultation

The existing `~/sxgc/ragc-ffi/src/lib.rs` is an AGC decompression C ABI:
`sxgc_agc_open`, `sxgc_agc_len`, `sxgc_agc_range`, `sxgc_agc_close`.
It exposes no SXI2 reader, FM interval, search, or branch enumeration. Loading it
with ctypes cannot supply Wikipedia consultation features.

The qlite Rust implementation (`xsa/src/qlite.rs`, upstream
139926a09db84d02c9877943d31fc8dcff52cb26) has `Reader::open`,
`Index::open`, `Index::ctx`, `Ctx::search`, and `Ctx::locate_steps`.
`Ctx::rank` is private. Search consumes mapped bytes in reverse order; the CLI
applies `reader.sigma` then reverses the pattern. Reimplementing the mapping
incorrectly would silently query a different string. These are in-process Rust
interfaces inside the executable, not a Python shared-library ABI. No upstream
files were modified, and no replacement xsa was implemented.

`c14_coverage.py` is the available CPU batch bridge. It decodes stored GPT-2
IDs through the exact byte alphabet inverse, samples token boundaries, searches
causal left suffixes, and emits a summary geometry at those boundaries:
255 right-extension counts, maximal supported left-context bytes, total outgoing
support, entropy, and coverage. Byte 0x1e is omitted from branch enumeration
because the CLI binary record framing uses it. It is a Wikipedia document
separator, so omitting it also removes separator continuations. This omission
must remain visible; this is not a complete 256-byte distribution. No target
byte enters feature construction; observed-next queries are measurement only.
EOS ID 50256 has no text-byte continuation and needs a dedicated neutral channel.
The geometry is token-boundary aligned, not a 50,257-token exact continuation
support distribution. Projecting a first-byte support probability to every
candidate sharing that byte is only a first-byte feature, never token probability.

The sampler rejects positions without 1024 preceding and 256 following bytes
inside a document and any CLI separator in that window. Coverage estimates thus
refer to eligible interior token boundaries, not all Pile bytes or all tokens.
The number of rejected draws is receipted. 1024 is a right-censoring ceiling.
The Wikipedia dump is raw serialized content, with markup/entity conventions;
exact overlap is representation dependent. Locate of 256-byte spans estimates
verbatim *span overlap*, not the fraction of documents attributable to Wikipedia.

## Upstream interface handoff (request only)

Expose a read-only persistent SXI2 handle plus batched length-delimited raw-byte
requests (including NUL and 0x1e). Retain index memory across batches. Return
original-byte counts with explicit sentinel/document-end handling and versioned
sigma mapping; accept no Pile target/future bytes in causal feature requests.
For each left-context suffix, return maximal support up to a caller cap, interval
width, 256 next-byte counts (right extensions), nonterminal outgoing count,
entropy, coverage, and censor flag. Preserve interval state for backward
extensions; forward continuation needs a corresponding index operation or
repeated searches, not the same prepend operation. Provide a batched token-trie
walk only if exact full-BPE candidate support is required. Define EOS separately.
Add brute-force oracle comparisons, absent/boundary-symbol tests, document-seam
rules, deterministic concurrency, and timings separating open/search/serialization.
This is an interface request, not a claim these APIs already exist.

No GPU was needed. The CLI kernel uses `-j 1`; index initialization may use its
own threads, which the receipts journal. Whole-process and kernel costs are
reported separately in the phase-1 result. A 40M-position arm needs 14,814.815
features/s to fit feature production inside 2700 seconds. Dense float32 storage
for 260 scalars alone is 41.6 GB decimal per 40M positions, excluding IDs and
metadata. A full float32 vocabulary vector would be 8.041 TB per arm. Reusing a
frozen sidecar across arms amortizes extraction; it does not make the cost zero.

## Observed upstream blocker

Two of 2,524,500 right-extension queries returned `18446744073709551615`
(UINT64_MAX), not a possible count in a 100,093,934,467-symbol index. Sample
2769's 1024-byte context has width 177; sample 9057's nine-byte context has width
2. Appending original byte 194 triggers the impossible width in both cases.
Raw branch pattern IDs are 699658 and 2287033. The source emits `r - l` without
checking the returned interval order; the underlying interval/rank cause has
not been isolated. Do not treat this as a proven fix prescription.

`count_failure_repro.patterns` contains the two original-byte patterns for an
upstream regression. Run the rebuilt xsa with `--count --patterns` on that file;
this file is a reproducer, not a claimed additional execution. The original
batch outputs are the execution evidence. `count_range_failures.json` and
`branch_consistency_failures.json` retain both failures. The initial independent
audit failed; two feature rows were then rejected explicitly. The hardened
Python bridge now rejects an impossible count before constructing distributions.
The receipt audit passing after quarantine does not make the query engine pass.
Require upstream repair and independent branch/absent-pattern regression gates
before training, in addition to the persistent-interface and budget prerequisites.
