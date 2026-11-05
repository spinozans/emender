# X43 A: store-preserving inference operator rounds

Registered before quality measurement. Frozen checkpoints: B1-v2
`9c87a6a306b708608d7f466dbd62c70c53263c5322f24963b91b94afc8f4def8`
and dense emender `7a2598c8c1e77f349be6be7af1d59ece7300a5345cb3dab1420e0274d2523019`.
No training, production kernel changes, checkpoint writes, or live union queries.

Interface: original batched forward at k=1 (required byte-exact log-probability
and per-target loss fingerprints); k>1 uses window-local, per-layer/head stores
carried through residual block rounds at the SAME byte before EMIT. B uses its
independent eager store oracle; dense uses native continuation handles. Observe
an input once in B; operator rounds never consume shifted targets. Dense's
native operator clock advances per round, including decay, without emitting a
byte. Slot budget never increases. Store resets at every window/document.

Sweep k={1,2,4}, extra rounds only where exact cached branch support >1.
Each layer round reads the immutable, exact cached BRANCH answer once and
performs UPDATE/SELECT. Cached reads are logical branch reads, not new index
RPCs; count both logical reads and zero live RPCs. Residual stream evolves
across rounds. Halt deterministically if no CORRECT in the layer round AND
maximum absolute slot-strength delta <1e-6, otherwise bound by k. Halting is
layer-local. No learned halt policy. Native R-CONFLICT is CORRECT.

Bounded diagnostic: first two windows from first and middle registered eval
cache shards, 128 bytes maximum each, frozen shard hashes checked. Gather raw
bytes and shifted targets from attested document offsets, verify document
hashes and boundaries. Report actual half membership. This is teacher-forced
next-byte scoring at generation positions, not free-running text quality or a
full-corpus estimate. All selected positions retained; fork-only NLL primary.
Use deterministic FP32 for both standard and iterative paths, seed 4301.
No postmeasurement threshold/sample changes. Card 3 or 5, broker only.

PA1: fork NLL decreases with increased maximum rounds (negative curve sign).
Report each paired delta, including worsening, flatness, nonmonotonicity, and
nonfinite failures. A small diagnostic cannot establish general improvement.
PA2: high-entropy forks require more rounds than low-entropy forks. Bins:
low H<=log(2), high H>log(2); report empty bins, round histograms, and means.
PA3: collapse watch: logical branch reads per eligible position do not decrease
as k increases. Record requests, actual rounds, immediate halts, bound hits,
and query volume; early halts must be explained by op telemetry.

The k=1 compatibility path preserves the original GEMM shapes by delegation;
it does not prove byte parity of a different scalar execution schedule. Record
that precision/scheduling confound explicitly for k>1. Existing C14 tests must
remain green; retained reference tests cover deterministic reset, causal prefix,
masked padding, invalid budgets, checkpoint immutability, and k=1 fingerprints.

Implementation correction before the valid dense sweep: dense receives zero
consultation features (as in its training and standing paired evaluation), while
fork labels/entropy still come from the cache. An initial invalid dense k=1
attempt passed nonzero features to its untrained projection; exclude that
attempt from every quality claim. Native continuation also requires all four
telemetry scratch buffers; an omitted scratch buffer caused compilation failure
before dense k=2 output. Retain both failed attempt receipts. No sample,
threshold, checkpoint, prediction, or B1 interface changes.
