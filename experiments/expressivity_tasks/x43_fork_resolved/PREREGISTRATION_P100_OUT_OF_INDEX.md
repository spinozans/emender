# P100 independent verification and out-of-index transfer

Registered before measurement. RESULTS_PILE_100GB.md certifies BUILD ONLY.
Index /mnt/nvme2n1/erikg/P100c/pile-head100.sxi2, raw pile.txt first
100000000000 bytes, virtual END. Use isolated serving executable from C14
serving preflight, existing sxgc and all mounted source bytes read-only.

Gate1:16 branch queries (four by-offset sample contexts at lengths8/16/32/64)
and one64-NUL negative control, independent brute-force scan of the ENTIRE raw
[0,100e9) region, in bounded32MiB chunks with overlap. Count every overlapping
occurrence, all256 next-byte frequencies and EOF; compare each complete native
branch histogram, width and EOF exactly. Scan reads source by offset, never
materializes corpus. Store exact hex patterns and raw source/index stat identity,
serving binary hash, streaming source SHA256, progress cursor and counts.
Fail closed: no heldout extraction/evaluation until ALL queries pass. This is a
sampled query certificate, not exhaustive SA/BWT or closure/T3 certification.

Gate2 after query verification: choose document-complete spans beyond100e9,
separated by raw0x1e delimiters; hash-match exclusions against every frozen
C14 stream-audit document SHA256, never source row order. Record source spans,
document hashes, raw-byte windows, masked padding and document-disjoint split.
Initial target4096 windows across >=64 documents, full-document hashes retained;
actual counts govern all results. Duplicate documents by content are excluded.

Predict positive transfer gain (off NLL minus on NLL) on out-of-index text;
roughly25–75% of each corresponding in-region gain survives, with substantial
arm/stratum variation. Prior36.6% support at16 bytes motivates short-suffix
transfer. This range is descriptive, not a confidence interval. All outcomes
results. Register before evaluating consulted/stacked transformer and emender,
unbounded twin, dense controls and available array-alone baselines. Frozen
weights and stack calibration must remain fixed. Report paired on/off gains,
support-length strata, and denominators on exactly the same heldout windows.

Causal limitation: excluding document byte spans removes direct indexing of
those occurrences; duplicated substrings may remain in the index. Cross-domain
and sample differences confound a literal causal coverage-vs-transfer subtraction.
Name in-region-minus-out-of-index gain an operational coverage difference, not
an identified pure causal component. No fine tuning or stack refitting on test.
No DiLoCo requirements apply. Supervisor tool unavailable; no contact claimed.
