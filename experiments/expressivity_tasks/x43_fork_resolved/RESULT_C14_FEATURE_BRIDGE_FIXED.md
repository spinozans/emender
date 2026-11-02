# C14 fixed-count feature bridge dry-run

Rebuilt xsa release at clean a94c2ceea9976696ad7e147cce4db720b223cef0,
equal to fetched origin/pile-week-v1. Exact former-sentinel repros return 0/0;
"domestic cat" returns 2493. No QUERY_COUNT_IMPOSSIBLE fired.

Extracted the consultation prefix: document 0, first 16 independent 128-byte
windows, all 2,048 real shifted targets. Reused the dense session's pile.txt
offset receipt, independently hash-checked document 0 against documents.jsonl,
and retained the finalized dense span receipt snapshot. No span-location scan
was duplicated. No raw corpus was materialized or written. No GPU used.

The extractor batches exact raw-byte suffix searches and all 256 right branches.
It uses CLI records, bounded argv for byte 30, and exact FASTA for dash-leading
byte-30 patterns when representable. This is repeated native backward search;
it cannot preserve interval state across CLI requests. NUL+30 patterns and
unrepresentable dash/newline cases fail closed. A universal persistent,
length-delimited native interface remains absent.

Byte 30 is actually queried. Counts retain its document-terminal mass; 39 rows
have positive terminal counts. Nonterminal support, entropy and the 260 features
exclude that mass explicitly. No Wiki document seam is used as a continuation.
Causal contexts reset at each Pile window; the registered 1024 cap is therefore
bounded by the 128-byte model window. Exact-count range, response IDs,
branch-sum, shape, feature arithmetic, and journal audits pass for 648,330
queries and 2,048 rows. Eight tests pass, including a toy full-alphabet oracle,
terminal policy, impossible branch sums, transport rejection, hash binding,
and document/window boundaries. This is not an independent full-index oracle.

Whole workflow: 130.098s, 15.742 features/s,
including 2.206s to rebuild the final
terminal-aware geometry from retained native answers. Five native processes cost
124.744s; their journaled query kernels are tiny by
comparison. CLI -j 1 does not isolate a CPU: default Rayon can use all 256 logical
CPUs. Native decoded run columns are 265,914,840,963 bytes, plus overhead; peak RSS
was not measured. GPU needs: none for extraction; training/transfer unbenchmarked.

Linear 40M scenario: 705.83 hours per cache, versus 14,814.815
features/s required inside 2700 seconds. At measured small-batch throughput,
2700 seconds produces about 42503 positions.
Neither is a scaling guarantee. Registered budgets are 2700 active seconds per
head, not fixed position counts: consultation actual positions remain unknown
until trained. No full-arm cache, training, isolated-core, GPU-transfer, or
large-batch benchmark is claimed. Dense float32 features alone are 41.6 GB/40M;
raw integer counts and JSON receipts require additional space. Frozen caches
can be shared between heads, but their cost remains payable.

16-byte support: 1173/1808 eligible positions (64.8783%), versus phase-1 36.5900%.
64-byte support: 0/1040, versus phase-1 2.3600%. 256/1024: no eligible windows,
not zero measured coverage. Median maximal context is 16 bytes; coverage is
100.0000%, and 289 contexts are censored by available reset history.
This is the first document's raw-byte prefix, unlike phase-1's random eligible
interior token boundaries: the difference is descriptive, not a corpus-wide
improvement or regression.

The initial extraction stopped on an argv option-marker transport limitation,
after two successful native batches. It produced no accepted feature cache.
The corrected exact FASTA fallback completed the second extraction. Both
attempts and their costs are retained; the successful benchmark excludes the
initial failed attempt. No count was coerced or quarantined.

Readiness: this validates the bounded extraction, not consultation launch.
Universal transport, an independent native full-alphabet fixture oracle,
complete train/valid caches, amortized large-batch extraction and transfer costs,
and actual training position counts remain unresolved. Original phase-1 records
are historical and unchanged. contact_supervisor and ledger messaging tools were
absent from this runtime; coordination used the shared dense offset artifacts.
