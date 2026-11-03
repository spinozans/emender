# C14 native serving preflight — incomplete full-size work order

Built xsa release at `88d4e0744a05359614d28db5f0ed160a677d8ef5` in the isolated shared clone
`/home/erikg/emender-scratch/c14-xsa-serving`, fetched origin/pile-week-v1,
and merged ff-only. The source tree is clean. The original ~/sxgc checkout was
not edited. Former-sentinel counts are 0/0; domestic cat is 2,493; the-space has
313,603,799 occurrences and 137 branches. Complete branch closure holds.
The server logs backward-search ranks, gather_ab=0 and lf_steps=0; no Wikipedia
corpus argument or corpus reads are used by the query process.

An independent overlapping bytes.find oracle agrees on all branches, interval
width and EOF for 261 native fixture patterns. All 256 query bytes are exercised,
including NUL, byte 30, ff, newline and leading dash. The format reserves an
absent byte for END, so the fixture has 255 present symbols (fe absent); this is
not an all-256-present corpus claim. Five new tests and six existing document/
padding tests pass. Transport rejects truncated frames, wrong IDs, incomplete
branch distributions and inconsistent sums. Maximal-suffix selection is tested
against an independent brute-force oracle, including terminal-only features.

## Measured extraction

Exactly 1,034,056 real training targets from the first 8,192 frozen
source-order windows / 219 documents were extracted in
260.307391 seconds: **3972.442 positions/s** end to end,
including native startup/shutdown, document/offset checks, suffix selection,
complete distributions, feature arithmetic and gzip writes. Native -j32 serving
reported 31,530.865 contexts/s across 7,033,954 queries:
6.802295 queries per output position. This differs from
a benchmark of already-selected contexts. CPU affinity allowed 256 logical CPUs;
this was not an isolated-core measurement. Native peak child RSS was
295,295,631,360 bytes. No GPU was used.

Every context ends before its target and stays in one independent model window.
Each used document is hash-checked against the frozen offset receipt. Wikipedia
separator byte 30 is measured and retained as terminal mass, excluded from
nonterminal probabilities, outgoing count, entropy and fork bit; context is
clipped after any separator to prevent Wikipedia seams. Empty nonterminal
support zeros all 260 inputs. The 260 registered inputs can be reconstructed
exactly from the sparse count receipt. No raw corpus was written.

The gzip cache is 49,545,274 bytes, outside git at
`/home/erikg/c14_consultation_features/serving_benchmark_8192/features.jsonl.gz`;
SHA256 `2eaa4a4ec590171ceef69e89f7c734687e21074eb5ff06e73d3146861cd11bba`. Arithmetic/shape/closure replay passed all
1,034,056 rows and rechecked the original immutable evaluation-window
hash `2ab51ce59ad94010d78f6817f29fa06fc5ee64feea766376ff0e35347f196d4d`.
Both dense checkpoint hashes and all frozen dependency hashes are unchanged.
This is cache arithmetic replay, not a full-index brute-force oracle.

| Diagnostic | Bounded raw-byte training prefix | Phase-1 token-boundary interiors |
|---|---:|---:|
| 16-byte support | 436,838 / 911,371 = 47.931962% | 36.59% |
| 64-byte support | 4,275 / 521,609 = 0.819579% | 2.36% |
| 256-byte support | No eligible contexts | 0.59% |
| 1024-byte support | No eligible contexts | 0.09% |

Median maximal context is 14 bytes; 132,416 rows are
censored by reset history. Nonterminal coverage is
1,030,696/1,034,056 (99.675066%).
There are 26,344 positions with positive separator mass.
Reset positions 1–16 have 130,420/
130,852 nonterminal support;
interior positions 17–128 have 900,276/
903,204.
These are different sampling populations. This prefix is neither full-stream
nor held-out coverage. The model resets cap available history at 128 bytes;
256/1024-byte eligibility is absent, not measured zero support.

## Full-size scope and uncompleted work

The frozen offset inventory contains **43,052,359,095 training targets** and
**434,204,084 validation targets**, in 340,231,036 and 3,431,242 windows.
The registered complete train/valid cache requirement is not satisfied.
At this prefix's rate, linear scenarios are 125.44 days for training features
and 1.27 days for validation features. The current gzip representation projects
to 2.0836 TB, versus 807.65 GB free at preflight. Dense float32 inputs would be
45.226 TB. These are implementation-specific extrapolations, not full-stream
measurements or proof that improved caching cannot fit. Cross-chunk supported-
context reuse and denser/dictionary storage remain implementation work; the
current benchmark does not justify starting a months-long exhaustive run or
allocating a cache beyond available space. No full-cache job was left running.

The new serving interface and native-query validation pass. This supersedes
the former transport blocker only. It does not satisfy the complete-cache launch
gate or establish a full extraction budget. No scope reduction, eval resampling,
window change, dense retraining, or lower-budget consultation proxy was made.
The work order is **incomplete**; no consultation training/evaluation, feature
transfer benchmark, coverage-stratified loss table, device-off comparison,
P1/P2/P3 verdict, coverage-scaling subset, or Pile-scale C13 stacking was run.

| Head | Frozen dense final-test nats/byte | Consultation |
|---|---:|---|
| Emender | 2.276134617 | Not run |
| Transformer | 1.404434206 | Not run |

The dense values are reused frozen evidence, not new measurements. No prediction
is accepted or refuted by this preflight. POSITIONING.md is untouched; no .pt or
large feature cache is committed. The exact requested full-size completion
commit title would overstate this result, so partial-work commits are labeled
as preflight evidence.

## Failures retained

Two undersized deterministic fixture attempts were refused by the native band
floor at 8/16 GiB. Increasing RAM did not fix the small-input floor. A larger,
highly repetitive fixture was interrupted during slow order-tape comparisons;
its three native phase groups were terminated explicitly. A 1.2 MB random,
255-symbol fixture then built and passed all 261 oracle patterns. The failed
logs and successful replacement are retained separately. The initial pytest
relative-import collection failure was corrected; the final tests pass.
Existing unrelated trailing-blank-line warnings from an unscoped git diff check
were left untouched; scoped checks apply to this work.

contact_supervisor/intercom tools were absent from the enabled catalog. No
WorksGood, dispatcher, worker, completion/review machinery or subagent was used.
The implementation remained in the attended session. Read-only corpus mounts,
broker-only GPU constraints and the frozen paired baseline were preserved.
