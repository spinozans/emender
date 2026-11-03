# C14 cache scaling continuation — consultation work remains incomplete

The registered full-cache gate remains unsatisfied. This continuation tests two
implementation changes before allocating a full-size job: exact bounded LRU
reuse of native branch replies across selector chunks, and sparse binary window
records in gzip. The fixture-validated selector and native executable are
unchanged. Native serving uses -j128. This is CPU work; no GPU lease is needed or
used. The destination is new and outside git; existing caches are not overwritten.

The sparse format stores document/start/window length and, per position, selected
context length, censor flag, EOF and every nonzero byte/count pair. Branch order
and uint64 counts are retained. Reconstructing the registered 260 inputs uses the
same arithmetic and terminal policy. It does not quantize probabilities. Empty
support still zeros all inputs. Query-cache eviction affects query cost only.
No Pile text is saved. Decoder tests reject truncated or invalid records and
exercise all 256 branches, terminal mass and large counts.

## Measured result

Exactly **1,034,056 positions in 8,192 frozen training windows** were extracted
in **169.832325 seconds (6,088.688 positions/s)**, including native startup,
source/document checks, selection, arithmetic, compression and shutdown. Native
-j128 serving processed 6,909,960 queries, versus 7,033,954 in preflight: the LRU
saved 123,994 queries. Peak native child RSS was 295,126,642,688 bytes.

Exact row equality against the original JSON preflight passed for every position,
including all branches, entropy, counts, censoring and window identities. All
coverage/length histogram and support-rate diagnostics match the original
summary exactly. This is cache replay; the 261-pattern native oracle from the
previous session remains separate evidence.

The sparse gzip file occupies **34,266,350 bytes (33.137809 bytes/position)**,
SHA256 `b9d526b731f3c547b06a83503f263ec19a864d62846a338654908586e0bfde72`,
at `/home/erikg/c14_consultation_features/reuse_j128_8192/features.sparse.gz`.
The complete measured pipeline is 1.532732 times faster and uses 69.161692% of
the original storage. Jobs, reuse and representation changed together; this is
not an ablation attributing the gain to one change.

Prefix-specific linear scenarios remain **81.84 days for training, 0.83 days for
validation, and 1.441 TB combined**, against 807.20 GB free. They are not full-stream
measurements or lower bounds for other implementations. This tested combination
still does not justify a full allocation or satisfy the registered launch gate.
No additional unique positions beyond the preflight prefix were extracted.

Evidence: [benchmark summary](receipts/c14_cache_scale/benchmark_summary.json),
[exact parity](receipts/c14_cache_scale/parity.json),
[scenarios and remainder](receipts/c14_cache_scale/scale_comparison.json),
[frozen identities](receipts/c14_cache_scale/frozen_identity.json),
[14 passing tests](receipts/c14_cache_scale/tests.txt).

## Limits and retained failure

The benchmark uses the same first 8,192 training windows as serving preflight,
not new full-stream or held-out evidence. The matching distribution establishes
self-consistency for that prefix only. Its 16/64-byte support rates remain
47.931962%/0.819579%, with median maximal context 14 bytes, versus phase-1
36.59%/2.36% and median 12. Phase-1 sampled token-boundary interiors; this samples
source-order raw bytes under independent 128-byte resets. These populations
are different. There is no eligible 256/1024-byte history in these windows;
absence of eligibility is not zero measured support.

The first new test invocation failed collection because the test used relative
imports without package collection. Absolute project imports corrected it; the
final 14 tests pass. The failure did not run extraction or affect dense artifacts.

No complete train/valid cache, GPU transfer benchmark, consultation training,
paired loss/support table, device-off comparison, P1/P2/P3 verdict, coverage
subset swap or Pile-scale C13 stacking is claimed. The registered complete-cache
requirement is not waived. The new sparse encoding is a tested storage prototype,
not a training integration. No full extraction job is left running, and no
months-long job or cache larger than available disk is allocated. The original
single-seed dense controls remain frozen. The requested completion commit title
would overstate this result; this continuation is committed as partial work.

The source/offset identities, both dense checkpoint hashes, frozen evaluation
windows and model dependency hashes are checked in `frozen_identity.json`.
Those identity checks reuse the existing dense validation; they do not rerun
inference or rehash all selected corpus bytes. CPU affinity includes 256 logical
CPUs and the benchmark is not an isolated-core measurement.

contact_supervisor/intercom is absent from the enabled tool catalog. No supervisor
question, WorksGood service, dispatcher, worker or subagent was used. Work stayed
in the attended session, with ~/sxgc and /mnt/nvme2n1 untouched. No .pt file or
large cache is committed.
