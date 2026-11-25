# Same surprise notes, less time writing them

The [registered CPU check](JOINT_READER.md) passed. On invented 4,096-byte windows, the exact count-based writer took **0.29–2.77 ms** instead of **15.76–20.05 ms** (medians). This changes *how* the two-note predictor is implemented, not its rule for guessing the next byte. It makes a no-refit test of that rule more practical; there is **no new evidence about language prediction** in this measurement.

| Input, 4,096 observed bytes | Python median / p90 (ms) | Native median / p90 (ms) | Median paired speed ratio | Distinct contexts / context-follower pairs |
|---|---:|---:|---:|---:|
| All zeros | 17.074 / 17.380 | 0.291 / 0.304 | 59.21× | 4 / 4 |
| Repeated `color: blue` / `color: black` pattern with binary bytes | 15.759 / 16.957 | 0.337 / 0.364 | 48.85× | 79 / 87 |
| Uniform pseudorandom bytes | 20.048 / 20.848 | 2.772 / 2.852 | 7.39× | 12,411 / 16,250 |

One AMD EPYC 7713 CPU thread, two warmups and 31 paired trials per fixture, alternating order. Each call built fresh tables and returned the byte tape and work counts, including the Python/native interface. All three made exactly 16,369 suffix lookups and updates. The largest table occurred on the random input and was slowest in both implementations. These are not measurements of the host, entire reader, streaming latency or natural throughput. In particular, the [LS](RESULT_LITERAL_SURPRISE.md) 21.50 ms natural-window timing was for the **old** writer; we didn't reopen those windows.

[`text_tape.cc`](text_tape.cc) uses separate namespaces for each suffix length, exact follower counts, FP64 interpolation, and writes before updating any counts. [`joint_reader.load_candidate(seed)`](joint_reader.py) binds the closed J coefficients to the matching W-B weight file and the verified build, in an **already admitted** worker. It returns scores from one prefill and the correctly bound marks only: no sham search or sibling scores. It is for a separately admitted evaluation, not the local default. U remains the confirmed reader alongside B.

The actual retained binary passed 21 synthetic CPU tests, including all-byte-score equality of the bound-only path with the LS reference, dense prefix oracles, prefix extension/reset, binary keys, saturation, rejection of changed builds/coefficients and wrong checkpoints. No GPU acquisition and zero natural documents. GCC 13.3.0 build 0.750 s; no fast math or fused multiply-add contraction. Independent reduction of the closed paired trial list reproduced all medians, p90s and ratios. [Compact receipt](receipts/joint_reader_cost.json), SHA-256 `25389c636ec9524888bf63f0a8301abaab449ba45aa6e4b427f0474bfc660c0c`; local code `a65ef57ef71d4fa9bb83fd018ddbf6ababbb60bd`. Compiler flags, source and library digests are in the receipt. No binary, timing journal or document memberships in the release.
