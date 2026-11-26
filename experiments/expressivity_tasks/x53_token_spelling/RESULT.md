# X53 — the rest of the token matters

**The same lookup bought more when it returned possible tokens rather than just the first byte.** On 262,144 held-out raw bytes, retaining the full spelling alternatives lowered loss from **2.36528 to 1.95812 nats/byte**: a gain of **.40715 [.39799, .41592]**, or 17.2% lower predictive loss. The readers made exactly the same 65,159 lookups. Their entire 256-way forecasts at those lookup positions were identical; the gain came *between* lookups.

**Where the gain came from:** giving the reader a general token spelling dictionary, but not the suffix node's preferences within each first-byte group, got it to **1.99838**. That accounts for 90.1% of the measured `.40715` reduction. Which suffix node had returned the alternatives still mattered: its preferences beyond the first byte added **.04026 [.03710, .04348]** nats/byte over the dictionary control. Both registered comparisons passed on all bytes and on within-token predictions. This is evidence for a useful token continuation record in a **fixed count reader**. It is not a result about learned attention, BPE, or a tokenized P100.

[Registration](PROTOCOL.md) · [pre-access CPU/source qualification](receipts/qualification.json) · [compact audited result](receipts/result.json) · [invented suffix-node oracle](../token_suffix_fanout/README.md)

## Comparison

Build a small token suffix index and the common byte predictor from one 4,097-byte raw window in each of 512 TRAIN documents. A token ends on the first non-ASCII-word byte (a byte outside `A–Z a–z 0–9 _`), delimiter included. Ignore the initial fragment through its delimiter and never commit a trailing unfinished token. The rule is causal, reversible on the indexed complete tokens, and deliberately simpler than a general tokenizer. A new document starts from scratch. No token ID or end marker is an external target.

At the first observed delimiter, read the root. At every subsequent observed delimiter, advance by the token that actually finished and read the longest matching token suffix; unknown tokens reset to root. The archive returns its content follower counts, backed off with four pseudo-occurrences distributed by BUILD root frequencies. Inside a token, keep only the alternatives matching the bytes observed so far. No additional suffix query or speculative child read is needed to filter a saved answer. Mix its next-byte law half-and-half with the common smoothed byte trigram; use the trigram alone if unsupported. Nothing updates on PROBE.

| Reader | What it knows between lookups | All bytes | At lookup | Inside token | Before first lookup | CPU wall s/M bytes |
|---|---|---:|---:|---:|---:|---:|
| BYTE | Byte context only; no token index access | 2.38911276 | 3.06583005 | 2.16162723 | 2.37103558 | 0.804 |
| FIRST | Same byte context; lookup's first-byte forecast only at the boundary | 2.36527665 | 2.96993397 | 2.16162723 | 2.37103558 | 8.388 |
| LEX | Contextual first forecast, then general BUILD spelling frequencies | 1.99837803 | 2.96993397 | 1.66472821 | 2.37103558 | 16.874 |
| SPELL | Contextual first forecast **and** that node's token preferences | 1.95812296 | 2.96993397 | 1.61020987 | 2.37103558 | 31.652 |

Loss is in nats per real byte in the named band (lower is better). PROBE had **1,024 new TRAIN documents**, one 257-byte window apiece: 65,159 lookup, 193,561 inside, 3,424 pre-lookup targets. The same reset, eligible prefix and lookup site were used for every reading arm. All route states/lengths and first forecasts were checked for equality at every real refresh, not just averaged afterward. All pre-lookup distributions were equal. FIRST and BYTE were identical between reads. Ratios below are not accuracy gains.

| Candidate : control | Band | Control minus candidate [95% interval], nats/byte |
|---|---|---:|
| SPELL : FIRST | all | +.40715368 [.39799205, .41592451] |
| SPELL : FIRST | inside | +.55141736 [.54119127, .56133280] |
| SPELL : LEX | all | +.04025506 [.03709695, .04347680] |
| SPELL : LEX | inside | +.05451834 [.05021479, .05889264] |
| LEX : FIRST | all | +.36689862 [.35896571, .37464082] |
| LEX : FIRST | inside | +.49689902 [.48860076, .50497816] |
| FIRST : BYTE | all | +.02383611 [.02041496, .02752948] |
| FIRST : BYTE | at lookup | +.09589609 [.08278336, .10977934] |
| SPELL : BYTE | all | +.43098979 [.42130807, .44034657] |

Intervals are 10,000 jointly paired whole-document bootstrap draws, RNG 5300, linear quantiles, with the byte denominator recomputed in each draw. The readers are deterministic; the interval measures document sampling uncertainty, not training-seed variability. The 90.1% is the descriptive ratio `(FIRST−LEX)/(FIRST−SPELL)` of all-byte loss reductions under these controls, not a fraction of all information in the index.

## Coverage and price

- All reading arms tokenized 1,024 streams, made 65,159 lookups (one per 4.02 target bytes **on average**, not a fixed cadence), and committed 64,135 complete query tokens after the first anchors. 56,723 were known. There were 8,436 root and 56,723 nonroot reads; mean matched depth 3.20 tokens including root. Of nonroot reads, 34,520 missed the per-arm immutable source-response cache and 22 had no content follower. No cross-arm feature reuse.
- General spelling evidence covered 178,236 of 193,561 inside predictions (92.1%). On the other 15,325 both spelling readers fell back to the byte model. The original **nonroot** fanout still had at least one matching continuation on 59,998 inside predictions (31.0%). SPELL and LEX differed at TV > `1e-12` on 55,077 inside positions in 1,015 documents; mean TV across *all* inside positions .06108. Distribution movement alone would not establish a gain; the contrasts above do.
- The archive represents 517,399 complete token occurrences and 58,495 distinct spellings (455,621 dictionary bytes); 655,645 suffix-automaton states, 1,087,974 transitions, 176,567 local spelling-trie nodes. The private numeric image, **including** the byte model, is 70,134,987 bytes. Accumulating/persisting it took 0.78 s excluding raw extraction; restoring counts and mounting the automaton/lexicon took 2.01 s. Peak worker RSS 1,287,844 KiB includes Python, data, indexes and validation; it is not the size of a serving page.
- FIRST and LEX each copied 58,149,511 response bytes; SPELL copied 283,033,559 (**4.87×**). Nonroot projection pages carry 256 four-byte counts, while full pages carry `(token ID,count)` pairs; root uses one tag referring to pre-mounted root data. Across the probe SPELL returned 35,371,050 nonroot pairs. Logical fanout including the shared root was 528,834,870 entries per reader; those were **not** all transported. Counts are actual in-process packed copies, not network throughput, and exclude the one-time dictionary mount.
- Chosen probe wall: BYTE 0.211 s, FIRST 2.199 s, LEX 4.423 s, SPELL 8.297 s. Of the reading-arm totals, timed token completion/navigation and source-response acquisition used 1.368 / 1.710 / 1.548 s (FIRST/LEX/SPELL); they do **not** include the full consumer-side spelling processing. Block execution order rotated across the four programs. These are local CPU reference clocks including prediction and scoring; an additional 0.738 s of comparison/progress work was outside them. SPELL took **3.77×** FIRST's time and **1.88×** LEX's. No claim of cheaper lookup from tokenization alone.

## Data boundary and decision

The next contiguous block after 49,254 whole-document SHA exclusions supplied BUILD then PROBE. The pinned flagship manifest and complete source document hashes/C14 shifts were checked. The index saw only BUILD windows (2,097,664 raw bytes); it did not see probe text. The count image and compiled index were sealed and independently witnessed before PROBE opened. The separate witness closed 1,048,576 chosen-path target exposures; recomputation from receipts confirmed the scores and intervals without reopening windows. Worker wall 46.72 s, no GPU, no final-test access. Whole-document exclusion does not exclude repeated passages.

**Keep full token branches as a quality reference, and keep the general spelling control as the cheaper baseline.** A promising engineering comparison is to hold the token law fixed while replacing repeated large page copies with reusable immutable handles/local prefix reductions. Check all 256 probabilities for parity and price the full path, including resident memory. This result neither shows that token retrieval beats dense byte P100 nor justifies building a 100 GB token index; the corpora, features and readers are different.
