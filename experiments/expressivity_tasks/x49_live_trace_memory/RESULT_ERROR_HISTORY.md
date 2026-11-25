# Remembering which bytes the model guessed wrong helped, a little

When `color: ` recurs, J reads what followed its latest match. EH leaves that read alone and looks at the *other* matches. Knowing what followed them, how surprising those bytes were and which ones the model missed lowered next-byte loss by **.002206 nats/byte over J** on 128 new documents. Also remembering **which wrong bytes it guessed** lowered loss by a further **.000131** [paired 95% interval .000068,.000196]. The fitted rule **suppressed** the old wrong guesses at all three checkpoints; it did not just promote them as plausible alternatives.

**Did the guesses belong to the right occurrences?** Replacing a wrong guess with one from an earlier mistake on the same observed byte with the same neural surprise code, where possible, cost **.000030 nats/byte** [.000013,.000046] with the reader held fixed. Letting the substitute reader fit its own coefficients still left the true history ahead by **.000023** [.000008,.000039]. Both contrasts were positive at every checkpoint. The substitute often retained the original guess, and often still provided useful information. This is evidence for a small benefit of the correct association, not evidence that the whole gain over J came from that association.

**Decision.** Both registered tests passed: pooled scalar history helped, and the identities of wrong guesses added something beyond it. HISTORY accounted for about **94% of ERROR's loss reduction relative to J** on this probe. That's a comparison of predictors, not a decomposition of stored information. The extra work was material: about **32 ms per 4,096 inputs just to prepare the extra tape and build the true pools** in this implementation. This is a probe on documents in the native **TRAIN split**, separate from fitting; it is not a native calibration or final confirmation. Keep B as baseline, U as native-final-confirmed and J as calibration-replicated. No default change or Lux nomination.

[Registered question, intervention and decisions](ERROR_HISTORY.md) · [implementation](ERROR_HISTORY_IMPLEMENTATION.md) · [public aggregate](receipts/error_history_result.json) · [pre-access CPU qualification](receipts/error_history_cpu_qualification.json)

## Prediction on every real byte

All readers used the same three frozen W-B weights, one fresh causal prefill per seed/document, H4096 and B's deepest eligible suffix and latest match. At each earlier arrival the bare host guessed **before** seeing the byte; after seeing it, the writer could mark a mistake. The new counts excluded the latest eligible occurrence, even when the key had no other match. No shallower fallback, target as a prediction feature, padding, cross-window memory, or extra neural read. COUNT adds relative follower counts and their interaction with J's confidence. HISTORY adds summed literal and neural surprise notes and counts of missed actual followers. ERROR also counts which incorrect guesses were made. SHIFT fits the same six-coefficient rule on displaced guesses; SWAP uses the displaced counts with ERROR's unchanged fit. The donor is the most recent strictly earlier mistake in this window on the same follower and exact neural code; fallback self. It preserves the source record's observed byte, both notes and mistake status, not necessarily the donor's age, text or literal code.

Nats per real shifted byte (lower is better). Same 128 documents and **524,288 targets per checkpoint**, including misses and singleton keys. The three checkpoints do not triple the number of independent documents.

| Reader | Mean NLL | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| B | 1.118561553 | 1.118398691 | 1.118682255 | 1.118603713 |
| U | 1.115461945 | 1.115276178 | 1.115730431 | 1.115379225 |
| J | 1.113716453 | 1.113466421 | 1.113944054 | 1.113738885 |
| COUNT | 1.113200414 | 1.112876077 | 1.113446967 | 1.113278198 |
| HISTORY | 1.111510247 | 1.111189720 | 1.111753466 | 1.111587554 |
| ERROR | 1.111378956 | 1.111095852 | 1.111595542 | 1.111445473 |
| SHIFT | 1.111402443 | 1.111130186 | 1.111617311 | 1.111459831 |
| SWAP | 1.111408636 | 1.111136869 | 1.111623618 | 1.111465421 |

Positive gain means lower loss for the first reader. Intervals use 10,000 jointly paired whole-document resamples at the three fixed weights, RNG 4915, each draw's own target denominator and linear percentiles. The four primary comparisons were fixed before access.

| Candidate:control | Gain | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| ERROR:HISTORY | +0.000131291 | [+0.000068064,+0.000195835] | +0.000093869 / +0.000157924 / +0.000142081 |
| ERROR:SHIFT | +0.000023487 | [+0.000008004,+0.000039135] | +0.000034334 / +0.000021769 / +0.000014358 |
| ERROR:SWAP | +0.000029680 | [+0.000013485,+0.000046325] | +0.000041018 / +0.000028075 / +0.000019948 |
| ERROR:J | +0.002337497 | [+0.001819410,+0.002933029] | +0.002370569 / +0.002348512 / +0.002293411 |
| HISTORY:COUNT | +0.001690167 | [+0.001366091,+0.002056831] | +0.001686356 / +0.001693500 / +0.001690643 |
| HISTORY:J | +0.002206206 | [+0.001722103,+0.002763529] | +0.002276700 / +0.002190588 / +0.002151330 |
| COUNT:J | +0.000516039 | [+0.000311416,+0.000753147] | +0.000590344 / +0.000497087 / +0.000460687 |
| SHIFT:HISTORY | +0.000107805 | [+0.000052661,+0.000165293] | +0.000059535 / +0.000136155 / +0.000127724 |
| ERROR:U | +0.004082989 | [+0.003334468,+0.004925290] | +0.004180327 / +0.004134889 / +0.003933752 |
| J:U | +0.001745492 | [+0.001436008,+0.002075797] | +0.001809757 / +0.001786377 / +0.001640340 |
| U:B | +0.003099609 | [+0.002729807,+0.003475551] | +0.003122513 / +0.002951824 / +0.003224488 |

## How much history was there?

Per checkpoint there were 512,831 latest hits with **245,130 correct copies** (47.80% of hits). The 11,457 misses and 143,586 singleton hits were scored unchanged from J. The other 369,245 predictions had older history, totaling 10,114,467 occurrence uses (27.39 per active query). An occurrence can be used at more than one query. Distinguish original arrivals, repeated uses and changes to the *pooled predictive input*:

| PROBE count | 4901 | 4902 | 4903 |
|---|---:|---:|---:|
| Guesses written | 524,160 | 524,160 | 524,160 |
| Mistakes at arrival | 213,307 | 213,871 | 214,092 |
| Mistake arrivals with an earlier matched donor | 89,173 | 89,467 | 90,150 |
| Arrivals where this changed the guessed byte | 60,799 | 61,236 | 61,555 |
| Older mistake uses in queries | 2,510,930 | 2,476,343 | 2,477,434 |
| Uses with a matched donor | 803,643 | 771,488 | 784,528 |
| Uses with a different guessed byte | 469,128 | 462,141 | 468,514 |
| Queries with a changed histogram | 119,763 | 120,881 | 121,351 |

All 128 documents at each seed had at least one changed histogram. The control actually changed about **22.84–23.15% of all predictions' input histograms** (32.43–32.87% of active histories); it does not reassign every historical mistake. Swaps can cancel in a sum. Gains above are **not** restricted to affected rows. The HISTORY:COUNT contrast tests the additional observed notes and miss counts *together*, not the causal contribution of either note on its own.

## Storage and work

The correct reader retains J's learned 32-float write and its two true note bytes (520 KiB of explicit tapes at 4K), plus a 4 KiB guess tape in this implementation. The tape alone does not price the history: the true table admitted 16,305 key occurrences per window across depths 8/4/2/1. Peak PROBE counts were **7,229 keys with a pending latest position each and 4,997 committed key/byte cells**. As one explicit logical accounting, a key of up to 8 bytes plus one length byte, uint16 pending position and uint16 older-count costs at most 13 bytes; one byte symbol plus three uint16 counts and two uint32 surprise sums costs 15 per cell. Combining even the separate maxima gives 173,028 extra bytes for these fields including the guess tape. That is **not** the size of the actual Python index: it excludes lookup/offset structures, object overhead, alignment and diagnostic counters. It neither includes J's shared index nor proves a compact deployed layout. The six installed ERROR coefficients are 48 FP64 bytes per checkpoint.

In the research panel the second index needed up to 2,551 committed sham cells, 7,229 pending keys and 1,364 donor keys; ERROR alone needs none of these. Packed per-query snapshots peaked at 428,000 bytes true and 90,016 displaced; joint FP64 feature columns peaked at 1,034,320 bytes before other workspaces. The J text writer peaked at 7,254 contexts and 9,500 context-follower pairs. These are individual peaks, not necessarily the same document. The [aggregate](receipts/error_history_result.json) separates FIT and PROBE and gives each checkpoint's work, peaks and timings.

**Mean component ms per 4,096-position PROBE window/checkpoint.** Shared host prefill 98.652; B 74.167; current trust fields 3.217; neural note build/read .464/.061; text writer 1.453; J correction 3.236. B/U comparison assembly 3.293 was outside J. Guess and tape preparation **1.238**, true history pooling **30.396**. Additional comparison-only sham pooling **15.301**; shared feature assembly including the sham **4.233**. All-256 responses COUNT/HISTORY/ERROR/SHIFT/SWAP **4.184 / 4.053 / 3.999 / 4.113 / 4.073**. These clocks come from one shared-prefill panel, not isolated serving paths. In particular one cannot subtract the HISTORY and ERROR response times to estimate the cost of collecting wrong guesses.

On FIT, mean ms/window for guess preparation / true pools / sham / features / four sufficient-statistic updates were 1.562 / 32.879 / 20.769 / 4.845 / 2.802. Shared host and B were 111.743 and 84.485. Solving all four small systems took 1.00–1.25 ms per seed; the all-seed seal took .340 s. No neural updates, no refit to U or J, no fitted SWAP.

## Access, cost and closure

CPU qualification was published at `04109858a1ffc91f7e00184008a20caeb7a4b72e` (local `ba72e13f5f3a8850bb539257d50609bbb1f6788d`) **before** natural reservation: 164 CPU checks, 654 native frontiers, three strictly loaded frozen checkpoints; full-size invented packet costs. On metadata, rederived 44,088 excluded whole-document SHAs and 22,419 eligible full native TRAIN documents, taking the next contiguous 256 FIT and 128 PROBE. Whole-document hash exclusion is not passage deduplication. FIT-only preparation checked 4,450,660 source bytes in 39.27 wall / 39.18 CPU s. All 12 fitted readouts, FIT journals and identities sealed at independently witnessed unit **768**, before any fresh PROBE text. PROBE then checked 1,993,890 source bytes; no persistent PROBE window file. An independent observer verified advancing real work through the terminal count of 1,152 natural seed/document prefills.

One independently funded, detached, renewing RTX 6000 Ada lease **released rc=0 at ≤348 card-s** of the 1,200 ceiling. 364.78 s inclusive scheduling wall; 665.56 owner and waited-child CPU s; driver and short parents another 16.90 CPU s. Peak device allocated/reserved 715,992,064 / 746,586,112 bytes, worker peak RSS 2,608,672 KiB. The CPU close checked the actual weight hashes, all twelve coefficient vectors against independent FIT-only Cholesky solutions, document-level support and index peaks, all eleven intervals from absolute losses, full exposure and the release ledger, without reopening the corpus. Both EH roles are spent. Public aggregate SHA-256 `60c270920302709dc57664c018db963db4946bc834f59f48f237d5d114ce00ec`; full private close SHA-256 `be743653fa8bf84b82184d3cd5617b217f05b7a505313188f65b1ec017ac20b3`. Fitted arrays, memberships and journals stay private.
