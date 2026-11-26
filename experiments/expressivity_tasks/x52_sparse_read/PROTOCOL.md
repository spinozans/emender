# X52 — can an earlier answer cover the three bytes between archive reads?

Register before new membership or outcomes. X51 established a large current-read gain in a small reader; its every-byte schedule left little evidence of useful reliance on old replies. This is a fixed read-budget comparison, **not** a learned skipping policy. No second/child lookups, previous weights, calibration or final access.

## Programs and information boundary

Predict external `x[t+1]` after seeing `x[0:t+1]` (zero-based in each new window). Scheduled sparse reads occur **only at t = 0, 4, …, 60**. There are 16 refresh and 48 skipped predictions in 64. Record the target's age from the last read as `t % 4`. Independently resolve only the scheduled literal prefixes, truncated to the X50 materialized depth of 8. No querying the other 48 answers then masking them, no aggregate/normalizer from skipped queries and no future or cross-window input. FULL can use the native causal prefix scan. Verify the sparse answers match the scheduled subset of the dense answers before interpreting quality or speed; different native batching costs are real work, not assumed away.

Same X51 one-layer width-32 model, two typed rows per observed byte, identical current byte/type/position shells. At a refresh only, add the learned projection of `sqrt(current_256_bin_pdf)` to the current answer row; base logits use `log(.99*pdf+.01/256)`. Otherwise the current row is an empty shell and base logits are a fixed, add-half, **FIT-only** target-byte unigram. A retained old answer is evidence for learned attention, **not** a stale direct prediction copied onto a later target. No action labels or other memory path. All previous observations remain available; reset completely at the window/document boundary.

| Trained arm | Actual XSA queries/window | Older visible answer content |
|---|---:|---|
| NONE | 0; no mount capability | None. Same shells and byte-only unigram throughout. |
| EVERY | 64; one at every byte | All prior replies. Full-rate quality/time reference, exact X51 KEEP at shared weights. |
| DISCARD4 | 16 | Current scheduled reply only; erase its content for every subsequent prediction. |
| KEEP4 | 16 | All earlier scheduled replies, including on skipped bytes; unscheduled slots never contain archive information. |

**ERASE4:** on KEEP4 PROBE only, re-score the same acquired packet with the same fitted weights but DISCARD4's older-value visibility. No new archive work. Keep the first prediction identical. Price intervention and receipt work separately from the chosen path. This tests reliance by the fitted reader; it does not substitute for a win over separately trained DISCARD4. A changed distribution alone isn't a gain.

## Population and fit

Same eligible out-of-index native TRAIN rank and window rule as X50/X51. Reproduce exclusions through PT; bind the X50 and X51 spent reservations and prove both are the consecutive prefix preceding this study. X51 public aggregate `bb8ea0b7fb85ecdccb0cdc58e04e3a04e164fc22c82af3c4abe1b92ae39bb4b0`, reservation `e8507b0ddf6bf5c4ffc7bdf96dc8b058e62758f6a38b7cde6f92cd19a737ace3`. **47,718 unique SHAs excluded** before X52. Take the next 512 FIT then 1,024 disjoint PROBE documents, or stop. No tuning, topping up, corpus duplication across roles or reuse at another offset. Verify published fixture/CPU qualification before new membership or raw access. Same pinned source and `documents.jsonl.gz` digest, offsets >=100,600,000,000, length >=4,097 and hashed aligned 4K-window start; check full document SHA and C14 shifts, take only its first 65 bytes at that start. The target positions lie outside the 100 GB P100 archive; this is not passage deduplication.

Seeds 5201/5202/5203. All four arms train from the same per-seed X50 Tape initialization/order, three epochs, document batches of 16, AdamW .003 / weight decay .01, clip 1, FP32 deterministic single-thread CPU. Rotate arm order by seed; new immutable-page LRU per arm/phase, no cross-fit feature cache. Output 256 next-byte logits at every position; optimize and score **all** 64 real targets, not just refreshes or hits. Counts for the unigram come only from the 32,768 shifted FIT targets and are frozen before optimization. All 12 terminal fits and the unigram must be sealed and independently witnessed before PROBE opens. One detached local CPU job; no GPU, remote submission, new final or automatic retry. Publish reviewed source and compact aggregates only, never private data, membership, corpus-derived per-query answers, per-document scores or fitted arrays.

## What decides the result?

Report all-byte, refresh, gaps pooled, and age 1/2/3 absolute NLLs for the four arms and fixed-weight ERASE4. Report full-rate and sparse base priors and unigram descriptively. Each PROBE seed has 65,536 real targets, of which 49,152 are gaps. Compute paired whole-document intervals, jointly resampling the same 1,024 windows across fixed seeds, 10,000 draws, RNG 5200, linear 95% percentiles. Gain = control minus candidate NLL, with the denominator appropriate to the stated band. A win requires all seed gains positive and lower bound >0; loss is symmetric, otherwise unresolved, not equivalence. No best-age or best-seed selection.

- **Primary:** KEEP4 vs DISCARD4 on the three skipped positions pooled. Also KEEP4 vs ERASE4 on those same positions at unchanged weights. Claim useful learned reuse *between lookups* only if both pass. Report the two contrasts over all bytes too. Report age-specific differences descriptively.
- **Quality paid:** EVERY vs KEEP4 on all bytes and on gaps; KEEP4, DISCARD4 and EVERY vs NONE on all bytes. If EVERY beats NONE, give the descriptive fraction `(NONE−KEEP4)/(NONE−EVERY)` at the matched dose. This is a quality/price comparison, not a noninferiority test or equal-time training comparison.
- **Price paid:** verify scheduled PDF and prior stream hashes against EVERY's scheduled subset and across sparse arms; verify equal FIT prior for skipped positions. Report true logical reads, frames/pages, chosen FIT/PROBE wall and CPU, receipt/diagnostic clocks separately. Report measured speedup or slowdown vs efficient full-rate batching, **not** a 4× speed claim from query counts. Report phasewise TV/changed rows (TV>`1e-6`) under frozen erasure, without filtering loss. No default or scale decision.
