# One archive read beat the learned follow-up on fresh text

**We built the loop.** The first read asks the mounted 100 GB XSA what follows the observed text. The model can inspect the answer, choose a hypothetical byte, and ask about the extended string *without* adding that byte to the document. Both answers and the chosen operand stay in the sequence that the model attends to when predicting real bytes. The question was whether learning that second question would earn its cost.

**On this probe, no.** On one 64-target window from each of 256 previously unused documents, asking a learned follow-up raised next-byte loss by **0.002731 nats/byte** compared with making one read (paired 95% interval for the **increase**: **0.000365 to 0.005117**). All three training seeds did worse. The learned program took **2.38× the batched CPU evaluation time**. It did not just reproduce the fixed rule: it chose a different byte from the most common continuation on **29.1%** of the probe queries, pooled over seeds.

The tested alternative with the best mean loss was *always* to ask about the most common continuation. It beat learned selection at every seed, but did **not** reliably beat a single read. Giving a second reader the same lookup budget while hiding archive answers from its query selector did not rescue it either. We keep one read as the economical choice **for this prototype**. This is not a result against string-kernel attention, against building a work tape, or against every possible learned search program.

[Registration](PROTOCOL.md) · [pre-access native and CPU checks](receipts/qualification.json) · [full redacted result](receipts/result.json) · [runnable reader](README.md)

## What ran

The archive operation pools exact next-byte counts from the root and materialized suffixes of lengths 1–8, weighted by `32**depth` before normalization. Every eligible corpus byte contributes at the root; matching materialized suffixes get extra weight. Missing materialization is not interpreted as proof of zero occurrences. The reduction is exact for this declared kernel and tied byte values, not dense learned attention over all corpus activations. The local neural context, in contrast, is ordinary causal attention over an append-only sequence of observed byte rows and typed read rows. It reaches 128 rows for ONE or 192 for a two-read program in the registered 64-byte context. No possible child's answer is fetched merely to score its candidacy.

Every fitted arm had the same 32-wide attention architecture, the same positive direct archive prior, the same paired initialization and document order, and the same training exposure. ONE uses parent answers only. GREEDY follows the parent histogram's largest mass. BLIND and ADAPT sample one hard action during fitting and use argmax at evaluation. BLIND's selector sees only actual preceding bytes; ADAPT's selector can see the current parent answer and all earlier chosen answers. Both two-read predictors see the genuine parent and chosen child. The action receives credit for the remaining window losses, not just the next label. The model never receives a correct query as supervision. All four arms must make their prescribed number of reads: this is a test of *which extension*, not learned stopping.

## All real next-byte losses

Lower is better; no hit, match, entropy or changed-action filter. PRIOR is the unfitted smoothed archive prediction, not a fifth trained model and not a no-archive control.

| Reader | Mean NLL | Seed 5001 | Seed 5002 | Seed 5003 |
|---|---:|---:|---:|---:|
| PRIOR | 1.39193934 | 1.39193934 | 1.39193934 | 1.39193934 |
| ONE | 1.37832303 | 1.37917200 | 1.37909295 | 1.37670415 |
| GREEDY | 1.37711613 | 1.37676598 | 1.37721408 | 1.37736832 |
| BLIND | 1.38162329 | 1.38061414 | 1.38098088 | 1.38327485 |
| ADAPT | 1.38105444 | 1.38146885 | 1.38206429 | 1.37963017 |

ONE lowered mean loss by .013616 relative to the archive prior. It used both observed bytes and archive replies; this comparison alone does not isolate which part of the tape produced that improvement. In particular we did not compare attention over past work with attention to only the newest read, or train a matched reader with the archive removed.

Gain = **control loss minus candidate loss**; positive favors the first name. Each of 10,000 bootstrap draws resamples whole paired documents jointly across all three fixed seeds (RNG 5000, linear percentiles). The three seeds did not supply three independent test corpora.

| Candidate:control | Gain, nats/byte | Paired 95% interval | 5001 / 5002 / 5003 |
|---|---:|---|---|
| ADAPT:ONE | −0.00273140 | [−0.00511686,−0.00036518] | −0.00229685 / −0.00297134 / −0.00292602 |
| ADAPT:BLIND | +0.00056886 | [−0.00050534,+0.00161403] | −0.00085471 / −0.00108340 / +0.00364468 |
| ADAPT:GREEDY | −0.00393831 | [−0.00658762,−0.00135572] | −0.00470287 / −0.00485021 / −0.00226185 |
| GREEDY:ONE | +0.00120691 | [−0.00152166,+0.00406599] | +0.00240601 / +0.00187887 / −0.00066417 |
| BLIND:ONE | −0.00330026 | [−0.00580328,−0.00084949] | −0.00144214 / −0.00188793 / −0.00657071 |

ADAPT departed from GREEDY on **5,164 / 3,517 / 5,623** of 16,384 queries at the three seeds, respectively. BLIND departed on 16,362 / 16,010 / 16,358. ADAPT:BLIND is unresolved, not equivalence. The separately fitted predictable GREEDY policy was better than learned selection; neither this comparison nor the action counts identify a single cause (different queries, sampled-versus-greedy fit, delayed credit, training dose, etc.).

## Price and scope

| Arm | FIT CPU-run wall s/M targets | PROBE CPU-run wall s/M targets | PROBE time / ONE |
|---|---:|---:|---:|
| ONE | 61.73 | 48.05 | 1.00 |
| GREEDY | 93.47 | 81.39 | 1.69 |
| BLIND | 90.67 | 66.25 | 1.38 |
| ADAPT | 140.14 | 114.57 | 2.38 |

These are observed batched times at equal exposure, including chosen lookups and local model work. Each seed made 16,384 parent reads per probe arm; every two-read arm also executed exactly 16,384 child reads. GREEDY and BLIND can batch chosen requests across time; ADAPT has a true serial dependency on earlier child replies. Prices are not single-request decode latency, GPU throughput, or quality at matched wall time. The [aggregate](receipts/result.json) includes per-seed walls, native page reads and the distinction between logical queries and transport batches. User-space page caches restarted for each arm/seed/phase, not the OS page cache.

**Data:** Excluded 45,414 already-spent document SHAs, then reserved the next 768 eligible TRAIN documents once. FIT used 512 documents, three passes (98,304 real targets per seed/arm). All 12 terminal fitted states were sealed and independently witnessed before the disjoint 256-document TRAIN PROBE opened (16,384 targets per seed/arm, one pass). Full-document hashes and C14 shifts were checked; FIT/PROBE respectively read 8,226,306 / 4,634,980 source bytes to verify them. The windows are beyond the mounted prefix: the archive did not contain the target positions themselves; duplicated text elsewhere is still possible. This is an out-of-fit pilot on native TRAIN, not a calibration/final, a general policy benchmark or a 150M-host result.

**Closure:** One detached CPU job, no GPU, 154.63 s worker wall including reservation and setup, 134.02 owner CPU s and 38.57 child CPU s. Native open 0.284 s; peak owner/max-child RSS 1,240,296/1,951,592 KiB (not additive). The observer witnessed advancing targets, the FIT seal and all 1,344 terminal batches (1,376,256 train-plus-probe target exposures), followed by owner exit. A separate receipt-only review reproduced the intervals using bootstrap multiplicities rather than indexed loss rows. Neither review reopened raw text. FIT and PROBE are spent; no tuning or retries on these windows. Public result SHA-256 `5440696c48fb6647e9f1efd1d1f1e046b69d82582906895d10600e8ecdfae260`.
