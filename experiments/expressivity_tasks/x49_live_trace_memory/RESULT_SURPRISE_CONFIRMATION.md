# X49-U held-out confirmation: the earlier forecast is worth remembering

**Use U alongside B for these three frozen models.** When a follower byte first appeared, U stored one byte describing how surprising it was under the model's preceding forecast. On 342 new native final documents, that record reduced loss by **.002800 nats/byte over current confidence alone** and **.002637 over a mark from a different, matched occurrence**. Against unmodified B the gain was **.003261**. All three comparisons passed the [registered decision rule](SURPRISE_CONFIRMATION.md): positive at each seed and a paired 95% interval entirely above zero. No coefficients or weights were changed for this test.

[Aggregate receipt](receipts/surprise_confirmation_result.json) · [closed-receipt audit](confirm_analysis.py) · [original fits and TRAIN probe](RESULT_SURPRISE.md).

## What was confirmed?

B retrieves the most recent eligible literal follower and its bound 32-D neural write. U leaves the address, byte, vector, and neural weights alone. At a hit, its four fixed coefficients use current confidence and the **old bare host's surprise for that particular follower** to adjust the odds of predicting the retrieved byte. The other 255 bytes keep their relative odds; on a miss, U predicts exactly as B does. N uses only the current confidence fields. S has U's capacity and the same current fields, but its old mark comes from the same observed byte in the same age bin, displaced when possible. S uses its own previously fitted coefficients. Neither the byte nor B's vector is displaced.

| Predictor | Held-out loss (nats per real byte) |
|---|---:|
| B: original reader | 1.149837 |
| N: current confidence | 1.149377 |
| S: current confidence + matched control mark | 1.149214 |
| **U: current confidence + correctly bound mark** | **1.146577** |

Positive gain = control loss minus candidate loss. Every comparison includes **all 525,209 shifted targets per seed**, not just hits or changes. For uncertainty, 10,000 joint draws resampled whole documents across predictors and fixed seeds (RNG 4908); each draw divided by its **own** number of real targets. The unit of replication is the document, not three copies of the document or half a million independent bytes.

| Candidate : control | Gain [paired 95% interval] | Gains by seed 4901 / 4902 / 4903 |
|---|---:|---:|
| **U:N** — incremental value of the past | **+.002800 [+.002529, +.003078]** | +.002808 / +.002721 / +.002871 |
| **U:S** — value of the correct occurrence | **+.002637 [+.002388, +.002893]** | +.002686 / +.002547 / +.002679 |
| **U:B** — gain over the original reader | **+.003261 [+.002981, +.003548]** | +.003249 / +.003163 / +.003370 |
| N:B — calibration with no historical mark | +.000461 [+.000386, +.000536] | +.000441 / +.000442 / +.000499 |
| S:N — displaced mark vs current only | +.000163 [+.000104, +.000223] | +.000122 / +.000173 / +.000192 |

Current calibration helped, and the control mark was not useless. But **neither explains away the gain from linking the forecast to the right occurrence**. The effect transferred from full TRAIN windows to fresh, shorter native final documents without a new fit. This comparison does not identify why the old mark works, show that surprise is the best mark, or compare U with a separately trained A on this final.

## Exposure, coverage and cost

The metadata-only reservation excluded 41,829 prior document SHAs. Of 663 eligible whole short final documents, it selected the next 342 in the existing rank immediately after W's spent prefix, stopping on the first whole document crossing 524,288 targets. **321 were left unreserved and unopened.** Reservation took 23.66 s wall / 23.65 CPU s and opened no text. Under the one evaluation lease, the worker verified 525,551 source bytes and both shifts. No padding, cross-document state, fabricated end-of-document target or persistent raw copy. This is full-document disjointness, not proof of passage-level deduplication.

| Real targets per document | Documents | Real targets | Hits | Misses |
|---|---:|---:|---:|---:|
| 1–15 | 0 | 0 | 0 | 0 |
| 16–127 | 4 | 372 | 197 | 175 |
| 128–1,023 | 112 | 64,628 | 57,642 | 6,986 |
| 1,024–4,095 | 226 | 460,209 | 442,454 | 17,755 |
| **Total per seed** | **342** | **525,209** | **500,293** | **24,916** |

- The follower was correct on 220,038 hits (43.98%). Every document had a hit, a miss, a correct copy, a displaced slot and a changed mark code. S displaced 406,193 slots; codes actually differed on 379,570 / 380,588 / 380,734 reads. U and S have separately fitted coefficients, so their scores can differ even at equal codes.
- The extra tape is **one uint8 per input**, reset at each document: 525,209 bytes cumulatively per seed, not simultaneously resident. It encoded 524,867 arrivals (first slot never read) and U read 500,293 marks. Of the arrival normalizations, 499,957 were shared with the current fields and 24,910 were extra. No written or read historical mark saturated. The current bare-host field saturated on 10 / 12 / 10 hits; the current B field never saturated. Tape size excludes B, host state and temporary scores.
- Mean times per seed/document: shared host prefill **111.37 ms**, ordinary B head **30.34 ms**, shared current fields **1.79 ms**, U/S mark construction **0.38 ms**, U mark read/feature **0.045 ms** (S rotation + read **3.51 ms**), N/U/S score adjustment **1.66 / 1.55 / 1.53 ms**. One host prefill served all four predictions. These are components of the batched experiment, **not** a measurement of standalone serving latency or throughput.
- One RTX 6000 Ada lease, released rc=0 at **≤185 card-seconds (3 min 5 s)**, 204.03 s inclusive scheduling wall, 312.64 owner plus waited-child CPU s. Peak device allocated/reserved 715,972,608 / 754,974,720 bytes; process peak RSS 2,753,828 KiB. The detached driver's witness verified owner, renewal and advancing actual target counts through 1,026 seed/document units. The earlier joint U fit/probe campaign cost **≤287 card-s**, separately; this confirmation paid for no fitting or training.

Public pre-reservation qualification `115ea71755feb02e168f92524b79b6fc79b3f234`, local qualified source `b034099576d15f276b5f15f8e8603165c6d91194`. The three checkpoint files and pre-PROBE fit-map (`67656fe3d1053a30519574c70e0c6258c4bced91089be171ffb1d832075156b9`) were bound before raw access. An independent reduction from absolute losses and document multiplicities reproduced all five paired comparisons; the audit also rehashed all three checkpoint files and verified terminal work and release without reopening text. Aggregate SHA-256 `9c9bbabdf8cbb146af32f6b031e2c946c3840834c42779c6fc8cfd135bbc5626`.

**Decision boundary:** B remains the comparison baseline; the unchanged U readout earned local use with its matched checkpoint. No universal host claim, no change to the Lux nomination decision, no further sampling for this test.
