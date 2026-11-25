# The third byte improved prediction; its tie to a particular occurrence did not pass the test

When the model sees `color: ` again, it can find an earlier `b` and use two notes about how surprising that `b` was. D uses those notes to reconstruct the vector the model would otherwise have stored. R also computes the real vector on arrival and saves **one byte describing what D missed**. On 128 new documents, that extra byte lowered next-byte loss by **.000553 nats/byte** over D [paired 95% interval .000382,.000718]. Adding just a fixed average correction did not explain the gain. The full vector still beat R by **.000329** [.000176,.000471]. R recovered about **63%** of the full vector's loss advantage over D on this probe (`R:D / J:D`), not 63% of its information.

**Did it matter that the byte belonged to this particular `b`? Not demonstrated.** S kept everything else fixed but used the byte of a different earlier `b` with *exactly the same two notes* and in the same age bin, when there was one. R:S was **+.00000027 [−.00001110,+.00001162]** nats/byte; two of three checkpoints had negative point estimates. All 128 documents contributed actual changes, but only about **5% of hit rows** had a different residual byte. S still used the true byte everywhere it could not find a match. This is neither an equivalence test nor evidence that a reader could dispense with computing the third byte. A lower reconstruction error did not settle this prediction question.

R beat U on these documents, but the registered rule required **all three** of R:D, R:G and R:S before considering transfer. **No calibration transfer, no new default, no scale nomination.** B remains the baseline; U remains native-final-confirmed for its three checkpoints, and J the full-vector candidate replicated on calibration. No final or calibration documents were used here.

[Prospective comparison](RESIDUAL_BYTE.md) · [implementation and isolation checks](RESIDUAL_BYTE_IMPLEMENTATION.md) · [verified aggregate](receipts/residual_byte_result.json)

## What was compared

One true document-local H4096 address and one causal host prefill per window/checkpoint. B/U/J use the full historical vector. D reconstructs it from the observed follower and two arrival notes. G adds the global mean residual to D; R adds the mean residual for the stored eight-sign code; S displaces only the residual code, not the address, follower or trust notes. Each of D/G/R/S uses the fixed J trust rule on its **own** raw scores. All **524,288 real shifted bytes** per checkpoint, including misses; never select only changed codes. The same 128 native TRAIN PROBE documents were scored at all three frozen W-B weights after fitting codebooks on 256 different native TRAIN documents.

| Reader | Mean NLL | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| B: full vector | 1.117895 | 1.117973 | 1.117214 | 1.118496 |
| U: full vector + neural trust note | 1.114830 | 1.114848 | 1.114280 | 1.115361 |
| J: full vector + both trust notes | **1.113055** | 1.113029 | 1.112467 | 1.113667 |
| D: two-note reconstruction | 1.113936 | 1.114005 | 1.113260 | 1.114544 |
| G: D + global correction | 1.113938 | 1.114002 | 1.113264 | 1.114547 |
| R: D + correct coded residual | 1.113384 | 1.113376 | 1.112731 | 1.114043 |
| S: D + conditionally displaced code | 1.113384 | 1.113376 | 1.112733 | 1.114042 |

Positive gain means lower loss for the first reader. 10,000 whole-document draws paired across readers and fixed checkpoints, RNG 4914, linear percentiles. Scores are per real byte, not top-choice accuracy. Seeds do not triple the number of independent documents.

| Candidate:control | Gain (nats/byte) | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| **R:D** | **+.000553** | [.000382,.000718] | .000629 / .000529 / .000501 |
| **R:G** | **+.000554** | [.000385,.000719] | .000626 / .000533 / .000504 |
| **R:S** | **+.00000027** | [−.00001110,+.00001162] | −.00000055 / +.00000258 / −.00000122 |
| J:R | +.000329 | [.000176,.000471] | .000347 / .000264 / .000376 |
| R:U | +.001446 | [.000956,.001984] | .001472 / .001549 / .001318 |
| R:B | +.004511 | [.003939,.005133] | .004597 / .004484 / .004453 |
| J:D | +.000882 | [.000590,.001155] | .000976 / .000793 / .000877 |
| D:U | +.000894 | [.000364,.001482] | .000844 / .001020 / .000817 |
| S:G | +.000554 | [.000385,.000718] | .000626 / .000530 / .000505 |
| G:D | −.00000143 | [−.00000586,+.00000298] | +.00000268 / −.00000362 / −.00000335 |
| J:U | +.001775 | [.001314,.002255] | .001819 / .001813 / .001694 |

## How often did the intervention act?

There were **513,020 hits, 11,268 misses and 252,097 correct copies** per seed. The exact-triple rule moved **39,177 / 38,872 / 39,050** slots; it changed the byte in **26,440 / 27,082 / 25,868** of them (5.04–5.28% of all hits across seeds). A singleton stays put; distinct slots can also agree on their code. Do not interpret S's result as the performance of an independently generated substitute for *every* residual byte. Relaxing the matching rule or restricting the prediction endpoint to hits would ask a different question.

Each FIT used **1,048,320 eligible writes** irrespective of later hit status. All 256 codes were present per seed, none singleton; the largest cells had 18,694 / 17,729 / 30,628 writes. All 256 codes were read on PROBE in both R and S, no fallback reads. Across 16,416,640 hit components per seed, mean reconstruction MSE was D .166604, G .166573, R .096797, S .100105. R recovered the old vector more accurately than S even though R's *all-byte prediction* advantage over S was unresolved. Clipping at reads affected 33,223–45,954 components for D, 10,569–11,723 for the additional global correction, 677,187–749,970 for R, and 677,301–750,197 for S. On PROBE the text note saturated 106 times, neural note zero per checkpoint. See the aggregate for all FIT/write counts and clipping denominators.

## What it cost

R needs **3 bytes per position plus 134,144 installed bytes** per checkpoint (D table 100,352 + axes 1,024 + residual means 32,768). The 128-byte global control and FIT sufficient statistics are separate. At 4,096 positions the explicit totals are **143 KiB R, 106 KiB D, 520 KiB J**. These exclude the common model, input, indices, count tables and workspaces. The text writer peaked at 7,311 contexts / 9,553 context-follower pairs per PROBE window; the sham index peaked at 4,079 positions and 3,723 distinct triples. Neither a smaller process footprint nor a speedup follows from the record sizes.

Mean ms per 4,096-position PROBE window/checkpoint: the extra R write work over D was **3.131 ms**: **1.246** to compute the true vector, **.394** for D and the subtraction, **1.491** to encode the eight signs. This excludes the text/neural notes (1.470/.587), which D also uses. The shared host prefill cost 97.618; address and byte code 43.038, direct head 25.751, current host field 1.584. At reads: .052 to gather codes, .388 to reconstruct D, and .188 to add R's residual and clip (G .144, S .201); .063 to gather the full control vector. S's exact-triple search and read cost 6.863 ms. B/D/G/R/S outer head times were 5.550/5.551/5.507/5.487/5.409; trust responses 8.038/4.793/5.108/4.824/4.834 (B includes both U and J). These are component clocks in one shared-prefill comparison, **not** separately timed serving paths.

FIT mean ms/window: host 99.063, text/neural marks 1.321/3.225, true vector 1.584, coarse write plus subtraction .519, sign encoding 1.534, group statistics .907. Final code means took .449–.589 ms per seed, storage 1.45–1.72 ms and the all-seed seal .050 s. No neural updates or U/J/D refits.

## Evidence and closure

Public qualification `918a7368bb37756c33f2811ede25ba952799c3a0` (local `ed1e5033d798fea6102acfaf58f65ac9cdfb564b`): 153 CPU tests, 654 native frontiers, three strictly loaded weights, their D table hashes and three unambiguous spectral bases **before** reservation. Replayed 43,704 whole-document SHA exclusions; 22,803 eligible full TRAIN documents before taking the next contiguous 256+128. This is not passage deduplication. FIT-only preparation checked 4,065,847 source bytes in 30.68 wall / 30.56 CPU s. Four invented checks/prefills per fitted checkpoint (12 total), separately from natural work. Saved/reloaded all three books, checked all-256 anchors and standalone R, then sealed at independently witnessed unit **768** before opening PROBE. PROBE checked 2,522,742 source bytes. No persistent PROBE text; 1,152 terminal natural seed/document prefills. No raw inputs reopened for closure.

One detached RTX 6000 Ada lease, renewed, **released rc=0 at ≤211 incremental card-s** of the separate 1,200 ceiling. 226.76 s inclusive scheduling wall, 416.08 owner/child CPU s; peak allocated/reserved device bytes 715,992,064 / 746,586,112, peak RSS 2,628,480 KiB. The close audit rehashed the three weight files, verified every code mean and the global control against FIT statistics, reproduced all 11 paired intervals from absolute sums, and independently checked occupancy, fallback counts, MSE denominators, terminal charge and release. Aggregate SHA-256 `5a8f76f6e2f9cb20e8f426b2a86957327d33b9a81d2d9c86229f1ef31af7849d`. Both RB roles are spent; do not reopen them, loosen the donor or tune the bit count to rescue the result.
