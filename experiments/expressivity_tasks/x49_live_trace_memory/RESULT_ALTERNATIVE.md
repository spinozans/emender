# X49-V: the old alternative often named a plausible byte, but did not earn another memory slot

**Keep the confirmed U predictor. Do not run a V confirmation.** U remembers how surprising an earlier follower was and uses that to decide how much to trust a copy. V remembered a second byte: the strongest alternative in that old forecast. On fresh out-of-fit documents it beat D, which calibrates only today's strongest alternative, by **.00002725 nats per real byte**. It did **not** establish an advantage over an equal-size readout using the wrong occurrence, or over leaving U alone. Both of those document-paired intervals crossed zero; one seed lost to U. This is an inconclusive incremental gain, not proof that the alternatives are equivalent.

[Prospective design](FORECAST_ALTERNATIVE.md) · [implementation and U-only adapter](ALTERNATIVE_IMPLEMENTATION.md) · [aggregate, coefficients and work](receipts/alternative_result.json) · [closed-receipt audit](alternative_report.py).

## Comparison at fixed copy probability

For three frozen W-B checkpoints with their exact confirmed U coefficients, fit D/V/K separately on 256 native TRAIN documents (1,048,576 real targets per seed). Seal all nine readouts before opening 128 disjoint native TRAIN PROBE documents (524,288 real targets per seed). No neural updates or U refits. At a hit, D can recalibrate the current best non-copy. V also sees the old forecast's best byte other than the observed follower; K gets that byte from a rotated occurrence with the same observed follower and age bin. K fits with its own intervention. All models use the **correct U surprise and B vector**; all leave U's exact score for copying fixed and redistribute only the mass among non-copies. On misses, all give U's answer. This is not a native final or a comparison to historical A/B on these documents.

| Predictor | PROBE loss (nats/byte) |
|---|---:|
| U: confirmed incumbent | 1.120784263 |
| D: current alternative only | 1.120791838 |
| V: current + correctly bound old alternative | 1.120764584 |
| K: current + matched wrong-occurrence alternative | 1.120778633 |

To make the small contrasts legible, **the following table uses millionths of a nat per real byte**, not nats per hit. Positive = control loss minus candidate loss. Percentile 95% intervals use 10,000 joint draws of whole documents across methods and fixed seeds (RNG 4909).

| Candidate : control | Gain [paired 95% CI], ×10⁻⁶ nats/byte | By seed 4901 / 4902 / 4903 |
|---|---:|---:|
| V:D — incremental historical proposal | **+27.253 [+3.898, +52.357]** | +6.863 / +35.208 / +39.690 |
| V:K — correct binding vs displaced binding | +14.049 [−7.254, +35.636] | +6.860 / +20.537 / +14.749 |
| V:U — drop-in gain | +19.679 [−8.710, +51.034] | −4.590 / +33.066 / +30.561 |
| D:U — calibrating today's alternative | −7.575 [−27.579, +12.126] | −11.453 / −2.142 / −9.129 |
| K:D — displaced proposal vs current only | +13.205 [+0.403, +26.233] | +0.003 / +14.671 / +24.940 |

**Why proposal accuracy isn't the result.** There were 512,715 hits and 11,573 misses per seed. On 236,372 hits the copy was correct, leaving 276,343 non-copy outcomes on which any change could help. On those, V's historical proposal named the target 63,748 / 63,660 / 63,670 times (about 23%); K's did so 32,332 / 32,187 / 32,243 times (about 12%). The current alternative alone was right 151,660 / 151,877 / 151,736 times (about 55%). These are proposal identities, **not** the predictors' top-1 decisions or their all-byte loss. A plausible old guess is not necessarily new information beyond the current probability distribution. The experiment tested whether the fixed readout could exploit it; it did not test every way of summarizing an earlier forecast.

K displaced 436,870 slots per seed; the extra bytes actually differed on 361,294 / 359,574 / 359,659 hits. V agreed with the current proposal on 247,178 / 250,390 / 248,775 hits and K on 92,877 / 95,278 / 94,498. All 128 documents had misses, non-copies, changed alternatives and some correct proposals of each kind. K's gain over D was small but positive; a wrong occurrence does not supply zero information. The primary V:D contrast alone therefore doesn't demonstrate occurrence-specific value. All five contrasts include every target, and the registered decision required V:D, V:K **and** V:U.

## Operations and disposition

- 42,171 earlier full-document SHAs were excluded, including both U roles and the U short confirmation. From 23,955 eligible native TRAIN documents the campaign took the next contiguous 256 + 128 full windows. CPU preparation read only FIT: 4,594,209 source bytes, 1,048,832 window bytes, 23.05 s wall / 22.96 CPU s. Under lease, all fits froze at the witnessed 768-unit boundary before PROBE read 2,006,126 source bytes. All shifts within one document, no passage-deduplication claim. No final reserved or opened by V.
- Extra alternative tape: one uint8 per observed input over U, or 4,096 bytes per full window; first slot unused. PROBE encoded 524,160 alternatives and read 512,715 bound alternatives per seed. Tape size is not total activation memory. Mean PROBE ms/window: host prefill 102.96, B head 82.14, U current fields 3.67, surprise construction/read 0.51/0.067, U response 3.19; current alternative 1.78, extra historical tape build 1.68, V read + feature 0.27, K rotation + read/feature 9.45; D/V/K score adjustments 3.61/3.77/3.75. FIT moment accumulation for all three readouts averaged 1.41 ms/window. These are batched diagnostic components, not serving latency.
- One RTX 6000 Ada lease, released rc=0 at **≤283 incremental card-seconds (4 min 43 s)**, 301.18 s inclusive scheduling wall and 564.12 owner plus waited-child CPU s. Peak device allocated/reserved 715,992,064 / 746,586,112 bytes, process peak RSS 2,604,024 KiB. Detached owner, renewed heartbeat and advancing real work witnessed through 1,152 terminal units. The separate earlier U campaigns cost ≤287 card-s for fitting/probing and ≤185 for its native confirmation; neither was rerun or charged to V.
- Qualified local source `8d63d3783070dacf522109ec6c5b33c56b9dc130`, public pre-reservation source and CPU evidence `379b645677b0002b4dda528e260df594f1c8e292`. Pre-PROBE fitted map `19719b6cce0ce38627436bae8654e08dbd035a522664f8c2326efe96e8eaea6f`. The audit rehashed the three actual weight files, reproduced all nine FIT solves by Cholesky, all five paired reductions from absolute policy losses by document multiplicity, and terminal charge/release. Aggregate SHA-256 `80770a965e37f9f70c8af1be107319c13108c345c9f2d1fbd1293a040e2da3c6`.

**V closes here.** No second sample, reopened window, V adoption or scale nomination. The separate held-out result for U stands.
