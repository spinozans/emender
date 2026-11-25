# Remembering the old model forecast helped even when we remembered the old text statistics

**Yes, on this diagnostic.** When a piece of text repeated, predicting what came next worked best when the model could recall two facts about the earlier continuation: how unexpected it was to the model, and how unexpected it was according to counts in that document. Adding the model's note to the text note improved average log loss by **.003674 nats per real byte**. Keeping the text note but retrieving the model's note from the wrong occurrence lost **.003236**. These comparisons used the same addresses, bytes and neural vectors. The advantage was small, agreed across three frozen checkpoints, and was measured on *all* next bytes, not just successful copies.

[Protocol](LITERAL_SURPRISE.md) · [implementation, including failed fixture](LITERAL_SURPRISE_IMPLEMENTATION.md) · [compact verified result](receipts/literal_surprise_result.json). **Decision:** the correctly bound neural note contains information this particular count-based note does not. The converse addition also helped: both notes beat the previously confirmed neural-only reader U by .002046 [.001491,.002755]. This merits a **separate no-refit confirmation**. U remains the local recommendation for its matched frozen models; no LS final was reserved or opened, no new default and no Lux nomination.

## What was compared

At each hit B retrieves an earlier follower, e.g. the `b` that followed a previous `color: `. The reader has to decide how much this supports predicting `b` *now*. N uses only current confidence to adjust B's copy odds. T adds the earlier follower's one-byte surprise under local text counts; H adds its one-byte surprise under the earlier neural forecast. J adds both correctly bound notes. S keeps the correct text note but uses a neural note from a different eligible occurrence of the same byte in the same age bin when possible. U uses the same features as H with its **original**, separately confirmed coefficients. All six have the same frozen host and B reader. Only copy odds can change; a miss is exactly B.

We fitted five small linear corrections per checkpoint, one fixed ridge-regularized Newton step at B, on 256 full native TRAIN windows (1,048,576 shifted targets per seed). All fifteen fits and the three original U identities were frozen and witnessed before the separate 128-document TRAIN probe opened. No neural weight changed. The new roles came from the next contiguous block of unique native TRAIN documents after 42,555 exclusions. No previously spent text was reopened. Document-SHA exclusion is not passage deduplication.

## Predictions on 524,288 disjoint probe bytes per checkpoint

Nats per byte, lower is better. Repeated evaluations of the same documents with three fixed weights are **not** independent document samples.

| Earlier note(s) available | Mean | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| N: neither | 1.144422 | 1.144342 | 1.143931 | 1.144992 |
| T: literal | 1.143249 | 1.143211 | 1.142739 | 1.143796 |
| H: neural, matched fit | 1.141697 | 1.141641 | 1.141391 | 1.142060 |
| J: literal + correct neural | **1.139575** | 1.139587 | 1.139271 | 1.139867 |
| S: literal + wrong-occurrence neural | 1.142811 | 1.142867 | 1.142307 | 1.143258 |
| U: confirmed neural-only reader, unchanged | 1.141622 | 1.141597 | 1.141347 | 1.141921 |

Gain = control loss minus candidate loss. Joint whole-document percentile intervals, 10,000 draws, RNG 4910. The first two rows were the joint primary test.

| Candidate:control | Gain | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| **J:T** | **+.003674** | [.003009,.004570] | .003623 / .003468 / .003929 |
| **J:S** | **+.003236** | [.002734,.003861] | .003280 / .003036 / .003391 |
| H:T | +.001552 | [.001061,.002034] | .001569 / .001349 / .001737 |
| T:N | +.001173 | [.000751,.001729] | .001131 / .001191 / .001196 |
| H:N | +.002724 | [.002211,.003381] | .002701 / .002540 / .002933 |
| S:T | +.000438 | [.000235,.000749] | .000343 / .000432 / .000538 |
| J:H | +.002122 | [.001528,.002934] | .002054 / .002120 / .002192 |
| T:U | −.001627 | [−.002180,−.001070] | −.001614 / −.001392 / −.001875 |
| J:U | +.002046 | [.001491,.002755] | .002009 / .002076 / .002054 |

The equally sized, equally fitted neural note beat the literal one. Both alone beat current confidence, and even the displaced neural note helped a little after the literal note. The larger **correct-minus-displaced** gain, with the literal note held fixed, is the binding evidence. Descriptively, the fitted J rule at all three seeds gave *opposite signs* to the two old surprises: at fixed current confidence and fixed other note, larger neural surprise raised copy odds, while larger literal surprise lowered them. That describes these readouts, not a universal rule that unexpected bytes recur.

## Coverage and cost

- Per checkpoint on the probe: 512,838 hits, 11,450 misses, 245,549 correctly copied targets. All 128 documents had hits, misses, correct copies, displaced slots and unequal codes. 435,406 slots displaced; the neural codes actually differed on 404,911 / 405,446 / 405,071 reads. Correct literal and neural codes differed on 487,999 / 488,277 / 488,224 reads. Hit-weighted Pearson correlations .4008 / .3974 / .3994. These are descriptive, not the effect size.
- Both writers stored one byte per input. Across the 128 windows per checkpoint the text writer made 524,288 order-zero updates and 2,095,232 context lookups plus the same number of context updates. It constructed 744,611 distinct context entries and 982,174 context/follower entries **summed across windows**; the peaks in one window were 7,619 and 10,001 respectively. The text mark saturated on 78 writes in 25 documents and 828 reads; no neural arrival mark saturated. FIT work and support are in the aggregate.
- Incremental mark construction per 4,096-target probe window averaged **21.50 ms for literal counts versus 0.52 ms for neural surprise from already available host outputs**. They are not comparable standalone language models. Shared host prefill was 100.63 ms and B construction 84.47 ms; J response 4.00 ms, including neither writer. These are measured batch components, **not** streaming decode latency, an isolated J-vs-U throughput benchmark, or a claim that a two-byte record is free.
- FIT-only CPU preparation opened 4,616,727 source bytes (24.74 wall / 24.65 CPU s), no PROBE text. After the all-seed seal at witnessed unit 768, the probe reader checked 3,631,004 source bytes. One shared prefill per seed/document, 1,152 terminal units. One RTX 6000 Ada lease, positively released rc=0, **≤300 incremental card-s** of the 1,200 cap, 315.47 s inclusive scheduling wall, 592.52 owner/child CPU s. Peak device allocated/reserved 715,992,064 / 746,586,112 bytes; process RSS 2,598,856 KiB. No retry or other acquisition.

Public source qualification `5bcfc13b2e28a4a9f383ff2a4c1dd199dd254ca7` (local `55d6f6ed75cb8f07e111cf628f419683b236ac3b`, 120 CPU tests, native literal parity and three strict real loads). Private FIT seal `443f3161f2dbfe075e6ece00329e6c2dc0ebedb19693100cf40292a672e47847`. From closed receipts the CPU audit rehashed three actual weight files, re-solved fifteen FIT systems by Cholesky, reproduced all nine paired intervals by independent multiplicity reduction, checked correlations from centered document moments and verified the charge and release. Aggregate SHA-256 `4e2789566883cbaf11131919bfa72781f0ac0c0d74a0c58289caea2728fe99a2`. No private memberships, raw windows, per-target predictions or weights in the release.
