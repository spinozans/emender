# X49-Q: a longer matching context did not improve on the latest write reliably

**Keep B's latest compatible neural write.** With the models, literal evidence and readout fixed, replacing its payload by the one with the longest matching left context (E) gave **+.00000163 nats per external byte** relative to latest (J), 95% paired document interval **[−.00002370, +.00002754]**. The gain changed sign across seeds. E's mean head-and-address time was 128.5 ms per document versus 85.8 ms for J. This does not establish equivalence; it gives no reason to pay the extra search cost here. This was a **TRAIN diagnostic**, not a native final or a Lux nomination.

[Prospective protocol](ADDRESS_DEPTH.md) · [implementation and retained fixture stop](QUERY_IMPLEMENTATION.md) · [verified aggregate, including work/stratum sums](receipts/query_result.json) · [receipt audit](query_report.py).

## What changed

One frozen, full-dose **W continued-B** host/head per seed (4901, 4902, 4903). The longest eligible literal suffix, its follower byte, the continuation histogram, and the latest-age and support features stayed fixed. J read the latest write. E read the compatible write maximizing the length of left-context match (at most 32); ties stayed as recent as possible. M moved on exactly the same frontiers as E. Where available it took a different compatible write with J's match length and E's log-age bin; otherwise it took E itself. Each policy read **one** neural value per hit, never several to choose among them. No weights were updated. The 256 scores agreed exactly wherever the corresponding policies had the same address, including misses.

## Result on all real shifted targets

128 fresh, full-length native TRAIN windows × 4,096 targets = **524,288 targets per seed/policy**, not nine independent corpora. Gains below are control NLL minus candidate NLL in **millionths of a nat per byte** (positive favors candidate). Intervals resample the 128 documents jointly across the three fixed seeds, 10,000 draws with RNG 4906. No hit-conditioned filtering.

| Comparison | Gain | Paired 95% interval | Seeds 4901 / 4902 / 4903 |
|---|---:|---:|---:|
| E:J — does extension beat latest? | +1.63 | [−23.70, +27.54] | +6.76 / +8.99 / −10.87 |
| E:M — is a longer match better than the matched substitution? | +15.39 | [−0.70, +31.21] | +18.06 / +17.05 / +11.05 |
| M:J — descriptive substitution effect | −13.76 | [−36.26, +9.34] | −11.30 / −8.06 / −21.92 |

Absolute NLL (nats/real byte) was **1.17203498 J, 1.17203335 E, 1.17204874 M**. E scored better than M in every seed, but the document interval includes zero. The point difference mostly reflects M doing worse than J, not E improving over J. This is not evidence that compatible writes are interchangeable, nor that B's earlier computation is unnecessary: the [B:A comparison](RESULT.md) changed the source of the computation, not the choice between eligible old writes.

## Where policies differed

Each stratum had support in all 128 documents. Contributions below are on the **all-byte denominator**, in millionths of a nat per byte; they add to the primary estimates before rounding. The aggregate also gives the absolute conditional NLL and its contribution for each policy.

| Stratum | Targets per seed (% of all) | E:J contribution | E:M contribution | M:J contribution |
|---|---:|---:|---:|---:|
| Literal miss | 11,399 (2.17%) | 0 | 0 | 0 |
| Hit, latest retained | 471,564 (89.94%) | 0 | 0 | 0 |
| Move, no matched alternative | 20,415 (3.89%) | +0.36 | 0 | +0.36 |
| Move, matched alternative | 20,910 (3.99%) | +1.26 | +15.39 | −14.12 |

E changed **41,325** addresses (7.88% of targets); 50.60% of those moves had a matched alternative. Without an alternative the mean left-context match length grew 6.37→9.57 bytes and the mean age grew 329→957 positions. With one, the length grew 5.38→8.47 and age grew 217→1,176; M's mean age there was 1,191. The matched absolute E/M age difference averaged **144.10** positions (max 1,838; signed M minus E mean +14.49). Binning is not exact age matching and does not randomize all source context. On those matched positions the conditional E:M gain was +.000386 nats/target, not +.000386 on the population. No retuning or expansion on these data.

## Work, cost and closure

- Metadata excluded 41,317 distinct prior document SHAs, leaving 24,467 eligible native TRAIN documents before selecting the first 128 by the registered rank. CPU preparation hash-checked 2,018,445 source bytes, 524,416 window bytes and both shifts (16.68 s wall, 16.66 CPU s). No calibration/final opened; Q's selected documents are spent. Full-document identity checks do not imply passage-level deduplication.
- Each seed/policy made 524,288 literal queries and writes, 512,889 neural reads, and scored all 11,399 misses. E and M each made 2,087,040 additional posting inserts and considered 1,422,859 extension slots per seed; M considered another 45,809 control slots. Those are observed-byte address operations, **not** candidate neural reads. J made no refinement search.
- One actual prefill per seed/document, **384 prefills, 1,152 policy evaluations**. Mean shared prefill 111.5 ms. Mean head + address times per document J/E/M **85.8/128.5/136.7 ms**, of which additional address search was **0/47.0/47.5 ms**. Each policy occurred first, second and third 128 times each. These are costs of this implementation, not native XSA speed or per-token serving latency.
- One local RTX 6000 Ada lease; **≤213 inclusive card-s** (0.0592 GPU-h), 231.67 s inclusive job wall from scheduling ticket, 418.28 owner + waited child CPU s. Device peak 715,992,064 allocated / 746,586,112 reserved bytes; process peak RSS 2,568,472 KiB. Work, CPU and paid device time are not additive costs. PID/starttime, detached ownership, renewal, all 384 actual completed units and release rc=0 were independently checked. No retry or remote job.
- Registered publicly at `9369cd760c653359aa435345d74c3da46d0aed4d`. Local qualified `76f9dbde8565edb04292537d8fc96f8c64c7c70b`, public source/CPU receipt `4d7320aa0d8307a72d945192e208b3c9a361789e` (85 tests; strict CPU loads of all three W-B checkpoints before selection). Closure re-hashed the three frozen files, reproduced the terminal summary, and independently reduced all three joint document bootstraps without text or inference. Aggregate SHA-256 `8d35f000f281659c84b3b7252b09b6f00cbc97d176a6a6cc021c30a1f68ff761`.

**Decision:** no replacement for J and no confirmation run for E. Preserve the distinction between useful bound values, which the earlier panel supported, and address refinement, which did not earn an improvement in Q.
