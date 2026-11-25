# Keeping both notes helped on new short documents, without refitting

**The benefit transferred, but was smaller.** When deciding whether to copy an earlier byte, the frozen two-note reader (J) used both what the model had expected at that earlier occurrence and what local text counts had predicted there. On 381 previously unused short documents, it beat the confirmed neural-note reader (U) by **.000969 nats per real next byte** [paired 95% interval .000686,.001254]. It also beat the neural-only reader whose coefficients were estimated on exactly J's fit population (H), by **.000997** [.000726,.001275]. Both gains were positive at every frozen checkpoint. The fitted rules and model weights were unchanged.

That's a **0.080% reduction in average log loss versus U**, not 0.080 percentage points more top-choice accuracy. It is about half the .002046 gain on the [earlier full-window TRAIN probe](RESULT_LITERAL_SURPRISE.md), a different population; we did not isolate length as the reason. Using the native writer cost about **0.58 extra seconds per million targets**, or **0.62%** on the shared-prefill path estimate. The conclusion is a replicated fixed candidate for these models, not a new default: **U is still the reader with native-final confirmation; B remains the baseline.** No final documents or remote allocation were used.

[Prospective protocol](JOINT_TRANSFER.md) · [implementation](JOINT_TRANSFER_IMPLEMENTATION.md) · [verified aggregate](receipts/joint_transfer_result.json)

## Quality

One new, locked native **calibration** cohort, used for scoring only. From 939 eligible short documents / 1,335,748 targets after 42,939 prior SHA exclusions, the registered prefix first crossed the dose at 381 documents / **525,674 real targets per checkpoint**. 558 calibration documents and all 321 unreserved short final documents were left outside this test. Whole-document SHA disjointness is not proof of passage deduplication. No tuning or extra fitting on these texts. One prefill and one B construction per seed/document. Predictions include all misses and precede target gather.

| Reader | Nats/byte (mean) | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| B: earlier vector and literal evidence | 1.212620 | 1.212188 | 1.211891 | 1.213780 |
| U: B with the original neural-note correction | 1.209690 | 1.209247 | 1.209132 | 1.210691 |
| H: neural note, frozen coefficients matched to J's FIT | 1.209718 | 1.209264 | 1.209158 | 1.210731 |
| J: correctly bound neural and count notes | **1.208720** | **1.208233** | **1.208206** | **1.209722** |

Gain is control minus candidate loss per real shifted byte. Resampled whole documents jointly across all readers and the three fixed checkpoints (10,000 draws, RNG 4912); each draw has its own real-target denominator.

| Candidate:control | Gain | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| **J:U** | **+.000969** | [.000686,.001254] | .001014 / .000926 / .000969 |
| **J:H** | **+.000997** | [.000726,.001275] | .001032 / .000952 / .001009 |
| J:B | +.003899 | [.003555,.004257] | .003956 / .003685 / .004057 |
| U:B | +.002930 | [.002672,.003194] | .002942 / .002759 / .003089 |
| H:U | −.000028 | [−.000073,.000017] | −.000018 / −.000026 / −.000040 |

Both predeclared checks passed. H:U did not establish an improvement by simply estimating the neural-only coefficients again. JT did not shuffle records; evidence about **correct occurrence binding** comes from LS, not a second intervention on these texts.

## What the second note cost

Shared = measured causal prefill, B scores/addresses and correctly bound neural fields. From exactly that packet we timed the complete U response or the native text-table construction, read and J response. We alternated which branch ran first; H was outside both clocks. Writer is **within** J time.

| Checkpoint | Shared (s) | U branch (s) | J branch (s) | Of J: writer (s) | Extra s / million targets | `(shared+J)/(shared+U)` |
|---|---:|---:|---:|---:|---:|---:|
| 4901 | 50.087 | .512 | .827 | .253 | .600 | 1.00624 |
| 4902 | 47.615 | .480 | .786 | .251 | .583 | 1.00637 |
| 4903 | 48.058 | .490 | .787 | .246 | .565 | 1.00612 |

Across these documents the native writer averaged **0.656 ms per seed/document**. J's branch itself took longer than U's; the percentage is small because the host and B work dominate. The ratio is *not* an independently run end-to-end throughput or per-token decode benchmark. No comparison here measures how many more training steps a different model could take with that time.

Per checkpoint there were 498,316 literal hits, 27,358 misses, and 206,567 correct copies; all 381 documents had each. The text writer made 525,674 order-zero and 2,096,981 context updates (and as many context lookups), constructed 916,042 contexts / 1,205,058 context-follower entries **summed over fresh windows**, and peaked at 6,531 / 8,571 entries in one window. 27 text codes saturated in 16 documents; no neural arrival mark saturated. These are logical table entries, not allocator bytes.

| Real target length | Documents | Targets | Hits | Misses |
|---|---:|---:|---:|---:|
| 1–15 | 0 | 0 | 0 | 0 |
| 16–127 | 4 | 301 | 139 | 162 |
| 128–1,023 | 153 | 88,740 | 78,936 | 9,804 |
| 1,024–4,095 | 224 | 436,633 | 419,241 | 17,392 |

## Receipts and limits

Public qualification `51e6fbab917098ca1b3a2044641424a26c24cfd9` (local `8285d5e87d1390358c7d77c8e1a70714c26d66be`): 132 CPU tests, native address parity, retained writer and three strict W-B loads. Metadata-only reservation 28.98 wall / 28.98 CPU s. Device shape/parity on invented 1/16/64/65/4,095-target cases passed before the one raw open; then 526,055 whole source bytes with both shifts checked. 1,143 seed/document units, zero fits and zero neural updates. One RTX 6000 Ada lease released rc=0 at **≤176 incremental card-s** of the 1,200 ceiling, 191.93 s inclusive scheduling wall and 293.67 owner/child CPU s. Peak allocated/reserved device bytes 715,972,608 / 754,974,720; peak process RSS 2,747,820 KiB. Earlier fitting/qualification costs are separate, not free and not charged to this lease again.

The CPU close audit rehashed the actual three weight files, verified the fixed reader, document/shift/journal identities, renewed detached owner, advancing target counts and positive release. It reproduced all five intervals from absolute losses and sample multiplicities, and both path estimates from per-document clocks, without reopening text. Aggregate SHA-256 `62db2e0beb85bee456b1fea2005d98b4f5684f6d41f493841ac3fc499b9a24d6`. Do not reopen this cohort, infer a scale nomination from the calibration result, or promote J on a claim about some other checkpoint.
