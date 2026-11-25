# When the same guess recurred, remembering its correction helped

**Which alternative mattered depended on what the model had guessed.** On 128 new documents, looking up what actually followed *this* wrong guess at earlier matches (PAIR) lowered next-byte loss by **.000400 nats/byte** over knowing the same unpaired counts (MARG), paired 95% interval **[.000314,.000489]**. MARG already knew how often the present guess failed at that key, how often it was right and which alternatives had been missed. It improved on the frozen ERROR reader by **.000330** [.000217,.000447]. PAIR's total gain over ERROR was **.000729** [.000581,.000886]. All three fixed checkpoints agreed, counting **every real next byte**.

**The link between guess and outcome mattered.** We exchanged outcomes between eligible pairs of old mistakes so that the counts of each guessed byte, counts of each actual correction and number of records available to the present guess were exactly unchanged. At PAIR's fixed coefficients this cost **.000494 nats/byte** [.000409,.000583]. A reader fitted on exchanged pairs still lost to PAIR by **.000391** [.000307,.000479]. Its own small gain over MARG was only .000008. That distinguishes a useful association from simply recognizing a frequent mistake. It doesn't prove that all the value is specific to this document rather than, for example, a recurring model-wide confusion.

**Tradeoff and decision.** MARG added **1.31 seconds per million real targets** beyond the shared ERROR path; PAIR added **5.78**. That's about **7.6% more time for PAIR than MARG**, or 10.0% more than ERROR, in this implementation. Both registered decisions passed: the cheap marginal adjustment helped, and the paired table earned a separately registered, no-refit transfer test. This was a disjoint **native TRAIN probe**, not a calibration or final confirmation. Neither rule becomes the default. B remains the baseline and U the native-final-confirmed reader; no Lux nomination.

[Prospective comparison](PAIRED_CORRECTION.md) · [implementation](PAIRED_CORRECTION_IMPLEMENTATION.md) · [pre-access CPU qualification](receipts/paired_correction_cpu_qualification.json) · [public aggregate](receipts/paired_correction_result.json)

## Comparison on every byte

For example, when the model predicts a space, MARG knows that space was previously a mistake at this literal pattern and that newline was a common missed alternative. PAIR additionally knows whether **newline actually followed the mistaken space prediction**, rather than a different wrong guess. This is an illustration, not a passage taken from the corpus. The query is the bare host's present top prediction, before the next byte arrives. The older guesses were likewise made before their followers arrived; only observed mistakes can enter the table. The latest eligible occurrence stays with J. We used the same weights, addresses, earlier scalar/vector features and thresholds as in [ET](RESULT_ERROR_TRANSFER.md); no document state is carried over.

ERROR is the frozen history reader; it keeps separate counts of wrong guesses and missed actual followers. MARG uses those counts to move score toward alternatives when this guess has failed before. PAIR adds a contrast between the alternatives after this particular wrong guess and the unpaired alternatives. Both PAIR coefficients were positive at every weight: the fitted direction favored the actual past corrections relative to the unpaired distribution. SHUF uses its own fit to exchanged pairs, SWAP uses the unchanged PAIR fit on the exchanged input. A successful exchange changes two edges, never creates a correct-guess edge, and never changes earlier predictions retroactively. Neither PAIR nor MARG needs the diagnostic control table to predict.

Loss in nats per real shifted byte; lower is better. All fits sealed across all seeds on **256 FIT documents** before the **128 PROBE documents** were opened. There were **524,288 targets per checkpoint** on PROBE, not three independent samples of documents.

| Reader | Mean NLL | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| B | 1.129747622 | 1.129588319 | 1.129829928 | 1.129824618 |
| U | 1.126885042 | 1.126740403 | 1.127036051 | 1.126878672 |
| J | 1.125243860 | 1.125091686 | 1.125334498 | 1.125305395 |
| HISTORY | 1.123101636 | 1.122990104 | 1.123092549 | 1.123222255 |
| ERROR | 1.122928894 | 1.122855616 | 1.122895001 | 1.123036067 |
| MARG | 1.122599010 | 1.122516839 | 1.122544687 | 1.122735505 |
| PAIR | 1.122199402 | 1.122138966 | 1.122151634 | 1.122307605 |
| SHUF | 1.122590659 | 1.122510813 | 1.122536745 | 1.122724420 |
| SWAP | 1.122693307 | 1.122621340 | 1.122630685 | 1.122827896 |

Positive gain = control minus candidate loss. 10,000 jointly paired whole-document draws, RNG 4917, draw-specific target count, linear percentiles, average over the fixed weights. The first four gains all had to be positive at every checkpoint with lower bound above zero for the PAIR decision. The fifth decided the MARG question independently.

| Candidate:control | Gain | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| PAIR:MARG | +0.000399609 | [+0.000313915,+0.000488595] | +0.000377873 / +0.000393053 / +0.000427900 |
| PAIR:SHUF | +0.000391258 | [+0.000307471,+0.000479099] | +0.000371847 / +0.000385111 / +0.000416815 |
| PAIR:SWAP | +0.000493905 | [+0.000409113,+0.000582687] | +0.000482374 / +0.000479051 / +0.000520291 |
| PAIR:ERROR | +0.000729493 | [+0.000580812,+0.000885716] | +0.000716650 / +0.000743367 / +0.000728462 |
| MARG:ERROR | +0.000329884 | [+0.000216944,+0.000447243] | +0.000338776 / +0.000350314 / +0.000300562 |
| SHUF:MARG | +0.000008351 | [+0.000002932,+0.000013962] | +0.000006026 / +0.000007942 / +0.000011085 |
| PAIR:J | +0.003044458 | [+0.002527871,+0.003589046] | +0.002952720 / +0.003182864 / +0.002997790 |
| ERROR:HISTORY | +0.000172742 | [+0.000107245,+0.000239846] | +0.000134489 / +0.000197548 / +0.000186188 |
| HISTORY:J | +0.002142224 | [+0.001767936,+0.002545803] | +0.002101581 / +0.002241950 / +0.002083140 |
| J:U | +0.001641182 | [+0.001254111,+0.002005286] | +0.001648718 / +0.001701553 / +0.001573277 |
| U:B | +0.002862580 | [+0.002527701,+0.003207652] | +0.002847916 / +0.002793877 / +0.002945946 |

## What changed at prediction time?

Per checkpoint: 512,861 latest hits, 243,136 correct latest copies, 11,427 misses and 143,711 singleton hits. 369,150 predictions had older history (16,126,299 older occurrence uses). A literal hit alone does not mean a usable correction. **About 32% of targets** found a past mistake that matched both the key and the current guess. Outside those rows all four PC rules returned ERROR unchanged. On about 82,000 rows per seed, the distribution of actual alternatives to that guess differed from the unpaired distribution; PAIR and MARG also have separately fitted first coefficients, so that is a feature-support count, not the number of different predictions. SWAP *holds both coefficients fixed*.

| PROBE support | 4901 | 4902 | 4903 |
|---|---:|---:|---:|
| Earlier guesses written | 524,160 | 524,160 | 524,160 |
| Mistaken at arrival | 214,128 | 214,397 | 215,122 |
| Correct at arrival | 310,032 | 309,763 | 309,038 |
| Predictions with a matching mistake | 169,563 | 169,909 | 169,654 |
| Matching older mistake uses | 992,141 | 998,648 | 970,973 |
| True conditional law differed from unpaired | 82,151 | 82,229 | 82,210 |
| Committed mistake records across keys | 542,902 | 543,496 | 545,798 |
| Adjacent mistake pairs considered across keys | 236,348 | 236,595 | 237,733 |
| Legal two-edge exchanges | 87,157 | 86,121 | 87,058 |
| Queries with a changed conditional histogram | 59,815 | 60,007 | 59,551 |
| Changed conditional uses (sum of positive bin differences) | 136,380 | 136,124 | 137,343 |
| Documents with a changed conditional histogram | 128 | 128 | 127 |

A single mistake can enter more than one literal key and can be used in later predictions repeatedly. Swapping two old edges is not the same as changing a queried histogram. Here exchanges changed it on **11.36–11.45% of all targets** (35.10–35.32% of those with a matching mistake). All unchanged queries remain in the loss and intervals.

## Price of the read

The common ERROR path includes one prefill, J, guess/query tapes and the true ERROR fields and response. MARG adds its marginal assembly and full response. PAIR adds its own sparse pair-table construction/query and contrast assembly and pays for its full response instead. These are component-based path estimates; diagnostic control construction and scoring are not required by either single reader.

| Weight | Shared ERROR s | MARG extra s/M real targets | PAIR extra s/M | PAIR path / MARG path | PAIR path / ERROR |
|---|---:|---:|---:|---:|---:|
| 4901 | 27.510 | 1.196 | 4.707 | 1.0654 | 1.0897 |
| 4902 | 31.068 | 1.358 | 6.575 | 1.0861 | 1.1110 |
| 4903 | 32.000 | 1.379 | 6.071 | 1.0752 | 1.0995 |
| Pooled | 90.578 | 1.311 | 5.785 | 1.0760 | 1.1004 |

On PROBE, mean per 4,096-target document/checkpoint: host prefill **107.63 ms**, complete post-host ERROR **128.25 ms**, marginal fields **1.99 ms**, true pair index **18.29 ms**, true pair features **0.52 ms**, MARG / PAIR responses **3.37 / 2.89 ms**. Comparison-only work included other base answers (8.08 ms), switched index (20.05 ms), control features and support (0.56 ms), SHUF / SWAP responses (2.91 / 2.89 ms). FIT statistics cost 0.80 ms per window in addition to building the features. Don't add the nested common-path clocks twice. These prices are not standalone throughput, a comparison against extra host training, or proof the Python indexes are necessary in this form.

The true index peaked at **7,311 pending latest-key positions, 2,223 key/guessed-byte rows and 3,466 distinct key/guess/correction cells** in a document (individual maxima across checkpoints). It returned 439,145 / 443,515 / 433,148 nonzero correction bins summed over PROBE queries. Packed true query snapshots peaked at 104,852 bytes; marginal / full paired FP64 feature arrays at 145,688 / 291,376 bytes. Guess and query tapes were 4,096 bytes each. These are not total allocator bytes or minimal storage. The diagnostic switched index peaked at 3,588 pair cells and up to 1,349 keys with an unmatched provisional mistake; no such table on a PAIR-only read. The [aggregate](receipts/paired_correction_result.json) includes work, timing, live state, snapshot and feature maxima by role and weight.

## Access and closure

The [public qualification](receipts/paired_correction_cpu_qualification.json) preceded natural reservation: public `0e6450c1c54669369f84b6328dcc28136ff5761d`, local `b8619f5829fa4a73cb93dd669269117b59ccda2f`. 192 CPU tests, 654 native address frontiers, all three strict CPU checkpoint loads and invented full-window pricing. Metadata replay excluded 44,833 document SHAs including all of ET, leaving 22,035 eligible full native TRAIN documents. The next contiguous 256 FIT and 128 PROBE were reserved, without alternative offsets. Whole-document exclusion does not prove passage deduplication. Preparation opened only FIT (4,019,232 source bytes, 36.31 wall / 36.21 CPU s). All nine readouts and FIT journals were sealed at independently witnessed unit **768** before PROBE opened; the latter verified 3,064,480 source bytes, with no persistent PROBE window file. Exactly 1,152 natural seed/document prefills across both roles. No neural updates, refits to the four earlier readers or access to native calibration/final text.

One separately admitted local RTX 6000 Ada lease released **rc=0 at ≤349 card-s** of its 1,200 ceiling; inclusive scheduling wall 364.09 s, 667.45 owner/child CPU s and 16.45 driver/short-parent CPU s. Peak device allocated / reserved 715,992,064 / 746,586,112 bytes, worker RSS 2,604,236 KiB. The independent CPU close rehashed the three weight files, checked FIT-only solutions against Cholesky, recomputed absolute losses and eleven paired intervals from sample multiplicities, replayed support/peaks/branch costs and checked a renewed detached owner, advancing work and the positive release. **Both PC roles are spent.** Public aggregate SHA-256 `80609171412a6c5682b9b57469a7750d5cbd432918994eac26a51fcfca60631d`, private close `91ef90da942bad7f68e59fa26895a43e553e15dac35e23e3c7872588950ba71e`. Fitted arrays, membership, guesses, associations and journals remain private.
