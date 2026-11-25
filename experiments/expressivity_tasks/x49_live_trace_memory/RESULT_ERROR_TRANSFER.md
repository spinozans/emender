# Remembering past mistakes helped on shorter documents, without refitting

**Both effects transferred.** When a byte pattern recurred, looking beyond its latest match to the older outcomes, their surprise notes and the model's miss counts (HISTORY) lowered next-byte loss by **.001731 nats/byte** over the latest-match reader J. Recording **which wrong byte the model guessed** (ERROR) lowered it a further **.000139** [paired 95% interval .000092,.000187]. The weights and readouts were frozen before these 361 new native calibration documents were opened. All three checkpoints improved, scoring every real byte, not just predictions where a history was found.

**How much depended on the guess being the right one?** Replace the guess with one from a matched earlier mistake where possible. Keeping ERROR's coefficients fixed, that substitution cost **.00001855** [.00000850,.00002856] nats/byte. Even a substitute reader with coefficients previously fitted for its own inputs lost by **.00001277** [.00000231,.00002322]. The substitute itself beat HISTORY. The evidence for *which guess belonged to this record* is real on this test, but smaller than the full ERROR:HISTORY difference. HISTORY alone supplied **92.6% of ERROR's loss reduction over J** here (94.4% on the [full-window TRAIN probe](RESULT_ERROR_HISTORY.md)). That is a comparison of predictions, not a percentage of information stored. Don't subtract absolute losses across the two populations to infer an effect of document length.

**Cost:** either history added about **9%** to the measured shared-J path. The incremental time difference between keeping and discarding the wrong-byte identity was small and **changed sign across checkpoints**. ERROR used more table cells. There is no demonstrated speed advantage or timing equivalence; this is one shared-prefill inference panel, not standalone serving throughput. These are now calibration-replicated readers for the three tested weights. **B remains the baseline, U the native-final-confirmed reader.** No default change, final access, Lux nomination or remote work.

[Prospective decisions](ERROR_TRANSFER.md) · [implementation](ERROR_TRANSFER_IMPLEMENTATION.md) · [public CPU qualification](receipts/error_transfer_cpu_qualification.json) · [redacted result aggregate](receipts/error_transfer_result.json)

## Prediction on the locked cohort

Metadata replay excluded 44,472 prior document hashes, including the whole JT prefix and both EH roles. From the remaining 558 eligible short native calibration documents (810,074 targets), the next contiguous block in JT order first crossed the dose at **361 documents, 524,554 real shifted targets per checkpoint**. The crossing document was kept whole. 197 eligible short calibration documents were not used. Exclusion by whole-document SHA is not passage deduplication. All eight readers saw the same window-local evidence; no state crossed a boundary. There was one host prefill per document/checkpoint, no fitting or weight update, and labels were gathered only after constructing all eight predictions.

Nats per real next byte; lower is better. B recalls the latest match's vector; U adds its neural surprise note; J also uses its literal count surprise. COUNT uses older follower counts at the selected key; HISTORY adds their true note sums and miss counts. ERROR also counts which bytes were wrongly guessed. Only the guessed byte is substituted in the control: the donor is the most recent strictly earlier mistake with the same observed follower and exact neural surprise code, otherwise the original guess. SHIFT uses its own prior fit; SWAP uses ERROR's fit. No donor work on either chosen path.

| Reader | Mean NLL | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| B | 1.172555883 | 1.171621551 | 1.172324863 | 1.173721235 |
| U | 1.169365256 | 1.168419170 | 1.169275998 | 1.170400601 |
| J | 1.168545622 | 1.167577442 | 1.168452129 | 1.169607296 |
| COUNT | 1.168089626 | 1.167085794 | 1.168001006 | 1.169182079 |
| HISTORY | 1.166814267 | 1.165860060 | 1.166726052 | 1.167856690 |
| ERROR | 1.166675626 | 1.165768560 | 1.166556342 | 1.167701975 |
| SHIFT | 1.166688393 | 1.165786924 | 1.166570216 | 1.167708037 |
| SWAP | 1.166694179 | 1.165791795 | 1.166575140 | 1.167715602 |

Gain = control minus candidate loss. Resampled whole documents 10,000 times, jointly across readers and weights, RNG 4916, with the draw's actual target count and linear percentile endpoints. Replication required positive gains at **every** fixed checkpoint and a paired lower bound above zero on **each** required contrast. The first four rows decide wrong-guess replication; the next two decide scalar-history replication. Both passed.

| Candidate:control | Gain | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| ERROR:HISTORY | +0.000138642 | [+0.000092143,+0.000186825] | +0.000091500 / +0.000169709 / +0.000154715 |
| ERROR:SHIFT | +0.000012767 | [+0.000002309,+0.000023223] | +0.000018365 / +0.000013874 / +0.000006062 |
| ERROR:SWAP | +0.000018554 | [+0.000008497,+0.000028564] | +0.000023236 / +0.000018798 / +0.000013627 |
| ERROR:J | +0.001869997 | [+0.001610759,+0.002135263] | +0.001808882 / +0.001895786 / +0.001905322 |
| HISTORY:COUNT | +0.001275359 | [+0.001092993,+0.001457822] | +0.001225734 / +0.001274954 / +0.001325389 |
| HISTORY:J | +0.001731355 | [+0.001484789,+0.001984156] | +0.001717381 / +0.001726077 / +0.001750606 |
| COUNT:J | +0.000455996 | [+0.000340981,+0.000572103] | +0.000491648 / +0.000451123 / +0.000425217 |
| SHIFT:HISTORY | +0.000125875 | [+0.000084464,+0.000168517] | +0.000073136 / +0.000155836 / +0.000148653 |
| ERROR:U | +0.002689631 | [+0.002271690,+0.003133125] | +0.002650610 / +0.002719655 / +0.002698627 |
| J:U | +0.000819634 | [+0.000571743,+0.001080146] | +0.000841728 / +0.000823869 / +0.000793305 |
| U:B | +0.003190627 | [+0.002942919,+0.003445015] | +0.003202381 / +0.003048866 / +0.003320634 |

## Which histories did the readers actually use?

There were 498,573 literal hits and 212,413 correct latest copies per checkpoint. On **25,981 misses and 159,892 singleton hits**, both histories returned J unchanged. On the other **338,681 predictions** there were **4,016,252 older occurrence uses** (11.86 per active query). A write can inform more than one later prediction. The control changed the *summed histogram* at **14.13–14.33% of all targets** (21.88–22.20% of active queries), fewer than the 22.84–23.15% of all targets on EH. Matching can leave an identical guess; even different guesses can cancel in a sum. The primary endpoint includes all unchanged rows.

| Intervention support | 4901 | 4902 | 4903 |
|---|---:|---:|---:|
| Guesses written | 524,193 | 524,193 | 524,193 |
| Mistakes at arrival | 214,488 | 214,851 | 215,546 |
| Mistake arrivals with a donor | 55,236 | 55,798 | 56,819 |
| Arrivals with a different guessed byte | 37,952 | 38,297 | 39,015 |
| Older mistake uses | 1,600,616 | 1,608,244 | 1,608,789 |
| Uses with a matched donor | 301,697 | 308,382 | 311,236 |
| Uses with a different guess | 195,511 | 198,092 | 203,771 |
| Queries with a changed histogram | 74,106 | 74,898 | 75,184 |
| Documents with any changed histogram | 358 | 355 | 352 |

| Real target length | Documents | Targets | Hits | Misses |
|---|---:|---:|---:|---:|
| 1–15 | 0 | 0 | 0 | 0 |
| 16–127 | 3 | 337 | 204 | 133 |
| 128–1,023 | 147 | 90,231 | 81,173 | 9,058 |
| 1,024–4,095 | 211 | 433,986 | 417,196 | 16,790 |

## What the two alternatives cost

Shared J includes the full causal host, B, both notes and J response. Starting from that common packet, each branch separately paid for bare-host top-1 preparation, its own index, features and full response. HISTORY converted guesses to miss flags, without indexing the wrong identities. Runs alternated which branch was timed first; controls and diagnostic rebuilds were outside both clocks. The ratios below include shared J once per estimated path, not twice.

| Weight | J (s) | HISTORY extra s/M real targets | ERROR extra s/M | ERROR − HISTORY s/M | (J+HISTORY)/J | (J+ERROR)/J | (J+ERROR)/(J+HISTORY) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4901 | 52.043 | 9.139 | 9.365 | +.226 | 1.09211 | 1.09439 | 1.00208 |
| 4902 | 50.023 | 8.945 | 9.060 | +.115 | 1.09380 | 1.09501 | 1.00111 |
| 4903 | 44.879 | 8.339 | 7.712 | −.627 | 1.09747 | 1.09014 | .99332 |
| Pooled | 146.945 | 8.808 | 8.712 | −.095 | 1.09432 | 1.09330 | .99907 |

Per document/checkpoint, shared J averaged 135.68 ms. HISTORY took another 12.80 ms (0.62 prep, 9.39 pool, 0.82 features, 1.94 response), ERROR 12.66 ms (0.61, 9.18, 0.89, 1.97, respectively). Diagnostic controls took another 26.49 ms and are not needed by either reader. The invented stress cases priced before access were not representative runtime guarantees. We haven't compared against spending the same time on new host training.

Each chosen index peaked at **6,827 keys** with a pending latest position. HISTORY peaked at **3,179 committed key/byte cells**; ERROR at **4,444** across seeds. Peak HISTORY/ERROR packed query snapshots were 313,400 / 408,868 bytes and FP64 feature arrays 587,600 / 779,616 bytes. Top-1 tape peaked at 3,920 bytes on either path; HISTORY also made a miss-flag tape of that size. These are individual component peaks, not total process, Python index or minimal deployed storage. Only the control kept donor keys (peak 1,176). The verified aggregate separates chosen/control work, clocks, peaks and affected-document counts by checkpoint.

## Access and close

Public registration `821a8547d4b56a098695c399a2bf532df88dfced`; public pre-access qualification `3125dc94fde8ee0f9e09e01aaa222ea52325ef4e`, local source `12bb816cffc3e9c8c727517d8652e5c2c21cc5e9`. 179 CPU tests, 654 native address frontiers, three strict weight loads and prospective branch prices on invented streams. Metadata-only reservation took 39.29 wall / 39.29 CPU s. Distinct local numeric authority with ≥32 GiB free; pre-admission projections using the prior JT charge plus invented extra work were 223–652 seconds, not measurements of this job. After freezing the actual rows and coefficient identities, the leased 1/16/64/65/4,095-input device/score checks passed before the one raw open. Verified whole-document hashes and true shifts on 524,915 source bytes.

A detached, renewed owner was independently witnessed through **1,083 real seed/document units** and their actual target sums. One RTX 6000 Ada lease **released rc=0 at ≤230 incremental card-s** (1,200 ceiling); 244.80 s inclusive scheduling wall, 435.97 owner/child CPU s, another 15.66 driver/short-parent CPU s. Peak device allocated / reserved 715,972,608 / 754,974,720 bytes, worker RSS 2,739,272 KiB. The separate CPU close rehashed all three weight files, verified the pre-open freeze and release ledger, independently reproduced the losses and eleven intervals from sums and sample multiplicities, and replayed the branch timing, support and peak reductions. No calibration text reopened. The ET cohort is spent. Public aggregate SHA-256 `0d58f3aa50fb6830e998747530043796f48a2c0d636b53452c48a4c3f480ef1f`; private close `24da7fa08c8b2c740a073c676acd6a76df5ef7fb48894d3791e6df8991fc66b0`. No fitted coefficients, membership, per-document outcomes or guesses in the release.
