# Remembering corrections helped again. Their text context mattered, but accounted for a smaller gain.

**The question:** Suppose the model expects a space. Earlier in the document it expected a space and got a newline instead. Is it useful to remember that pairing? And is it important that the mistake happened *after the same stretch of text*? Space/newline is an illustration, not a corpus example.

**The answer on 197 fresh short calibration documents:** yes to both, without changing the models or refitting the readers. Knowing which correction actually followed the current guess at the matched text pattern (**PAIR**) beat knowing only how often the guess failed there and which alternatives were common (**MARG**). The gain was **.000281 nats per next byte** (paired 95% interval **[.000192,.000371]**), about **0.024% lower prediction loss**. This replicates the direction of the [earlier TRAIN probe](RESULT_PAIRED_CORRECTION.md). The gain is small; it is not a percentage of guesses made correct.

**Most of that improvement did not require a pair table at every pattern.** One table for the document *so far* (**PREFIX**), still using local counts to decide which alternatives were plausible, beat MARG by **.000211** [.000136,.000291]. That is **75% of PAIR's measured loss reduction over MARG**, not 75% of all predictive information or of all correct guesses. PAIR beat PREFIX by the remaining **.000070** [.000012,.000127], at **2.1% more time** on the measured shared prediction path. All three frozen checkpoints agreed. PREFIX had lower measured time and fewer occupied pair cells here, but was not as accurate. Both are hand-designed readers of the model's past forecasts; this is not evidence the model invented a memory system or learned which records to keep.

[Registered test](CORRECTION_TRANSFER.md) · [implementation and retained failed fixture](CORRECTION_TRANSFER_IMPLEMENTATION.md) · [pre-access qualification](receipts/correction_transfer_cpu_qualification.json) · [redacted full aggregate](receipts/correction_transfer_result.json)

## What the controls tell us

All readers saw the same genuine prefix in the same frozen host. Only mistakes that were old enough were eligible. The latest eligible local follower was already used by the common reader. PAIR used only older local mistake pairs; PREFIX maintained one delayed prefix table and discounted that latest record for the lookup. PREFIX did **not** use the whole document in advance, another document, or a corpus-wide confusion table. It used the same local support, unpaired feature and coefficients as PAIR. This is a comparison against that particular fixed cheaper alternative, not every possible context-independent rule.

When we exchanged outcomes between eligible older mistakes while keeping both sets of counts fixed, PAIR lost **.000444** [.000331,.000563] with its coefficients unchanged (SWAP). A reader previously fitted on exchanged pairs (SHUF) still lost to PAIR by **.000283** [.000192,.000374]; SHUF itself was indistinguishable from MARG on this test (gain −.0000015 [−.0000080,+.0000048], mixed checkpoint signs; not equivalence). Even without the table of pairs, MARG beat the earlier frozen ERROR reader by **.000360** [.000259,.000463]. The four registered PAIR comparisons all passed; the separate MARG replication and both PREFIX comparisons against MARG/ERROR passed too. The PAIR:PREFIX difference passed the separately specified directional check. None depended on scoring only the positions where a table changed the answer.

### Absolute loss and uncertainty

One untouched document per sample. All **285,520 real shifted bytes per checkpoint**, including misses and singleton matches, contributed. The three checkpoints predict the same documents; they are not three times as many independent samples. Loss is negative log probability per actual next byte (lower is better).

| Reader | Mean | 4901 | 4902 | 4903 |
|---|---:|---:|---:|---:|
| B | 1.183196775 | 1.183713594 | 1.182720987 | 1.183155744 |
| U | 1.179921110 | 1.180422153 | 1.179552391 | 1.179788784 |
| J | 1.178933734 | 1.179316239 | 1.178560938 | 1.178924025 |
| HISTORY | 1.176891608 | 1.177179886 | 1.176523187 | 1.176971752 |
| ERROR | 1.176748783 | 1.177081734 | 1.176341974 | 1.176822642 |
| MARG | 1.176388574 | 1.176712047 | 1.175957725 | 1.176495949 |
| PAIR | 1.176107433 | 1.176420494 | 1.175676088 | 1.176225717 |
| SHUF | 1.176390029 | 1.176716059 | 1.175959381 | 1.176494646 |
| SWAP | 1.176551745 | 1.176917773 | 1.176113083 | 1.176624378 |
| PREFIX | 1.176177509 | 1.176491332 | 1.175753783 | 1.176287410 |

Gain = control loss minus candidate loss; positive favors the first name. Intervals use 10,000 jointly paired whole-document resamples, RNG 4918, actual number of shifted targets in each draw, linear percentiles, average over the fixed checkpoints.

| Candidate:control | Gain | Paired 95% interval | 4901 / 4902 / 4903 |
|---|---:|---|---|
| PAIR:MARG | +0.000281140 | [+0.000192188,+0.000370838] | +0.000291553 / +0.000281636 / +0.000270232 |
| PAIR:SHUF | +0.000282596 | [+0.000192390,+0.000373550] | +0.000295566 / +0.000283293 / +0.000268929 |
| PAIR:SWAP | +0.000444312 | [+0.000330548,+0.000563114] | +0.000497280 / +0.000436995 / +0.000398661 |
| PAIR:ERROR | +0.000641350 | [+0.000491944,+0.000790402] | +0.000661240 / +0.000665885 / +0.000596924 |
| MARG:ERROR | +0.000360209 | [+0.000259142,+0.000463242] | +0.000369687 / +0.000384249 / +0.000326693 |
| PAIR:PREFIX | +0.000070076 | [+0.000012126,+0.000127332] | +0.000070838 / +0.000077695 / +0.000061693 |
| PREFIX:MARG | +0.000211065 | [+0.000135926,+0.000290576] | +0.000220715 / +0.000203941 / +0.000208539 |
| PREFIX:ERROR | +0.000571274 | [+0.000441682,+0.000703341] | +0.000590402 / +0.000588190 / +0.000535231 |
| SHUF:MARG | −0.000001455 | [−0.000007974,+0.000004836] | −0.000004012 / −0.000001656 / +0.000001303 |
| PAIR:J | +0.002826301 | [+0.002406110,+0.003259998] | +0.002895745 / +0.002884850 / +0.002698307 |
| ERROR:HISTORY | +0.000142825 | [+0.000077776,+0.000207878] | +0.000098152 / +0.000181214 / +0.000149110 |
| HISTORY:J | +0.002042126 | [+0.001729938,+0.002349428] | +0.002136353 / +0.002037751 / +0.001952273 |
| J:U | +0.000987376 | [+0.000643494,+0.001314922] | +0.001105914 / +0.000991453 / +0.000864760 |
| U:B | +0.003275666 | [+0.002902785,+0.003673267] | +0.003291441 / +0.003168596 / +0.003366959 |

## How often did the association matter?

Per checkpoint, **271,453** positions had a literal match (116,799 correct latest copies); 14,067 missed, and 87,036 hits had only one eligible record. 184,417 positions had older history, used with full multiplicity (2,115,450 older occurrence uses). Only **25.0–25.1% of all predictions** found a previous wrong guess that matched both the current guess and the selected text pattern. All four PC rules and PREFIX left ERROR unchanged where there was no such mistake.

| Support | 4901 | 4902 | 4903 |
|---|---:|---:|---:|
| Earlier guesses written | 285,323 | 285,323 | 285,323 |
| Mistaken / correct at arrival | 117,620 / 167,703 | 117,702 / 167,621 | 117,729 / 167,594 |
| Predictions with a matching older mistake | 71,383 | 71,574 | 71,386 |
| Matching older mistake uses | 244,849 | 247,982 | 239,643 |
| True correction law differed from local unpaired alternatives | 36,023 | 36,146 | 36,275 |
| Adjacent mistake pairs considered across local keys | 110,502 | 110,569 | 110,573 |
| Legal exchanges | 43,587 | 43,303 | 43,596 |
| Queries with changed conditional histogram after exchange | 26,825 | 26,721 | 26,791 |
| Changed uses (sum of positive bin differences) | 53,828 | 53,756 | 54,464 |
| Documents with a changed histogram after exchange | 193 | 193 | 193 |
| Queries where local vs prefix TV exceeded `1e-12` | 38,662 | 38,693 | 38,561 |
| Documents where local vs prefix TV exceeded `1e-12` | 192 | 191 | 192 |
| Sum of local vs prefix total-variation distances | 12,181.33 | 12,121.39 | 12,136.45 |
| Eligible arrivals written as prefix mistakes | 116,235 | 116,323 | 116,352 |
| Queries discounting a latest mistake from prefix counts | 42,381 | 42,630 | 42,416 |

An exchange and a changed query are different units. The adjacent exchange altered the actual queried histogram on about **9.4% of all positions**; PAIR and PREFIX's correction distributions differed on about **13.5%**. The latter averaged about **0.17 total variation on matching queries** (about 0.043 over all targets). This distance compares the correction distributions, not the final 256-way predictions. Counts do not imply that every change improved the loss. The prefix scan admitted 282,176 sufficiently old arrivals per checkpoint, once each; local pair-table writes could occur at multiple suffix depths. The prefix reader kept the same local marginal counts as MARG; only its estimate of the pairing was key-blind.

## Price of each choice

One fresh host prefill and full ERROR prediction per checkpoint/document. Each chosen branch built its own marginal features and full response; PAIR also built its local sparse table, PREFIX its single delayed table. Branch order rotated through the six permutations. The extra diagnostics, including exchanged associations, were outside all three chosen clocks. All time comparisons below include the same common path once, **not** three independently run serving systems.

| Checkpoint | Common ERROR s/M targets | MARG extra s/M | PAIR extra s/M | PREFIX extra s/M | PAIR path / PREFIX path |
|---|---:|---:|---:|---:|---:|
| 4901 | 98.952 | 1.045 | 5.937 | 3.095 | 1.0279 |
| 4902 | 108.945 | 1.124 | 4.948 | 3.348 | 1.0142 |
| 4903 | 107.573 | 1.123 | 5.695 | 3.331 | 1.0213 |
| Pooled | 105.157 | 1.097 | 5.527 | 3.258 | 1.0209 |

Pooled, PAIR cost **2.269 extra seconds/million real targets vs PREFIX**. Compared with shared-plus-MARG, the PAIR path took 4.17% more time and PREFIX 2.03% more time. Common ERROR averaged 152.41 ms/document; chosen MARG/PAIR/PREFIX added 1.59/8.01/4.72 ms respectively. Comparison-only work cost 22.49 ms/document (15.52 s/M). These measurements do not compare against spending that time training the host.

| Extra logical state / returned data | 4901 | 4902 | 4903 |
|---|---:|---:|---:|
| Peak local key/guess rows | 1,684 | 1,682 | 1,672 |
| Peak local key/guess/correction cells | 2,377 | 2,419 | 2,400 |
| Peak prefix guess rows | 79 | 81 | 75 |
| Peak prefix guess/correction cells | 502 | 511 | 499 |
| Local nonzero bins returned over all queries | 154,607 | 156,330 | 153,114 |
| Prefix nonzero bins over all queries | 209,066 | 210,092 | 207,887 |
| Peak local packed snapshot bytes | 47,493 | 47,613 | 47,469 |
| Peak prefix rate-array bytes | 61,992 | 61,152 | 61,296 |

Peak PAIR and PREFIX feature arrays were the same size within each weight (123,984 / 122,304 / 122,592 bytes); MARG used half as many feature bytes. A smaller live pair table does not guarantee a smaller returned array, nor do logical cell counts measure Python allocator use. The aggregate separates per-document peaks from totals of work.

## Scope and closure

Public qualification `b83f1063559f7c0323216666e29274015f13e2d0` preceded natural reservation. The metadata census rederived 45,217 prior excluded SHAs and the complete remaining **197** short native calibration documents after the 381 JT and 361 ET documents. Whole-document exclusion does not guarantee deduplicated passages. Length bands (real targets/document): 1–15: 1 document / 13 targets; 16–127: 6 / 416; 128–1,023: 72 / 44,769; 1,024–4,095: 118 / 240,322. The smaller dose was fixed before membership; don't infer a length effect by comparing different cohorts.

The readouts were fixed before the single natural open. Device parity on five invented lengths preceded it. Read 285,717 source bytes, with no padding, state carry, fits, neural updates or native final access. Metadata-only reservation took 36.73 wall / 36.55 CPU s. One detached RTX 6000 Ada job, 591 witnessed document/checkpoint units, **≤137 inclusive card-s**, released rc=0. Inclusive scheduling wall 152.00 s, owner/child CPU 256.68 s, driver/short parents 15.77 CPU s. Peak allocated/reserved device bytes 715,972,608 / 754,974,720; worker RSS 2,760,296 KiB. The receipt-only close rehashed the three weight files and independently reproduced the losses, paired intervals, target counts, support, peaks and branch costs without reopening text. Public aggregate SHA-256 `25dbd6ee5dc29b71a770b040e6e661bdf28dcb5503cf7d734b67553ae5e52fc6`; private close `a0c56ff05b49d7fb53a1fe6230cd3176a9a11e4b85aacefb4f5a8ef8e26916b8`.

**Decision:** paired corrections now have a disjoint TRAIN probe and a no-refit calibration replication on the three tested checkpoints. PREFIX is a useful lower-cost comparison; local pairing was measurably better here. The short calibration inventory under these rules is spent. No reuse, native-final confirmation, default change or scale nomination. B remains the baseline; U remains the native-final-confirmed local reader.
