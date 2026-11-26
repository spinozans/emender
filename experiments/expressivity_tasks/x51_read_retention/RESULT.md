# The current archive answer was valuable. Saving every past answer added very little.

**Does looking it up help now? Yes.** On 65,536 real next bytes from 1,024 fresh documents, giving a small reader the current 100 GB XSA answer cut prediction loss from **3.09848 to 1.40494 nats/byte**, even when the answer disappeared immediately afterward. That's a reduction of **1.69355** [1.65984, 1.72869], or about 55% of this byte-only reader's loss. It is **not** a percentage of guesses made correct. The current read cost about **22×** the batched CPU evaluation time of this inexpensive no-archive model. The archive supplies substantial prior knowledge: its smoothed next-byte histogram *alone* scored 1.42249. Don't attribute the whole gain to the learned attention layer or assume it would be as large over a pretrained LM.

**Does it help to keep the answers? Two comparisons give different levels of evidence.** When we **trained** one reader to keep answers and another to discard them, KEEP won at all three seeds, but by only **.000087 nats/byte** [.000036, .000137], about **0.006%** of DISCARD's loss. When we took the **same trained KEEP weights** and hid old answers at prediction time, the mean gain from leaving them visible was **.000008** [−.000039, +.000058]. Hiding them hurt at one seed and helped at two. The old answers really were on the model's path: erasure changed nearly every later distribution, usually by a small amount. It did **not** reliably make predictions worse. These results support the small separately fitted retention advantage, **not** the stronger registered claim that the fitted model reliably benefits from consulting old answers at inference. Nor do they prove that old answers have zero value.

**Practical implication:** for this reader and training dose, almost all the measured archive gain came from having a fresh answer at the current position. No-read was much faster. A useful next question is whether old answers can *substitute* for some expensive new reads. Here both reading arms received a fresh answer at **every** byte; no read was ever skipped, and no sparse lookup schedule was tested.

[Registered test](PROTOCOL.md) · [implementation](README.md) · [pre-access qualification](receipts/qualification.json) · [full aggregate and prices](receipts/result.json)

## The two controls

All three models see the same actual byte history and attend over the same number of typed rows (two per observed byte, up to 128). NONE gets no source capability or archive logits; it starts from an add-half unigram counted only on genuine FIT targets. DISCARD and KEEP both start from the current X50 direct archive prior and add a learned 256-way residual. They issued exactly the same lookups: normalized answer/primary prior hashes agreed over every FIT order and over the whole probe. DISCARD's earlier answer slots carry only their byte/type/position embeddings. KEEP adds the original histogram projection to those same slots. The raw observations never change, nor can an erased answer hide inside a deeper contextual layer. The first prediction in a window has no earlier slot to read.

DROP is an intervention on the frozen KEEP fit, **not** another trained model: erase earlier values after exactly the same lookups, do not change the current answer, and do not refit. That distinction matters. A training comparison can move the learned weights as well as the input seen during evaluation; a fixed-weight intervention tests use by that particular reader. We didn't determine the cause of the small difference between the two comparisons.

## Absolute and paired loss

Lower absolute loss is better. No positions were excluded from the endpoint. Each fixed seed predicted the *same* windows; the probe isn't three independent corpora.

| Reader or prior | Mean NLL | 5101 | 5102 | 5103 |
|---|---:|---:|---:|---:|
| NONE | 3.09848312 | 3.06573857 | 3.11615143 | 3.11355937 |
| DISCARD | 1.40493668 | 1.40228298 | 1.40612397 | 1.40640308 |
| KEEP | 1.40484965 | 1.40207403 | 1.40610169 | 1.40637324 |
| DROP (KEEP weights) | 1.40485755 | 1.40229566 | 1.40604176 | 1.40623524 |
| Archive prior only | 1.42248920 | 1.42248920 | 1.42248920 | 1.42248920 |
| FIT unigram only | 3.54214091 | 3.54214091 | 3.54214091 | 3.54214091 |

Gain is control minus candidate loss (positive favors the first name). Jointly resample 1,024 whole paired documents with replacement, 10,000 draws, RNG 5100, linear percentiles, all three fixed seeds. Registered comparisons, not a search for favorable subsets:

| Candidate:control | Gain, nats/byte | Paired 95% interval | 5101 / 5102 / 5103 | Decision |
|---|---:|---|---|---|
| KEEP:DISCARD | +0.00008703 | [+0.00003640,+0.00013745] | +0.00020896 / +0.00002228 / +0.00002984 | win |
| DISCARD:NONE | +1.69354644 | [+1.65984435,+1.72869206] | +1.66345559 / +1.71002745 / +1.70715629 | win |
| KEEP:NONE | +1.69363347 | [+1.65992076,+1.72875730] | +1.66366455 / +1.71004974 / +1.70718613 | win |
| KEEP:DROP | +0.00000790 | [−0.00003896,+0.00005791] | +0.00022163 / −0.00005993 / −0.00013800 | unresolved |

At unchanged KEEP weights, all 1,024 first predictions per seed were identical with and without erasure. Of the 64,512 later positions, **64,282 / 64,490 / 64,052** exceeded the registered `1e-6` total-variation threshold. Every document had a changed later prediction at each seed. Mean TV on later positions was **.001848 / .001820 / .000833**. This measures the change in the 256-way predictions, not a change in top guess or evidence that the movement helped. Costs and support never filtered the loss comparison.

## Exposure and cost

| Arm | FIT chosen wall s/M | PROBE chosen wall s/M | Probe parent reads per seed | Probe native page rows per seed |
|---|---:|---:|---:|---:|
| NONE | 6.09 | 2.38 | 0 | 0 |
| DISCARD | 57.24 | 51.57 | 65,536 | 277,301 |
| KEEP | 55.68 | 51.35 | 65,536 | 277,301 |

No child reads. KEEP and DISCARD each requested the same 277,301 page rows in 123 batches per probe seed. Extra fixed-weight diagnostic time was 0.503 / 0.520 / 0.511 s outside the chosen KEEP clocks. These are batched, single-CPU-thread model clocks at matched exposure, including real lookup and optimizer work where applicable; the full aggregate separates receipt, diagnostic and owner CPU costs. KEEP/DISCARD wall ratios changed direction across seeds. We do **not** infer that keeping costs less, that discarding saves physical RAM in vectorized training, or that this measures live single-request latency.

The CPU qualification preceded selection. Both X50 roles and all predecessors were excluded by whole-document SHA (46,182 excluded). The next contiguous 512 FIT and 1,024 disjoint PROBE TRAIN documents supplied one 64-target window each, beyond the indexed prefix. Whole-document hashes and C14 shifts were verified at extraction (10,175,161 FIT / 17,776,264 PROBE source bytes). Duplicate passages elsewhere in the archive remain possible. Three epochs over FIT gave 98,304 fitted target exposures per seed/arm. The detached CPU process sealed all nine fits plus the FIT-only unigram at 884,736 witnessed target exposures before any PROBE payload opened. The additional fixed KEEP-weight predictions required no new source reads. No calibration/final access or GPU allocation.

The worker completed 1,440 batches (1,474,560 chosen-path target exposures) and exited. Including reservation, extraction, mount and close, worker wall was 86.43 s, owner CPU 75.04 s, child CPU 21.99 s, native open 0.365 s. Peak owner/max-child RSS 1,248,340/1,957,872 KiB (not additive). Receipt-only close and a separate multiplicity-weighted reconstruction verified all four intervals and nine fitted score journals without reopening natural text. Public aggregate SHA-256 `bb8ea0b7fb85ecdccb0cdc58e04e3a04e164fc22c82af3c4abe1b92ae39bb4b0`.

Both roles are spent. This is a local, out-of-fit TRAIN probe on a 32-wide reader with 64-byte histories, not a general finding about all retained values, long documents, or scale. No default change, confirmation or rerun.
