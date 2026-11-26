# X52 — can an old answer cover the three bytes between lookups?

**A saved answer helped the reader that learned to use it, but it did not replace a fresh read.** With archive access on only one byte in four, erasing old answers from the same fitted model hurt its predictions on the three intervening bytes. Training with those answers present, however, did not consistently beat training without them. The sparse reader ran **2.82× faster** than the every-byte reader in batched CPU evaluation, but kept only **22.8%** of its loss reduction relative to no archive. This is a quality/time trade, not the same quality for a quarter of the cost.

## What was held fixed

[Protocol](PROTOCOL.md) · [pre-access source and CPU gate](receipts/qualification.json) · [aggregate results](receipts/result.json) · [X51: fresh reads everywhere](../x51_read_retention/RESULT.md).

The probe was 1,024 new TRAIN documents: one reset 65-byte window per document, 65,536 real shifted targets per seed, all outside the mounted P100 prefix. No reuse of either X51 role. After 47,718 whole-document SHA exclusions, the next 512 documents were FIT and the next 1,024 PROBE. All twelve new fits and the FIT-only byte unigram were sealed and independently witnessed before probe text was opened. Three fixed seeds (5201–5203); same initialization, order and exposure per seed. No tuning, extra probe, GPU or final-test access. Exclusion by document SHA does not exclude repeated passages.

At positions 0,4,…,60 the sparse readers both requested a real archive answer. At positions 1–3 after each read they made **no new lookup**. The only base prediction on those gaps was the same FIT unigram. KEEP4 could attend to previously deposited `sqrt(pdf)` projections; DISCARD4 had the same byte history and typed slots but no older answer content. ERASE4 applies DISCARD4's visibility at KEEP4's unchanged weights using the same current evidence. EVERY is a matched fresh fit with one read per byte and retention; NONE has no archive. No child reads. The scheduled answer and prior streams matched EVERY's scheduled subset in all fits and probes.

## Loss on every real target

Mean NLL in nats/byte across the fixed seeds (lower is better). “Refresh” is the sparse schedule; EVERY also reads in the gaps. There were 16,384 predictions per age and 49,152 gap predictions per probe seed. All tokens scored here are original bytes, not tokenizer pieces.

| Reader | All bytes | Refresh | Gaps pooled | 1 byte after | 2 after | 3 after |
|---|---:|---:|---:|---:|---:|---:|
| NONE | 3.08200136 | 3.09959912 | 3.07613544 | 3.07968921 | 3.08083762 | 3.06787950 |
| EVERY | 1.41186125 | 1.46581412 | 1.39387695 | 1.43507918 | 1.39203333 | 1.35451834 |
| DISCARD4 | 2.70127970 | 1.52586415 | 3.09308489 | 3.09575224 | 3.09903875 | 3.08446368 |
| KEEP4 | 2.70126362 | 1.52647210 | 3.09286079 | 3.09570758 | 3.09880055 | 3.08407425 |
| ERASE4 (no refit) | 2.70175861 | 1.52648486 | 3.09351653 | 3.09651694 | 3.09916367 | 3.08486898 |
| Unigram alone | 3.53800322 | 3.54686695 | 3.53504864 | 3.54159150 | 3.53663403 | 3.52692038 |
| Every-byte prior alone | 1.42809925 | 1.49711664 | 1.40509346 | 1.44832592 | 1.40287062 | 1.36408383 |
| Sparse prior alone | 3.02556564 | 1.49711664 | 3.53504864 | 3.54159150 | 3.53663403 | 3.52692038 |

Gain = control loss minus candidate loss per target in the named band; positive favors the first reader. Paired 95% whole-document bootstrap intervals (10,000 draws, RNG 5200), averaging over fixed seeds, not claiming independent replications from each byte. The registered directional rule also requires all seed signs to agree.

| Candidate : control | Band | Gain [95% interval] | Seed gains (5201 / 5202 / 5203) | Decision |
|---|---|---|---|---|
| KEEP4 : DISCARD4 | gaps | +.00022409 [.00002068, .00043321] | +.00049246 / −.00010904 / +.00028887 | **Unresolved** |
| KEEP4 : ERASE4 | gaps | +.00065573 [.00040996, .00091669] | +.00186508 / +.00000679 / +.00009533 | **Win** |
| KEEP4 : DISCARD4 | all | +.00001608 [−.00014053, .00017159] | +.00021637 / −.00034647 / +.00017834 | Unresolved |
| KEEP4 : ERASE4 | all | +.00049499 [.00030492, .00069534] | +.00139604 / +.00001070 / +.00007823 | Win |
| EVERY : KEEP4 | all | +1.28940238 [1.26584986, 1.31504556] | +1.29129028 / +1.30131979 / +1.27559706 | Win |
| EVERY : KEEP4 | gaps | +1.69898384 [1.66756427, 1.73337792] | +1.70173921 / +1.71520543 / +1.68000689 | Win |
| KEEP4 : NONE | all | +.38073774 [.37241578, .38955959] | +.38469121 / +.37865355 / +.37886846 | Win |
| DISCARD4 : NONE | all | +.38072166 [.37236476, .38950503] | +.38447484 / +.37900002 / +.37869012 | Win |
| EVERY : NONE | all | +1.67014012 [1.63947459, 1.70328112] | +1.67598149 / +1.67997333 / +1.65446552 | Win |

The two primary gap comparisons answer different questions. **Fixed-weight reliance passed**: the same KEEP4 reader used past information to lower its gap loss, albeit by only about .021%, with most of the effect at seed 5201. Every first prediction stayed identical. At TV > `1e-6`, erasure changed 49,144 / 49,150 / 49,120 gap predictions across seeds, in every document; movement is not the evidence for better prediction, the loss contrast is. **A consistent benefit over separately fitting DISCARD4 did not pass**, despite the positive pooled gap interval: seed 5202 went the other way. Over all bytes even the interval crossed zero. The joint criterion for useful reuse as a training choice failed. Unresolved is not equivalence.

## Actual work and decision

Chosen-path wall seconds per million real targets, summed over the three seeds. FIT includes optimization. Archive resolution, pages, model and loss are in the chosen clocks; evidence hashing, receipt work and the ERASE4 counterfactual are separate. The archive clock is batched, in a shared OS cache, with a fresh model-facing page LRU per arm and phase. These are not single-byte decode latencies or matched-wall fits.

| Reader | Reads per probe seed | Prefix frames per probe seed | Page rows / row batches per probe seed | FIT s/M | PROBE s/M |
|---|---:|---:|---:|---:|---:|
| NONE | 0 | 0 | 0 / 0 | 5.62 | 1.98 |
| EVERY | 65,536 | 1,024 | 276,295 / 119 | 56.66 | 50.29 |
| DISCARD4 | 16,384 | 16,384 | 69,435 / 64 | 22.21 | 17.11 |
| KEEP4 | 16,384 | 16,384 | 69,435 / 64 | 22.23 | 17.80 |

There really were 75% fewer answer distributions. The sparse resolver sent one suffix frame per scheduled prefix, whereas the dense scan produced 64 prefix rows per document frame. Both native request geometry and page traffic matter. The measured speedup was 2.82× for KEEP4 on probe, not 4×. The on-probe ERASE4 diagnostic added 0.810 s across the three probes and no reads. The detachment/closure witness recorded 1,920 chosen batches, 1,966,080 scored target exposures, worker wall 79.84 s, 67.71 owner + 19.41 child CPU s, zero GPU. Acquisition, chosen CPU, audit and erasure clocks by arm/phase are in the aggregate receipt.

**Decision:** If the objective is predictive quality for this small reader, keep the current read. Use quarter-rate only as an explicitly cheaper, lower-quality option. The measured keeper benefit does not justify changing defaults. This tested carrying a **one-byte forecast**, not carrying alternative multi-byte continuations or a followable token-suffix node. No inference about token-aligned lookup, large models or scale follows from it.
