# A whole-corpus read followed edited answers without any lexical hits

**Both co-trained corpus heads passed the [registered invented binding test](PROTOCOL.md) in all three seeds.** LOCAL did not follow an edit. DENSE scored all eight corpus rows with the exact dot product; INDEXED used only the positive background summary, with **zero exact row refinements**. Both returned learned vectors to a learned LM, not successor-token votes. [All per-seed measurements](receipts/result.json).

Across 224 held-out pairs of archive editions, swapping two values preserved every prompt, every key, every row weight, and the multiset of values. There were 448 changed-key comparisons per fit. “Followed” requires getting the old answer right before the swap **and** the new answer right after it on the **same live prefix**, unconditionally.

| Native training/eval law | Edited answer NLL ↓ | Edited answer accuracy | Edits followed / 448, seeds 5701 / 02 / 03 | Gain vs stale archive, changed-key NLL |
|---|---:|---:|---|---:|
| LOCAL | 2.08078 | 12.50% | 0 / 0 / 0 | 0 |
| DENSE | 0.09098 | 96.35% | 392 / 448 / 448 | +11.15321 |
| INDEXED | 0.06736 | 95.85% | 448 / 363 / 448 | +9.15909 |

Table losses and percentages are equal-seed means. Edited answer NLL was 0.27174 / 0.00008 / 0.00113 for DENSE and 0.00298 / 0.19723 / 0.00187 for INDEXED: **no consistent ordering between corpus laws**. Cue NLL was about 2.080 for all three, close to the irreducible `ln(8)` for a shared prompt with eight balanced possible next cues. On all shifted targets (cue **and** answer) LOCAL / DENSE / INDEXED scored 2.08034 / 1.08549 / 1.07372. On unchanged keys both original and edited predictions were right in 12.50% / 96.35% / 95.29% of pairs, respectively.

## Useful read does not mean faithful approximation—or literal addressing

- Swap **only the law** at the same frozen weights and on the same edited edition: DENSE-trained weights scored **4.57735** answer NLL under INDEXED instead of 0.09098. INDEXED-trained weights scored **0.59092** under DENSE instead of 0.06736. Mean next-token TV was **0.754 / 0.322**, conditional corpus-row TV **0.578 / 0.246**, respectively. Finite rank 32 was *not* a drop-in approximation to the fitted dense reader. Co-training the approximate read from the start mattered here.
- At answer time the corpus had high joint mass (mean 0.946 DENSE, 0.992 INDEXED). But **three of six corpus fits** had perfect edited-answer accuracy while **never ranking the intended corpus row first**: DENSE 5703, INDEXED 5701 and 5703. Their mean conditional mass on that row was only **0.00538 / 0.01122 / 0.00474**, respectively. Accurate decoding of a learned aggregate does not identify which row it used.
- Each edition is a permutation: knowing the other seven values determines the eighth. A model could read a mixture of the *other* values and decode the omitted one. A two-key swap cannot isolate an intended-row change from the donor-row change. This result establishes use of the relative archive bindings; it does **not** establish direct successor copying or success on arbitrary edits with repeated values.

A separately registered [independent-row test](../binding_edits/RESULT.md) subsequently resolved the exclusion ambiguity on fresh repeated-value books, without reopening this probe. It reports the three failed frozen transfers as well as the new co-trained fits.

## What was measured

One CPU thread, 6,778 trainable scalars per model. All parameter tensors saw nonzero gradients in all fits. Same initial states, 1,024 updates / 16,384 real targets within seed; 9 fits total, **60.83 s** in the measured training loops (per fit LOCAL 5.58–5.66 s, DENSE 6.40–6.45 s, INDEXED 8.21–8.26 s). These tiny-shape costs are not a scale benchmark. All nine snapshots verified before the single PROBE opening; freeze SHA `6843161006903a7e12923b1177aeb0aff06b310fff110b5cecf570b5311bb838`. [28 passing pre-fit CPU checks](receipts/qualification.json). A separate receipt-only check verified paired doses, counts, local invariance, cue entropy bound, NLL decompositions and the per-seed primary. No natural data, CUDA launch, GPU retry or additional card-seconds. This is a mechanism result, not a reversal of the [negative natural four-page result](../NATURAL_RESULT.md).
