# Independent edits distinguish reading a row from decoding its absence

**The trained whole-corpus head can follow a change to one archive answer, with no matching token and all seven other rows fixed.** On fresh books with repeated values, DENSE trained on that task followed every isolated edit in every seed; INDEXED followed **86.51%, 100%, 100%**. In contrast, three models that had been perfect on the earlier **two-key permutation swap** followed **zero** isolated edits. The difference is identifiable in the learned logit response, not just in a heatmap.

[Protocol](PROTOCOL.md) · [numeric receipt, including every seed and failed transfer](receipts/result.json) · [CPU tests and receipt audit](README.md). “Followed” means both the original and new answers are top-1 on the *same* prompt; no conditioning on the original answer having been correct. LOCAL had 12.5% answer accuracy but **zero** joint successes on either intervention.

## Closed comparisons

Each fit saw the same 1,792 manufactured probe editions. For each of 28 key pairs we crossed the two values independently in all 64 ways, with six fixed background values including duplicates. This allowed 3,584 variable-key answers, **25,088** directed one-row rewrites, and **3,136** directed swaps that preserved the entire value histogram. These are counts over a finite design, not independent natural-document trials. No query had any lexical refinement; every corpus row contributed via its exact score (DENSE) or the positive feature summary (INDEXED).

| Training books → new repeated-value probe | Active answer NLL ↓ | Single-row edits followed, seeds 5701 / 02 / 03 | Same-bag swaps followed, seeds 5701 / 02 / 03 | Registered all-seed result |
|---|---:|---|---|---|
| Permutations, LOCAL | 2.08078 | 0 / 0 / 0% | 0 / 0 / 0% | baseline |
| Permutations, DENSE (frozen transfer) | 5.25211 | 87.50 / 100 / **0%** | 87.50 / 100 / 2.30% | **no**, seed 5703 failed |
| Permutations, INDEXED (frozen transfer) | 7.26544 | **0** / 83.21 / **0%** | 1.21 / 82.14 / 2.61% | **no**, seeds 5701 and 5703 failed |
| Repeated values, LOCAL | 2.09761 | 0 / 0 / 0% | 0 / 0 / 0% | baseline |
| Repeated values, DENSE | **0.00014** | 100 / 100 / 100% | 100 / 100 / 100% | **passed** |
| Repeated values, INDEXED | **0.05152** | 86.51 / 100 / 100% | 85.33 / 100 / 100% | **passed** |

NLL is the equal-seed mean in nats per active answer. For the new fits, edited vs stale archive improved NLL on isolated edits by **15.0512** nats for DENSE, **11.7474** for INDEXED. The intended row received mean **joint** masses 0.9851 / 0.9362 respectively (including local and no-read competition). INDEXED seed 5701 missed 306 of 3,584 active answers; do not report all indexed fits as perfect. Clamping only the queried row to the same value for every alternative brought accuracy to **12.5% for all 18 fits**. Getting the same-bag swap right distinguishes selective reading from merely noticing the marginal histogram changed.

## Why a permutation swap could give the wrong impression

Suppose each color appears once. A reader can give very little weight to the requested slot, aggregate the *other* colors, then favor the color least represented in the read. Swapping two colors makes that reader change its answer correctly. Replacing just the requested color while allowing repeats is different.

For this fixed vector head, write `g_ij` for the joint mass that query `i` gives row `j`. The response to changing answer `a` to `b` factors exactly into an address weight and a learned decoder contrast `κ(a,b)` ([derivation](PROTOCOL.md)):

- **Change only row `i`:** the log odds of `b` versus `a` change by `g_ii κ(a,b)`.
- **Swap with row `j`:** the same log odds change by `(g_ii − g_ij) κ(a,b)`.

For the three [formerly perfect swap solvers](../semantic_tail/RESULT.md) that never ranked the intended key first (DENSE 5703, INDEXED 5701 and 5703), **all 28 unordered answer contrasts were negative**. On average the original two-row operation moved the relative odds in the right direction on the new books (log-odds changes **+20.30 / +19.02 / +17.40** for the swaps), but changing *only* the target row moved them in the wrong direction (**−0.80 / −1.64 / −0.60**). None got both endpoints right on even one of the 25,088 isolated edits. The other three transferred corpus models had **zero** negative contrasts and each passed the registered rule individually. This is evidence of decoding by exclusion in specific fits, not a guess based on where attention peaked.

All six replacement-trained corpus models had positive decoder contrasts for every answer pair. They learned to decode the intended value without needing all answers to be distinct. The largest difference between the analytic log-odds change and native scored logits was **0.00002282** across all fits and edits. Direct probability changes, paired NLL, same-bag controls, and known support were all checked; attention mass alone would not establish this result.

## Scope and cost

The old nine synthetic snapshots remained fixed; only their training files were loaded, not their original probe scores. Nine fresh fits used the same starts, architecture, optimizer and **16,384 real shifted targets per fit**, including cues, with 1,024 distinct FIT editions rather than 128 repeated eight times. A successful training recipe, **not** a controlled estimate of the effect of diversity alone. No tuning or retries. New training loops took **60.33 s** altogether on one CPU thread. Thirty pre-fit checks passed; all 18 weights and paired starts/orders were bound before one new scoring opening. New freeze SHA `9474f398c9e2b90c77d5abd73dbfe129cd5d713d0a909d0a14fb3c2177bbeccb`. The receipt-only audit checked counts, directed-pair symmetry, decoder and mass identities, and each primary decision.

**Use independent row edits as well as swaps when testing binding.** This establishes a capability of the co-trained finite-rank vector reader on eight invented keys, not fidelity to dense attention at arbitrary weights, a language-model quality win, or an all-position GPU speedup. No natural documents accessed and no additional card-seconds.
