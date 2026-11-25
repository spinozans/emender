# X49-W: the inner binding matters to L, but L does not beat recomputation

**Decision: keep B as the local reader.** A second lookup really carried an earlier computation into a write that was later read. Scrambling just that inner computation hurt L. But on new documents, learning to write with the earlier value did **not** outperform learning to write with a matched current-row value. Both enriched writers slightly improved on B at full exposure, and both lost to B on average at actual common training time. Do not adopt L, attribute its entire gain over B to the older operand, or infer a Lux case from the positive sham effect.

[Prospective design](WRITE_READ.md) · [implementation and data boundary](WRITE_IMPLEMENTATION.md) · [aggregate receipt](receipts/write_result.json) · [closed-receipt verifier](write_report.py).

## The direct comparison

B recalls the vector attached to the latest eligible literal follower. P and L use the same literal addresses, observed bytes, counts and 7,264 additional parameters. When writing a record, P recomputes the inner operand from the writer's current host and the earlier follower byte; L reads the **base** vector from the earlier writer. Neither recurses. All three started from the same trained B weights **within each seed**, with fresh moments. They made 1,024 updates on 4,194,304 shifted targets apiece. H is the lookup horizon for each hop, not a truncation of the causal host.

The final contains 377 previously unspent **whole short documents** in the native final split. Each cell scores the same **562,712 real next bytes** including misses, with no padding or end target. Primary loss is total NLL / total targets. Intervals are paired document bootstraps with 10,000 common draws (RNG 4905), each divided by its own target count, averaged over the three paired training seeds. They describe document uncertainty for these endpoints, not a large sample of independent training runs.

| Full-dose all-byte NLL (lower is better) | H128 | H1024 | H4096 |
|---|---:|---:|---:|
| B: continued reader | 1.336956 | 1.211269 | 1.190048 |
| P: current-row inner value | 1.336803 | 1.211084 | **1.189850** |
| L: earlier inner value | **1.336797** | 1.211109 | 1.189882 |

**Gain = control minus candidate**, in nats per real byte. Common wall uses the registered pair-specific durable checkpoints, not a cost estimate or a learning-curve interpolation.

| Candidate : control | Full-dose H4096 [95% CI] | Full-dose gains by seed (4901/02/03) | Common-wall H4096 [95% CI] |
|---|---:|---|---:|
| L : P | −.000033 [−.000135, +.000051] | −.000152 / −.000013 / +.000067 | −.001935 [−.002231, −.001620] |
| L : B | +.000166 [+.000108, +.000241] | +.000206 / +.000133 / +.000158 | −.001867 [−.002699, −.000794] |
| P : B | +.000199 [+.000077, +.000365] | +.000358 / +.000146 / +.000091 | −.002454 [−.003126, −.001601] |

At fixed exposure the data support a benefit from **enriching the write**, not a benefit specific to using the earlier inner value instead of recomputing. The L:P interval includes zero; that is not an equivalence proof. All three common-wall contrasts were negative in seeds 4901 and 4903 and positive in 4902. Checkpoint slack never exceeded 3.51%. The pairs have different common budgets, so their wall contrasts need not sum. There is no consistent improvement in L's relative advantage when increasing H128 to H4096: L:P interaction −.000039 [−.000097,+.000010], L:B +.000007 [−.000058,+.000090]. The receipt includes all H-specific contrasts, seed effects, intervals and selected steps. No new effect-size gate was imposed.

## What the interventions establish

At H4096, per seed, 535,344 outer lookups hit and **27,368 missed**. On 496,586 targets (88.25% of *all* targets), the outer read selected a write that itself had a literal hit. P and L had the same two-hop addresses. On 66,652 targets the two hops together reached over 1,024 positions back. This is a frequently available path, not a rare-operation explanation for the null against P.

| H4096 intervention, wrong minus correct | Mean NLL increase [95% CI] |
|---|---:|
| B: outer payload only | +.002585 [+.002385,+.002779] |
| P: outer payload only | +.002806 [+.002595,+.003012] |
| L: outer payload only | +.002770 [+.002560,+.002976] |
| **L: inner operand only** | **+.000033 [+.000023,+.000044]** |

The outer intervention reassigned 432,948 retrieved vectors per seed. The inner-only intervention changed the donor at 432,948 **writes**; a later read consumed an altered write on **392,466 predictions**. It did not change the selected outer slot, follower byte, suffix histogram or direct head. All seeds lost under inner rotation. L thus uses the binding, but P shows that sensitivity is not enough to justify *where the inner computation came from*. The larger outer effect applies to all three readers; it is not an estimate of L's incremental advantage.

| Document length in shifted targets | Documents | Real targets | Docs with a two-hop opportunity |
|---|---:|---:|---:|
| 1–15 | 1 | 14 | 0 |
| 16–127 | 8 | 611 | 7 |
| 128–1,023 | 132 | 77,394 | 132 |
| 1,024–4,095 | 236 | 484,693 | 236 |

The receipt gives pooled age and length diagnostics (no loss for a singleton length bin). The misses and the shortest document **are** in the primary loss. The short-document population is different from the spent long-window X49 and R tests. Full-document SHA exclusions do not prove passage-level deduplication. Do not compare absolute NLL across those boards as if they were paired.

## Cost and closure

All nine training leases finished and released: B ≤2,789, P ≤2,857, L ≤2,882 card-s across their three seeds. Shared preparation took 11.764 wall s (attributed equally to nine trajectories). Full training pre-update NLLs were B 1.159655, P 1.159609, L 1.159598; these are not the endpoint test losses. The final froze nine full-dose and eight additional actual common-wall checkpoints **before opening the source**. It completed 6,409 endpoint-documents in one final lease, correct H4096 only on the eight extra endpoints. Final ≤1,323 card-s; **total incremental ≤9,851 card-s (2.736 GPU-h)** including training and evaluation, vs the registered 18,900 card-s operational safeguard. No retry, no remote job. Earlier B training is a common sunk cost, not included again. The three endpoints share the same test bytes; repeats across seeds, arms and interventions are not additional independent data.

On this implementation the full-dose H4096 correct head took about 30.48/31.55/31.89 ms per short document for B/P/L, separate from mean host prefill 108.27/105.36/107.94 ms. These are serial diagnostic timings including eligibility/donor work in the scored cells, not a serving speed claim. Byte counts, memory peaks, host time and inclusive training clocks are in the receipt.

Local source `a8eda5bf15228c1f198176cb40bfd68cfb7900fc`, qualified public source `939789452e969e18d9febfe557c1e1d075f2446e`; frozen map `719cc795e6972e95b7b19e29f9a92b2ae6b21e4833a54cdac9cfe8f9057f16ea`. A CPU-only audit reproduced the closed analysis, independently reduced the three primary paired bootstraps, verified all 17 selected weight hashes, nine terminal recovery hashes and ten witnessed release charges. No document text was reopened. Audit output SHA `538811441a9fb0dba5502a10d6b9ca550a179c463003b623a86433794f6b8b0d`. The 377 test documents are spent; the other 663 short documents were not opened.
