# Attention over a corpus, with optional local memory

**Define the answer as if the query could see every admissible memory row. Then approximate the computation, not the definition of the memory.** A MEM (maximal exact match) can be a good way to *find* rows worth expensive attention without being a condition for a row to have any attention weight. This is a design reference, not a change to the four-page natural panel.

## What is a row? What is the query?

At prediction time `t`, the live host forms `q_t = Q(x_≤t)`. The query always knows the observed local context. Independently choose which sources to **expose as keys/values**:

- `C`: the complete allowed external corpus. For a predictive index, put one row at each eligible **within-document token boundary**. The key represents the context ending there. The simplest value is a point mass on its recorded successor token. Other choices include a distribution over extensions, an embedded span, or what a passage writes into a reader. Ends are structural unless the token model explicitly scores EOS.
- `L_t`: optional local memory. For a successor value at local boundary `j`, require `j+1 ≤ t`. For ordinary hidden-state values, rows through `t` may be used if their entire computation is causal. Never expose `x_(t+1)` or the unobserved remainder of the live document as index data.
- Optionally expose a single **no-read** row with a learned or calibrated logit. Its value can be zero for a neural hint, or the base LM distribution in a token mixture.

Thus corpus-only, local-only, or local **and** corpus use the same operation. A local recurrent query is not the same as exposing local K/V rows. This is cross-attention to specified, document-reset memory encodings, not a claim to reproduce running an arbitrary transformer on the concatenation of every document.

## The exhaustive target

For a row `i`, let `b_i > 0` be its base mass. A concrete dense reference is `s_ti = q_tᵀ k_i / √d`, `w_ti = b_i exp(s_ti)`. Other kernels are possible, but name the reference before claiming an approximation to it. For each source `s` compute

```text
Z_s = Σ_{i∈s} w_ti                   (how much attention mass?)
U_s = Σ_{i∈s} w_ti v_i               (what answer does that mass carry?)
Z   = Σ_s Z_s
α_s = Z_s / Z
r_t = (Σ_s U_s) / Z = Σ_s α_s r_s,   r_s = U_s / Z_s for nonempty s.
```

A source can return **`(log Z_s, r_s)`** without returning its rows. Softmax the source log masses to obtain the exact joint read. Empty support has zero mass, not an identity passage. If the only exposed source is `C`, its share is always one: use local or no-read competition to ask “how strongly does this query favor the archive?” A learned source gate after separate softmaxes is a different rule unless it recovers these shares. `α_C` is attention mass, not proof that using the archive improves loss.

With one-hot successor values, the read is a distribution: `p(v) = Σ_i (w_ti/Z) 1[successor(i)=v]` (plus the no-read row's weighted LM law if enabled). With vector values, the live token head consumes `r_t` and need not copy a word. With `v_i(S_t)=F_i(S_t)` the same weighting defines a **generalized attention read with live-conditioned values**; ordinary static-value attention is its special case.

**Multiplicity matters.** Under ordinary per-row softmax (`b_i=1`), 100 corpus rows of score 0 compete evenly with one local row of score `ln 100`. Keeping four corpus rows and renormalizing gives the corpus `4/104`, not `1/2`. Exact grouping of the 100 tied scores needs their count and mean value, not 100 transfers. If database size should not confer this prior, one alternative is `b_i=π_s/|s|` within each nonempty source. That changes the target to a source-balanced reference; it is not ordinary per-row attention. Duplicate bytes, equal scores and equal values are three different notions.

## How MEMs and their extensions can implement an approximation

For direct copy evidence prefer occurrences aligned to the **current prediction boundary**. An internal MEM can propose relevant documents, move attention among boundary rows, or supply evidence to a learned reader. A semantic key can deserve high weight with no exact match at all. Union overlapping proposals by row identity. A continuation handle can expose the conditional distribution beyond its branch; distributing one row's mass among possible extensions should not silently create extra mass at every depth.

An exact **lexical** baseline illustrates the distinction. Give every endpoint weight `a₀ > 0`, then add `a_ℓ ≥ 0` when its context matches the last `ℓ` observed tokens. For matching suffix intervals `I_ℓ`, `Z_C = a₀ |C| + Σ_ℓ a_ℓ |I_ℓ|`; the numerator uses the corresponding value sums. Use every supported suffix length without a top-k page cutoff. The intervals are intentionally nested: each endpoint has one weight equal to its background plus its length bonuses. Counts and follower histograms suffice for successor values under this law, not for arbitrary semantic attention.

One concrete approximation to dense dot-product attention:

1. Give **every corpus row** a cheap positive background score `ŵ_i = b_i φ(q_t)ᵀ ψ(k_i)`. For fixed keys and values, precompute `Σ_i b_i ψ(k_i)` and `Σ_i b_i ψ(k_i) v_iᵀ`. Two contractions give the whole corpus's approximate `(Ẑ, Ũ)` without querying each row separately.
2. Let `H_t` be the union of addresses found by MEMs and selected extensions. Use the **true** attention score on these rows. Correct *both* sums: `Ẑ_C = Ẑ + Σ_{i∈H_t}(w_i−ŵ_i)` and `Û_C = Ũ + Σ_{i∈H_t}(w_i−ŵ_i)v_i`. This replaces the background contribution; it does not add each match twice. Non-hit rows still contribute. Local attention can stay exact and merge with the corrected corpus mass. At full coverage this recovers the dense reference; at partial coverage it is only as good as the remaining background. The features need to approximate the exponential dot product for this to approximate **that** attention, not merely to be positive.
3. If arbitrary per-row values or neural scores prevent such summaries, another option is exact matched rows plus probability-sampled residual rows with known, nonzero inclusion probabilities. Estimate numerator **and** denominator; their normalized ratio is not generally unbiased. Alternatively, explicitly choose a sparse lexical kernel as the target. That is exact attention for a different scoring law, not evidence of matching dense semantic attention.

There is no promise of sublinear exact work for arbitrary keys, values and queries. The global feature/value table also has a price: feature rank × value width, potentially very large for a full token distribution. With GDN‑2's fixed-control affine values one may summarize coefficients and contract with the current seed. For arbitrary nonlinear E97 seeded values a static value mean is not sufficient. Any learned corpus summaries have a weight version; rebuilding or differentiating through them during co-training is a separate cost. A stale detached summary is not an exact current-weight training pass.

[Five invented CPU checks](test_corpus_attention_law.py) verify the 100-row mass example and truncation identity, exact suffix-interval summation, and the global background correction for no, partial and full refinement. They check the algebra, **not** how accurate learned features are or how fast a corpus-scale index runs.

The [CPU implementation and qualification](corpus_kernel/README.md) now exercise this on actual token occurrences. The indexed path sums disjoint exact rows and positive tail summaries to avoid cancellation. The [Triton port](corpus_kernel/TRITON.md) implements batched interval lookup and tiled read/first backward. Optional internal MEM proposals advance to a within-document boundary aligned to the current frontier before scoring. The [all-frontier training path](corpus_windows/TRITON.md) shares one current-weight bank across every real shifted prediction in a batch of document windows, including its backward pass. These implementations are separate from the registered natural panel.

On a [co-trained invented alias task](semantic_tail/RESULT.md), the corpus head followed two-key archive edits even when **no** live token occurred in the archive. The indexed reader used only the feature background. A frozen swap of indexed and dense laws substantially altered predictions; learning a useful background is not the same as reproducing dense attention. Several fits decoded the correct token without ranking its row first. On a [separate independent-edit test](binding_edits/RESULT.md), three of those permutation-trained fits moved the new-answer odds in the wrong direction when **only its own row** changed. With repeated-value co-training, every dense and indexed seed passed both independent rewrites and same-bag swaps; neither lexical hits nor unique values were required.

**Real language compression:** the [raw-byte pilot](language_compression/RESULT.md) co-trained LOCAL, suffix COUNT, exhaustive and indexed readers on identical one-pass text. On 524,416 fresh probe bytes they required **3.59961, 3.59429, 3.73405 and 3.75254 ideal bits/original byte**, respectively. Both learned corpus readers lost to LOCAL. The fused Triton path accelerated a complete indexed optimizer step **1.60× over eager indexed PyTorch**; on the natural fits exact GEMM was faster than indexed. These are two distinct results.

## What would demonstrate approximation?

First compute exhaustive attention (tiled if necessary) on a tractable memory with **the same** rows, priors, encodings, query, temperature and masks. Compare `log Z_C`, local/corpus/no-read shares, attention mass on omitted rows, and errors in the weighted read and next-token distribution. For a pure truncation renormalized on retained rows, the total variation distance from full attention **over row IDs** equals the omitted mass `ε`; for values of norm ≤ `B` the read error is at most `2Bε`. Without a bound or estimate of that mass, a top-k list supplies no such certificate. Test high-entropy queries, ties, frequent weak matches, rare anchors, nonlexical matches and empty local memory. Separately ask whether the co-trained approximation improves held-out next-token loss at its measured training and serving price. Those are different tests.
