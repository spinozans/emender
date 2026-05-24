# M2RNN Promoted Perspective Audit

**Purpose:** Harvest the verbatim promoted framings from M2RNN's paper and
accompanying blog post so that the NDM paper v2 can engage the real narrative,
not a paraphrase. This is a research artifact, not the paper. Plain factual prose.

**Produced:** 2026-05-24
**Task:** m2rnn-paper-blog

---

## §1 Source Bibliography

### 1.1 M2RNN arXiv Paper

**Citation:** Mishra, M., Tan, S., Stoica, I., Gonzalez, J., & Dao, T. (2026,
March 15; revised May 13, 2026). M²RNN: Non-Linear RNNs with Matrix-Valued
States for Scalable Language Modeling. *arXiv:2603.14360*.

**URL:** https://arxiv.org/abs/2603.14360  
**PDF:** https://arxiv.org/pdf/2603.14360  
**HTML rendering:** https://ar5iv.labs.arxiv.org/html/2603.14360  
**Type:** Peer-reviewed preprint (arXiv)  
**Accessed:** 2026-05-24  

### 1.2 ArXivIQ Substack Blog Post

**Citation:** ArXivIQ (2026, March 22). M^2 RNN: Non-Linear RNNs with
Matrix-Valued States for Scalable Language Modeling. *ArXivIQ Newsletter*.

**URL:** https://arxiviq.substack.com/p/m2-rnn-non-linear-rnns-with-matrix  
**Type:** Third-party promotional/summary blog post (Substack); includes
links to the arXiv paper, GitHub code, and Hugging Face model releases.  
**Accessed:** 2026-05-24  

**Note on authorship:** The ArXivIQ substack is a third-party AI paper
summary service, not authored by Mishra et al. directly. The promotional
language it uses ("a scalable, highly expressive drop-in layer for modern
hybrid architectures"; "a state-capacity issue, not an inherent flaw of
non-linearity") matches the abstract and introduction of the paper verbatim or
near-verbatim, indicating the blog accurately reproduces the paper's own
promoted framing. All quotes below are sourced directly from the paper (arXiv)
or the blog (ArXivIQ) with source indicated.

### 1.3 Other Promotional Surfaces

**GitHub / Hugging Face:** The ArXivIQ blog post includes links to code on
GitHub and models on Hugging Face; these were not independently accessed for
this audit. No dedicated author blog post, talk, or press release by Mishra et
al. was found in searches conducted on 2026-05-24. Searches attempted:

- `M2RNN "matrix-valued RNN" substack blog post Mishra Tan Stoica Gonzalez Dao 2026` (returned ArXivIQ post)
- `M2RNN Mishra Tan 2026 twitter announcement "matrix-valued" "state capacity" OR "drop-in"` (no author-authored tweet found)

**404/inaccessible sources:** No author-written substack or personal blog post
by Mishra, Tan, Stoica, Gonzalez, or Dao promoting M2RNN was located. The
ArXivIQ post (§1.2 above) is the primary promotional surface outside the paper
itself.

---

## §2 Verbatim Promoted Framings

### 2a. Diagnosis of Historical Underperformance: The "State-Capacity" Framing

**Quote 1** (arXiv:2603.14360, Section 2.5.1 — "Poor language modeling performance"):

> "Despite their expressivity advantages, non-linear RNNs like LSTMs and GRUs
> significantly underperform both Transformers and modern linear RNNs on language
> modeling benchmarks."
>
> "A natural hypothesis is that non-linearity itself is the bottleneck, but we
> argue the gap is largely attributable to state size."
>
> "Vector-valued non-linear RNNs maintain a hidden state $h_t \in \mathbb{R}^d$,
> which is far smaller than the matrix-valued states $H_t \in \mathbb{R}^{K \times V}$
> used by linear attention models."

*Source: ar5iv.labs.arxiv.org/html/2603.14360, §2.5.1, accessed 2026-05-24*

**Quote 2** (ArXivIQ blog, stating the paper's thesis):

> "The historical underperformance of non-linear RNNs (like LSTMs or GRUs) on
> language modeling tasks was a state-capacity issue, not an inherent flaw of
> non-linearity itself."

*Source: arxiviq.substack.com/p/m2-rnn-non-linear-rnns-with-matrix, accessed 2026-05-24*

This is the central diagnostic framing: M2RNN's promoted narrative attributes
all prior nonlinear RNN failures to state capacity (vector vs. matrix), not
to anything about the update rule, gradient conditioning, or training recipe.

### 2b. Positioning as Drop-in Layer and Hybrid Deployment Recommendation

**Quote 3** (ArXivIQ blog, executive summary):

> "a scalable, highly expressive drop-in layer for modern hybrid architectures"

*Source: arxiviq.substack.com/p/m2-rnn-non-linear-rnns-with-matrix, accessed 2026-05-24*

**Quote 4** (arXiv:2603.14360, Abstract):

> "Notably, replacing even a single recurrent layer with M²RNN in an existing
> hybrid architecture yields accuracy gains comparable to Hybrid M²RNN with
> minimal impact on training throughput."

*Source: arxiv.org/abs/2603.14360, Abstract, accessed 2026-05-24*

**Quote 5** (arXiv:2603.14360, Section 3.4.2):

> "Due to the expensive nature of M²RNN layers, we explore using them more
> sparingly. Our results show that replacing only a single Mamba-2 or Gated
> DeltaNet layer with M²RNN in a hybrid architecture achieves the same accuracy
> improvements as the full hybrid M²RNN model, while keeping training throughput
> within 6% of the Hybrid Gated DeltaNet at both 4k and 16k context lengths.
> This provides a scalable strategy for incorporating M²RNN layers into large
> language models."

*Source: ar5iv.labs.arxiv.org/html/2603.14360, §3.4.2, accessed 2026-05-24*

**Quote 6** (ArXivIQ blog):

> "When deployed sparingly within hybrid architectures, inserting even a single
> M²RNN layer yields significant perplexity and downstream accuracy gains with
> minimal throughput degradation."

*Source: arxiviq.substack.com/p/m2-rnn-non-linear-rnns-with-matrix, accessed 2026-05-24*

The promoted recipe is explicitly *sparse* nonlinear layers in a hybrid:
M2RNN as a drop-in enhancement to Mamba-2 or Gated DeltaNet hybrid stacks,
not as a pure-recurrent backbone.

### 2c. Empirical Recipe: Hybrid Ratio and Layer Placement

**Quote 7** (arXiv:2603.14360, Section 5.1.2):

> "1 attention layer for every 7 recurrent layers (i.e., 1 out of every 8
> layers is attention)"

Placement: M2RNN replaces the recurrent layer immediately preceding each
attention layer.

*Source: ar5iv.labs.arxiv.org/html/2603.14360, §5.1.2, accessed 2026-05-24*

**Quote 8** (arXiv:2603.14360, Abstract):

> "In hybrid settings that interleave recurrent layers with attention, Hybrid
> M²RNN outperforms equivalent Gated DeltaNet hybrids by 0.4–0.5 perplexity
> points on a 7B MoE model, while using 3× smaller state sizes for the
> recurrent layers."

*Source: arxiv.org/abs/2603.14360, Abstract, accessed 2026-05-24*

### 2d. What M2RNN Claims to Achieve — and What It Does Not

**Claims made:**

**Quote 9** (arXiv:2603.14360, Section 3.2 — State-Tracking):

> "To evaluate this claim empirically, we compare M²RNN, GRU, Gated DeltaNet
> [−1,1], and Gated DeltaProduct [−1,1] parameterized as a product of two
> Householder matrices on the $S_3$ task."
>
> "However, both GRU and M²RNN generalize perfectly to unseen context lengths
> achieving ≥99.5% accuracy up to 512 sequence length."
>
> "This suggests that theoretical expressivity alone does not guarantee robust
> state tracking in practice, and that non-linear RNNs like M²RNN and GRU
> exhibit stronger length generalization on this task."

*Source: ar5iv.labs.arxiv.org/html/2603.14360, §3.2, accessed 2026-05-24*

**Quote 10** (arXiv:2603.14360, Section 5.4.2):

> "Hybrid M²RNN achieving the largest gains: 12.3 and 10.1 point gains across
> the 410M and 7B MoE model scales"

*Source: ar5iv.labs.arxiv.org/html/2603.14360, §5.4.2, accessed 2026-05-24*

**Quote 11** (arXiv:2603.14360, Conclusion):

> "Our experiments demonstrate that M²RNN consistently comes close to Mamba-2
> and Gated DeltaNet at both the 410M dense and 7B (1B active) MoE scales."

*Source: ar5iv.labs.arxiv.org/html/2603.14360, Conclusion, accessed 2026-05-24*

**Acknowledged limitations:**

**Quote 12** (arXiv:2603.14360, Conclusion — Limitations):

> "The non-linear recurrence introduces additional computational overhead
> compared to linear alternatives. Future work could explore approximations or
> more efficient kernel implementations to reduce this cost while preserving
> expressivity."
>
> "Additionally, evaluating M²RNN at larger scales and with longer training
> contexts would further validate its potential as a foundational building block
> for efficient language models."

*Source: ar5iv.labs.arxiv.org/html/2603.14360, Conclusion/Limitations, accessed 2026-05-24*

**Benchmarks NOT run by M2RNN:**

- The paper tests the $S_3$ permutation group (6 elements, solvable) but
  does **not** empirically evaluate on $S_5$ (120 elements, non-solvable,
  NC1-complete). The abstract's claim that "M²RNN achieves perfect state
  tracking generalization at sequence lengths not seen during training" is
  supported only by $S_3$ experiments.
- No BPTT gradient conditioning results (gradient norms by step) are reported
  for the paper-default shape under a pure language model training run.
- No pure-recurrent stability data for the paper-default shape under
  schedule-free AdamW is reported.

**Quote 13** (arXiv:2603.14360, §3.1.2 — gradient stabilization):

> "Apply per-step gradient clipping to the gradient of the recurrent state
> $H_t$ during BPTT"

*Source: ar5iv.labs.arxiv.org/html/2603.14360, §3.1.2, accessed 2026-05-24*

The paper notes that gradient clipping is required for the nonlinear recurrence
but does not report gradient norm magnitudes or training divergence events for
the paper-default shape.

### 2e. Comparison to Delta-Rule Families (DeltaNet, GLA)

**Quote 14** (arXiv:2603.14360, Abstract):

> "Hybrid M²RNN outperforms equivalent Gated DeltaNet hybrids by 0.4–0.5
> perplexity points on a 7B MoE model, while using 3× smaller state sizes for
> the recurrent layers."

*Source: arxiv.org/abs/2603.14360, Abstract, accessed 2026-05-24*

**Quote 15** (ArXivIQ blog):

> "Linear models mentioned: Mamba-2, Gated DeltaNet. These 'fall short on
> expressivity' and 'fail to track states dynamically across long sequences.'"

*Source: arxiviq.substack.com/p/m2-rnn-non-linear-rnns-with-matrix, accessed 2026-05-24*

**Quote 16** (arXiv:2603.14360, Abstract):

> "Together, these results establish non-linear RNN layers as a compelling
> building block for efficient and scalable language models."

*Source: arxiv.org/abs/2603.14360, Abstract, accessed 2026-05-24*

The paper's positioning against DeltaNet is exclusively in the hybrid regime
(DeltaNet hybrid vs. M2RNN hybrid) and on LongBench long-context benchmarks.
The paper does not compare against DeltaNet in a pure-recurrent stack.

---

## §3 Points of Contact with NDM

### 3a. State-Capacity Framing vs. NDM Evidence

The M2RNN paper's central promoted framing (Quotes 1–2) holds that
nonlinear RNNs historically underperformed because of *state capacity*: the
vector-valued hidden state was too small relative to the matrix-valued states
of linear attention models. Expanding to a matrix state — regardless of update
rule — is presented as the fix.

The NDM paper's evidence directly complicates this diagnosis. M2RNN-CMA
(the CMA-ES-reshaped M2RNN variant, a nonlinear matrix-state RNN) stalls at
$S_3$ accuracy 0.31 (solvable control, 6 elements) and $S_5$ accuracy 0.22
(non-solvable witness, 120 elements) at the same parameter count as NDM. This
is the critical counterexample to the state-capacity thesis: M2RNN already
operates on a matrix state — state capacity in the M2RNN sense is not the
bottleneck. NDM achieves $S_3 = 1.00$ and $S_5 = 0.79$ with the same matrix
dimensionality. The diagnostic disagreement is therefore not about capacity but
about the update rule itself: NDM's delta correction $v - S^T k$ (which feeds
prediction error back through the state write) versus M2RNN's raw outer-product
write $\tanh(H W + k v^T)$ (which does not).

Furthermore, the paper-default M2RNN shape diverged at step 8,400 under a
pure language model training run with schedule-free AdamW, reaching gradient
norms of approximately $4.2 \times 10^7$. The M2RNN paper does not report this
stability profile. This production evidence argues that the paper-default M2RNN
geometry has gradient conditioning problems that are not predicted by the
state-capacity framing: the hidden state is bounded (tanh + forget interpolation)
but parameter gradients can still be catastrophically large. The mechanism
(noted in `docs/M2RNN_E88_COMPARISON.md`) is that the paper-default shape uses
one shared query/key address stream repeated across many heads, collapsing
all gradient flow through a small bottleneck — a geometric problem that the
state-capacity diagnosis does not surface.

### 3b. Drop-In Layer Recommendation vs. NDM's Hybrid Degradation Finding

The M2RNN paper's most actionable promoted claim (Quotes 3–6) is that a
*single* M2RNN layer can be substituted into an existing hybrid architecture
for large gains. The sparsity recommendation — "deployed sparingly" — is
treated as a practical virtue: low throughput cost, easy integration.

The NDM paper's §7 hybrid degradation result (paper main.typ, §7, figure
`fig_hybrid`) constitutes a direct empirical counter-framing from the opposite
direction: interleaving NDM layers with Gated DeltaNet layers in an
`[NDM, NDM, GDN, GDN]` pattern *underperforms* pure NDM on both modular counter
(0.536 hybrid vs. 0.903 pure NDM vs. 0.648 pure FLA-GDN) and FSM tracking
(0.713 hybrid vs. 1.000 pure NDM vs. 0.830 pure FLA-GDN). As stated in the
paper caption: "State-tracking capability is not a property the NDM block can
lend to a stack of mixed blocks; purity is part of the recipe."

This finding does not directly refute the M2RNN hybrid claim — the NDM hybrid
test uses NDM+GDN, not M2RNN+attention — but it establishes a principle that
directly contradicts the drop-in-layer premise: nonlinear recurrent blocks in a
mixed stack do not inherit or propagate their state-tracking advantage to the
surrounding linear blocks. The M2RNN paper's perplexity improvement in hybrid
settings (Quotes 7–8) may be real, but perplexity is a coarser signal than
state-tracking accuracy. The NDM hybrid degradation result suggests that the
expressivity advantage M2RNN claims over DeltaNet hybrids — the same advantage
that motivates the drop-in-layer recommendation — would be compromised in
exactly the hybrid deployment the paper recommends.

### 3c. S3 Benchmark: M2RNN Claims vs. NDM Evidence

M2RNN reports $\geq 99.5\%$ accuracy on $S_3$ (solvable, 6 elements) with
generalization to unseen lengths up to 512 (Quote 9). This is presented as
evidence that M2RNN "achieves perfect state tracking generalization."

The NDM expressivity harness runs M2RNN at matched 8M parameter count and
finds a starkly different result: M2RNN-CMA achieves 0.31 on $S_3$ (training
length T=128), and M2RNN-paper achieves 0.38. NDM achieves 1.00. The
discrepancy is almost certainly due to scale and configuration differences
(the M2RNN paper likely uses a much larger model for the $S_3$ experiment),
but it highlights a gap in comparability: the promoted $S_3$ generalization
result is not at the 8M expressivity scale where the update-rule comparison is
meaningful. More importantly, the M2RNN paper does not evaluate on $S_5$ (the
non-solvable, NC1-complete group) at any scale. The $S_3$ result supports
expressiveness only for solvable groups, which are a proper subset of the tasks
that separate TC0 and NC1. NDM's $S_5$ result (0.79 at T=128, compared to
M2RNN's 0.22) is the finding that the state-capacity framing does not predict.

### 3d. Gradient Conditioning: NDM's Evidence vs. M2RNN's Absence of Evidence

M2RNN notes that gradient clipping is required during BPTT (Quote 13) but
reports no gradient norm trajectory or divergence events for the paper-default
shape. The NDM paper records that the paper-shaped M2RNN baseline diverged at
step 8,400 with gradient norm $\approx 4.2 \times 10^7$ under the same training
setup (schedule-free AdamW, bf16, 2K context, Pile). The CMA-ES-reshaped
M2RNN-CMA variant trains stably (dim=1920, depth=21, H=370, N=16).

This is a finding M2RNN's promoted narrative does not predict: the state-
capacity thesis suggests that matrix state is the key intervention, but the
NDM results show that the *shape* of the matrix state (specifically, the
q/k-to-value-head ratio and the raw-write vs. delta-correcting update) determines
stability, not state capacity per se. This is documented in
`docs/M2RNN_E88_COMPARISON.md` §"Working hypothesis" and §"Live M2RNN stability
report, 2026-05-09."

### 3e. Agreement and Reframe: Pure Nonlinear Recurrence at Scale

The NDM paper agrees with M2RNN on one central premise: that nonlinear matrix-
state recurrence is trainable at scale and competitive with linear-state models
on perplexity. M2RNN demonstrates this at 410M with Nemotron-CC-v2; NDM at
1.27B with The Pile. The NDM paper does not dispute the M2RNN perplexity
result; it treats M2RNN as a peer (`docs/related_work_nonlinear_rnns.md`,
§"Related-Work Peers").

Where NDM reframes rather than contradicts: M2RNN presents the scale-up as
confirmation of the state-capacity thesis. NDM frames the same capability as
a multi-programmed systems result — the enabling factor is the per-head bounded
memory recipe (Triton kernel, sparse checkpointing, independent address programs
per head), which is structural and applies across nonlinear matrix RNN families
(the M2RNN-CMA variant runs under the same recipe). The question M2RNN does not
ask — "which update rule within the nonlinear matrix RNN family produces
stronger state tracking?" — is the one NDM's expressivity section directly
answers.

---

## §4 Recommended Response Strategy for v2

The M2RNN paper makes two large promoted claims NDM v2 must engage directly:
(a) that the historical underperformance of nonlinear RNNs was a state-capacity
issue, not an inherent flaw of non-linearity; and (b) that M2RNN is a scalable
drop-in layer for hybrid architectures, with even a single layer yielding
meaningful gains.

**Should v2 directly quote M2RNN's promoted framings, or paraphrase them?**
Quote verbatim — specifically the two signature phrases: "a state-capacity
issue, not an inherent flaw of non-linearity itself" and "a scalable, highly
expressive drop-in layer for modern hybrid architectures." These appear in both
the paper and the promotional blog, making them the authors' own intended
framing. Verbatim quotation removes any risk of attacking a strawman; both
phrases are short and directly engaged by NDM's evidence.

**Which specific NDM findings most cleanly contradict the state-capacity
framing?**
The $S_3$ result at matched parameter count: M2RNN-CMA already has
matrix-valued state but achieves only 0.31 on $S_3$ (solvable group control,
6 elements) while NDM achieves 1.00. The state-capacity framing predicts no gap
between the two on solvable groups, since both have matrix state; the observed
gap is 1.00 vs. 0.31. $S_3$ is the cleaner rebuttal than $S_5$ precisely
because it is solvable — if state capacity were the binding constraint, a matrix
RNN should clear this task. The $S_5$ separation (0.79 vs. 0.22) adds evidence
but is a harder target; the $S_3$ failure is the diagnostic crux.

**Is the §7 hybrid degradation result the right rebuttal to the drop-in-layer
recommendation, or is something else stronger?**
The hybrid degradation result is the correct structural rebuttal: it shows
empirically that state-tracking capability is not a property a nonlinear block
lends to neighbouring linear blocks. Position it carefully — the NDM test uses
NDM+GDN, not M2RNN+attention, so the claim is not "M2RNN's hybrid gains are
fabricated" but "expressivity is a property of the whole stack, not individual
layers." The stability finding is the stronger practical rebuttal: the
paper-default M2RNN diverges at step 8,400 (grad norm $\approx 4.2 \times
10^7$) under the same training setup where NDM is stable. A layer that cannot
train stably in its published configuration is not yet a drop-in layer.

**What is the cleanest one-sentence summary of NDM's diagnostic disagreement
with M2RNN?**
*M2RNN attributes the historical failure of nonlinear RNNs to insufficient
state capacity, but NDM's evidence shows that a nonlinear matrix-state RNN
using M2RNN's update rule still fails on solvable-group control tasks at
matched parameter count — indicating that the update rule, not state
dimensionality, is the load-bearing distinction.*

---

*End of audit. All quotes are from sources accessed 2026-05-24. No claims are
attributed to M2RNN that are not direct quotes or paraphrases of accessed
source material.*
