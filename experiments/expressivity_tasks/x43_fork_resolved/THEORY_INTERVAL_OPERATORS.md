# Theory note: the interval-operator algebra over the corpus, and LLMs as its approximations

Recorded Nov 5, 2026, from the operator's design discussions (this channel).
Git is the binding; no predictions here are pre-registered — those get bound
per-experiment when run. Receipt references are to the committed ledger.

## Part I — The algebra

### The ground structure

Over corpus T, the suffix-array/xsa machinery gives, for any byte pattern P,
the interval I(P) of suffixes prefixed by P, with exact occurrence counts and
exact left/branch extension structure. Backward search, branch enumeration,
and the suffixient (fork) structure are the exact primitives. A branch query
returns the exact next-byte distribution over the corpus given P — measured
at 22.7k contexts/s with zero corpus reads (serving receipts, X43).

### The LM as dynamics of a weighted interval set

A language model's state at any moment is formalized as a weighted set of
intervals S = {(I1,w1)..(Ik,wk)} — a set of hypotheses about where the
current text sits in the corpus's pattern space. The LM is the dynamics of S:

| Operator | Interval-set semantics | Realization |
|---|---|---|
| EXTEND(a) | advance all patterns by emitted byte a | exact (index) |
| BRANCH | expand an interval into its extension children with counts | exact (index) |
| SELECT/PRUNE | keep the intervals that matter | policy — the learnable seat |
| SUBSUME | collapse a too-specific low-support interval to its parent | deterministic |
| EXPIRE | remove intervals that can no longer anchor the context | deterministic |
| UPDATE | reweight on contradicting evidence | rule-given |
| EMIT | next-byte distribution = weight-mixture over the set's branches | learned readout |

The registered consultation ladder (16/64/256/1024-byte suffix supports) is
the degenerate case: one context, nested intervals, fixed weights. The
operator view generalizes to arbitrary — including memory-composed —
patterns with learned weights under a bounded k.

### The particle-filter identity

A weighted hypothesis set, propagated exactly (EXTEND/BRANCH), reweighted by
evidence (UPDATE), resampled under a budget (MERGE/EXPIRE/SELECT), is a
particle filter. The proposal: **the LM is a particle filter over the
corpus's decision space, in which the xsa makes propagation exact.** The
χSA bounds (chi << r <= n; the decision structure is sparse) bound the state
complexity the filter needs — the same bound that makes a bounded slot store
sufficient.

### The emender realization (already built)

The five memory ops realize the algebra's set dynamics:

| Emender op | Interval-set meaning |
|---|---|
| APPEND | a hypothesis interval enters the set |
| CORRECT | UPDATE on contradiction (residual learning) |
| POP | EXPIRE by pattern ownership (implemented as the B1 suffix-ownership component) |
| DECAY | staleness weighting |
| MERGE | consolidation of indistinguishable hypothesis classes |

The physical slot budget is the beam width k (the frontier size). PB3's
occupancy telemetry measured sampled frontier size; removing the budget cost
4.96x per step, with peaks 81/110/81/96 and nonbinding cap 257. Nearly identical
early sampled means do not establish frontier bloat as the cause of that cost
or a live-occupancy scaling law. Learning lives only in the weights (frozen basis,
learned coordinates; routing detached) — which the algebra identifies as
SELECT's policy and EMIT's readout. The zero-learned-management spine (F12)
is the theory's natural shape, not a constraint.

Receipt supports: C9 tier-1 (suffix backoff = SUBSUME, succeeds where
full-context re-entry fails); the exact-walk failure (14% forced continuations
= EMIT without UPDATE); PB4 coupling (weights that cannot distinguish
"interval present" from "interval informative"); C14-B measured triangle: budgeted 1.584700014, unbudgeted 1.716496444,
append-only 2.246020115, trained on 71,677,042 / 14,455,380 / 75,394,452
byte targets respectively. The 0.661320101 budgeted advantage over append-only
supports lifecycle value in this fixed-time recipe, without isolating operations.
Unbudgeted achieves 1.716496444 with roughly fivefold less data: the exposure
confound cuts in its favor as data-efficiency evidence. Budget pressure itself
cannot be separated cleanly from lifecycle value, exposure, capacity and trajectory;
that historical unresolved status is superseded by the matched twin below. Full strata and
limits: [RESULT_C14_ABLATION_TRIANGLE.md](RESULT_C14_ABLATION_TRIANGLE.md).

Single-seed exposure accounting: [data-efficiency receipt](RESULT_C14_DATA_EFFICIENCY.md)
reports all six endpoints. Unbudgeted has 9.045x the passive dense-anchor NLL
improvement per training target; this endpoint statistic is not matched-loss
sample complexity. Heavy trained compute and budget-pressure causality remain
confounded.

Measured budget/beam-width verdict (PT1–PT3): the single-seed matched twin
reaches **1.565270853**, **−0.019429161** nats/byte versus budgeted at exactly
71,677,042 targets. The registered negative sign favors compute shaping rather
than additional curation value from pressure in this recipe. This is a small
margin without uncertainty, not a general result about optimal beam width.
Allocated capacity, implementation, trajectory and 5.96x active training time
remain confounded. Early 9.045x dense-anchor improvement per target saturates
to 2.3170x passive's endpoint normalization, while translating to slightly better
matched-exposure loss; no sample-complexity law follows. Aggregate lifecycle
value versus append-only (0.661320101) remains supported in the fixed-time
recipe; append-only's exposure is not exactly matched. All 66 rows, reset/interior
and support/fork breakdowns, kernel equivalence and terminal coverage are in
[the matched harvest](RESULT_C14_B_MATCHED.md).

## Part II — LLMs of other types as approximations of the algebra

### The general claim

Training never gives a model the algebra — only the corpus as a linear
stream. The only place the corpus's exact interval structure can accumulate
is the weights. Therefore: **trained LLM weights are compressed, lossy
approximations of the suffix-array operator algebra over the corpus they
trained on, learned without access to it.** The proposed representational
limit is exactness where the algebra is trivial (single-branch / forced spans)
and approximation where it is fork-dense. Finite training need not attain
that limit: bound experiment B below refutes near-perfect forced recall at
its registered budget, while retaining an entropy-dependent accuracy ordering.

### Family-by-family mapping

The terminal comparison of these families is proposed in the
[six-arm 1.3B registration DRAFT](PREREGISTRATION_1P3B.md). It is awaiting the pure
nomination and operator sign-off; this pointer records no scale result or launch
authorization.

- **Transformer (dense)**: KV cache = a per-document dynamic index, built
  exactly at inference but unbounded (the program's receipted unbounded arm);
  learned attention keys = fuzzy matching = approximate intervals; induction
  heads (Olsson et al. 2022) = learned match-then-continue circuits =
  approximate backward search. C13's tier-1 re-copying observation is a
  scoped receipt, not an architecture law: **the head asymmetry inverts at
  Pile scale; the transformer is the better array-reader.**
  C14's registered Pile-scale consultation datum gives transformer device-on
  **1.266063829** versus dense **1.404428829** nats/byte: gain
  **+0.138364999**, compared with Emender's **+0.766873539**. Transformer P1
  **PASS**, P2 **FAIL** (device-off minus dense
  **+0.058468789**). The Emender supported-target advantage
  **does not hold** against a consulted transformer (0.717143862 versus
  0.811639725). This is a single-seed, equal-active-time comparison with unequal
  exposure, not an architecture-only causal claim. Full signs and strata:
  [C14 head-asymmetry verdict](RESULT_C14_TRANSFORMER.md) and
  [complete first-pass ladder](RESULT_C14_FIRST_PASS.md). The conditional
  array baseline is 0.663434002: supported-target distances are transformer
  +0.053709860, Emender +0.148205723. The array's total is infinite, so these
  are conditional distances, not distances to a total-LM oracle or a bound.
  Transformer soft stacking reaches **1.263566972** vs consulted alone
  **1.266063829** and dense+calibration **1.299507807**: soft P3 **PASS**,
  but the residual is only **−0.002496858** nats/byte (about 0.20%). The
  small residual after consultation replicates on the better reader. This
  is descriptive evidence consistent with most of this fusion recipe's
  benefit already captured by consultation; absorption into weights remains
  a causal interpretation, not a measured mechanism. Literal exact-walk P3
  still **FAILS**: 30,751,940 forced-zero targets give infinite total loss.
- **Emender with an explicit store and array**: the retained evidence is
  operator causality, unbounded-store data-efficiency, and bounded-state
  theory, with the scopes and matched-exposure receipts above. None of these
  entails being the best array-reader. A larger gain over a worse dense head
  does not rescue that claim. The linked frozen stacking comparison measures
  the residual after consultation; it cannot by itself identify what training
  absorbed or establish an architecture-only mechanism.
- **SSM / RNN / fixed-state**: the state is a fixed-budget compression of
  the interval set. Paper 1's theorem (fixed-state models of any size are
  automata in the finite-precision regime) reads here as: the approximation
  class is bounded by the state, not by the algebra. GDN2's proxy-scale wins
  and the budget-bound depth receipts are the empirical face.
- **kNN-LM / Memorizing Transformers**: explicit partial BRANCH with fuzzy
  learned keys; a datastore of (k,v) = APPEND without the rest of the
  algebra.
- **Infini-gram + neural interpolation**: exact BRANCH at a fixed ladder (the
  degenerate interval set); the measured interpolation gain is a direct
  measurement of the neural approximation's residual on exact branch counts.
- **Retrieval-speculative (REST)**: EXTEND/EMIT approximated by exact-match
  proposal with neural verification.
- **Constrained decoding (tries/automata)**: EMIT constrained by operators
  over a different structure (a grammar, not the corpus).
- **M2RNN / attention hybrids**: partial adoption of the unbounded operator
  (routing around the fixed-state wall) — the hybrid patch as an implicit
  admission of the algebra's necessity.

### The string-kernel face

An LM's context embeddings are a learned feature map over substring space.
The exact kernel — similarity of contexts by their continuation behavior —
is computable by the xsa operators (branch distributions are the feature).
Embedding similarity is the approximate shadow of interval similarity; the
alignment between the two is measurable (kernel-alignment probe).

### Signatures already in hand

- C11 copy law: trained on a stream inside its own index, the model's
  globally optimal solution is to BE the interval structure (verbatim recall).
  Memorization/dedup literature (Carlini et al.; Lee et al.) reframed:
  verbatim learning motivates the hypothesis of exactness on forced spans.
  C11 alone does not establish that finite training attains this limit.
- The 12-byte-median support regime at Pile scale: the shared short-suffix
  structure is what cross-corpus consultation transfers (74% supported
  targets) — the shallow rungs of the ladder are the corpus-independent part
  of the algebra.
- P2/PB4 device-off penalties: the weights' approximation of "features
  present" vs "features informative."

### Predictions (binding status recorded per item)

1. **Forced-span verbatim law — bound experiment B**: registered in
   `5da0c320`; see [RESULT_FORCED_SPAN.md](RESULT_FORCED_SPAN.md). The prediction
   is **refuted** on a 16-document, finite4-byte-context self index with
   a tier-1 neural emender trained for180 seconds. Final-panel byte accuracies
   forced/low/high are 0.809874/0.791501/0.689320; whole forced-run
   exact accuracy is 0.476804, against
   the registered .99 threshold. Both document-disjoint recall panels were
   in training; no unseen-document claim. This bounds the finite-budget
   prediction and does not establish maximal-suffix or asymptotic exactness.
2. **Interpolation-residual scale**: the NLL gain from fusing exact branch
   counts into a trained LM predicts the LM's approximation error on branch
   structure (the fusion receipts give the first points of this curve).
3. **Kernel-alignment probe**: context-embedding similarity correlates with
   suffix-behavior similarity, with correlation strength increasing with
   training exposure and decreasing with branch entropy.
4. **The approximation-distance ladder** (the paper figure): NLL against the
   exact algebra's oracle, by architecture class — dense (in-weights only),
   +KV (per-doc index), +fuzzy access (kNN), +exact fixed ladder
   (infini-gram-style), +explicit operators (ours). Every rung is already
   separately measured in this program's receipts except the ladder drawn as
   one figure.

### What this makes of the program

The fork-resolved architecture is the point where the model stops
approximating the algebra and starts holding it: the array carries the
operators exactly; the weights carry only what was always theirs to carry —
the weights. Paper 1 bounds what fixed-state approximations can ever do;
this note defines what all trained LLMs are approximating; the fork-resolved
system is the explicit realization. POSITIONING.md's empty intersection is
re-derived at a higher level: no neighbor defines the target computation the
way this algebra does.

## Provenance

Operator design discussions, Nov 4-5 2026: (i) the structured-search /
flanking-memory-pattern proposal (context extensions conditioned on the
store's state; the search state as an interval set); (ii) the operator
formalization request (LLM as operators over the xsa; back out the
operators; realize in emender, learnable only where the weights are);
(iii) the generalization to other LM families as approximations of
string-kernel / suffix-array kernel algebra over their training corpora.
Recorded by the assistant from that discussion; git is the binding.

## Part III — Tokenization, and compute per decision (recorded Nov 5, from the same discussion)

### Tokenization is the static approximation of the suffixient structure

A tokenizer (BPE/unigram) is a fixed, corpus-derived collapse of byte runs
into units — in algebra terms, a frozen SUBSUME/EMIT policy pre-baked from
corpus-average statistics: a "token" is the merge table's bet that a byte run
is usually forced (single-branch), so the model need not decide within it.
The suffixient structure gives exactly this per-context and exactly: the
chi-positions are where decisions exist; all other spans are forced. Byte
level + xsa therefore REPLACES tokenization with its exact, adaptive form:
free emission through forced spans (54% of Pile eval positions are forced —
114,751,649/214,097,022), decisions only at forks, "vocabulary" = the
corpus's local chi-structure instead of a merge table's guess. Tokenizer
pathologies read as approximation error: glitch tokens and arithmetic/
spelling failures are contexts where the frozen forced-span bet collides
with a live fork. The program's standing byte-level constraint is thereby
theoretically explained, not merely imposed: byte-level + xsa is the exact
system; a tokenizer is its lossy pre-compression.

### Thinking per decision, not per token

The algebra separates thinking from writing. The unit of compute is the
fork (where the loss mass lives: 66-82% of NLL); forced spans emit at zero
model cost. TEST-TIME COMPUTE = operator rounds over the interval set
between emissions: each round a BRANCH query (44us, exact counts), a
SELECT/UPDATE pass over the store (2.5ms/step measured), a hypothesis
extension. Thoughts live in the weighted interval set (the store is the
scratch space), not in emitted tokens — standard CoT externalizes its
working because a standard LM has no other state to think in. Accounting:
total compute ~ decisions x rounds-per-decision; with ~46% of positions
being decisions, several operator rounds per decision fit the same
envelope a token model spends single-passing every unit. Receipts already
on the rounds-vs-quality curve: single-round EMIT fails (the exact-walk's
14% forced-byte wrongness = EMIT without UPDATE); iteration helps (C9
suffix backoff). Candidate experiment (unbound): sweep bounded operator
rounds per fork at generation on frozen arms; receipt rounds and quality.

## Part IV — Multi-step before emit (recorded Nov 5, same discussion)

Three mechanisms: (1) micro-steps — run the block k times per input byte
with the store evolving across rounds, emit after round k (mechanically an
inner loop; the store is the inter-round accumulator); (2) operator rounds
at decisions only — BRANCH/UPDATE/SELECT per round at forks, forced spans
free between (C9 backoff is already a 2-round example: fail, re-query
shorter, succeed); (3) the inverse — one step, multiple emissions (forced
spans, 54% of positions). Architectural distinction: a transformer's only
inter-round state is its residual stream (CoT externalizes because there
is no other scratch); the emender's store IS the per-decision working
memory — thoughts land in slots as APPEND/CORRECT, EMIT is the conclusion.

Halting: learned halting (ACT/PonderNet) is a learned management decision
(F12 tension) and a gate subject to the receipted collapse law ("gates
that can abstain will abstain"). The deterministic alternative: iterate
until the store stops changing (no CORRECT fires, weight delta < epsilon),
bounded by k_max — the ops' event telemetry (already receipted in PB3)
is the halting signal itself.

Expressivity: rounds add compute, not state — the budget does not grow
with k, so the boundary theorems (Paper 1) are untouched; pondering in a
fixed-budget store buys serial depth without buying state. The transformer
alternative (growing KV) is the receipted unbounded arm. State is what the
theorems bound; thinking is what rounds buy.

Cost: k rounds at decisions only (~46% of positions), forced spans free:
k=3 rounds-at-decisions ~ 1.9x total compute for 3x serial depth at every
decision. This remains a proposed accounting envelope, not a measured speedup.

**Bound experiment A:** registered in `2c939baf`, inference-only store-carrying
machinery, frozen B1-v2 and dense emender, k={1,2,4}. See
[RESULT_OPERATOR_ROUNDS.md](RESULT_OPERATOR_ROUNDS.md) and the hashed receipts
in `receipts/theory_rounds/results.json`. The k=1 interface reproduces standard
outputs and losses byte-exact. On 169 forks in 512 calibration targets, B1 fork
NLL is 1.912022 / 2.012623 / 2.165866; dense is
2.451763 / 2.923047 / 3.554979. PA1's negative curve sign is refuted here.
PA2 is indistinguishable at the k=4 bound; PA3 shows increasing query volume,
without collapse. Extra serial depth with the existing frozen policy is not
automatically useful. These are teacher-forced positions with cached BRANCH
reads, scalar execution for k>1, layer-local halting, and no free-running or
state-conditioned branch search claim. Forced positions still receive one
model round. Failed attempts remain failures in the receipts.

The separate training experiment with fixed micro-steps k={1,2,4}, matched
protocol and actual steps/bytes, remains unbound and unexecuted.

Single-seed trained-heavy-compute counterpoint: unbudgeted reaches 1.716496444
on 14,455,380 targets with about 4.96x budgeted active time per position.
Unlike the frozen-policy inference-round intervention, its weights were trained
throughout that expensive computation. This does not establish useful serial
depth or trained-round benefits; implementation overhead and capacity remain
confounded. See [data-efficiency receipt](RESULT_C14_DATA_EFFICIENCY.md).

C14-B occupancy-heavy profile reaches the production-kernel boundary. The
257-slot control launches a padded 512-slot native tile with compiler-reported
n_spills=4078; eight native launches account for 84.86% of measured self CUDA
time, including checkpoint recomputation. A wrapper-only token-unbind change
is byte-exact on the real trained batch and a synthetic 128-live-slot probe;
replay work falls, but whole-step compute throughput does not improve in the
samples. This is implementation evidence, not intrinsic cost or a live-slot
scaling law. The matched-exposure twin now supplies the registered sign, with capacity,
implementation, compute and trajectory still confounded; see Part I.
See [occupancy-heavy profile](RESULT_C14_UNBUDGETED_PROFILE.md), including the
X31 re-certification bill and the untouched exact original resume state.
