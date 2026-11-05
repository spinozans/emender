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
occupancy telemetry measured frontier size; the unbudgeted 4.96x cost receipt
measured frontier bloat. Learning lives only in the weights (frozen basis,
learned coordinates; routing detached) — which the algebra identifies as
SELECT's policy and EMIT's readout. The zero-learned-management spine (F12)
is the theory's natural shape, not a constraint.

Receipt supports: C9 tier-1 (suffix backoff = SUBSUME, succeeds where
full-context re-entry fails); the exact-walk failure (14% forced continuations
= EMIT without UPDATE); PB4 coupling (weights that cannot distinguish
"interval present" from "interval informative"); C14-B ablation (the ops carry
0.66 nats causally — the set dynamics are load-bearing, not decorative).

## Part II — LLMs of other types as approximations of the algebra

### The general claim

Training never gives a model the algebra — only the corpus as a linear
stream. The only place the corpus's exact interval structure can accumulate
is the weights. Therefore: **trained LLM weights are compressed, lossy
approximations of the suffix-array operator algebra over the corpus they
trained on, learned without access to it.** The approximation is exact where
the algebra is trivial (single-branch / forced spans) and lossy where it is
fork-dense.

### Family-by-family mapping

- **Transformer (dense)**: KV cache = a per-document dynamic index, built
  exactly at inference but unbounded (the program's receipted unbounded arm);
  learned attention keys = fuzzy matching = approximate intervals; induction
  heads (Olsson et al. 2022) = learned match-then-continue circuits =
  approximate backward search. Our C13 receipt — the transformer re-copies as
  coverage rises — is the approximation preferring retrieval.
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
  verbatim learning is exactness on forced spans — the approximation is exact
  where the algebra is trivial.
- The 12-byte-median support regime at Pile scale: the shared short-suffix
  structure is what cross-corpus consultation transfers (74% supported
  targets) — the shallow rungs of the ladder are the corpus-independent part
  of the algebra.
- P2/PB4 device-off penalties: the weights' approximation of "features
  present" vs "features informative."

### Candidate predictions (NOT pre-registered; bind per-experiment when run)

1. **Forced-span verbatim law**: verbatim recall accuracy is near-perfect on
   single-branch (forced) spans and degrades monotonically with branch
   entropy. Testable at our scales against any self-indexed training corpus.
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
