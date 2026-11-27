# Attention over an XSA-style index: exact skeleton, moment ladder, certificates

Companion to [CORPUS_ATTENTION.md](CORPUS_ATTENTION.md). This states precisely what is
**exactly computable** from pattern-count/interval primitives, what the nearest approximation
to full attention is at each budget, and what no suffix-aligned structure can reach. It is a
mathematical write-up only: no code, natural membership, or GPU work is authorized here;
implementations need their own registration and qualification. "XSA-style" means an index
that answers, for any pattern `P`: its occurrence interval `I(P)`, its count, its child-edge
counts, and successor-conditioned counts `count(Pv)`.

## 1. The law

Rows are eligible endpoints: within-document positions with a real successor. At prediction
time `t` the live model forms query `q`; row `i` has key `k_i` and value `v_i`; the sources
corpus `C`, local `L`, and no-read compete in one normalizer:

```
Z_C = Σ_{i∈C} b_i exp(q·k_i/√d),  U_C = Σ_{i∈C} b_i exp(q·k_i/√d) v_i,
r_t = (U_C + U_L + U_skip) / (Z_C + Z_L + Z_skip).
```

`b_i` is part of the **law**, not the approximation. With `b_i = 1` a pattern seen 100×
gets 100× mass (multiplicity matters); with `b_i = π_C/|C|` the corpus competes on quality.
At |C| ≈ 10^11 the per-row choice is worth `ln|C| ≈ 25.3` nats of prior mass and dominates
the read's behavior. Every claim below must be read *relative to a fixed choice of `b`*.

## 2. The count skeleton is exact, not approximate

Let `P_ℓ` be the last `ℓ` observed bytes, `I_ℓ` the rows whose context ends with `P_ℓ`
(occurrences of `P_ℓ`, shifted to the following position; `I_0` = all rows), and `ℓ_i` a
row's suffix-match depth. **Restrict the score family to depth-only scores**
`s_i = g(ℓ_i) + log b`. Then with `m_ℓ = |I_ℓ| − |I_{ℓ+1}|` (rows of depth exactly `ℓ`)
and exact-depth successor histograms

```
h^ex_ℓ(v) = count(P_ℓ v) − count(P_{ℓ+1} v),
```

every quantity is one index lookup and the law is **exactly**:

```
Z_C = b Σ_{ℓ=0}^{D} m_ℓ e^{g(ℓ)},
p_C(v) = Σ_ℓ e^{g(ℓ)} h^ex_ℓ(v) / Z_C        (D = deepest live match).
```

Writing `p_ℓ(v) = h^ex_ℓ(v)/m_ℓ` makes the identification explicit:
`p_C = Σ_ℓ λ_ℓ p_ℓ` with `λ_ℓ ∝ e^{g(ℓ)} m_ℓ` — **the count law is attention restricted to
depth-only scores, i.e. exactly the Jelinek–Mercer/Katz family of order-mixed n-gram
smoothers**, with the mixing weights coupled to depth and count. X43-style consultation is
the hard-selection limit (`g` → step). The pilot's COUNT arm is a cousin in a *different
placement* (an output-level log-linear mixture gated on live state); the skeleton inside the
attention softmax has never been run as an arm. Counts must be taken over eligible endpoints
only (document ends carry no successor). Cost per query: one root-to-node walk of depth `D`
collecting child counts — `O(D · alphabet)`, exact, at any corpus size, forever. This is the
zeroth order and the mandatory control: it is what every learned reader must beat.

## 3. Beyond depth-only scores: the moment ladder

Let learned keys be arbitrary functions of row context. Group rows into nodes `u` (any
query-independent grouping; §5). Write `k_i = k̄_u + δ_i`, `t = q/√d`. The node's mass is a
**moment-generating function** of its key measure:

```
Z_u = b n_u e^{t·k̄_u} E_{k∈u}[e^{t·δ}],
log E e^{t·δ} = ½ tᵀΣ_u t + κ₃(t)/6 + O(‖t‖⁴),
```

with `Σ_u` the within-node key covariance and `κ₃` its third cumulant. The ladder:

* **Order 0** — counts: `Z_u = b n_u e^{g(ℓ_u)}`. Exact for depth-only scores and histogram
  values; otherwise biased.
* **Order 1** — mean key and mean value: `log Z_u ≈ log(b n_u) + t·k̄_u`, `r̂_u = v̄_u`.
  Error `O(within-node key dispersion × ‖t‖)`.
* **Order 2** — Gaussian closure plus **value tilt**:
  `log Z_u ≈ log(b n_u) + t·k̄_u + ½ tᵀΣ_u t`,
  `r̂_u ≈ v̄_u + C_u t` with `C_u = Cov_u(k, v)` (a `d_v×d` matrix).
  The tilt corrects the systematic bias of mean-value pooling: softmax reweights rows toward
  high-scoring keys, so the pooled value drifts along the key–value regression. Typical-case
  relative error is `O(κ₃ ‖t‖³)` — small once queries are normalized.

Sufficient statistics are `(n_u, Σk_i, S_u=Σk_ik_iᵀ, Σv_i, T_u=Σv_ik_iᵀ)`, with
`k̄ = Σk/n`, `Σ_u = S/n − k̄k̄ᵀ`, `v̄ = Σv/n`, `C_u = T/n − v̄k̄ᵀ`. **All merges are linear
and associative** — a parent is the sum of its children's five statistics. This is simpler
than the current log-sum-exp tree merges: no peak stability, no exponentials, exact at every
level. The currently implemented rank-32 positive-feature bank is a different order-1
quadrature (fixed random directions); measured read RMS vs exact was ~1.2e−3 at pilot
geometry, so approximation fidelity there was already near — the ladder's value at scale is
the *certificate* (§4), the *tilt*, and query-adaptive refinement.

## 4. Certificates and budgeted refinement

Accumulate at build time the node radius `R_u = max_i ‖k_i − k̄_u‖`. Then for every query:

```
log Z_u ∈ log(b n_u) + t·k̄_u ± R_u ‖t‖      (two-sided, rigorous).
```

This turns the read into **certified branch-and-bound**: refine nodes where
`n_u e^{t·k̄_u} (e^{+R_u‖t‖} − e^{−R_u‖t‖})` is largest; stop when unresolved mass is a
chosen `ε`; then read TV error ≤ `ε` and `‖r − r̂‖ ≤ 2B_v ε` (the existing bound). Provably
near in both friendly regimes — **peaked** attention (deep match: the exact interval is small,
moments handle the rest) and **flat** attention (read ≈ corpus mean value). The adversarial
middle — high-scoring rows scattered across subtrees with heterogeneous values — is a
theorem-shaped wall for *every* sublinear method, not an engineering gap. The measurable
diagnostic, computable wherever exact attention is: **the fraction of top-mass rows lying
outside the matched suffix interval**. That number is the ceiling of any suffix-aligned
approximation and the exact quantity a learned core must justify itself against.

## 5. Where the nodes come from

* **Query-side exact interval**: rows of `I_D` get per-row exact scores. At corpus scale a
  common trigram has `|I_D|` in the millions — the exact core itself must refine. The
  natural refinement is **left extension**: rows of `I_ℓ` partition by the byte preceding the
  match into occurrence sets of the extended patterns `c·P_ℓ`, whose counts are the *same
  primitive*. Recursing leftward groups matched rows by ever-longer shared context — exactly
  the conditioning a context-keyed encoder wants — until tip groups fit the per-query budget;
  interior groups use order-2 summaries, tips use per-row scores.
* **Bank-side grouping**: a query-independent hierarchy over all rows is the tree of shared
  left context (occurrence sets of patterns, nested as patterns extend leftward) — all counts
  are forward-pattern counts. The implemented fixed 128-row dyadic leaves are the
  content-agnostic fallback that needs no tree; content-adaptive nodes tighten dispersion at
  the same mass, which is the trade the ladder wants.

## 6. What is not reachable

Suffix alignment certifies relevance that *ends at the frontier*. Gapped, paraphrastic or
earlier-anchored relevance does not concentrate in any subtree; its mass is transverse. For
that mass the options are a second structure (learned features over keys — the current
rank-32 bank is one) or probability sampling with inclusion weights, reported as such. No
suffix-index approximation can be near when the transverse mass is large; §4's diagnostic
measures it.

## 7. Cost and scale walls

Order-2 build cost is `O(N(d + d² + d_v d))` ≈ 4× the current rank-32 feature build
(`d = d_v = 64`: ~8.4k flops/row). Read cost is `O(cover × d²)` — negligible. Backward is
`O(N d)` per update with rank-1 adjoints (§8). Node statistics are ~8.3k floats (≈33 KB)
per node; full moments are fine at pilot scale, sketched (random-projection) covariances at
multi-MB archives. The per-update **current-weight rebuild wall is unchanged**: at 10^11
rows no law that re-encodes every row per update is runnable. The registered policy choices
— two-timescale refresh, `K`-step staleness with measured staleness bias, or sampled
summaries — are separate decisions; the pilot-scale and multi-MB ladder does not need them.

## 8. Adjoint anchors (derive from a torch autograd oracle first, then port)

For a touched node with mass adjoint `β_u = ∂L/∂log Z_u` and value adjoint `γ_u = ∂L/∂r̂_u`:

```
∂log Z_u/∂k_i = (1/n_u)[ t + (t·(k_i − k̄_u)) t ],
∂r̂_u/∂v_i    = (1/n_u)[ 1 + (t·(k_i − k̄_u)) ]  (along γ_u),
```

plus cross terms of the same rank-1 shape (`γ_u` into `k_i` via `C_u`, `β_u` into `v_i` via
the tilt, `t`-parallel pieces through `k̄`, `Σ`, `C`). Every per-row adjoint is `O(d)` and
structured; the multi-output VJP through the one shared current-weight graph stays the
training path.

## 9. Measurement plan (to be registered on its own)

1. At 16,384 rows, where exact is computable: arms EXACT / current features / order-2 /
   skeleton (order 0 *inside the softmax*) / LOCAL. Measure per rung: relative `log Z_C`
   error, read TV, next-token KL, achieved certificate `ε` per budget, and the transverse
   mass fraction. This prices each rung honestly before any scale-up.
2. At multi-MB BUILD where exact dies: skeleton vs order-2 vs LOCAL, equal dose, documents
   disjoint, all the panel discipline. Decision reading: if the skeleton matches order-2,
   the learned residual is not paying; if order-2 is `ε`-near exact but both lose to LOCAL,
   the deficit is the law or the prior, not the approximation; if order-2 is far from exact
   at small `ε`, the transverse mass is large and a second structure is required.
