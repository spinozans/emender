# Prior art for a chunkable polynomial **state-lift** recurrence

**Task:** `lift-prior-art` · **Date:** 2026-06-09 · **Web access:** available; every arXiv ID
and DOI below was web-verified this session (search + page fetch). Where a source could **not**
be independently re-verified in this session it is flagged `[UNVERIFIED-ID]` so we never cite a
number we did not confirm.

---

## 0. The object, and the distinction that matters

We are considering a recurrence whose **state is lifted by a finite polynomial / monomial feature
map** — e.g. carry not just `S` but the degree-≤2 monomials `[S, S⊗S]` — chosen so that a
**bounded-degree nonlinear** state→state dynamic becomes **linear (affine) in the lifted
coordinates**, hence **associative ⇒ parallel-scannable / chunkable** (chunked-matmul, GDN-2-class
throughput). The promise: *more-than-linear expressivity without a per-step pointwise nonlinearity*
(the thing that ordinarily breaks chunkability).

Two superficially-similar lines must be kept **separate**, because only one is our object:

- **(a) Polynomial feature map on the INPUTS (queries/keys) of linear attention.** You replace the
  softmax kernel `exp(qᵀk)` by `φ(q)ᵀφ(k)` where `φ` is a polynomial / Taylor / power feature map.
  The recurrent **state** `S = Σ φ(kₜ) vₜᵀ` then *accumulates additively* and is **linear in its own
  previous value**. The non-linearity lives in the input map, never in the state→state transition.
  **Well-trodden.** *Not our object.*

- **(b) Polynomial lift of the RECURRENT STATE itself.** The transition is nonlinear **in the
  previous state** (`Sₜ` depends on `Sₜ₋₁ ⊗ Sₜ₋₁`, products of state coordinates), and we lift the
  state with degree-k monomials so the lifted update is again linear and scannable. **This is our
  object.** This is, mathematically, **Carleman / Koopman linearization** of a nonlinear recurrence,
  truncated to finite degree.

The crisp test that separates (a) from (b): **does a state coordinate at time `t` depend
multiplicatively on a state coordinate at time `t−1`?** In (a) the answer is *no* (state is an
additive accumulator); in (b) the answer is *yes*.

---

## 1. Line (a) — polynomial / higher-degree kernel feature maps on q/k (NOT our object)

This is the mature, crowded line. Map for completeness so we don't confuse it with (b).

| Work | arXiv / venue (verified) | What it does | Relation to (b) |
|---|---|---|---|
| Katharopoulos et al., *Transformers are RNNs* | **2006.16236** (ICML 2020) | `φ(q)ᵀφ(k)` with `φ=elu+1`; additive outer-product state `S=Σφ（k)vᵀ`; recurrent + O(N) form | Defines the additive-state template. State is **linear** in itself. |
| Choromanski et al., *Rethinking Attention with Performers* (FAVOR+) | **2009.14794** (ICLR 2021) | random-feature `φ` that unbiasedly approximates `exp` | feature map on inputs |
| Schlag, Irie, Schmidhuber, *Linear Transformers Are Secretly Fast Weight Programmers* | **2102.11174** (ICML 2021) | state = fast weights; adds the **delta rule** (write correction) | still linear-in-state transition; ancestor of DeltaNet |
| Zhang et al., *Hedgehog & the Porcupine* | **2402.04347** (ICLR 2024) | **learnable** feature map mimicking softmax spikiness | feature map on inputs |
| Arora et al., *Based* | **2402.18668** (ICML 2024) | **2nd-order Taylor** feature map `φ(k)≈[1,k,k⊗k/√2]` ⇒ state holds `k⊗k` outer products | **boundary case** — degree-2, but the quadratic is of the **input** `k`, not the state |
| Kacham, Mirrokni, Zhong, *PolySketchFormer* | **2310.01655** (ICML 2024) | degree-p **polynomial** attention made linear-time via sketching | feature map on inputs |
| Buckman, Gelada, Zhang, *Symmetric Power Transformers* (Manifest AI) | manifestai.com article (2024); peer companion *Conformal Transformations for Sympow*, **2503.03269** (2025) | `φ = p`-th **symmetric tensor power** of q/k ⇒ recurrent state is the `p`-th symmetric power accumulator (size ~`dᵖ`), chunkable | **closest boundary case** — a genuine degree-p **state of monomials**, but the monomials are of the **input** `k`, accumulated **additively**; transition still linear-in-state |

**Subtlety to put on record (so we don't fool ourselves):** Based and Symmetric-Power *do* produce
a state whose coordinates are degree-2/degree-p **monomials** — `Σ (kₜ⊗kₜ) vₜᵀ`. Superficially this
looks like an "`S⊗S` state". It is **not**: the tensor power is taken of the **incoming key**, then
**added** to the state. No state coordinate is a product of *previous-state* coordinates. These are
firmly in line (a). They are the strongest evidence that *"polynomial state of size dᵖ, chunkable"*
is already a solved, shipped idea — **as long as the polynomial is of the inputs.**

---

## 2. Line (b) — lifting the RECURRENT STATE itself

Four sub-clusters, ordered by closeness to our exact object.

### (b-i) Input-side polynomial state (the d^p accumulators) — already done, but it's really (a)
Covered above (Based, Symmetric Power, PolySketchFormer). **Verdict for this sub-cluster: done and
shipped.** If our "degree-2 lift" turns out to be *of the inputs*, we are reinventing Based /
Symmetric-Power and should stop. This is the honest "don't reinvent it" warning the task asked for.

### (b-ii) Carleman / Koopman state-monomial lift — the *exact math* of our object
Carleman linearization lifts a polynomial dynamical system `ẋ=f(x)` (f polynomial) into an
**infinite linear** system in the monomials `[x, x⊗x, x⊗x⊗x, …]`, truncated to finite degree;
Koopman is the operator-theoretic generalization (monomials are one choice of "observables").
*"The Carleman technique uses monomials of the state as observables in constructing the Koopman
operator."* This is **precisely** "lift the state by degree-k monomials so nonlinear-in-state becomes
linear-in-lifted-state."

Verified anchors:
- Takeishi, Kawahara, Yairi, *Learning Deep NN Representations for Koopman Operators* — **1708.06850**
  (learn the lifting `φ` end-to-end so dynamics are linear in `φ`-space).
- *Globalizing the Carleman linear embedding method for nonlinear dynamics* — **2510.15715** (2025);
  control-theory framing, confirms Carleman = finite monomial state-lift.
- (canonical, knowledge-cited, `[UNVERIFIED-ID]`) Lusch, Kutz, Brunton, *Deep learning for universal
  linear embeddings of nonlinear dynamics*, Nature Comms 2018 — the deep-Koopman archetype.

**What is missing in this cluster:** every realized Koopman/Carleman sequence model I could verify is
built for **dynamical-system identification / control / forecasting**, and is trained/evaluated as a
**linear recurrence in the lifted space that is fitted once**, *not* as a token-mixing layer of a
language model with a **chunked / parallel-scan training kernel** and **input-dependent (gated)
transitions**. The math of (b) is old and well-understood; the **realization of (b) as a chunkable,
gated, learned sequence-mixing primitive is what is absent.**

### (b-iii) Parallelizing a *genuinely nonlinear* recurrence — by Newton, not by lift
A distinct, active mechanism: keep the nonlinear-in-state recurrence and parallelize its *evaluation*
with a fixed-point / Newton solver, rather than making it associative.
- Lim et al., *Parallelizing Non-linear Sequential Models over the Sequence Length* (**DEER**) —
  **2309.12252** (ICLR 2024). Newton fixed-point; cost **cubic in state size**.
- Gonzalez, Warrington, Lim, Linderman, *Towards Scalable and Stable Parallelization of Nonlinear
  RNNs* — **2407.19115** (NeurIPS 2024): **quasi-DEER** (linear in state) + **ELK** (Kalman-stabilized).
- Gonzalez, Kozachkov et al., *Predictability Enables Parallelization of Nonlinear State Space Models*
  — **2508.16817** (NeurIPS 2025): ties parallelizability to the Lyapunov exponent / PL-conditioning;
  predictable systems evaluate in `O((log T)²)`.

**Why this is *not* our object:** these are **iterative Newton solvers** (multiple passes,
state-size-cubic or special conditioning), **not** a single associative-scan over a lifted state.
They parallelize *despite* nonlinearity; we want to *remove* the nonlinearity by lifting so a
**one-pass chunked matmul** suffices. Important to cite as the competing way to get "nonlinear +
parallel," and as evidence the field actively wants this capability.

### (b-iv) Genuinely nonlinear-in-state recurrences that are NOT chunkable
The classical second-order / multiplicative / bilinear / tensor RNNs — these *have* state×state (or
state×input) products, i.e. true (b)-style dynamics, but **none is parallel-scannable**:
- Sutskever, Martens, Hinton, *Generating Text with RNNs* (**mRNN**, multiplicative RNN), ICML 2011 —
  input-gated multiplicative transition via 3-way tensor factorization. `[UNVERIFIED-ID]` (no arXiv;
  ICML 2011 PDF verified).
- Krause et al., *Multiplicative LSTM* — **1609.07959**.
- Ebrahimi & Memisevic, *Revisiting Bi-Linear State Transitions in RNNs* — **2505.21749**
  (NeurIPS 2025): hidden units as **active** (multiplicative) participants; bilinear updates form a
  state-tracking complexity hierarchy with **Mamba at the lowest (linear) center**. Strong modern
  evidence that **state×state multiplicativity buys expressivity** — but it is realized **sequentially**.

Adjacent expressivity-via-structured-*linear*-transition (chunkable, but transition is still **linear
in state** — so not (b)): GLA (**2312.06635**), Gated DeltaNet (**2412.06464**), DeltaProduct
(**2502.10297**, products of Householders, tunable expressivity, *explicitly chunkwise-parallel*).
These show the field's current frontier for "more expressive but still chunkable" is **richer linear
transition matrices**, **not** a nonlinear-in-state lift. Log-Linear Attention (**2506.04761**) adds a
hierarchical chunk structure — again linear-state.

---

## 3. VERDICT

**Has the chunkable polynomial-**STATE-lift** recurrence (b) been built? — No, not as our exact
object. With one heavy caveat.**

1. **If "degree-2 lift" means a degree-2 feature map of the INPUTS** giving a `d²`-sized additively-
   accumulated state → **YES, fully done and shipped**: this is **Based** (2nd-order Taylor) and
   **Symmetric Power Transformers** (p-th symmetric power), both chunkable, both ~GDN-class. *Do not
   reinvent this.* It is the most likely way to accidentally rebuild prior art.

2. **If "lift" means lifting the state's OWN previous value** (`Sₜ ← f(Sₜ₋₁⊗Sₜ₋₁, xₜ)`, made linear
   and scannable by carrying degree-k state monomials) → **the math exists (Carleman/Koopman) but the
   realization as a gated, chunk-parallel, learned sequence-mixing layer does NOT appear in the
   verified literature.** The pieces are all present and *separately* shipped:
   - degree-k monomial state, chunkable — present, but **of inputs** (b-i / a);
   - monomial-lift-makes-nonlinear-linear — present as **Carleman/Koopman**, but **not** as a chunked
     LM kernel (b-ii);
   - nonlinear-in-state + parallel — present, but via **Newton solvers**, not associative scan (b-iii);
   - genuinely state×state multiplicative dynamics — present, but **sequential / not chunkable**
     (b-iv: mRNN, mLSTM, bilinear RNN).
   **Nobody has verifiably joined "true state⊗state nonlinearity" with "single-pass chunkable
   associative scan" via a finite polynomial state lift, as a trained sequence model.**

### Residual novelty for us
- **The combination is open:** *gated-delta backbone + finite degree-2 lift of the **recurrent
  state** + a chunked associative-scan kernel.* No verified work does the lift on the **state**
  (vs inputs) while keeping a one-pass chunk kernel.
- **Capability framing is open:** Carleman/Koopman work targets dynamical-systems forecasting/control;
  bilinear-RNN work (2505.21749) targets **state-tracking** but sequentially. Framing a degree-2
  state-lift as *the* chunkable route to the state-tracking / bounded-counting capability that linear
  transitions provably lack (DeltaProduct/bilinear-hierarchy results) is a defensible novel angle.
- **Honest risks to pre-empt:** (i) **truncation** — a degree-2 state lift is **not closed**: the
  update of `S⊗S` generates degree-3/4 terms, so any finite lift is *approximate* for genuinely
  nonlinear dynamics (Carleman truncation error). We must state which dynamics we claim to represent
  *exactly* vs approximately. (ii) **`d²` blow-up** — the lifted state is `O(d²)`; this is the exact
  cost wall Sympow hit (state too big past `p≈4`). Our novelty must show the lift buys capability the
  `d²`-input-accumulators (Based/Sympow) don't already buy — otherwise we've reinvented (b-i).

---

## 4. Prioritized reading list (closest to (b), read in this order)

1. **Symmetric Power Transformers** — Buckman, Gelada, Zhang (Manifest AI, 2024) + **Conformal Sympow
   2503.03269**. *The* shipped "degree-p chunkable polynomial state." Read first to know exactly what
   the input-side lift already gives, so our state-side lift is differentiated.
2. **Based — 2402.18668** (Arora et al.). The degree-2 incarnation; cleanest statement of "2nd-order
   feature map ⇒ d² chunkable state." Our nearest neighbour on the (a) side.
3. **Revisiting Bi-Linear State Transitions in RNNs — 2505.21749** (Ebrahimi, Memisevic, NeurIPS 2025).
   The expressivity case for **true state×state** multiplicativity, with the Mamba-at-the-center
   hierarchy. The "why bother with (b)" argument.
4. **Predictability Enables Parallelization of Nonlinear SSMs — 2508.16817** + **DEER 2309.12252** +
   **quasi-DEER/ELK 2407.19115**. The competing mechanism (parallelize nonlinearity by Newton); tells
   us what we must beat (single-pass chunk vs iterative solver) and when nonlinear-parallel is hard.
5. **Carleman/Koopman lift** — Takeishi et al. **1708.06850** and *Globalizing Carleman* **2510.15715**.
   The exact math of a finite monomial state-lift and its truncation behaviour. Source of our
   theory + the honest truncation caveat.
6. **DeltaProduct 2502.10297** + **Gated DeltaNet 2412.06464** + **GLA 2312.06635**. The chunkable
   frontier *with linear-state transitions*; the baseline family our state-lift must out-express at
   equal throughput.
7. **PolySketchFormer 2310.01655**. If `d²` state blows up, this is the prior art on **sketching** a
   high-degree polynomial state down — a likely tool, and prior art to cite if we sketch.
8. **Transformers are RNNs 2006.16236** + **Fast Weight Programmers 2102.11174**. Foundations; the
   additive-state template and the delta-rule our gated-delta backbone descends from.

---

## 5. Bottom line (plain)

- Line **(a)** (polynomial feature maps on q/k) is **mature and crowded** — Performer, Katharopoulos,
  Hedgehog, Based, PolySketchFormer, Symmetric-Power. Report it; do not build it.
- The **boundary case** — a `dᵖ`-sized chunkable state of degree-p **input** monomials — is **shipped**
  (Based, Symmetric Power). **The single most likely way to reinvent prior art is to lift the inputs
  and call it a state lift.** Guard against this explicitly.
- Our **exact object** — a finite degree-2 lift of the **recurrent state's own previous value**,
  preserving chunkability — is **mathematically known (Carleman/Koopman) but, in the web-verified
  literature, has NOT been realized as a gated, chunk-parallel, learned sequence-mixing kernel.**
  The competing realized routes are either (b-iii) Newton solvers (multi-pass, not associative) or
  (b-iv) sequential bilinear/multiplicative RNNs (not chunkable).
- **Residual novelty is real but narrow:** gated-delta + degree-2 **state** lift + chunked
  associative-scan kernel, framed around state-tracking/counting capability — *provided* we show it
  beats the `d²` input-accumulators (Based/Sympow) on capability and confront the truncation-closure
  and `d²`-cost objections head-on. If we cannot beat Based/Sympow on capability, we have reinvented
  (b-i) and should stop.

---

## References (all IDs web-verified this session unless marked `[UNVERIFIED-ID]`)

- Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs.* arXiv:2006.16236 (ICML 2020).
- Choromanski et al. *Rethinking Attention with Performers.* arXiv:2009.14794 (ICLR 2021).
- Schlag, Irie, Schmidhuber. *Linear Transformers Are Secretly Fast Weight Programmers.* arXiv:2102.11174 (ICML 2021).
- Zhang, Bhatia, Du et al. *The Hedgehog & the Porcupine.* arXiv:2402.04347 (ICLR 2024).
- Arora et al. *Based: Simple linear attention LMs balance the recall-throughput tradeoff.* arXiv:2402.18668 (ICML 2024).
- Kacham, Mirrokni, Zhong. *PolySketchFormer.* arXiv:2310.01655 (ICML 2024).
- Buckman, Gelada, Zhang. *Symmetric Power Transformers.* Manifest AI, 2024 (https://manifestai.com/articles/symmetric-power-transformers/).
- *Conformal Transformations for Symmetric Power Transformers.* arXiv:2503.03269 (2025).
- Yang et al. *Gated Linear Attention.* arXiv:2312.06635 (ICML 2024).
- Yang, Kautz, Hatamizadeh. *Gated Delta Networks (Gated DeltaNet).* arXiv:2412.06464 (ICLR 2025).
- Siems et al. *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder Products.* arXiv:2502.10297 (2025).
- Guo et al. *Log-Linear Attention.* arXiv:2506.04761 (2025).
- Lim et al. *Parallelizing Non-linear Sequential Models over the Sequence Length (DEER).* arXiv:2309.12252 (ICLR 2024).
- Gonzalez, Warrington, Lim, Linderman. *Towards Scalable and Stable Parallelization of Nonlinear RNNs (quasi-DEER/ELK).* arXiv:2407.19115 (NeurIPS 2024).
- Gonzalez, Kozachkov et al. *Predictability Enables Parallelization of Nonlinear State Space Models.* arXiv:2508.16817 (NeurIPS 2025).
- Krause et al. *Multiplicative LSTM for sequence modelling.* arXiv:1609.07959 (2016).
- Sutskever, Martens, Hinton. *Generating Text with Recurrent Neural Networks (mRNN).* ICML 2011. `[UNVERIFIED-ID]` (no arXiv; conf. PDF verified).
- Ebrahimi, Memisevic. *Revisiting Bi-Linear State Transitions in Recurrent Neural Networks.* arXiv:2505.21749 (NeurIPS 2025).
- Takeishi, Kawahara, Yairi. *Learning Deep NN Representations for Koopman Operators.* arXiv:1708.06850 (NeurIPS 2017).
- *Globalizing the Carleman linear embedding method for nonlinear dynamics.* arXiv:2510.15715 (2025).
- Lusch, Kutz, Brunton. *Deep learning for universal linear embeddings of nonlinear dynamics.* Nature Comms 2018. `[UNVERIFIED-ID]` (canonical, not re-verified this session).
