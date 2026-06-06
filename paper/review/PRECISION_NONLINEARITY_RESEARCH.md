# Finite-precision nonlinearity, serial-vs-parallel scan, and linear-recurrence expressivity

**Task:** `research-precision-nonlinearity` (agent-1018, model claude:opus). Deep
literature + conceptual review. **`paper/main.typ` was NOT edited.**

**Web access:** AVAILABLE and used. Every citation in this document was located
*and verified on the web in this session* (WebSearch + direct fetch of arXiv
abstract pages / ACL Anthology / DOI landing pages / publisher pages). Five
parallel research sub-agents (a WG/Workflow fan-out, one per research thread 1–5)
each performed independent WebSearch and returned only works they had seen on the
web; the lead author then **independently re-verified every identifier**,
including direct `arXiv/abs/...` fetches for all 2024–2026 preprints and the
numerical-dynamics references, before admitting any citation. Citations the
sub-agents could not confirm were dropped and are recorded in §8 as honest gaps,
not invented. **No reference here is from memory alone; none is fabricated.** If a
claim could not be tied to a verifiable source, it is labelled an *inference* or an
*open question* rather than stated as established.

---

## 0. The motivating empirical wrinkle (why this review exists)

The paper argues (`main.typ:520–536`) that a *linear-state* recurrence "at fixed
precision and width is a regular-language recognizer that lives inside TC⁰ and
therefore cannot solve non-solvable-group word problems," with the symmetric group
$S_5$ (120 elements, smallest non-solvable group) as the canonical NC¹ witness via
Barrington's theorem. The intended separation is: **linear-state ⊆ TC⁰**,
**nonlinear-state can reach NC¹**, and (assuming TC⁰ ≠ NC¹) the two are distinct.

The wrinkle is in the paper's own experiment (`review/S5_SYMMETRIC_RESULTS.md`).
After symmetric per-architecture CMA tuning, the **linear-state** variant
`e88-linear` (`linear_state=1`, `use_gate=1`) *won* the S5 slot:

| Model | S5 T=128 | T=256 | T=512 | T=1024 |
|-------|---------:|------:|------:|-------:|
| **e88-linear** (linear state, winner) | **0.9997** | 0.7515 | 0.3909 | 0.2002 |
| e88-tanh (nonlinear state)            | 0.9888 | 0.6296 | 0.3216 | 0.1678 |
| GDN                                   | 0.5446 | 0.2801 | 0.1441 | 0.0759 |
| M²RNN-CMA                             | 0.1655 | 0.0858 | 0.0479 | 0.0276 |
| random (1/120)                        | 0.0083 | 0.0083 | 0.0083 | 0.0083 |

Two facts must be held together. (i) `e88-linear` reaches **0.9997 at the training
length** T=128 — far above the nonlinear baselines, exactly where the linear model
is "supposed" to be unable to track $S_5$. (ii) Its accuracy **decays toward chance
as length grows** (0.75 → 0.39 → 0.20 at 256/512/1024), the hallmark of a model
that has *not* acquired a length-robust $S_5$ algorithm. The runs used the serial
training/eval recipe in `train_hybrid.py`.

The author's hypotheses for reconciling (i) with the TC⁰ ceiling:

- **H1 — serial rounding is a per-step nonlinearity.** In finite/floating-point
  precision every forward-in-time step applies rounding/truncation, a tiny
  nonlinearity. A *serially executed* "linear" recurrence is therefore not exactly
  linear-in-time; accumulated rounding may make it weakly nonlinear and lift it
  above its idealized TC⁰ ceiling at bounded length.
- **H2 — parallel/associative scan reassociates the rounding.** Floating-point
  addition/multiplication is non-associative, so running the *same* recurrence via
  a Blelloch-style associative scan groups the operations differently. The
  serial-in-time nonlinearity does not accumulate the same way; parallel-scan
  execution should behave closer to a *true* linear-in-time model.
- **H3 — the "linear won S5" result may be a serial-precision artifact.** If H1/H2
  hold, a parallel-scan re-run (and/or a precision sweep) would disambiguate, and
  the paper's TC⁰-vs-NC¹ argument — which assumes *idealized* linearity — would be
  unaffected once the artifact is removed.

This document maps what the literature **does** and **does not** establish about
each, separates established results from inferences from genuine open questions,
proposes concrete disambiguating experiments tied to the E88 variant, and states
the consequences for the paper's expressivity claim.

---

## 1. Bottom line (read this first)

**On H1 (serial rounding = useful nonlinearity that escapes TC⁰).**
*Plausible mechanism, NOT an established theorem.* The premises are individually
well-supported: (a) finite precision genuinely changes the computational class of a
recurrence, not just its accuracy [Weiss-Goldberg-Yahav 2018; Siegelmann-Sontag
1995; Chiang-Cholak-Pillay 2023; Li-Cotterell 2025]; (b) rounding turns benign
iterated maps into qualitatively different discrete dynamical systems with emergent
cycles/attractors [Diamond-Kloeden-Pokrovskii-Vladimirov 1995;
Pingel-Schmelcher-Diakonos 1996]; (c) the quantization `round()` is a genuine
non-differentiable nonlinearity [Jeong-Xin-Yin 2025]. **But no verified work proves
the specific bridge** — that accumulated serial rounding supplies enough *usable*
nonlinearity to lift a linear-state recurrence out of TC⁰ and solve $S_5$. The
foundational TC⁰ theorems explicitly assume *clean / controlled-error* arithmetic
within the precision budget and do not model rounding as exploitable state
(Merrill-Petty-Sabharwal 2024 say so in as many words; see §6.3). So H1 is a live,
unclosed conjecture, not a citable result.

**On H2 (parallel scan reassociates the rounding).**
*Numerically established at the level of bits; unstudied at the level of $S_5$
expressivity.* That serial-recurrence and parallel/chunkwise-scan executions of the
*same* linear recurrence are **not bit-identical** in finite precision is textbook
numerical analysis [Goldberg 1991; Higham 2002; IEEE-754-2019; Blelloch 1990;
Demmel-Nguyen 2013; Shanmugavelu et al. 2024]. The entire modern linear-attention
family is *built* on three numerically non-equivalent forms — parallel, recurrent,
chunkwise [Katharopoulos 2020; RetNet 2023; GLA 2024; Mamba-2/SSD 2024; DeltaNet
2024], and practitioners keep SSM state in fp32 precisely because the serial bf16
recurrence accumulates error [Mamba 2023; TFLA 2025]. **What is missing** is any
work tying this reassociation to a *change in state-tracking expressivity* — i.e.,
showing serial-vs-parallel execution flips $S_5$ accuracy. That is exactly the
disambiguating experiment H3 calls for (§7).

**On H3 (the result is a serial-precision artifact).**
*Possible, but there is a stronger, better-established competing explanation that
must be ruled out first.* Two confounds independent of rounding-nonlinearity:

1. **Finite training length makes the asymptotic separation moot.** TC⁰ ⊊ NC¹ is an
   *asymptotic* statement; every fixed-length instance of the $S_5$ word problem is
   trivially in TC⁰ (indeed in AC⁰). Winning at T=128 is **not** evidence against
   the theorem, and the observed **decay toward chance with length is precisely the
   asymptotic failure showing through**. On this reading the paper's own data
   already reconciles with theory *without needing H1–H3 at all.*
2. **`e88-linear` is an *input-dependent* linear recurrence, not a diagonal SSM.**
   It has `use_gate=1`. The recent state-tracking theory is unambiguous that the
   relevant lever for a linear-in-state recurrence is **the structure of the
   (still-linear) transition operator** — input-dependence, dense vs diagonal
   transitions, and the eigenvalue range — *not* a state nonlinearity and *not*
   rounding [Cirone et al. 2024; Grazzi et al. 2025; Khavari et al. 2025;
   DeltaProduct/Siems et al. 2025; Movahedi et al. 2025; Shakerinava et al. 2026].
   An input-dependent dense linear recurrence can track far more than a diagonal one
   *in exact finite-precision arithmetic*, with no appeal to rounding.

So H3's "artifact" framing is plausible but **under-determined**: the experiments in
§7 are needed to separate a genuine precision artifact (H1/H2) from the more
mundane "input-dependent linear recurrence solves a fixed-length instance, then
fails to extrapolate" explanation.

**Consequence for the paper (full treatment in §6).** The clean dichotomy
"linear-state ⟹ TC⁰ ⟹ cannot do $S_5$" is an **asymptotic, idealized-arithmetic**
claim. It is *not* contradicted by a finite-precision, finite-length serial win,
and the paper already hedges with "at fixed precision and width" (`main.typ:524`).
The honest exposure is **not** that the theory is wrong, but that the **empirical
claim "the linear model is held away from $S_5$ by the complexity argument" is
overstated at training length** unless the length-extrapolation collapse is
foregrounded and the input-dependence/precision confounds are controlled.

---

## 2. Thread 1 — Precision and RNN/automata/transformer expressivity

**Established (real citations).**

- **Infinite-precision ceiling.** A finite RNN with *rational* weights and unbounded
  time is Turing-complete; with *real* weights it is super-Turing (P/poly)
  [Siegelmann & Sontag 1995; Siegelmann 2003]. Precision is the load-bearing
  assumption — the construction stores an unbounded stack in the unbounded-precision
  bits of an activation. Remove unbounded precision and the proof collapses. This is
  the canonical "why precision matters" baseline.
- **Finite precision changes the *class*, not just accuracy.** Under finite
  precision and linear time, LSTM and ReLU-Elman RNNs implement unbounded counters
  ($a^n b^n$, $a^n b^n c^n$) while GRUs and squashing-RNNs cannot
  [Weiss, Goldberg & Yahav 2018]. The mechanism — an additive, non-saturating cell
  state acting as a counter — is a *nonlinear-in-time* device. This is the closest
  classical analogue to H1: what a recurrence can track at finite precision is
  governed by whether its state update can build unbounded discrete structure.
- **Automata / saturation abstraction.** Saturated networks (step-function
  nonlinearities) map RNN variants to automata; LSTMs behave like counter machines
  and are *not* "rationally recurrent" (weighted finite automata), separating them
  from QRNN/SSM-like rational recurrences [Merrill 2019; Merrill, Weiss, Goldberg,
  Schwartz, Smith & Yahav 2020]. Linear-state recurrences are essentially
  rational/weighted-automaton recurrences, which the hierarchy places strictly
  below counter machines.
- **Log-precision transformers ⊆ uniform TC⁰.** Transformers with $O(\log n)$
  precision are simulable by logspace-uniform constant-depth threshold circuits
  [Merrill & Sabharwal 2023, *Parallelism Tradeoff*], with a matching FO[Majority]
  characterization [Merrill & Sabharwal 2023, *A Logic for Log-Precision
  Transformers*]. The framing is a *parallelism tradeoff*: any architecture as
  parallelizable as a transformer inherits TC⁰ limits — the conceptual core of H2.
- **Fixed precision is even weaker.** Fixed-precision transformer encoders recognize
  only uniform TC⁰ / FOC[+,MOD] [Chiang, Cholak & Pillay 2023]; saturated-attention
  transformers with floating-point values are in TC⁰ [Merrill, Sabharwal & Smith
  2022]; fixed-precision transformer LMs are exactly a small temporal-logic fragment
  and fail to generalize beyond it [Li & Cotterell 2025]. **Fixing precision
  *removes* power relative to log/infinite precision.**
- **Bounded-precision Turing completeness needs unbounded *total* state.** A
  54-neuron fixed-precision RNN is Turing-complete *only* by adding a dynamically
  growing external memory module [Chung & Siegelmann 2021]. Caveat for H1: fixed
  precision per neuron does **not** buy Turing power for free; a fixed-width linear
  recurrence has bounded total state and cannot escape this way.
- **Maps and surveys.** Hard-attention cannot robustly model PARITY/Dyck [Hahn
  2020]; the assumption-dependence of all these results (fixed vs log vs infinite
  precision; hard vs soft attention; uniformity) is mapped in [Strobl, Merrill,
  Weiss, Chiang & Angluin 2024].

**Inference.** The dividing line for $S_5$ is genuinely the *state map* (and the
transition structure), not parameter count; the theory predicts the *asymptotic*
failure of linear-state $S_5$, which is fully consistent with a finite-precision,
finite-length serial run nonetheless appearing to win.

**Open (named honestly).** None of these works analyze whether *serial finite-
precision rounding* injects a per-step nonlinearity that lets a nominally-linear
recurrence creep above TC⁰ at bounded length, nor how associative-scan
reassociation changes accumulated error/expressivity. The theory idealizes the
linear recurrence as exactly linear. H1/H2/H3 are outside its model.

---

## 3. Thread 2 — Does finite precision ADD or REMOVE power? (rounding as nonlinearity)

**Established — precision controls the language class (both directions).**
Beyond Thread 1: fixed precision *removes* power (Li & Cotterell 2025; Chiang et
al. 2023), while the *number system* sets the ceiling — rational ⟹ Turing, real ⟹
super-Turing [Siegelmann 2003]. Normalization handling is decisive for whether a
capability is realized: with layer-norm a transformer recognizes PARITY perfectly,
settling an open question [Chiang & Cholak 2022]. At finite precision, single-layer
diagonal complex SSMs cannot express non-Abelian state tracking, and $k$ layers
reach exactly the solvable groups [Shakerinava, Khavari, Ravanbakhsh & Chandar
2026] — precision regime determining the *algebraic* class reachable.

**Established — rounding changes the *dynamics* of iterated maps.** This is the
numerical-dynamics literature and it strongly supports H1's *premise* (rounding is
not a benign perturbation):

- Finite-precision discretization of chaotic maps causes **collapse to fixed points
  / short cycles**, with statistics matching *random mappings with a single
  attracting centre* — the discretized system is a qualitatively different
  dynamical object than the real-arithmetic one [Diamond, Kloeden, Pokrovskii &
  Vladimirov 1995].
- Two chaotic trajectories **coalesce/synchronize under roundoff but never under
  infinite precision** — a behavior created purely by rounding
  [Pingel, Schmelcher & Diakonos 1996].
- The quantization `round()` is genuinely non-differentiable and must be surrogated
  by the straight-through estimator [Jeong, Xin & Yin 2025] — confirming
  quantization injects a per-step nonlinearity (in *training*).

**Inference vs established for H1.** Verified: precision moves recurrences between
classes; rounding restructures dynamics; FP is non-associative. **Inference, not a
theorem:** that serial FP rounding of a *linear-state* recurrence constitutes a
controlled nonlinearity sufficient to leave TC⁰ and solve $S_5$. No verified paper
makes that exact statement.

**Open.** (a) No theorem isolates serial rounding as the mechanism that lifts a
linear recurrence out of TC⁰ for $S_5$. (b) No work proves a quantized *linear*
recurrence *gains* a language class (e.g., counting) purely from rounding — the STE
results are about trainability, not added expressivity. (c) The shadowing-lemma
framework ("a rounded linear recurrence shadows a nonlinear one") is the natural
bridge but has, as far as could be verified, **never been applied to argue an
expressivity (complexity-class) gain**, only trajectory closeness. (d) Whether the
per-step rounding is better modeled as zero-mean stochastic noise (averages out) or
as a systematic, input-correlated nonlinearity (would matter for expressivity) is
unsettled.

---

## 4. Thread 3 — Floating-point non-associativity & serial-vs-parallel scan

**Established fact 1 — FP non-associativity.** Rounding makes FP arithmetic violate
real-arithmetic identities; $(a+b)+c \neq a+(b+c)$ in general [Goldberg 1991, ACM
Computing Surveys; Higham 2002, *Accuracy and Stability of Numerical Algorithms*,
esp. Ch. 4 summation; IEEE-754-2019 states associativity of addition does not always
hold]. Summation error bounds depend on the *order* of summation.

**Established fact 2 — parallel scan changes the summation order.** Blelloch's
work-efficient up-sweep/down-sweep scan groups operands differently from a serial
left-fold [Blelloch 1990]. A scan is exact only when the operator is associative;
for FP "+", which is only *approximately* associative, the serial recurrence and the
parallel-scan recurrence compute **different rounded results** — the precise
mechanism behind H2.

**Established fact 3 — this is a known, only-specially-fixable reproducibility
problem.** Dynamic parallel scheduling + FP non-associativity makes even simple sums
non-reproducible across orderings; order-independent ("reproducible") summation
requires special algorithms [Demmel & Nguyen 2013, ReproBLAS], and the
non-determinism is documented end-to-end through GPU/PyTorch deep-learning pipelines
[Shanmugavelu et al. 2024]. **Absent** such handling (standard SSM/linear-attention
kernels do not use it), serial vs parallel reduction/scan differ.

**SSM / linear-attention specifics.** The family is explicitly built around three
numerically non-equivalent forms:

- *parallel vs recurrent* duality via associativity of matrix products
  [Katharopoulos et al. 2020];
- explicit *parallel / recurrent / chunkwise-recurrent* trichotomy [RetNet 2023];
- *chunkwise* (parallel within a chunk, recurrent across chunks) training in GLA
  [2024], DeltaNet [2024], Gated DeltaNet [2025], Mamba-2/SSD [2024]; S5 itself is
  trained via parallel associative scan [Smith, Warrington & Linderman 2023].

On precision specifically: Mamba keeps the recurrent/`A_log`/cumulative quantities
in **fp32** and its official guidance warns SSMs are sensitive to recurrent dynamics
and recommends fp32 parameters under AMP [Gu & Dao 2023]; SSD passes/accumulates the
inter-chunk state in fp32 to control the error a serial bf16 recurrence would
accumulate [Dao & Gu 2024]; tiled/chunkwise linear-RNN/xLSTM kernels must validate a
perplexity match against the reference, and repeated bf16↔fp32 recasting across
chunks grows error — i.e., **chunk size directly modulates accumulated rounding**
[Beck, Pöppel, Lippe & Hochreiter 2025].

**Synthesis for H1/H2/H3.** The numerical-analysis literature *firmly* supports that
serial-recurrence and parallel/chunkwise-scan executions of the same linear
recurrence are not bit-identical in finite precision (H2's premise), and that
re-running under a different (parallel-scan or fp32) regime is a legitimate
disambiguation (motivating H3). It supports only the *weaker* H1 claim that rounding
makes the realized map order-dependent and history-dependent; it does **not**
establish that this yields $S_5$ expressivity.

**Open.** No cited work (i) proves rounding yields enough expressivity to escape
TC⁰; (ii) performs the serial-vs-parallel-scan $S_5$ experiment; or (iii) quantifies
whether the serial-vs-parallel bf16/fp32 discrepancy is *large relative to the
decision margins* of the $S_5$ task — i.e., big enough to flip accuracy. These are
exactly the measurements §7 proposes.

---

## 5. Thread 4 — Is the linearity exactly vs approximately realized, and does it matter?

**Established — "linear-state" is exactly linear in the state; the transition is
input-dependent.** The canonical linear-attention recurrence
$S_t = S_{t-1} + \phi(k_t) v_t^\top$ applies the nonlinearity $\phi$ to the *input*,
not the state [Katharopoulos et al. 2020]. DeltaNet,
$S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ with L2-normalized
keys $k_t = \mathrm{SiLU}(W_K x_t)/\lVert\cdot\rVert_2$, has SiLU / normalization /
gating act on the *input* to form an input-dependent (generalized Householder)
transition, while $S_{t-1}\mapsto S_t$ stays exactly affine in $S_{t-1}$ [Yang et
al. 2024]. Gating/decay in Gated DeltaNet is a per-step input-dependent *scaling* of
the linear state, not a state nonlinearity [Yang, Kautz & Hatamizadeh 2024].
**Normalization does not inject state nonlinearity; it injects input-dependence into
a still-linear operator.** This matters for H1: whether `e88-linear` is "weakly
nonlinear-in-time" is a question about *finite-precision realization* of that linear
operator, not about an algebraic state nonlinearity.

**Established — the transition *structure* (not state nonlinearity) is the
state-tracking lever, in finite precision.**

- SSMs (incl. selective/Mamba) are in TC⁰ and cannot do $S_5$ in one forward pass;
  the proposed minimal fixes are **nonlinearity OR input-dependent transitions**
  [Merrill, Petty & Sabharwal 2024].
- **Finite-precision** linear RNNs with only positive eigenvalues cannot solve
  parity; extending the eigenvalue range $[0,1]\to[-1,1]$ unlocks parity, modular
  arithmetic, and permutation composition [Grazzi et al. 2025] — *precision is in
  the theorem statement.*
- Solving parity requires **both** input-dependence **and** negative eigenvalues;
  input-independent non-negative-eigenvalue stacks fail even with depth [Khavari et
  al. 2025].
- Multiple Householder micro-steps per token give diagonal-plus-rank-$n_h$
  transitions; by Cartan–Dieudonné enough reflections realize any orthogonal matrix
  (incl. permutations / $S_5$), so finite-precision state-tracking provably improves
  with $n_h$ **while staying linear-in-state** [DeltaProduct / Siems et al. 2025].
- Densifying the (linear) transition via fixed-point iteration reaches SOTA on the
  $A_5$ and $S_5$ state-tracking benchmarks [Movahedi et al. 2025]; diagonal linear
  RNNs cannot state-track but input-dependent *dense* linear recurrences with rank
  $\sim\log N$ match nonlinear-RNN expressivity [Cirone et al. 2024].

**Inference for H1–H3.** The *established* mechanism for a linear model "winning
$S_5$" is **eigenvalue range / input-dependence / multi-step (dense) composition —
properties of the ideal finite-precision recurrence**, not rounding-nonlinearity.
Since `e88-linear` is gated (input-dependent), this is the **leading competing
explanation** to H1/H3 and must be controlled (see §7, E5).

**Open.** (a) No peer-reviewed work measures whether *serial vs parallel-scan FP
execution* changes $S_5$ accuracy. (b) No theorem frames per-step rounding as a
"tiny nonlinearity" lifting a linear-state recurrence into NC¹. (c) No quantitative
analysis of whether L2/RMS-norm and decay-clamping, *as realized in FP*, perturb the
eigenvalue spectrum enough to matter. (d) Which mechanism explains a given empirical
"linear-won-$S_5$" — learned negative/complex eigenvalues, multi-substep
composition, or an FP artifact — has not been isolated in any verified publication.

---

## 6. Thread 5 — S5 / Barrington / NC¹ and the precision hypothesis

### 6.1 The group-theoretic anchor
Bounded-width (width-5) polynomial-size branching programs recognize exactly
(nonuniform) NC¹, with the construction crucially using the *non-solvability* of
$S_5$; consequently the word problem of every finite non-solvable group is
NC¹-complete, $S_5$ canonical [Barrington 1989; STOC 1986]. This is a discrete
circuit result — **no precision hypothesis**; it supplies the lower-bound *target*
the linear-recurrence separations aim at.

### 6.2 The TC⁰ upper bound for SSMs is stated in a *log-precision* model
[Merrill, Petty & Sabharwal 2024] prove iterated addition, iterated product, and
matrix powering over $c\log n$-bit floats are in L-uniform TC⁰, hence linear /
diagonal / non-gated SSMs (S4/S6) are in L-uniform TC⁰ (their Lemma 4.1, Thms
4.2/4.4). The headline separation (their Cor. 4.7): *assuming TC⁰ ≠ NC¹, no
log-precision SSM with the S4/S6 architecture can solve the word problem for $S_5$
or any NC¹-hard problem.* They explicitly contrast that simple **nonlinear** RNNs
*can* express $S_5$ via standard finite-state constructions [Minsky 1954; Merrill
2019] — the linear-vs-nonlinear dichotomy the paper rests on.

### 6.3 Precision is load-bearing — and the authors say rounding is outside the model
Verbatim from [Merrill, Petty & Sabharwal 2024]: when the datatype "has a fixed
number of bits … then these operations are easily seen to be in L-uniform TC⁰. As
Merrill and Sabharwal (2023) argue, however, finite-precision datatypes severely
limit the expressivity of neural architectures from a formal perspective …
motivating the use of parameterized datatypes that can (approximately) represent any
number." Implications:

1. **Genuine fixed precision (fp16/fp32/bf16) is even weaker — still inside TC⁰** by
   their own remark. So fixed precision *by itself* does **not** explain an
   empirical $S_5$ win; if anything it tightens the ceiling.
2. **The theorems assume controlled-error arithmetic within the budget.** They do
   *not* model accumulated rounding as exploitable state. H1 (serial rounding as
   adversarially-usable extra nonlinearity) is therefore **neither proved nor
   forbidden** by these theorems — it lives outside their formalization. **No
   verified work proves that finite-precision *serial* execution stays in TC⁰ once
   rounding is treated as usable nonlinearity.** This is the single most important
   genuine gap for the paper's argument.

### 6.4 Independent confirmation that the precision/width hypothesis is explicit
[Liu, Ash, Goel, Krishnamurthy & Zhang 2023] (ar5iv full text verified): *solvable*
semiautomata admit $O(1)$-depth transformer "shortcuts," and the obstruction to
constant-depth simulation is precisely the presence of non-solvable groups such as
$S_5$. Their lower bound (Thm 4) is precision-parameterized: no $O(\log T)$-precision
transformer with depth independent of $T$ and width poly$(T)$ can continuously
simulate non-solvable automata, and constant-depth simulation is hard unless TC⁰ =
NC¹. So the impossibility is conditioned on **(i) TC⁰ ≠ NC¹ and (ii) an
$O(\log T)$-precision, poly-width, $T$-independent-depth hypothesis** — a second,
independent statement that the precision/width assumptions are load-bearing.

### 6.5 Not all "linear-state" models are equally weak
Diagonal linear RNNs cannot state-track, but input-dependent *dense* selective
recurrences match nonlinear-RNN expressivity [Cirone et al. 2024]. The gated,
input-dependent `e88-linear` is in the *more expressive* subclass — a confound
separate from precision.

### 6.6 What this means for the paper's claim (the requested explicit note)
Putting §6.1–6.5 together:

- **The paper's TC⁰-vs-NC¹ separation is sound as an asymptotic, idealized-arithmetic
  statement** and is *not contradicted* by `e88-linear`'s finite-length win. The
  separation is conditional (TC⁰ ≠ NC¹) and asymptotic; a fixed length-128 instance
  is trivially solvable, and the **length-decay to ~0.20 at T=1024 is the predicted
  asymptotic failure**. The hedge "at fixed precision and width" (`main.typ:524`) is
  doing real work and should stay.
- **If H1–H3 hold, the theory still survives, but a specific *empirical* sentence
  becomes unsafe.** The paper says the nonlinear recurrence "reaches something a
  linear one provably cannot … where a linear-in-time recurrence is held away from
  the task by a classical complexity argument" (`main.typ:213–217`). The complexity
  argument holds *asymptotically and for the idealized linear model*; it does **not**
  by itself explain or predict the linear model's *behaviour at training length*,
  where `e88-linear` in fact **beat** the nonlinear `e88-tanh` (0.9997 vs 0.9888).
  Three non-exclusive readings of that win — (a) input-dependent/dense transition
  expressivity [Cirone/Grazzi/Movahedi], (b) finite-length triviality, (c) an FP
  serial-rounding artifact [H1/H2] — are currently **not distinguished**. Asserting
  the separation as the *explanation* of the experimental ranking, without
  controlling (a) and (b), overstates what the experiment shows.
- **Cleanest honest framing for the paper.** Keep the asymptotic TC⁰/NC¹ argument as
  a *theoretical* statement; report the $S_5$ result primarily through
  **length-extrapolation** (where the linear model collapses and, if it holds, the
  nonlinear model degrades more gracefully), not through a single training-length
  number; and explicitly note that a linear-state model's training-length success on
  a fixed-length $S_5$ instance is *expected* and is **not** a violation of the
  separation. If the experiments in §7 show the linear win is precision/serial-order
  dependent, that becomes a *positive* contribution: direct evidence that the
  idealized-linearity assumption is the right one to make and that serial execution
  can flatter a linear model.

---

## 7. Concrete disambiguating experiments (tied to E88) and what each proves

All are runnable with the existing harness (`train_hybrid.py`,
`eval_s5_symmetric_winners.py`) on the pinned `e88-linear` config. None require
touching `main.typ`. List ordered by information-per-GPU-hour.

**E1 — Parallel/associative-scan re-execution of `e88-linear` at matched precision.**
Because `e88-linear` is an input-dependent *linear* recurrence, its state update
$h_t = A_t h_{t-1} + b_t$ admits a Blelloch associative scan (compose the
$(A_t,b_t)$ affine maps). Run S5 eval two ways at **identical** dtype: (i) the
current serial time-loop; (ii) a chunkwise/associative-scan evaluation. Compare
S5 accuracy at T∈{128,256,512,1024}.
- *If serial ≫ scan*: the serial execution order is contributing to the apparent
  $S_5$ success beyond the ideal linear recurrence → **direct support for H1/H2/H3**;
  the "linear won $S_5$" headline is partly a serial-precision artifact.
- *If serial ≈ scan*: the win is a property of the (input-dependent linear) model
  itself, **not** serial rounding → H1/H2 are *not* the driver; attribute to §6.5 /
  Thread 4 mechanisms.
This is the **single most decisive** experiment and exactly the test the literature
has never run.

**E2 — Precision sweep fp64 / fp32 / bf16 (serial path).**
Re-run serial S5 eval (and, ideally, training) at three precisions.
- *Accuracy rises as precision drops* (bf16 > fp32 > fp64): rounding is *helping* →
  strong, surprising support for H1 (rounding-as-useful-nonlinearity).
- *Accuracy flat or rises with precision* (fp64 ≥ fp32 ≥ bf16): the win is not
  rounding-driven; fp64 serial ≈ "near-ideal linear" → **H1 disfavored.**
fp64 serial doubles as a practical proxy for the idealized linear-in-time model.

**E3 — Truly-linear scan baseline (the idealized TC⁰ control).**
Implement the *exact* ideal linear recurrence used by `e88-linear` (same learned
$A_t,b_t$) evaluated in fp64 via parallel scan — the cleanest realization of "what a
genuine linear-in-time model computes." Compare to `e88-linear` serial.
- Quantifies the *gap* between the idealized linear model and the deployed serial
  one. A large gap that closes under E1/E2 localizes the effect to serial FP order.
- If the fp64-scan ideal *already* reaches high training-length S5, the win is
  intrinsic to the linear operator (input-dependence/eigenvalues), **not** precision.

**E4 — Length-crossover vs mantissa bits.**
For each precision in E2, find the length $T^\*$ at which accuracy crosses (say) 0.5.
The theory predicts an asymptotic collapse; this measures *where* it sets in and how
$T^\*$ scales with mantissa bits.
- A clean dependence of $T^\*$ on mantissa bits would be novel evidence that the
  finite-length "success window" is precision-bounded — partially closing the
  *quantitative* open question in §3/§4.

**E5 — Control the input-dependence confound (ablate gating / eigenvalue range).**
Re-run with `use_gate=0`, and separately constrain transition eigenvalues to
$[0,1]$ (no negative/complex), per [Grazzi et al. 2025; Khavari et al. 2025].
- *S5 win disappears when gating/negative-eigenvalues are removed*: the win is the
  **input-dependent-dense-transition** mechanism (§6.5 / Thread 4), **not** rounding
  → H3's "precision artifact" framing is wrong; the mundane explanation wins.
- *S5 win survives the ablation but vanishes under E1/E2*: points back to a genuine
  serial-precision effect.
E5 is what separates "interesting precision physics" from "ordinary input-dependent
linear recurrence," and it is the control most likely to be raised by a referee.

**What the full battery establishes.** E1+E5 jointly identify *which* of the three
explanations (serial-FP artifact / input-dependent-linear expressivity /
finite-length triviality) drives the result; E2+E4 quantify the precision
dependence; E3 provides the idealized-linear yardstick the paper's theory assumes.
Together they convert the current ambiguous "linear won S5" into a defensible claim
of one specific kind.

---

## 8. Honest gaps — works suspected but NOT cited (no fabrication)

The sub-agents flagged, and the lead did **not** include as citations, the
following (recorded so the gap is visible rather than papered over):

- **A formal result that serial FP rounding lets a *linear* recurrence escape TC⁰ /
  solve $S_5$.** No such paper was found. This is the central H1 gap. If it existed
  it would be the most relevant citation in the document; its apparent non-existence
  is itself a finding (the paper's E1/E5 experiments would be novel).
- **A head-to-head serial-vs-parallel-scan study of $S_5$ (or any non-solvable
  group) state-tracking accuracy.** Not found — the H3 disambiguation appears
  genuinely untested in the literature.
- **A shadowing-lemma argument for an expressivity (not trajectory) gain under
  rounding.** Not found; the connection appears unmade.
- **Martin & Cundy, "Parallelizing Linear Recurrent Neural Nets Over Sequence
  Length" (arXiv:1709.04057).** Surfaced as the canonical parallel-scan-for-linear-
  RNN reference but its page was *not* fetched to confirm title/authors/year, so it
  is **not** cited here. (Verify before adding to `refs.bib`.)
- **Hua et al. 2022 (FLASH / "Transformer Quality in Linear Time") as origin of the
  chunkwise form**, and **stochastic-rounding error-accumulation work (e.g. Gupta et
  al. 2015)** — named from memory by sub-agents but not web-confirmed this session;
  excluded.
- **Specific quantified bf16-vs-fp32 GitHub issues in `state-spaces/mamba` /
  `fla-org/flash-linear-attention`** — only the README/general fp32 guidance was
  confirmed, not a specific quantified issue number; excluded.
- **Earlier Diamond–Kloeden "spatial discretization of dynamical systems" papers** —
  strongly suspected (same authors) but exact title/year not fetched; excluded in
  favor of the verified Physica D 1995 paper.

---

## 9. Bibliography (every entry verified on the web this session)

**Precision & expressivity — foundations / RNNs / transformers / SSMs**

1. H. T. Siegelmann, E. D. Sontag. *On the Computational Power of Neural Nets.* J.
   Comput. Syst. Sci. 50(1):132–150, 1995 (COLT 1992). DOI:10.1006/jcss.1995.1013.
2. H. T. Siegelmann. *Neural and Super-Turing Computing.* Minds and Machines, 2003.
   DOI:10.1023/A:1021376718708.
3. G. Weiss, Y. Goldberg, E. Yahav. *On the Practical Computational Power of Finite
   Precision RNNs for Language Recognition.* ACL 2018 (short), 740–745.
   arXiv:1805.04908 · ACL Anthology P18-2117.
4. W. Merrill. *Sequential Neural Networks as Automata.* ACL 2019 DeepLo Workshop.
   arXiv:1906.01615 · ACL Anthology W19-3901.
5. W. Merrill, G. Weiss, Y. Goldberg, R. Schwartz, N. A. Smith, E. Yahav. *A Formal
   Hierarchy of RNN Architectures.* ACL 2020, 443–459. arXiv:2004.08500 · ACL
   Anthology 2020.acl-main.43.
6. W. Merrill, A. Sabharwal, N. A. Smith. *Saturated Transformers are Constant-Depth
   Threshold Circuits.* TACL 10:843–856, 2022. arXiv:2106.16213 · ACL Anthology
   2022.tacl-1.49.
7. W. Merrill, A. Sabharwal. *The Parallelism Tradeoff: Limitations of Log-Precision
   Transformers.* TACL 11:531–545, 2023. arXiv:2207.00729 · ACL Anthology
   2023.tacl-1.31.
8. W. Merrill, A. Sabharwal. *A Logic for Expressing Log-Precision Transformers.*
   NeurIPS 2023. arXiv:2210.02671.
9. W. Merrill, J. Petty, A. Sabharwal. *The Illusion of State in State-Space
   Models.* ICML 2024 (PMLR v235). arXiv:2404.08819.
10. M. Hahn. *Theoretical Limitations of Self-Attention in Neural Sequence Models.*
    TACL 8:156–171, 2020. arXiv:1906.06755 · DOI:10.1162/tacl_a_00306.
11. D. Chiang, P. Cholak, A. Pillay. *Tighter Bounds on the Expressivity of
    Transformer Encoders.* ICML 2023 (PMLR v202), 5544–5562. arXiv:2301.10743.
12. D. Chiang, P. Cholak. *Overcoming a Theoretical Limitation of Self-Attention.*
    ACL 2022 (long). arXiv:2202.12172 · ACL Anthology 2022.acl-long.527.
13. L. Strobl, W. Merrill, G. Weiss, D. Chiang, D. Angluin. *What Formal Languages
    Can Transformers Express? A Survey.* TACL 12, 2024. arXiv:2311.00208 · ACL
    Anthology 2024.tacl-1.30.
14. S. Chung, H. T. Siegelmann. *Turing Completeness of Bounded-Precision Recurrent
    Neural Networks.* NeurIPS 2021. OpenReview:IWJ9jvXAoVQ ·
    proceedings.neurips.cc/paper/2021/hash/ef452c63f81d0105dd4486f775adec81.
15. J. Li, R. Cotterell. *Characterizing the Expressivity of Fixed-Precision
    Transformer Language Models.* NeurIPS 2025. arXiv:2505.23623.

**S5 / Barrington / NC¹ and automata learning**

16. D. A. Barrington. *Bounded-Width Polynomial-Size Branching Programs Recognize
    Exactly Those Languages in NC¹.* J. Comput. Syst. Sci. 38(1):150–164, 1989 (STOC
    1986). DOI:10.1016/0022-0000(89)90037-8 · STOC DOI:10.1145/12130.12131.
17. B. Liu, J. T. Ash, S. Goel, A. Krishnamurthy, C. Zhang. *Transformers Learn
    Shortcuts to Automata.* ICLR 2023. arXiv:2210.10749.

**Linear-recurrence state-tracking: transition structure & eigenvalues**

18. A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret. *Transformers are RNNs: Fast
    Autoregressive Transformers with Linear Attention.* ICML 2020 (PMLR v119).
    arXiv:2006.16236.
19. N. M. Cirone, A. Orvieto, B. Walker, C. Salvi, T. Lyons. *Theoretical
    Foundations of Deep Selective State-Space Models.* NeurIPS 2024.
    arXiv:2402.19047.
20. R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, M. Pontil. *Unlocking
    State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025 (Oral).
    arXiv:2411.12537.
21. J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi.
    *DeltaProduct: Improving State-Tracking in Linear RNNs via Householder
    Products.* 2025. arXiv:2502.10297.
22. B. Khavari, M. Shakerinava, J. Khullar, J. Huang, F. Rivest, S. Ravanbakhsh, S.
    Chandar. *Parity Requires Unified Input Dependence and Negative Eigenvalues in
    SSMs.* 2025. arXiv:2508.07395.
23. S. Movahedi, F. Sarnthein, N. M. Cirone, A. Orvieto. *Fixed-Point RNNs:
    Interpolating from Diagonal to Dense.* NeurIPS 2025 (Spotlight). arXiv:2503.10799.
24. M. Shakerinava, B. Khavari, S. Ravanbakhsh, S. Chandar. *The Expressive Limits
    of Diagonal SSMs for State-Tracking.* ICLR 2026. arXiv:2603.01959.

**SSMs / linear attention: the three execution forms & precision**

25. J. T. H. Smith, A. Warrington, S. W. Linderman. *Simplified State Space Layers
    for Sequence Modeling (S5).* ICLR 2023. arXiv:2208.04933.
26. A. Gu, T. Dao. *Mamba: Linear-Time Sequence Modeling with Selective State
    Spaces.* 2023 (COLM 2024). arXiv:2312.00752.
27. T. Dao, A. Gu. *Transformers are SSMs: Generalized Models and Efficient
    Algorithms Through Structured State Space Duality (Mamba-2 / SSD).* ICML 2024.
    arXiv:2405.21060.
28. Y. Sun, L. Dong, S. Huang, S. Ma, Y. Xia, J. Xue, J. Wang, F. Wei. *Retentive
    Network: A Successor to Transformer for Large Language Models (RetNet).* 2023.
    arXiv:2307.08621.
29. S. Yang, B. Wang, Y. Shen, R. Panda, Y. Kim. *Gated Linear Attention
    Transformers with Hardware-Efficient Training (GLA).* ICML 2024. arXiv:2312.06635.
30. S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. *Parallelizing Linear Transformers
    with the Delta Rule over Sequence Length (DeltaNet).* NeurIPS 2024.
    arXiv:2406.06484.
31. S. Yang, J. Kautz, A. Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with
    Delta Rule (Gated DeltaNet).* ICLR 2025. arXiv:2412.06464.
32. M. Beck, K. Pöppel, P. Lippe, S. Hochreiter. *Tiled Flash Linear Attention: More
    Efficient Linear RNN and xLSTM Kernels.* NeurIPS 2025. arXiv:2503.14376.

**Floating-point non-associativity, scans, reproducibility**

33. D. Goldberg. *What Every Computer Scientist Should Know About Floating-Point
    Arithmetic.* ACM Computing Surveys 23(1):5–48, 1991. DOI:10.1145/103162.103163.
34. N. J. Higham. *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM,
    2002. DOI:10.1137/1.9780898718027.
35. IEEE. *IEEE Standard for Floating-Point Arithmetic (IEEE 754-2019).*
    DOI:10.1109/IEEESTD.2019.8766229.
36. G. E. Blelloch. *Prefix Sums and Their Applications.* Tech. Report CMU-CS-90-190,
    Carnegie Mellon University, 1990.
37. J. Demmel, H. D. Nguyen. *Fast Reproducible Floating-Point Summation.* IEEE
    ARITH-21, 2013. (ReproBLAS line; see also *Algorithms for Efficient Reproducible
    Floating Point Summation*, ACM TOMS, DOI:10.1145/3389360.)
38. S. Shanmugavelu, M. Taillefumier, C. Culver, O. Hernandez, M. Coletti, A.
    Sedova. *Impacts of Floating-Point Non-Associativity on Reproducibility for HPC
    and Deep Learning Applications.* 2024. arXiv:2408.05148.

**Rounding-as-nonlinearity / numerical dynamics / quantization**

39. P. Diamond, P. Kloeden, A. Pokrovskii, A. Vladimirov. *Collapsing Effects in
    Numerical Simulation of a Class of Chaotic Dynamical Systems and Random Mappings
    with a Single Attracting Centre.* Physica D 86:559–571, 1995.
40. D. Pingel, P. Schmelcher, F. K. Diakonos. *Roundoff-Induced Coalescence of
    Chaotic Trajectories.* 1996. arXiv:chao-dyn/9607017 (Phys. Rev. E).
41. H. Jeong, J. Xin, P. Yin. *Beyond Discreteness: Sample Complexity Analysis of
    Straight-Through Estimator for 1-bit Quantization.* 2025. arXiv:2505.18113.

*(In-text-only, verified inside ref. 9, not independently re-fetched: M. Minsky,
*Neural Nets and the Brain-Model Problem* / finite-state RNN constructions, 1954 —
cited as the "RNNs can express $S_5$" source; included here for completeness, not as
a primary verified entry.)*

---

## 10. Provenance / how to reproduce the verification
- Fan-out: 5 parallel sub-agents (one per research thread), each restricted to
  WebSearch-confirmed citations, returning a structured schema with a
  `verifying_url` and a self-reported confidence per citation.
- Lead re-verification: every 2024–2026 arXiv id was fetched directly at
  `arxiv.org/abs/<id>` to confirm title + author list + year (2404.08819,
  2411.12537, 2502.10297, 2402.19047, 2505.23623, 2508.07395, 2503.10799,
  2603.01959, 2408.05148, 2503.14376, 2505.18113); classical/journal entries
  confirmed via ACL Anthology, DOI landing pages, JCSS/ScienceDirect, ACM DL, and
  publisher pages. Numerical-dynamics entries confirmed via publisher/Semantic
  Scholar listings and the resolving arXiv (chao-dyn) PDF.
- Anything not confirmed this session was dropped and recorded in §8. Zero citations
  are from memory alone.
