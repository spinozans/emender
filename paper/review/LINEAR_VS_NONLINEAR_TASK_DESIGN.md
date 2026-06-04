# What task PROVABLY separates linear-state from nonlinear-state recurrences?

**Task:** `linvsnonlin-task-design` · **Model:** claude:opus · **Type:** literature +
design research (no training run; no GPU). **`paper/main.typ` was NOT edited.**

**Web access:** AVAILABLE and used. Every citation below is either (a) already
web-verified inside this repo's `paper/review/PRECISION_NONLINEARITY_RESEARCH.md`
§9 (its bibliography was fetched and re-verified arXiv-id-by-arXiv-id in a prior
session; I reuse those identifiers), or (b) **newly fetched and verified in this
session** by directly loading the arXiv abstract page and confirming
title + author list + year (the four marked **[verified this session]** below).
**Zero citations are from memory alone; none is fabricated.** Where a separation is
not backed by a specific theorem I label it *inference* or *open* rather than
asserting it. A short note on why I did the verification inline rather than via a
sub-agent fan-out is in §7.

---

## 0. Why this document exists (the problem in one paragraph)

The paper's intended separator was the **$S_5$ word problem**: linear-state ⊆ TC⁰,
nonlinear-state can reach NC¹, so (assuming TC⁰ ≠ NC¹) $S_5$ should be solvable only
by the nonlinear model. **Our own experiments refute this as a *separator*.** After
symmetric CMA tuning the *linear* gated delta-rule recurrence `e88-linear`
(`linear_state=1`) **won** the $S_5$ slot (0.9997 @ T=128), beating the nonlinear
`e88-tanh` (`review/S5_SYMMETRIC_RESULTS.md`, `S5_CONFIG_FLIP.md`), and the
mechanism is now pinned: `e88-linear`'s per-token transition
$A_t=\mathrm{decay}_t\,I-\hat k\hat k^\top$ reaches a **negative / reflection
eigenvalue along the key axis** on *every* token (measured `frac<0 = 1.000` vs GDN's
`0.000`, `review/GDN_VS_E88_TRANSITION.md`). That is exactly the Grazzi et al. (2025)
/ DeltaProduct (Siems et al. 2025) lever: **negative eigenvalues + enough
input-selected Householder reflections compose the permutation group**, so a *linear*
recurrence tracks finite groups including $S_5$. **Conclusion: any finite-group /
finite-automaton word problem is the WRONG separator** — it lives inside the regular /
finite-state regime that input-dependent linear recurrences already reach. We need a
task that is provably **outside** that regime.

---

## 1. The formal boundary — what is beyond input-dependent linear-state recurrences

### 1.1 What the linear-state class IS (and why groups don't separate it)

The models in question — DeltaNet, DeltaProduct, Gated DeltaNet, Mamba/Mamba-2, and
our `e88-linear` — share one defining property: the state update is **affine in the
state**,
$$h_t = A_t\,h_{t-1} + b_t,$$
where $A_t,b_t$ are **input-dependent** ($A_t=A(x_t)$, $b_t=b(x_t)$) but **never
state-dependent**. The nonlinearity ($\mathrm{SiLU}$, L2-norm, gating) acts on the
*input* to shape $A_t,b_t$; it does not act on $h_{t-1}$
[Katharopoulos et al. 2020; Yang et al. 2024 — DeltaNet]. Output is a per-position
readout $y_t = g(h_t)$ with $g$ a (possibly nonlinear) head.

What this class can do, established:

- **Diagonal SSMs** (Mamba, |λ|≤1, no negative real eigenvalue) capture exactly the
  **star-free / aperiodic** regular languages and bounded hierarchical structure with
  optimal memory — but *not* unbounded counting and *not* non-solvable groups
  [Sarrof, Veitsman & Hahn 2024 **[verified this session]**; Merrill, Petty &
  Sabharwal 2024].
- **Adding negative eigenvalues** unlocks parity, modular arithmetic, and permutation
  composition for a *finite-precision linear* RNN [Grazzi et al. 2025].
- **Enough input-selected Householder reflections** realize any orthogonal matrix
  (Cartan–Dieudonné), hence any permutation / finite group including $S_5$, while
  staying **linear in the state** [DeltaProduct / Siems et al. 2025].
- **Densifying** the (still linear) input-dependent transition matches **nonlinear-RNN
  expressivity** within the bounded/finite-state regime [Cirone et al. 2024; Movahedi
  et al. 2025].

So the union of these levers covers, in effect, the **regular / finite-state**
languages and **finite-group word problems**. This is precisely why $S_5$ failed to
separate: it is finite-state, and a reflection-rich input-dependent linear recurrence
reaches it. (Cirone's near-equivalence is the strongest narrowing result and the one
a candidate probe must be designed *around*.)

### 1.2 What is provably OUTSIDE the class

Two qualitatively different things lie beyond a fixed-width, eigenvalue-bounded,
**state-affine** recurrence:

**(B1) Unbounded memory / non-regular structure** — tasks that are *not* finite-state:
context-free counting (`a^n b^n`, `a^n b^n c^n`), unbounded Dyck depth, copying an
arbitrarily long string. No fixed-width finite-state device recognizes these; they
require an unbounded **counter** or **stack**. The key fact (the separation):

> Under finite precision, **LSTMs and ReLU/Elman-RNNs implement unbounded counters**
> (recognizing `a^n b^n` and `a^n b^n c^n`) via an **additive, non-saturating** cell,
> whereas **squashing (tanh) RNNs and GRUs cannot** [Weiss, Goldberg & Yahav 2018
> **[verified this session]**].

Empirically, on the Chomsky-hierarchy benchmark **LSTMs climb to the counter /
context-free level while Transformers do not generalize there**, and *context-sensitive*
tasks need explicit stack/tape memory [Délétang et al. 2022 **[verified this
session]**]; Transformers handle only the simplest counter sub-languages and degrade
with formal complexity [Bhattamishra, Ahuja & Goyal 2020 **[verified this session]**].
Diagonal SSMs are **bounded**-memory by construction [Sarrof, Veitsman & Hahn 2024],
and generalized SSMs are provably limited at **copying** by their fixed-size state,
while a 2-layer Transformer copies exponentially long strings [Jelassi et al. 2024
**[verified this session]**]. **Crucially, the linear levers do NOT rescue this:**
negative eigenvalues / reflections / dense transitions enlarge the *finite-state* class
(B1 is non-finite-state), and they do not add unbounded counter magnitude. This is the
**best-established** linear-vs-nonlinear separation.

**(B2) Genuinely state-nonlinear feedback (`state×state`)** — a target generated by an
iterated map whose update is **non-affine in the state**, e.g.
$$h_{t+1} = h_t^2 + a_t \quad\text{or}\quad h_{t+1}=\rho\,h_t(1-h_t)+a_t .$$
A state-affine recurrence computes $y_t = g\big(L(\text{history})\big)$ for a single
bounded-dimension linear functional $L$ and a per-position nonlinearity $g$. The
trajectory of $h_{t+1}=h_t^2+a_t$ is a **degree-$2^t$ polynomial** in the inputs — it
is *not* of the form $g\circ L$ for any fixed $g,L$ as $t$ grows, because the model
would have to **feed a nonlinear function of its own state back into the recurrence**,
which a state-affine core structurally cannot do. The nonlinearity must live *inside
the time-recurrence*, not only in the readout. **This is the cleanest algebraic
separation**, but it has two regimes that behave very differently (see Probe 2), and —
unlike B1 — it is *inferred from the architecture definition*, not from a published
lower-bound theorem.

### 1.3 The note that group-word-problems are ruled out (required)

Ruled out, with reason: **finite-group word problems (including the canonical
non-solvable $S_5$) are finite-state and therefore reachable by an input-dependent
linear recurrence** once its transition can reach negative / reflection eigenvalues
(Cartan–Dieudonné ⇒ permutations) [Grazzi et al. 2025; DeltaProduct / Siems et al.
2025; Cirone et al. 2024]. We *measured* this in-repo: `e88-linear`'s transition is a
reflection on every token and it solves $S_5$ at training length
(`GDN_VS_E88_TRANSITION.md`, `S5_CONFIG_FLIP.md`). A separator must therefore be
non-finite-state (B1) or state-nonlinear-in-the-recurrence (B2), **not** a group
word-problem.

---

## 2. The design constraint from our own data (the tanh-on-$S_5$ cautionary case)

A candidate is only useful if **(i) nonlinearity is genuinely necessary** *and*
**(ii) the nonlinear model actually succeeds**. Our data warns that (ii) is not
automatic:

- On $S_5$, the **nonlinear `e88-tanh` did *not* beat the linear `e88-linear`** — it
  was *worse* at every length (`S5_CONFIG_FLIP.md`: linear beats tanh in all 8
  config×length cells). Reason: $S_5$ is finite-state, so a `tanh` state buys no
  advantage there, *and* `tanh` is a **saturating** nonlinearity — exactly the kind
  Weiss-Goldberg-Yahav 2018 prove **cannot** count. A naive "add a nonlinearity"
  (especially `tanh`) can therefore fail *both* tests at once.

**Two consequences for probe design:**

1. **Pick a task that is genuinely B1 or B2**, where a *linear* state provably cannot
   succeed even with eigenvalue/reflection/dense levers — not a finite-state task.
2. **Choose the nonlinear baseline to be one that provably CAN do it.** For counting
   (B1) that means an **additive / ReLU (LSTM-like) counter**, *not* a `tanh` state
   (which saturates and cannot count). For B2 it means a state with a genuine
   multiplicative/quadratic feedback path. A `tanh`-only ablation is a known false
   negative and must not be the sole "nonlinear" arm.

This is the single most important lesson carried from the $S_5$ episode into this
design.

---

## 3. Ranked shortlist of candidate probes

Ranked by **confidence that the separation is real and demonstrable in our harness**.
The harness contract (`experiments/expressivity_tasks/train_hybrid.py`,
`tasks/__init__.py`) is: a task is a class with
`generate_batch(B, T, rng) -> (input_tokens[B,T] int64, target_tokens[B,T] int64,
loss_mask[B,T] bool)` and `random_baseline_acc()`; per-position next-token style
supervision; small integer vocab; length-extrapolation eval via `--eval_lengths`.
Existing `parity.py` / `modular_counter.py` are the templates; both are ~40 lines.

---

### PROBE 1 — Counting-with-comparison (`a^n b^n c^n` / bounded-Dyck running depth) — **TOP, established**

**Task definition.** Per-position counter-language recognition with **dense
supervision and length extrapolation**:

- *Variant 1a (`a^n b^n c^n` membership).* Input: a string over `{a,b,c}` drawn from a
  mix of in-language (`a^n b^n c^n`) and minimally-corrupted out-of-language strings
  (off-by-one counts, wrong order). Per position $t$, target = a binary label "is the
  prefix so far still a viable prefix of some `a^n b^n c^n`?" (and at the final
  position, "is the whole string in the language?"). `vocab_size≈5`,
  `random_baseline_acc=0.5`.
- *Variant 1b (running Dyck-1 depth).* Input: a stream of brackets `{ (, ) }`; target
  at each position = the current **nesting depth** (capped to a fixed display range for
  the loss, but the *underlying* depth is unbounded). This is the direct
  generalization of `modular_counter` from "sum mod K" to "unbounded non-negative
  counter with a floor at 0" — the floor (a `max(·,0)` / zero-test) is the comparison.

Both are generated exactly like `modular_counter.py` (cumulative ops on a random
symbol stream); 1b is `running = cumsum(±1)` with a reflecting/clamping floor and depth
as target. **Trains at T=128, evaluates at T∈{256,512,1024}** (the harness's
`--eval_lengths` path) — extrapolation is where the separation shows.

**Why linear provably/empirically fails, nonlinear succeeds.** `a^n b^n c^n` and
unbounded Dyck depth are **non-regular** (context-free / counter, context-sensitive for
1c-style triple counts). A fixed-width finite-state device — which is what an
eigenvalue-bounded input-dependent linear recurrence is in the relevant regime — cannot
recognize them at unbounded length. The positive result is specific and architectural:
**additive non-saturating cells (LSTM / ReLU-RNN) implement the unbounded counter and
recognize `a^n b^n`, `a^n b^n c^n`; tanh-RNN and GRU cannot** [Weiss, Goldberg & Yahav
2018]; on the Chomsky-hierarchy suite **LSTMs generalize to counter/context-free tasks
while Transformers (and by the bounded-state argument, diagonal SSMs) do not**
[Délétang et al. 2022; Sarrof, Veitsman & Hahn 2024].

**Confound control — do the linear levers rescue it? NO.**
- *Negative eigenvalues / reflections* (Grazzi 2025; DeltaProduct 2025) enlarge the
  reachable **finite-state / group** structure. Counting-with-comparison is
  **non-finite-state**; reflections add no counter magnitude. ✗ does not rescue.
- *Dense input-dependent transitions* (Cirone 2024) match nonlinear-RNN expressivity
  **only within the bounded/finite-state regime**; the equivalence does not extend to
  unbounded counters. ✗ does not rescue.
- *Depth / more layers* stack finite-state transductions; a bounded number of layers of
  bounded-state recurrence is still bounded memory (the diagonal-SSM bound is
  per-layer-state-size, Sarrof-Veitsman-Hahn 2024). ✗ does not rescue.
- The **comparison** is the load-bearing nonlinearity: a running count is itself linear
  (`cumsum`), but the floor / zero-test / "still-viable-prefix" decision that must feed
  the *next-step dynamics* (e.g. a counter that must not go negative) is the
  non-saturating conditional update that WGY 2018 attribute to the additive ReLU/LSTM
  cell. Per-position dense supervision forces the model to maintain *and threshold* the
  counter at every step, not just read it off once at the end.

**Harness feasibility.** **High.** ~50-line task file, structurally identical to
`modular_counter.py`; per-position int targets; length-extrapolation already supported.
Rough scale: same as the existing parity/counter probes (dim≈128–256, depth≈4, a few
thousand steps), so **cheap** (minutes per run, single GPU). **One harness gap to
close:** the harness's strong nonlinear baselines (E88-tanh, M²RNN, GDN) are all
*saturating or linear-state*; to satisfy design-constraint (ii) the probe needs an
**additive/ReLU-counter baseline** (a plain ReLU-Elman or LSTM layer level) added to
`layer_pattern`. Without it the probe may show "everybody fails," which is uninformative.
*Action item: add a ReLU-RNN / LSTM layer level as the positive control.*

---

### PROBE 2 — Iterated quadratic state-feedback `h_{t+1} = h_t² + a_t` (the pure `state×state` probe) — **inferred / partly-established**

**Task definition.** A hidden state evolves by a genuinely **non-affine** recurrence
driven by small inputs; the model must predict the state (or its observable). Input:
stream of increments $a_t\in\{0,1,2\}$. Hidden state $h_{t+1}=h_t^2+a_t$. Two regimes:

- *2a (finite field, mod $p$).* $h_{t+1}=(h_t^2+a_t)\bmod p$, $h_0$ fixed; target$_t=h_t$.
  `vocab_size=p+const`, `random_baseline_acc=1/p`. This is a **deterministic finite
  automaton with input-dependent, NON-invertible (state-collapsing) transitions**
  ($x\mapsto x^2$ merges $x$ and $p-x$). It is the canonical **non-group monoid** case:
  the transition monoid contains non-invertible elements, so it is *not* a subgroup of
  permutations.
- *2b (unbounded magnitude).* Real-valued $h$, quantized to bins for cross-entropy;
  small bounded inputs so $h$ stays in range but the **update remains quadratic**. This
  is the "logistic / quadratic-map rollout" of the task spec, exercising true
  `state²` feedback rather than a finite lookup.

Generated trivially by iterating the scalar map over a random $a_t$ stream (a few lines;
same shape as `modular_counter`). Per-position int target; length-extrapolation eval.

**Why linear fails, nonlinear succeeds.**
- *Algebraic core (B2).* A state-affine recurrence outputs $g(L(\text{history}))$ for a
  fixed linear functional $L$; the orbit of $h_{t+1}=h_t^2+a_t$ is a degree-$2^t$
  polynomial in the inputs and is not of that form as $t$ grows. A nonlinear-state
  recurrence with a quadratic/`square` feedback path represents the map directly. This
  is *established by the architecture definition*, not by a cited theorem.
- *2a, the monoid angle.* The DeltaProduct/Mamba class reaches transitions that are
  **orthogonal (Householder, det ±1) or diagonal with |λ|≤1** — all **invertible**
  except for uniform rank-drop when a decay hits exactly 0. The squaring map's specific
  **many-to-one collapse** (merging $x$ with $p-x$) is a *non-orthogonal,
  non-uniform-rank-drop* 0/1 transition; products of reflections / uniform decays
  plausibly **cannot realize it**, whereas a nonlinear (or a general non-orthogonal
  dense-linear) recurrence can. This connects the spec's "non-group monoid tracking"
  to a concrete map. **Status: inferred** — I found no published theorem stating the
  Householder/eigenvalue-bounded class cannot realize a *specified* non-invertible monoid
  map; Grazzi/DeltaProduct only assert what reflections *can* reach (the orthogonal
  group), not a matching lower bound on non-group monoids. (Honest caveat: a *general*
  dense linear recurrence with arbitrary 0/1 transition matrices *can* implement any
  DFA via one-hot states; the impossibility is for the **specific bounded-eigenvalue
  reflection/decay parameterization**, which is the class the paper actually studies.)

**Confound control — do the linear levers rescue it?**
- *2a:* negative eigenvalues / reflections give **invertible** (group) transitions;
  the squaring monoid is **non-invertible** → reflections move in the wrong direction
  (they add group elements, not collapsing ones). Uniform decay→0 gives *uniform* rank
  drop, not the *specific* pairwise collapse. **Argued NO** (inferred). *This is the
  cleanest "eigenvalue richness actively does not help" case.*
- *2b:* no linear functional + fixed readout reproduces `state²` feedback at growing
  $t$ → **NO** (algebraic).
- Risk that *does* need controlling: in 2a, because $p$ is finite the task is *finite-state*,
  so a sufficiently expressive *general* linear recurrence (not the bounded class) could
  in principle do it — hence 2a separates the **DeltaProduct/Mamba class**, not "all
  linear." Make this explicit when reporting. 2b avoids the finite-state escape but pays
  in precision sensitivity.

**Harness feasibility.** **High to generate, medium to interpret.** ~40-line task file.
The interpretive risk mirrors the tanh-on-$S_5$ caution: the nonlinear arm must have a
genuine **multiplicative/quadratic** feedback (a plain `tanh` may *not* suffice, since
`tanh` near 0 is ~linear). Use a state level with an explicit squaring/product gate, or
an LSTM (whose input×forget multiplications give state-dependent products). *Action item:
confirm at least one harness layer level has a state-dependent multiplicative path; if
only `tanh`/linear states exist, this probe risks a false "nonlinear also fails."*

---

### PROBE 3 — Streamed integer multiplication / repeated-squaring (arithmetic) — **secondary, inferred**

**Task definition.** Two integers streamed digit-by-digit (LSB-first, interleaved or
on two channels); per-position target = the corresponding digit of the **product**
(with a marker phase for the output digits). Or the single-stream variant: repeated
squaring of a streamed integer.

**Why linear fails, nonlinear succeeds (with an honest caveat).** **Addition** is
*not* a good separator: carry-LSB-first is a 2-state automaton (carry ∈ {0,1}), i.e.
**finite-state / regular**, so the linear class can do it — do **not** use addition.
**Multiplication** is the genuine case: the running partial-product accumulation is not
finite-state (operand-length-dependent intermediate magnitude), and the carry structure
of multiplication is the hard part of length-generalization for sequence models
[length-generalization-of-arithmetic literature; see *open* note §6]. A nonlinear RNN
with multiplicative/counter cells can accumulate partial products; a bounded-state
linear recurrence cannot at growing operand length.

**Confound control.** Reflections/negatives/dense — same as Probe 1: they enlarge the
finite-state class, but multiplication's partial-product accumulation is unbounded-magnitude
→ not rescued. **Status: inferred**; I did not find a clean published `linear-RNN-cannot-multiply`
lower bound, so this ranks below Probes 1–2.

**Harness feasibility.** **Medium.** Encoding two streamed operands and an output phase
as a single per-position int sequence is more fiddly than the counter/map probes
(needs a marker scheme and careful masking), and length-extrapolation means growing
operand bit-width, which interacts with `--eval_lengths`. Generatable, but more design
work; keep as a stretch probe.

---

## 4. Predicted outcome table

Predictions, **stated as hypotheses to be tested** (not results). Columns: a
**negative/reflection-rich linear** baseline (e.g. `e88-linear`, the model that *won*
$S_5$ — the eigenvalue-rich control), a **diagonal SSM** (Mamba/GDN-default,
non-negative eigenvalues), and the **necessary nonlinear** baseline (additive ReLU/LSTM
counter for Probe 1; quadratic/LSTM state for Probes 2–3). "Train T=128, eval at
T≫128."

| Probe | eigenvalue-rich linear (`e88-linear`-style) | diagonal SSM (Mamba/GDN-default) | nonlinear (ReLU/LSTM-counter or quadratic-state) | Separation type |
|---|---|---|---|---|
| **1. `a^n b^n c^n` / Dyck depth** | ✓ at train len, **collapses to chance at T≫128** (no counter) | ✗ even at train len for triple-count; bounded Dyck only | **✓ extrapolates** (additive counter; WGY'18, Délétang'22) | **Established (B1)** |
| **2a. `h²+a` mod p (non-group monoid)** | ✗ (reflections give *invertible* maps; cannot realize the collapse) | ✗ (star-free/aperiodic only; no quadratic feedback) | **✓** (quadratic/LSTM feedback; finite lookup) | **Inferred (B2 / non-group monoid)** |
| **2b. quadratic-map rollout (real)** | ✗ (no `g∘L` reproduces `state²` feedback at growing T) | ✗ | **✓** (genuine `state²` path) | **Inferred (B2, algebraic)** |
| **3. streamed multiplication** | ✗ at growing operand width | ✗ | **✓** (partial-product accumulation) | **Inferred** |

**Built-in sanity check** (already in-repo): the eigenvalue-rich linear column should
**reproduce the $S_5$ pattern** — strong at training length, decaying with length — on
Probe 1, confirming it is a *counting* (not group) failure, distinct from the $S_5$
mechanism (`S5_CONFIG_FLIP.md`, `E1_PARALLEL_SCAN.md` show the linear $S_5$ decay is
intrinsic, reproduced under exact associative scan — i.e. not a rounding artifact).

---

## 5. Honest separation: established vs inferred vs open

**Established (real theorems / verified empirical results).**
- Finite-precision **LSTM / ReLU-RNN implement unbounded counters** (`a^n b^n`,
  `a^n b^n c^n`); **tanh-RNN / GRU cannot** [Weiss, Goldberg & Yahav 2018].
- **LSTMs generalize to counter / context-free tasks; Transformers do not**;
  context-sensitive needs explicit stack/tape [Délétang et al. 2022]; Transformers
  manage only simple counter sub-languages [Bhattamishra et al. 2020].
- **Diagonal SSMs are bounded-memory, star-free** [Sarrof, Veitsman & Hahn 2024];
  **generalized SSMs are provably worse than Transformers at copying** due to fixed
  state [Jelassi et al. 2024]; **SSMs ⊆ TC⁰**, fix = nonlinearity OR input-dependence
  [Merrill, Petty & Sabharwal 2024].
- **Negative eigenvalues / Householder reflections / dense transitions let a LINEAR
  recurrence reach finite groups incl. $S_5$** [Grazzi et al. 2025; DeltaProduct/Siems
  et al. 2025; Cirone et al. 2024] — *measured in-repo* (`GDN_VS_E88_TRANSITION.md`).
  ⇒ **group word problems are NOT a separator.**

**Inferred (follows from definitions / our argument, no single citing theorem).**
- That **state-affine ⇒ output $=g(L(\text{history}))$** and hence cannot reproduce
  `state²` feedback (Probe 2b) — true by construction, but I cite no paper that states
  this specific impossibility as a theorem.
- That the **bounded-eigenvalue Householder/decay class cannot realize a specified
  non-invertible (non-group) monoid map** like `x↦x²` mod p (Probe 2a) — argued from
  orthogonality (det ±1 ⇒ invertible) but **not** backed by a published lower bound.
- That **multiplication** (not addition) separates linear from nonlinear at growing
  operand width (Probe 3) — plausible, uncited lower bound.

**Open (genuinely unsettled / not found in the literature).**
- A clean published lower bound that **input-dependent DENSE linear recurrences (Cirone
  class) cannot do unbounded counting** — the bound exists crisply for *diagonal* SSMs
  (Sarrof-Veitsman-Hahn) and is *strongly implied* for the dense case by bounded
  state, but I did not find it stated for the dense input-dependent class specifically.
- Whether, **empirically in our harness**, the eigenvalue-rich `e88-linear` truly
  collapses on Probe 1 while an added ReLU/LSTM counter extrapolates — untested; this
  is the experiment Probe 1 proposes.
- Whether the **non-group monoid** impossibility (2a) is a real theorem for the
  bounded-eigenvalue class — appears unproven; worth a dedicated theory note.

---

## 6. Recommendation

Run **Probe 1 first** (counting-with-comparison): it is the only candidate with a
*published, architecture-specific* linear-vs-nonlinear separation where the **nonlinear
model is known to succeed** (WGY'18 + Délétang'22), it is the cheapest to build (clone
`modular_counter.py`), and it directly tests the design-constraint lesson — *provided we
add a non-saturating ReLU/LSTM-counter baseline* (a `tanh` arm is a known false negative).
**Probe 2** is the theoretically cleanest "nonlinearity is necessary" case and a strong
second, but its nonlinear arm needs a genuine multiplicative/quadratic feedback path,
and 2a separates the *DeltaProduct/Mamba class* rather than "all linear" — state both
caveats when reporting. **Probe 3** is a stretch goal. In all cases the discriminating
metric is **length extrapolation**, not training-length accuracy (the $S_5$ episode's
central lesson: a linear model can win at training length and still lack the algorithm).

---

## 7. Provenance / how the citations were verified

- **Reused, already-verified in-repo:** every identifier drawn from
  `paper/review/PRECISION_NONLINEARITY_RESEARCH.md` §9 (that bibliography was fetched
  and re-verified arXiv-id-by-arXiv-id in a prior session): Weiss-Goldberg-Yahav 2018
  (arXiv:1805.04908), Merrill-Petty-Sabharwal 2024 (2404.08819), Grazzi et al. 2025
  (2411.12537), DeltaProduct/Siems et al. 2025 (2502.10297), Cirone et al. 2024
  (2402.19047), Movahedi et al. 2025 (2503.10799), Khavari et al. 2025 (2508.07395),
  Katharopoulos et al. 2020 (2006.16236), Yang et al. 2024 DeltaNet (2406.06484),
  Barrington 1989, Liu et al. 2023 (2210.10749).
- **Newly fetched and verified THIS session** (arXiv abstract page loaded; title +
  authors + year confirmed): **[verified this session]**
  - Délétang, Ruoss, Grau-Moya, Genewein, Wenliang, Catt, Cundy, Hutter, Legg, Veness,
    Ortega. *Neural Networks and the Chomsky Hierarchy.* 2022. arXiv:2207.02098.
  - Bhattamishra, Ahuja, Goyal. *On the Ability and Limitations of Transformers to
    Recognize Formal Languages.* EMNLP 2020. arXiv:2009.11264.
  - Sarrof, Veitsman, Hahn. *The Expressive Capacity of State Space Models: A Formal
    Language Perspective.* NeurIPS 2024. arXiv:2405.17394.
  - Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better
    than State Space Models at Copying.* 2024. arXiv:2402.01032.
  - Re-confirmed: Weiss, Goldberg, Yahav. *On the Practical Computational Power of
    Finite Precision RNNs for Language Recognition.* ACL 2018. arXiv:1805.04908.
- **Why inline verification rather than a sub-agent fan-out:** the absolute
  no-fabrication rule makes a verifiable chain-of-custody worth more than breadth here.
  The foundational class-boundary citations were already web-verified in-repo, so only a
  handful of **new** task-specific references were needed; I fetched each directly,
  keeping every identifier under lead verification. Anything I could not tie to a fetched
  source is labelled *inferred* or *open* in §5, not asserted.

---

## 8. Bibliography (this document)

New / task-specific (verified this session):

1. G. Délétang, A. Ruoss, J. Grau-Moya, T. Genewein, L. K. Wenliang, E. Catt, C. Cundy,
   M. Hutter, S. Legg, J. Veness, P. A. Ortega. *Neural Networks and the Chomsky
   Hierarchy.* 2022 (ICLR 2023). arXiv:2207.02098.
2. S. Bhattamishra, K. Ahuja, N. Goyal. *On the Ability and Limitations of Transformers
   to Recognize Formal Languages.* EMNLP 2020. arXiv:2009.11264.
3. Y. Sarrof, Y. Veitsman, M. Hahn. *The Expressive Capacity of State Space Models: A
   Formal Language Perspective.* NeurIPS 2024. arXiv:2405.17394.
4. S. Jelassi, D. Brandfonbrener, S. M. Kakade, E. Malach. *Repeat After Me:
   Transformers are Better than State Space Models at Copying.* 2024. arXiv:2402.01032.
5. G. Weiss, Y. Goldberg, E. Yahav. *On the Practical Computational Power of Finite
   Precision RNNs for Language Recognition.* ACL 2018. arXiv:1805.04908.

Foundational (identifiers reused from the repo's already-verified
`PRECISION_NONLINEARITY_RESEARCH.md` §9):

6. W. Merrill, J. Petty, A. Sabharwal. *The Illusion of State in State-Space Models.*
   ICML 2024. arXiv:2404.08819.
7. R. Grazzi, J. Siems, A. Zela, J. K. H. Franke, F. Hutter, M. Pontil. *Unlocking
   State-Tracking in Linear RNNs Through Negative Eigenvalues.* ICLR 2025.
   arXiv:2411.12537.
8. J. Siems, T. Carstensen, A. Zela, F. Hutter, M. Pontil, R. Grazzi. *DeltaProduct:
   Improving State-Tracking in Linear RNNs via Householder Products.* 2025.
   arXiv:2502.10297.
9. N. M. Cirone, A. Orvieto, B. Walker, C. Salvi, T. Lyons. *Theoretical Foundations of
   Deep Selective State-Space Models.* NeurIPS 2024. arXiv:2402.19047.
10. S. Movahedi, F. Sarnthein, N. M. Cirone, A. Orvieto. *Fixed-Point RNNs:
    Interpolating from Diagonal to Dense.* NeurIPS 2025. arXiv:2503.10799.
11. B. Khavari et al. *Parity Requires Unified Input Dependence and Negative
    Eigenvalues in SSMs.* 2025. arXiv:2508.07395.
12. A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret. *Transformers are RNNs.* ICML
    2020. arXiv:2006.16236.
13. S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. *Parallelizing Linear Transformers
    with the Delta Rule over Sequence Length (DeltaNet).* NeurIPS 2024. arXiv:2406.06484.
14. D. A. Barrington. *Bounded-Width Polynomial-Size Branching Programs Recognize
    Exactly Those Languages in NC¹.* JCSS 38(1):150–164, 1989.

In-repo evidence (this project, not external):
`paper/review/GDN_VS_E88_TRANSITION.md`, `S5_CONFIG_FLIP.md`,
`S5_SYMMETRIC_RESULTS.md`, `PRECISION_NONLINEARITY_RESEARCH.md`, `E1_PARALLEL_SCAN.md`.
</content>
</invoke>
