# S5 MECHANISM SYNTHESIS — what explains the linear-state E88 win on S5?

**Task:** `s5-mechanism-synthesis` · model claude:opus · pure synthesis (no GPU).
**Inputs read (all seven required + two causal follow-ups):**
`PRECISION_NONLINEARITY_RESEARCH.md`, `S5_SYMMETRIC_RESULTS.md`, `S5_CONFIG_FLIP.md`,
`E1_PARALLEL_SCAN.md`, `E2_PRECISION_SWEEP.md`, `E5_ABLATE_INPUTDEP.md`,
`M2RNN_LINEAR_ABLATION.md`, plus `EIGENVALUE_CAUSAL_TEST.md` and
`GDN_VS_E88_TRANSITION.md` (the causal capstone the diagnostic battery produced).
**`paper/main.typ` was NOT edited** — implications are specified, not applied.

Every quantitative claim below traces to a committed result. Headline figures were
spot-checked against the raw committed JSONs
(`results/eigenvalue_causal_20260604/summary.json`: GDN+neg seed-mean S5
1.0000/1.0000/0.9999/0.9916; `results/s5_symmetric_20260603/eval/e88-linear_S5_seed42.json`:
S5 0.9991/0.6723/0.3462/0.1829 — matching the per-seed values cited in the source docs).

---

## 0. The thing to be explained

After symmetric per-architecture CMA tuning (`S5_SYMMETRIC_RESULTS.md`), the
**linear-state** variant `e88-linear` (`linear_state=1`, `use_gate=1`) won the S5 slot
at training length and led at every extrapolation length:

| Model (S5, seed-mean) | T=128 | T=256 | T=512 | T=1024 |
|---|---:|---:|---:|---:|
| **e88-linear** (linear state) | **0.9997** | 0.7515 | 0.3909 | 0.2002 |
| e88-tanh (nonlinear state) | 0.9888 | 0.6296 | 0.3216 | 0.1678 |
| GDN | 0.5446 | 0.2801 | 0.1441 | 0.0759 |
| M²RNN-CMA | 0.1655 | 0.0858 | 0.0479 | 0.0276 |
| random (1/120) | 0.0083 | 0.0083 | 0.0083 | 0.0083 |

Two facts to hold together: (i) `e88-linear` is at ceiling (0.9997) at the training
length T=128 — exactly where a model "supposed" to be TC⁰-bounded should not be able to
track S₅; (ii) its accuracy **decays toward chance with length** (0.20 at T=1024) — the
signature of a solution that has not acquired a length-robust S₅ algorithm. The
literature review (`PRECISION_NONLINEARITY_RESEARCH.md`) named five candidate
explanations. The diagnostic battery was built to adjudicate among them. It does.

---

## 1. Adjudication of the five candidate explanations

### Candidate 1 — the state-nonlinearity KNOB (tanh vs linear). **MINOR contributor; refutes the "tanh-latching is necessary" reading.**

**Evidence (config-flip 2×2, `S5_CONFIG_FLIP.md`; M2RNN ablation, `M2RNN_LINEAR_ABLATION.md`).**

The config-flip holds geometry/lr fixed and toggles only `linear_state`:

| S5 @T128 (seed-mean ± SD) | tanh (ls=0) | linear (ls=1) |
|---|---|---|
| **config T** | A: 0.9888 ± 0.0111 | D: **0.9998 ± 0.0002** |
| **config L** | C: 0.9967 ± 0.0042 | B: 0.9997 ± 0.0005 |

- Linear > tanh in **all 8** config×length cells; the edge is small at T=128 (all four
  cells ≥ 0.989, at ceiling) and **grows with extrapolation** (config T: D−A = +0.0110
  @T128 but +0.214 @T256).
- For E88/delta-correction the knob is a **small, same-direction edge**: linear is
  slightly better, but **the tanh variant also solves S₅** (0.9888 at T=128).
- For the M²RNN raw-write family the same knob matters *a lot* and in the **same
  direction** — removing the tanh nearly doubles S5@T128 (16.6 → 31.1) and lifts S3 off
  chance (`M2RNN_LINEAR_ABLATION.md`). tanh *hurts* a raw-write state-tracker.

**Supports:** the state-nonlinearity is not load-bearing for the E88 win; where it
matters (raw-write), *removing* it helps. **Rules out:** that the S₅ result requires a
tanh "latch." The win is, if anything, slightly *better* without the tanh — the opposite
of a "needs both halves" story. The knob is real but secondary, and its sign is the
reverse of the paper's framing (see §3).

### Candidate 2 — the CMA CONFIG (geometry/lr; rough ~300-step search). **RULED OUT as the driver.**

**Evidence (config-flip 2×2, `S5_CONFIG_FLIP.md`).** Holding the knob fixed and toggling
the config:
- At tanh: C−A = +0.0079 @T128 (config L slightly better).
- At linear: B−D = −0.0001 @T128, **−0.092 @T256** (config **T** better, increasingly so
  at length).

The config effect is **small, sign-flipping, and dominated in magnitude by the knob** at
every extrapolation length — an interaction, not a main effect. The `e88-linear` win is
**knob-driven, not config-driven**.

*Positive by-product (an honest finding, not the explanation):* cell **D** = config-T +
linear is the strongest S₅ cell at every T≥256 (0.8436/0.4384/0.2255), beating the
*published* `e88-linear` (cell B, config L). The published linear arm sits on a *mildly
suboptimal* geometry — direct evidence the per-arm ~300-step CMA undersampled the
geometry space and under-credited the linear knob's true ceiling. This implicates the
search budget, not the config axis, as a confound in the *original ranking*.

### Candidate 3 — INPUT-DEPENDENCE / eigenvalue range / dense transition. **STRONGLY SUPPORTED. This is the dominant mechanism, demonstrated causally.**

This is the leading literature explanation (Cirone 2024; Grazzi 2025; Khavari 2025;
DeltaProduct/Siems 2025; Movahedi 2025 — `PRECISION_NONLINEARITY_RESEARCH.md` §5/§6.5).
Two experiments converge on it from different angles, and a third turns correlation into
causation.

**(3a) Input-dependent transition is necessary (`E5_ABLATE_INPUTDEP.md`).** Replacing the
input-dependent Mamba decay with a learned per-head **constant** decay
(`decay_mode=constant`, param-matched, still linear-state, eigenvalues still in (0,1))
collapses S5@T128 **0.9997 → 0.3987** and S3 too (1.0000 → 0.4027). Removing the
*output gate* (`use_gate=0`) leaves the win **fully intact** (0.9997 → 0.9999, and
*better* extrapolation). So the load-bearing channel is the **input-dependent transition
(selectivity)**, not the output gate. This is Cirone-style selectivity, shown necessary.

**(3b) Negative / reflection eigenvalue is causally necessary AND sufficient
(`EIGENVALUE_CAUSAL_TEST.md`, built on `GDN_VS_E88_TRANSITION.md`).** This is the
strongest single result in the battery. The S5 winner (e88-linear) and the S5 loser (GDN)
are separated by exactly one property: the **sign of the along-key eigenvalue** of the
per-token transition. e88-linear's update `A_t = decay·I − k̂k̂ᵀ` has along-key eigenvalue
`decay − 1 < 0` (a *reflection*, verified directly on the running operator: 100% negative,
min −1.0000); GDN's `A_t = g(I − βk̂k̂ᵀ)` with `β∈(0,1)` is confined to `g(1−β) > 0`
(100% positive). Flipping the sign reachability **moves the win, on both models, in both
directions**:

| Configuration | can reach negative along-key eig? | S5 128 | S5 256 | S5 512 | S5 1024 |
|---|---|---:|---:|---:|---:|
| GDN baseline | NO (positive) | 0.5446 | 0.2801 | 0.1441 | 0.0759 |
| **GDN + neg eig** (`allow_neg_eigval=True`) | YES | **1.0000** | **1.0000** | **0.9999** | **0.9916** |
| e88-linear baseline | YES (reflection) | 0.9997 | 0.7515 | 0.3909 | 0.2002 |
| **e88-linear clamped** (`decay·(I−k̂k̂ᵀ)`, eig=0) | NO | **0.2690** | 0.1357 | 0.0736 | 0.0399 |
| e88-linear raw_write (eig=decay>0) | NO | 0.2626 | 0.1380 | 0.0733 | 0.0401 |

**Two independent models, five configurations: every config that can reach a negative
along-key eigenvalue solves S₅; every one that cannot, fails.** Adding negatives to GDN
*creates* the win (0.5446 → 1.0000) by a single flag; removing them from e88-linear
*destroys* it (0.9997 → 0.2690). This is the Grazzi (2025) / DeltaProduct (Siems 2025)
negative-eigenvalue / reflection mechanism, demonstrated causally on two real gated
delta-rule models.

**Reconciling 3a and 3b (a subtlety the synthesis must make explicit).** E5 reported that
the eigenvalue is "already in (0,1) by construction, so the negative-eigenvalue mechanism
is architecturally ruled out." That statement is about the **decay scalar** (the
*perpendicular*/diagonal eigenvalue, which is indeed in (0,1)). The eigenvalue-causal test
shows the decisive eigenvalue is a **different** eigenvalue of the *same* operator: the
**along-key** eigenvalue of `decay·I − k̂k̂ᵀ`, which is `decay − 1 < 0` and lives in the
*reflection (delta-correction) term*, not in the decay. So E5's "(0,1) by construction"
correctly describes the decay but *understated* the model: e88-linear **does** reach a
negative eigenvalue, via the `−k̂k̂ᵀ` reflection. The eigenvalue-causal test corrects and
completes E5. Together they give the **unified Khavari (2025) condition** — state tracking
needs **both input-dependence AND negative eigenvalues** — each half shown necessary by an
independent ablation (E5 kills input-dependence → collapse; eigenvalue-causal kills the
negative eigenvalue → collapse).

**Supports:** input-dependent dense transition with a reachable negative (reflection)
eigenvalue is the mechanism. This is the dominant, causally-isolated explanation.

### Candidate 4 — FINITE-LENGTH triviality (fixed-length S₅ ∈ TC⁰; decay with length = the asymptotic ceiling). **PARTIALLY SUPPORTED for e88-linear, but materially complicated by the GDN+neg counterexample.**

**Evidence (length curves across all experiments).** Every experiment reproduces the same
qualitative shape for e88-linear and its relatives: ceiling at T=128, monotone decay
toward chance by T=1024 (e88-linear 0.9997 → 0.2002; serial=scan identical, E1; fp64 ≈
bf16 identical, E2). On the standard reading
(`PRECISION_NONLINEARITY_RESEARCH.md` §1, §6.6) this *is* the asymptotic TC⁰ failure
showing through, and the fixed-length T=128 win is *expected and theory-consistent*: a
length-128 S₅ instance is trivially in TC⁰ (indeed AC⁰).

**But the GDN+neg result breaks the "linear must decay" inference.** Unlocking negative
eigenvalues does not merely lift the T=128 number — it **essentially removes the
length-extrapolation collapse**: GDN+neg holds **0.9916 at T=1024** (vs GDN-baseline
0.0759, vs e88-linear 0.2002). A linear-state recurrence with reflections produces a
*length-robust* S₅ solution here. So:
- The **decay of e88-linear** is a property of the *particular learned solution* on its
  *particular operator structure* (decay-on-identity, β=1, where the reflection magnitude
  is not freely composable across tokens), **not** an absolute ceiling forced by
  "linearity."
- "Fixed-length triviality" correctly explains why a *fixed-length win is not a theorem
  violation*, but it does **not** explain the win, and it is **not** the reason a given
  linear model decays — a sibling linear model (GDN+neg) does not.

**Supports:** the finite-length framing is the correct way to *reconcile the win with the
theorem* (the win is expected, not anomalous). **Rules out:** using length-decay as
evidence that "linear-state ⟹ must collapse." The length axis is governed by the
transition's eigenvalue structure, not by the state being linear.

### Candidate 5 — SERIAL-PRECISION ARTIFACT (H1/H2: serial rounding as usable nonlinearity; parallel-scan reassociation). **RULED OUT.**

**Evidence (E1 serial-vs-scan + E2 precision sweep).**
- **E1 (`E1_PARALLEL_SCAN.md`):** the same trained e88-linear evaluated via the serial
  time-loop and via an *exact associative scan* of the same affine model gives
  **bit-identical accuracy in fp32** and **Δacc ≤ 0.002 in bf16** (≪ ±0.05–0.09 seed
  noise) at every length — even though bf16 reassociation measurably perturbs the logits
  (per-token argmax agreement falls to 0.76 @T=1024). The win and its decay are intrinsic
  to the recurrence, **not** the serial execution order. (Caveat: the scan is exact *only*
  because `linear_state=1`; the tanh sibling is not associatively scannable — so the
  winner is literally a linear scan.)
- **E2 (`E2_PRECISION_SWEEP.md`):** sweeping the *same* weights at fp64/fp32/bf16 gives
  **flat** accuracy (paired bf16−fp64 < 1.5pp, sign-flipping at T=512/1024, ≪ seed SD).
  The fp64-serial idealized-linear proxy reproduces **both** the T=128 win **and** the
  length decay essentially identically to bf16. There is a faint, honestly-reported
  bf16 ≳ fp32 ≳ fp64 lean at short lengths (the *direction* H1 predicts), but it is far
  too small (<1.5pp) to be a 45-point separation from baselines.

**Supports:** nothing in favor of H1/H2. **Rules out:** the serial-precision / rounding /
execution-order artifact as the driver. The win survives exact arithmetic (fp64) and a
completely different FP reduction order (parallel scan). H1/H2 are rejected on this axis.

---

## 2. Ranked, honest conclusion

1. **DOMINANT — input-dependent transition with a reachable negative (reflection)
   eigenvalue** (Candidate 3). Causally isolated in both directions on two models. This is
   the unified Cirone/Grazzi/Khavari/DeltaProduct mechanism: selectivity (E5) **and**
   negative/reflection eigenvalue (eigenvalue-causal test), each shown necessary, jointly
   sufficient (GDN+neg = both → solves S₅ length-robustly).
2. **MODULATING — finite-length triviality** (Candidate 4): the correct lens for
   *reconciling the fixed-length win with the asymptotic theorem* (the win is expected),
   but **not** the cause of the win, and **not** the reason any particular linear model
   decays — GDN+neg does not decay.
3. **MINOR — the state-nonlinearity knob** (Candidate 1): a small, real, *same-direction*
   edge for E88 (linear ≳ tanh) and a large one for raw-write (linear ≫ tanh). Its sign
   refutes a "tanh-latching is required" reading.
4. **RULED OUT — the CMA config** (Candidate 2): small, sign-flipping, knob-dominated.
   (By-product: the published linear arm sits on a suboptimal geometry; the ~300-step CMA
   undersampled — a caveat on the *original ranking margin*, not on the mechanism.)
5. **RULED OUT — serial-precision artifact H1/H2** (Candidate 5): E1 (serial=scan) + E2
   (flat vs precision, fp64≈bf16) reject it cleanly.

**What remains open.**
- *Why e88-linear (decay-on-identity, β=1) decays with length while GDN+neg does not* is
  not fully resolved. The eigenvalue *sign* is reachable in both; the difference is likely
  in how freely the reflection *composes* across tokens (β=1 fixed vs β∈(0,2) learned) and
  in the trained solution's robustness — a quantitative open question (a DeltaProduct
  `n_h`-style "how many composable reflections" analysis would close it).
- *Clean S₃-specificity within the E88 family.* Both E88 negative-eigenvalue removal routes
  (clamp-to-0, raw_write) also damage S₃ via mechanisms orthogonal to the sign
  (retention-zeroing; loss of delta-correction). The clean positive-eigenvalue S₃ control
  is **GDN**, which solves S₃ (0.9243) with positive eigenvalues — so S₃-specificity of
  the sign lever rests on the GDN arm (unambiguous there).
- *Allowing negative/complex eigenvalues inside E88* (a `decay=2σ−1` or complex-diagonal
  parameterization) was documented as a future code change, not faked — the complement of
  the clamp experiment.

### Reconciliation with the asymptotic TC⁰/NC¹ theory

The theory is **sound and untouched**. Barrington (1989): S₅ word problem is NC¹-complete.
Merrill–Petty–Sabharwal (2024): log-precision **diagonal / non-input-dependent** SSMs
(S4/S6) are in L-uniform TC⁰, so *assuming TC⁰ ≠ NC¹* they cannot solve S₅ asymptotically
— and the authors explicitly state the fix is **nonlinearity OR input-dependent
transitions**. The crucial point the experiments make precise: **`e88-linear` is on the
expressive side of that "OR."** It is an *input-dependent dense* linear recurrence with
*reflection* (negative-eigenvalue) transitions — exactly the subclass Cirone (2024) shows
matches nonlinear-RNN expressivity and Grazzi/DeltaProduct (2025) show can track
permutation groups *while staying linear-in-state*. So:

- A fixed-length-128 S₅ win by `e88-linear` is **fully consistent** with the theorem (the
  instance is in TC⁰; asymptotically the linear model's *particular* solution decays).
- The theorem does **not** say "any linear-state recurrence cannot do S₅." It bounds the
  *diagonal / positive-eigenvalue / non-input-dependent* class. The relevant expressivity
  axis is the **transition operator's structure (input-dependence + eigenvalue range)**,
  **not** whether there is a `tanh` on the state. The experiments relocate the dividing
  line from "linear-vs-nonlinear *state*" to "weak-vs-expressive *transition*."

---

## 3. Implications for the paper (specified, NOT applied — `main.typ` untouched)

### (a) `main.typ:213–217` — "the nonlinear recurrence reaches something a linear one provably cannot … a linear-in-time recurrence is held away from the task by a classical complexity argument." **MUST be softened.**

As written this asserts the complexity argument as the *explanation of the experimental
ranking*. The experiments contradict the strong reading on three counts:
1. At training length the **linear-state** model *beat* the nonlinear one (0.9997 vs
   0.9888) — the ranking is the *reverse* of "linear is held away."
2. A linear-state recurrence with negative eigenvalues (**GDN+neg**) solves S₅
   **length-robustly** (0.9916 @ T=1024) — "a linear one provably cannot" is false for the
   *input-dependent, negative-eigenvalue* linear subclass; the provable bound is for the
   diagonal/positive class.
3. The win/decay is intrinsic to the linear operator (E1 serial=scan, E2 fp64≈bf16) — not
   a precision artifact propping up an otherwise-incapable linear model.

**Recommended softening.** Frame the complexity argument as an **asymptotic,
idealized-arithmetic** statement about the **diagonal/positive-eigenvalue (non-input-dependent)**
linear class, and report the empirical S₅ result through **length-extrapolation**, not a
single training-length number. Something like: *"Asymptotically, a diagonal or
positive-eigenvalue linear-in-time recurrence is held away from S₅ by a classical
complexity argument (TC⁰ ⊊ NC¹, assuming the separation). At fixed training length the
distinction is not visible — every sufficiently expressive model, including input-dependent
linear ones, solves the length-128 instance — so the separation is reported via
length-extrapolation behavior rather than the training-length score."* Do **not** claim the
complexity argument explains why the nonlinear model out-ranks the linear one at training
length; it does not (the linear one ranks higher).

### (b) §1 "State tracking needs both [delta correction + tanh-with-latching]" (`main.typ:243`, with the ingredient set `main.typ:233–246`, and the update form `main.typ:24` `S ← tanh(d S + k(silu(v) − Sᵀk)ᵀ)`). **MUST change.**

The empirical S₅ winner is **`e88-linear` — the variant with the `tanh` removed.** The
tanh-bearing sibling also solves S₅ (0.9888) but ranks *below* the linear variant at every
length, and on the raw-write family the tanh *actively hurts* (M²RNN: removing tanh nearly
doubles S5@T128). So "state tracking **needs** both" — read as "the tanh-latch half is
necessary" — is **not supported**: the tracking survives, and slightly improves, when the
tanh latch is removed.

What the experiments show is actually necessary (each demonstrated by an ablation that
destroys the win):
- **delta-correction** (read-modify-write) — removing it (`raw_write`) destroys S₅ (and S₃);
- **a reachable negative / reflection eigenvalue** — clamping it to ≥0 destroys S₅;
  adding it to GDN creates S₅ (eigenvalue-causal test);
- **input-dependent transition (selectivity)** — making the decay constant destroys S₅ (E5).

**Recommended change.** Either (i) **narrow the "needs both" claim to a statement about the
realizability *construction*** (the Lean orthonormal-key proof uses the tanh form, and
*that proof* uses latching) while stating plainly that **empirically the linear-state
variant matches or beats it**, so the tanh latch is *sufficient-in-the-proof* but *not
empirically necessary* for S₅; or (ii) **replace the second ingredient.** The empirically
load-bearing trio is **delta-correction + input-dependent (selective) transition +
reachable negative/reflection eigenvalue**, not delta-correction + tanh-latch. If the
paper wishes to keep the latching theory (Theorem set F), it should be presented as a
property of the *deployed tanh E88* and its *retention* behavior — **not** as a *necessary*
condition for the S₅ separation, which the linear winner refutes.

### (c) Recommended honest framing (for whoever edits the paper)

1. **Keep TC⁰/NC¹ as theory.** Barrington + Merrill–Petty–Sabharwal stand; state the bound
   as asymptotic, idealized-arithmetic, and **about the diagonal/positive-eigenvalue
   non-input-dependent linear class**.
2. **Report S₅ via length-extrapolation**, and state explicitly that **a fixed-length
   linear win is expected** (the length-128 instance is in TC⁰) and is *not* a theorem
   violation.
3. **Fold in the mechanism the experiments isolated:** the S₅ separation among these
   models is governed by the **transition operator** — input-dependence (selectivity) and
   **reachability of a negative / reflection eigenvalue** — not by a state nonlinearity.
   This is the Cirone/Grazzi/Khavari/DeltaProduct axis, here shown *causally* on the
   paper's own architectures.
4. **Decouple the precision question:** state that serial-vs-parallel execution and
   fp64/fp32/bf16 were tested and the result is *invariant* (E1/E2) — pre-empting the
   "is it a rounding artifact?" referee objection with a clean negative.
5. **Correct the linear-vs-nonlinear framing:** the empirical evidence is that the
   *state* nonlinearity is **not** the lever (linear ≳ tanh for E88; linear ≫ tanh for
   raw-write). The paper's contribution should be stated in terms of **delta-correction +
   transition structure**, which is robust, rather than the **tanh latch**, which the
   experiments do not support as necessary.

### (d) NEW positive contribution the experiments produced

**The eigenvalue-causal test is a novel, self-contained finding that deserves its own
paragraph (and arguably its own figure).** It is a clean, *bidirectional* causal
demonstration on two real gated delta-rule models that **negative-eigenvalue / reflection
reachability is the S₅ lever**:
- **Adding** reachable negative eigenvalues to GDN (one flag, `allow_neg_eigval=True`)
  **creates** the S₅ win — 0.5446 → 1.0000 @T=128 — and **erases the length-extrapolation
  collapse** (0.0759 → **0.9916** @T=1024).
- **Removing** them from E88-linear (sign-isolating clamp, verified on the running
  operator) **destroys** the win — 0.9997 → 0.2690.
- Across **two models × five configurations**, *every* config that can reach a negative
  along-key eigenvalue solves S₅ and *every* one that cannot fails — a perfect separation.

This is more than a control: it is a **positive result** — an experimental, causal
confirmation of the Grazzi (2025) / DeltaProduct (Siems 2025) negative-eigenvalue
state-tracking theory on real trained models, including the striking **length-robust S₅
from a *linear* recurrence** (GDN+neg @T=1024 = 0.99). It reframes the paper's S₅ story
from "nonlinear beats linear" to "**reflection-reachability** (a property of the *linear*
transition operator) is what tracks the non-solvable group," which is a stronger and more
defensible claim than the current one. A secondary novel by-product
(`S5_CONFIG_FLIP.md` cell D) is that the published `e88-linear` geometry is mildly
suboptimal — the linear knob's true ceiling is higher than the ~300-step CMA reported.

---

## 4. Validation against task checklist

- [x] **All seven input docs read** (+ the two causal follow-ups that the battery
  produced); every quantitative claim traces to a committed result, with two headline
  numbers spot-checked against raw JSONs (§0).
- [x] **All five candidate explanations adjudicated** with support/rule-out and real
  numbers (§1), **ranked honest conclusion** (§2), **reconciled with TC⁰/NC¹ theory**
  (§2, end): the bound holds asymptotically for the diagonal/positive class; e88-linear is
  the input-dependent/negative-eigenvalue subclass the theorem's "OR" exempts; fixed-length
  win is expected.
- [x] **Paper implications specified WITHOUT editing `main.typ`:** 213–217 softening (3a);
  §1 "needs both" change (3b); recommended honest framing (3c); the eigenvalue-causal
  novel contribution (3d).
- [x] This document committed; **not pushed** (per task).
