# X22T mapping receipt — our arms against Merrill et al. (2026), Defs. 2/3/5/10

**Source (primary, read this session).** Merrill, Jiang, Li, Lin, Sabharwal, *Why Are
Linear RNNs More Parallelizable?* Proceedings of the 43rd International Conference on
Machine Learning, Seoul, PMLR 306:88230–88260, 2026. Full proceedings text read from a
local copy (`/tmp/lunaroute-doc-73dc0d7c-4820-45e1-b8b1-d884aa6af0b2.md`). Definitions
below are transcribed from that text; section/theorem numbers are the paper's.

**Status of this receipt.** Every mapping claim has two sides: (i) the paper-side
definition, quoted; (ii) our-side object, with a repo file:line citation **and** a Lean
theorem that proves the algebraic/syntactic match, where available. Algebraic shape
does not establish transfer of the paper's rational-approximation complexity bounds to
exact reals, nor realization by a configured Emender activation.

**Pass-2 update (operator directive: re-prove, do not cite).** The mapping claims are no
longer citations. R1 lives in `formal/lean/ElmanProofs/Expressivity/ArmMapping.lean`, R3/R4
in `WeightedAutomata.lean`, R2 in `CounterMachine.lean`. Each claim below now carries the
declaration that proves it, and every place where our re-proof *cannot* reproduce the paper
is marked **DIVERGENCE**. The paper's complexity results and four-layer construction remain citations.
The counter-machine result is an additive-update `StreamModel` embedding, not a
completed closed-form transition proof. R4 gives homogeneous-product identities and
padding/depth arithmetic, not circuit-class membership.

---

## 0. The precision column — read this first

| | Merrill et al. | Emender X-series |
|---|---|---|
| datatype | rational semiring `⟨ℚ,⊕,⊗⟩` (Def. 1), with poly/log-precision variants of `exp`, `√` (Def. 10) | `ℝ`, exact in Lean; bf16/fp32 in the trained models |
| precision | **Def. 10**: `p_h(n) = max_{w∈Σⁿ} |encode(h(w))|` — bits as a function of *input length*; log (`O(log n)`) and poly (`O(nᶜ)`) are the two studied regimes, both **growing** | `BoundedAbstraction M n`: a **fixed** number of channel values `n = 2^B`, independent of length (X21T) |
| object | language recognizer, final-token readout (Def. 9) | `StreamModel`: one shared readout per token on words **and** on `ω`-streams (X21T/X22T) |

**Consequence (the pivot of the regime theorem).** Their log precision is *not* our
fixed budget: `O(log n)` bits means `2^{O(log n)} = poly(n)` values at length `n`. So
their entire analysis lives on the **growing-precision axis**, which our X20T/X21T/X22T
theorems do not address, while our fixed-budget theorems address a regime they do not
study. The two developments are **complementary regimes of the same object**, and the
Lean file says exactly that: `PrecisionRegime.log2_schedule_is_growing` proves `log₂` is
a growing schedule, and `PrecisionRegime.precision_regime_split` proves the fixed-budget
half. The counting implication in that skeleton is an
**impossible-global-abstraction corollary**: its premise gives an all-length global
abstraction for each `n`, and already `h 0` contradicts
`countModel_no_omega_finite_state`, for every schedule (growing or not). The
length-restricted abstraction bridge to Def. 10 is **UNPROVED**; the skeleton does
not show that growing budget buys counting.

---

## 1. Arm A — our `fla-gdn` (linear gated-delta): **DPLR LRNN instance**

### Paper side

> **Definition 4 (LRNN).** Given `x₁,…,xₙ ∈ K^d`, an LRNN head computes matrix-valued
> states `S₁,…,Sₙ ∈ K^{d×d}` via `S_t = A_t(x_t)·S_{t−1} + b_t(x_t)`, where
> `A_t(x_t) ∈ K^{d×d}` and `b_t(x_t) ∈ K^d` have arithmetic closed forms independent
> of `t`.

> **Definition 5 (DPLR).** An LRNN is *DPLR* if it satisfies `A_t = D_t − k_t v_tᵀ`,
> where `D_t` is diagonal, `k_t, v_t` are vectors, and all have an arithmetic closed
> form in terms of `x_t`.

> **DeltaNet** (Yang et al. 2024): `A_t = I − β_t k_t k_tᵀ`, `b_t = v_t k_tᵀ`, where
> `k_t, v_t, β_t` depend on `x_t` via projection. […] Grazzi et al. (2025) show how, for
> fixed-precision DeltaNet, the range of `β_t` is crucial for expressivity: restricting
> `β_t ∈ (0,1)` prevents recognizing parity, but `β_t ∈ (0,2)` allows it.

### Our side

`fla`'s GatedDeltaNet head with `allow_neg_eigval=True` computes (per head, per token)

```
S ← diag(exp g_t) · S                 # multiplicative decay, g in log space
S ← S + β_t · k_t ⊗ (v_t − Sᵀ k_t)    # rank-one delta write
```

i.e. `S_t = diag(exp g_t)·(I − β_t k_t k_tᵀ)·S_{t−1} + β_t k_t v_tᵀ`, so

```
A_t = diag(exp g_t) · (I − β_t k_t k_tᵀ) = D_t − k̃_t ṽ_tᵀ,
      D_t = diag(exp g_t),  k̃_t = β_t ⊙ exp g_t ⊙ k_t,  ṽ_t = k_t.
```

* Repo: `.venv/lib/python3.12/site-packages/fla/ops/gated_delta_rule/fused_recurrent.py:136`
  (`b_h *= exp(b_g)`), `:146–150` (delta + write, `b_v = b_beta * (b_v − Σ b_h·b_k)`,
  `b_h += b_v ⊗ b_k`); `fla/ops/gated_delta_rule/chunk.py:274`
  (`beta = fused_beta_sigmoid(beta_raw, scale=2.0 if allow_neg_eigval else 1.0)`);
  `fla/layers/gated_deltanet.py:73–77` (docstring: `allow_neg_eigval` "the beta will be
  multiplied by 2", citing arXiv:2411.12537).
* Emender wrapper: `ndm/models/complex_eig_head.py:239–241`
  (`beta = gdn.b_proj(x).sigmoid(); if self.allow_neg_eigval: beta = beta * 2.0`).
* **Lean proof of the DPLR identity** (R1): `PrecisionRegime.gatedDelta_isDPLR` and the
  sharper `ArmMapping.gdnMatrix_isDPLR`, on the *hand-rolled* matrix algebra of
  `ArmMapping.matMul`/`diagM`/`oneM`.
* **Lean proof that our recurrence *is* a Def. 4 LRNN step** (R1):
  `ArmMapping.gdnStep_eq_lrnnStep` — `gdnStep g β k v S = lrnnStep (gdnMatrix g β k) (β ⊙ k ⊗ v) S`,
  where `gdnStep` is the fla kernel recurrence transcribed coordinatewise.
* **Lean proof of the DeltaNet special case** (R1): `ArmMapping.gdnMatrix_zero_decay`
  (`gdnMatrix 0 β k = deltaNetMatrix β k`, i.e. `I − β k kᵀ` exactly) and
  `ArmMapping.gdnStep_zero_decay_eq_deltaNet`.
* **Lean proof of the per-channel orientation** (R1):
  `ArmMapping.gdnMatrixPerChannel_isDPLR` — `D − k (D k)ᵀ` is DPLR too (see divergence 4).
* **Lean proof of the β range**: `PrecisionRegime.negEigval_beta_range` —
  `2 · sigmoid z ∈ (0,2)` for every `z`. `allow_neg_eigval=True` therefore sits in the
  Grazzi et al. `β ∈ (0,2)` (parity-capable) range, not the `(0,1)` range.

### Verdict: **MATCH on the parameterization, with four differences**

| | paper DeltaNet | paper RWKV-7 | our fla-gdn (`allow_neg_eigval`) — all three `IsDPLR` in Lean |
|---|---|---|---|
| `A_t` | `I − β k kᵀ` (`D_t = I`) | `diag(w_t) − λ_t κ_t (a_t ⊙ κ_t)ᵀ` | `diag(exp g_t)(I − β k kᵀ)` |
| DPLR? | yes | yes | **yes (proved)** |
| `D_t` | trivial | nontrivial gate | nontrivial gate `exp g_t` — closer to RWKV-7's shape than to DeltaNet's |
| `β` range | `(0,1)`; `(0,2)` with neg-eigval | — | `(0,2)` **(proved)** |

Differences that matter and are *not* covered by the paper's theorems:

1. **Decay diagonal.** Our `D_t = diag(exp g_t) ≠ I`: our arm is DPLR with a nontrivial
   time-varying gate, like RWKV-7, not plain DeltaNet. Both are DPLR, so the paper's
   *upper* bound (Thm. 3: any LRNN over ℚ is in `PNC¹`) applies only under
   **paper-compatible rational-approximation semantics**, not automatically to our
   exact-real model; their
   *lower* bounds (Thm. 5/6: 4-layer DeltaNet/RWKV-7 simulate iterated `3×3` matrix
   multiplication and every WFA) are proved for their specific layer stacks with a
   router + token buffer + `O(n²)` arithmetic steps — **not** for a single Emender head.
2. **Depth/breadth of the construction.** Their completeness results use 4 stacked
   layers, a finite-window router (`t mod 2m`), a buffer, and scratch coordinates. Our
   arm is a single recurrent head (one layer) inside a residual MLP stack, `d = 16–64`
   per head. Nothing here claims our head is `PNC¹`-complete.
3. **Precision.** Their results are over `ℚ` at log or poly precision. Our arm is
   bf16/fp32 — a *fixed* budget, so X22T's fixed-budget collapse applies and the paper's
   growing-precision lower bounds do not.
4. **The parity statement is theirs.** `β ∈ (0,2)` is what *licenses* parity at fixed
   precision per Grazzi et al.; we prove only the range, not the parity recognition.

**Placement in their Figure 1, under paper-compatible rational-approximation
semantics only:** the *DPLR LRNN* upper-bound box (`PNC¹`). Exact-real transfer is
unproved; WFA completeness of their sufficient four-layer construction is not
claimed for a single configured head.

---

## 2. Arm B — our E97 `tanh`-state recurrence: **exact-real closed-form shape; semantics-qualified upper bound only**

### Paper side

> **Definition 2 (Arithmetic Closed Form).** `f : K^k → K` has an *arithmetic closed
> form* if it can be written as a finite `√`-composition of `⊕, ⊗, max, exp` and `·`.

> **Definition 3 (Nonlinear RNN).** Given `x₁,…,xₙ ∈ K^d`, an RNN computes recurrent
> states `h₁,…,hₙ ∈ K^d` via `h_t = f(h_{t−1}, x_t)`, where `f` is a closed-form
> arithmetic function.

> **Proposition 1 (Upper Bound).** Let `M` be an RNN over ℚ with `s = O(poly(n))`
> precision. Then the language recognized by `M` is recognizable by a Turing machine
> using `O(max{s, log n})` space and `poly(n)` time. **Corollary 1.** Poly-precision
> RNNs can be simulated in `P` and log-precision RNNs can be simulated in `L`.

> **Theorem 1 (Turing-Completeness).** For any multi-stack machine `M`, there exists a
> one-layer **MLP RNN** that recognizes the same language as `M`.
> **Theorem 2.** There exists a one-layer log-precision **MLP RNN** that solves sorted
> deterministic graph connectivity, an `L`-complete problem.

### Our side

E97's state nonlinearity (`state_activation`) is applied to the whole pre-activation:

```
S_t = f(decay_t ⊙ S_{t−1} + β_t · outer(delta_t, k_t)),   f = tanh (default)
```

* Repo: `ndm/models/e88_fla_hybrid.py:1356–1368` (`_apply_state_activation`;
  the docstring labels `'tanh'` saturating and `'relu'/'softplus'` non-saturating;
  these implementation labels are not counter-machine realization proofs); `:310`, `:833`
  (`S = tanh(decay * S + beta * outer(delta, k))`); `ndm/models/typed_head_mixture.py:228`
  (`e97_state_nonlin: str = 'tanh'`) and `:454` (`state_activation=self.e97_state_nonlin`).
* **Lean proof of closed form:** `PrecisionRegime.tanhStreamModel_isClosedFormRNN`
  exhibits a single `ArithClosedForm` expression with
  `f(h, x) = tanh(A_x·h + B_x)` and the two-vector `(A_x, B_x)` as the token encoding,
  using `ArithClosedForm.eval_tanhForm` (`tanh x = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)`, from
  `Real.tanh_eq`). The scalar statement transfers to the matrix case coordinatewise.
* **The Taylor-datatype reading** (R1): `ArmMapping.expTaylorForm` +
  `ArmMapping.eval_expTaylorForm` build their Def. 1 `exp` as an `N`-term Taylor
  truncation inside the closed-form language, and `ArmMapping.expTaylor_two_ne_exp`
  proves that truncation is **not** `Real.exp` (at `x = 1`, `2 < e`). Our models use exact
  `Real.exp`; this is a discrepancy only, not computational-class strictness or an
  upper-bound transfer theorem — see divergence 5.

### Verdict: **exact-real closed-form shape; UPPER bounds require paper-compatible rational-approximation semantics**

| | our E97 `tanh` state | paper's nonlinear RNN (Thm. 1/2) |
|---|---|---|
| closed form `f` | **yes (proved)** | definition |
| activation | `tanh` (saturating, `|S| < 1`) | **ReLU MLP** (non-saturating) |
| upper bounds (Prop. 1 / Cor. 1) | **apply only under paper-compatible rational-approximation semantics**; no exact-real transfer theorem | theirs |
| lower bounds (Thm. 1, Thm. 2, Cor. 2: multi-stack machines, L/P-completeness) | **do not transfer verbatim** — proved for ReLU MLPs, whose counter registers are unbounded above | theirs |
| state shape | matrix `head_v_dim × head_k_dim` per head, elementwise `tanh` | vector `h_t ∈ K^d`; same class, bookkeeping only |

Under **paper-compatible rational-approximation semantics only**, the paper
places the saturating nonlinear shape at `⊆ L` (log precision) / `⊆ P` (poly precision),
with **no lower bound** for it. No transfer to our exact-real model is proved. Whether a `tanh`-state RNN at
growing precision reaches counter machines is **open** (X22T open item O10), and the
literature's counter-machine constructions (their Lemma 3) are explicitly ReLU-based.
X21T's `tanh_one_dim_two_fixed_points` / `tanhStreamModel_state_bounded` describe
fixed points and bounded state values, not a growing-precision counter impossibility
or an upper-bound transfer theorem.

**Placement in their Figure 1, under paper-compatible rational-approximation
semantics only:** the *Nonlinear RNN* upper-bound side (exact-real transfer unproved); the "Counter machine" automaton class is **not** claimed for our `tanh` arm.

---

## 3. Our stack augmentation — a **one-counter machine**

* Repo-side object: `MemoryBoundary.stackModel` (three-phase control × `List Unit`).
* **Lean proof:** `PrecisionRegime.stackModel_run_eq_counterModel` — the unary stack's
  contents collapse to its depth, giving a `StackPhase × ℕ` model with
  push = `m+1`, pop = `m−1` (underflow at 0), identical output stream on every word.
* This is the counter-machine model of Fischer et al. / Weiss et al. / Merrill et al.
  Lemma 3, i.e. the model class their **Theorem 1** (poly-precision MLP RNN simulates a
  multi-stack machine) and **Theorem 2** (log-precision MLP RNN simulates a counter
  machine) act on. Their theorem supplies an abstract ReLU-MLP realization, not
  a realization by a configured Emender head. Our `cmModel_fold` instead proves an
  exact `StreamModel` embedding of the defined additive-update machine; its discrete
  control, Boolean equality decisions, and arbitrary lookups remain. Only
  `zeroMaskForm` has an arithmetic-expression correctness theorem. Closed-form
  transition assembly, even for one counter, and reset updates from their Lemma 3
  remain unproved (O14).
* Caveat, stated plainly: the RNN that their theorem produces has a **ReLU** gate. Our
  E97 `tanh` configuration is not that RNN; E97 `relu`/`softplus` configurations are
  **CANDIDATE HOSTS** requiring a realization proof, not covered configurations.
  A single configured elementwise activation (`e88_fla_hybrid.py:1356–1374`) is not
  a ReLU MLP transition, and the paper gives no softplus construction.

---

## 4. Taxonomy placement (their Figure 1) and what the X21E receipts mean

**The following upper-bound placements require paper-compatible rational-approximation
semantics; no exact-real transfer is proved.** Their Figure 1 rows: `TC⁰`(S4/Mamba) → `NC¹`(PD-SSM) → `PNC¹`(**DPLR: DeltaNet, RWKV-7**)
→ `L`(**nonlinear RNN, log precision**) → `P`(poly precision); automaton classes: — /
DWFA / WFA / counter machine / Turing machine.

| our arm | Figure-1 box | basis |
|---|---|---|
| `fla-gdn` (`allow_neg_eigval=True`) | **DPLR LRNN, `PNC¹` upper bound under paper-compatible rational-approximation semantics only**; no single-head WFA completeness | algebraic `gatedDelta_isDPLR` + cited Thm. 3; exact-real transfer unproved |
| `e97_state_nonlin='tanh'` (E97/E88) | **Nonlinear RNN**, `⊆ L`/`⊆ P` only under paper-compatible rational-approximation semantics | exact-real closed-form shape + cited Prop. 1/Cor. 1; exact-real transfer unproved |
| `e97_state_nonlin='relu'/'softplus'` | **CANDIDATE HOSTS**, not established `L`/`P` lower-bound placements | realization proof required: single elementwise activation ≠ ReLU MLP; paper gives no softplus construction |
| `stackModel` (framework construction only) | **Counter machine** | `stackModel_run_eq_counterModel` (proved) |
| all of the above *at a fixed budget* | **finite automata (both arms)** | X21T collapse + `PrecisionRegime.fixed_budget_collapses_to_finite_state` (proved) |

**The X21E receipts are empirical observations, not fixed-budget theorem predictions.**
X21E measured trained champions retaining only start-anchored shallow prefixes at
extended horizons and learning the cumulative 3721-state automaton only 2–6 steps deep.
The fixed-budget theorems provide neither a 3721-state capacity obstruction for these
models nor a learnability bound, and prove no necessary causal explanation of this
shallow-prefix learning. These bf16/fp32 receipts do not establish growing-precision
class placements. No training was launched by X22T.


---

## 5. Re-proof status table (pass 2) — what is proved, what diverges

| claim | Lean declaration | status |
|---|---|---|
| our gdn step is a Def. 4 LRNN step | `ArmMapping.gdnStep_eq_lrnnStep` | **proved** |
| our gdn step matrix is Def. 5 DPLR | `ArmMapping.gdnMatrix_isDPLR` | **proved** |
| zero decay ⇒ DeltaNet's `I − β k kᵀ` | `ArmMapping.gdnMatrix_zero_decay` | **proved** |
| per-channel decay ⇒ DPLR (other orientation) | `ArmMapping.gdnMatrixPerChannel_isDPLR` | **proved** |
| `allow_neg_eigval` ⇒ `β ∈ (0,2)` | `PrecisionRegime.negEigval_beta_range` | **proved** |
| Grazzi parity in `(0,2)`, not in `(0,1)` | — | **NOT re-proved** (theirs) |
| `tanh` step is a Def. 2 closed form | `ArithClosedForm.eval_tanhForm` | **proved** |
| E97 tanh arm is a Def. 3 RNN | `PrecisionRegime.tanhStreamModel_isClosedFormRNN` | **proved** |
| their Taylor `exp` ≠ `Real.exp` | `ArmMapping.expTaylor_two_ne_exp` | **proved** |
| every WFA is computed in one layer | `WeightedAutomata.wfaModel_readout` | **proved** |
| deterministic automaton = `0/1` WFA = X21T one-hot model | `WeightedAutomata.detWFA_state_eq_onehot` | **proved** |
| the literal `cycle3` matrix is not DPLR in its specified coordinates | `WeightedAutomata.cycle3_not_dplr`, `dplr_cycle_identity` | **proved (new obstruction)** |
| LRNN state = iterated matrix product | `WeightedAutomata.lrnn_hom_state` | **proved** |
| associative ⇒ balanced evaluation of depth `log₂n+1` | `WeightedAutomata.prodL_balanced_eval` | **proved** |
| the defined additive-update machine is exactly embedded in a `StreamModel` (not closed-form assembly) | `CounterMachineSim.cmModel_fold` | **proved** |
| the zero test is an *exact* closed form | `CounterMachineSim.zeroMaskR_int`, `eval_zeroMaskForm` | **proved** |
| the `StreamModel` construction has balanced-block discrimination (also accepts `abab`) | `CounterMachineSim.cmModel_separates_balanced` | **proved** |
| "log precision suffices" | `CounterMachineSim.cmModel_reachable_card` (`3·(n+1)` states) | **proved as cardinality, not bits** (divergence 3) |
| their Lemma 3's MLP synthesis, reset updates, complete closed-form assembly (including one counter) | — | **NOT reproduced**; O14, only the arithmetic zero-mask gadget is proved |
| their Thm. 6's 4-layer router+buffer | — | **NOT reproduced**; cited sufficient construction, not proved necessary |
| their Thm. 3: LRNN ⊆ `PNC¹` | — | **NOT reproduced**; only the associative-scan reduction (O13) |
| their Thm. 1: poly-precision ⇒ `P`-complete | — | **NOT reproduced** (no circuit/space complexity model in Lean) |

### The five divergences, in one place

1. **`b_t` arity.** Def. 4 says `b_t ∈ K^d` (vector); their own DeltaNet/RWKV-7 instances
   use rank-one *matrices*. We formalize both (`lrnnStep`, `lrnnStepVec`) and match the
   instances. Either their Def. 4 is loose or their instances are outside it.
2. **`D_t` and decay placement.** Their DeltaNet has `D_t = I`; ours is `diag(exp g_t)`, and
   with per-channel gates the rank-one term flips orientation (`D − k (D k)ᵀ`). All are DPLR,
   but they are different matrices, so "the DPLR arm" is not a single object.
3. **"Log precision" is re-proved as cardinality.** Their account is in bits (`p_h(n)`);
   ours is `3·(n+1)` reachable states for the witness machine. No rounding or bit-length
   model.
4. **Their ε-tolerance is unnecessary in our class.** Their Lemma 3 builds ReLU equality
   checks with tolerance `1/3`; our closed form has `max` (Def. 2 allows it) and tests zero
   *exactly* on integer-valued reals (`zeroMaskR_int`). This is a gadget result only:
   `cmRealStep` still has discrete control, Boolean equality decisions, and arbitrary
   lookup functions, so no assembled closed-form transition is established.
5. **Exact `exp` vs Taylor `exp`.** Their datatype truncates; we do not.
   `expTaylor_two_ne_exp` establishes a discrepancy only, not computational-class
   strictness and not a transfer theorem. Every cited upper-bound placement here is
   qualified to paper-compatible rational-approximation semantics; exact-real
   transfer remains unproved.

### Two findings the operator asked for

* **Literal matrix obstruction only, not behavioral strictness.** The cycle identity
  `A i j A j l A l i = A i l A l j A j i` holds for every DPLR matrix and fails for
  the literal `cycle3` permutation matrix in its specified coordinates. The reviewer's
  rot3-is-DPLR counterexample shows why this scope matters:
  `B = [[0,-1],[1,-1]] = diag(-1,0) - (1,1)ᵀ(-1,1)` **is DPLR** and cycles
  `(1,0) → (0,1) → (-1,-1) → (1,0)`, realizing the X21T three-cycle.
  Thus the paper's four-layer router+buffer construction is **sufficient, not proved
  necessary**; no behavioral strictness or minimal-depth theorem follows.
* **Exact embedding with an arithmetic zero-mask gadget, not a completed Lemma-3
  construction.** The `StreamModel` stores counters in unbounded real coordinates;
  `zeroMaskForm` arithmetically implements `max 0 (1 − |r|)`. No complete transition
  expression is proved, including for the one-counter witness. This is neither a
  growing-precision impossibility for encoded `tanh` counters nor a realization
  theorem for E97 ReLU/softplus configurations; those remain **CANDIDATE HOSTS**.

### X22TR corrections register

All eight narrowing applications are recorded with the reviewer's counterexample in
`THEOREMS.md` §7, corrections 14–21: (1) impossible global abstraction / unproved Def.-10
bridge; (2) additive-update embedding / unproved closed-form assembly; (3) semantics-
qualified upper bounds / discrepancy only; (4) literal matrix obstruction / rot3-is-DPLR /
sufficient four-layer construction; (5) candidate ReLU/softplus hosts; (6) balanced-block
discrimination; (7) geometric separation, not a robust decoder or finite-precision
dynamics; (8) no shallow-prefix capacity or learnability prediction. For (7),
`enc_robust_decode` proves nearby-corner uniqueness, but `tanhCornerModel.out` requires
exact corner equality and otherwise emits the initial-state output.
