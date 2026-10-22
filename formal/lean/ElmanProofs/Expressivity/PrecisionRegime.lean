/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.MemoryBoundary
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Data.Nat.Log
import Mathlib.Data.Fin.Tuple.Basic

/-!
# The precision regime split, and the arm mapping to Merrill et al. (2026)

`Why Are Linear RNNs More Parallelizable?` (Merrill, Jiang, Li, Lin, Sabharwal;
ICML 2026, PMLR 306:88230–88260) classifies RNN parameterizations by circuit
complexity, with the *precision of the state channel as a function of input length*
(Merrill et al. Def. 10: `p_h(n)` bits at length `n`) as the pivot:

* any LRNN over `ℚ` is in `PNC¹` (their Thm. 3), log-precision LRNNs in
  `AC⁰[ENC¹]` (Thm. 4);
* poly-precision nonlinear RNNs simulate multi-stack machines (Thm. 1, = Turing
  complete) and are `P`-complete (Cor. 2); log-precision nonlinear (ReLU-MLP) RNNs
  solve an `L`-complete problem (Thm. 2, via counter machines);
* DPLR LRNNs simulate every WFA and solve `PNC¹`-complete problems (Thm. 5, 6).

`PNC¹ ⊆ L`, so `PNC¹ ≠ L` (an open conjecture) is exactly what separates the two
arms at growing precision.

This file proves the parts of that picture that live inside *our* framework, and
carries the (at the time) uncited parts as explicit hypotheses. **Superseding note
(X22T pass 2):** the mapping lemmas and the constructive simulations are now
*independently re-proved* in `ArmMapping.lean` (R1), `WeightedAutomata.lean` (R3, R4) and
`CounterMachine.lean` (R2). These are narrower than complete paper re-proofs:
`cmModel_fold` is an additive-update embedding, not assembled closed-form arithmetic
(O14), and `cycle3_not_dplr` is a literal matrix obstruction, not behavioral strictness
or four-layer necessity. We have homogeneous-product identities and padding/depth
arithmetic (`prodL_balanced_eval`), not circuit-class membership (O13). All cited
upper-bound placements require paper-compatible rational-approximation semantics;
no exact-real transfer is proved.

## 1. Precision schedules and the fixed-budget regime (proved here)

`PrecisionSchedule = ℕ → ℕ` is schedule notation inspired by Def. 10. The bridge
from length-restricted bit precision to our global abstraction predicates is unproved.
`IsFixedBudget p` is `p = O(1)`; `IsGrowingBudget p` is `p` unbounded — their log and
poly precision are *growing*, so their whole analysis sits on the growing side.

* `fixed_budget_collapses_to_finite_state` — **at a fixed budget, uniformly in the
  length, every arm collapses to finitely many states** (X21T's collapse, in schedule
  form and without a surjectivity hypothesis).
* `countModel_needs_growing_budget` — impossible-global-abstraction corollary: its
  premise supplies global all-length abstractions, and already `h 0` contradicts
  `countModel_no_omega_finite_state` for any schedule, growing or not. No
  length-restricted Def.-10 schedule lower bound or counting realization follows.
* `log2_schedule_is_growing` — `O(log n)` precision is a growing schedule.

## 2. Mapping our arms onto their definitions (proved here)

* `tanhStreamModel_isClosedFormRNN` — our `TANH(1)` stream model *is* a Merrill et
  al. Def. 3 nonlinear RNN: its step is one arithmetic closed form (`ArithClosedForm`,
  mirroring their Def. 2) in the previous state and the token encoding. `tanh` is
  closed form because `tanh x = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)` (`Real.tanh_eq`).
* `gatedDelta_isDPLR` — our fla-gdn / gated-delta step
  `diag (exp g) · (I − β k kᵀ)` *is* their Def. 5 DPLR form `D − k̃ ṽᵀ`.
* `negEigval_beta_range` — the `allow_neg_eigval` configuration gives `β ∈ (0,2)`,
  the range in which Grazzi et al. (2025) show fixed-precision DeltaNet recognizes
  parity (as cited by Merrill et al.).
* `stackModel_run_eq_counterModel` — our stack augmentation is a **one-counter
  machine** in their (Fischer et al. / Weiss et al.) sense, so their Thm. 1/2
  machinery applies to it; we do not re-prove Turing completeness.

## 3. The regime split and the conditional arm separation

* `precision_regime_split` — a regime skeleton, not a Def.-10 schedule bridge:
  fixed budget ⇒ finite automata under the abstraction hypothesis; the counting
  implication has an impossible global premise. It does not show that growing
  budget buys counting; the length-restricted Def.-10 bridge is unproved.
* `arm_separation_at_growing_precision` — the cited half, as a conditional statement
  whose hypotheses are named after the theorems that supply them: DPLR `⊆ PNC¹`
  (their Thm. 3), a nonlinear-arm witness outside `PNC¹` (their Thm. 2 + `PNC¹ ≠ L`),
  hence the nonlinear arm is *not* contained in the DPLR arm.

Scoping: nothing here re-proves a Merrill et al. theorem, and no complexity class is
defined in Lean. The cited statements enter as explicit hypotheses so that the
conditional structure is visible, and every Lean-proved statement is about our own
`StreamModel`/`BoundedAbstraction` objects.
-/

namespace PrecisionRegime

open Function
open LinTanhExpressivity
open StreamingExpressivity
open MemoryBoundary

/-! ## 1. Precision schedules -/

/-- Schedule notation inspired by Merrill et al. (2026, Def. 10): `p n` denotes a
putative total bit budget at length `n`. The length-restricted precision bridge is
unproved; `BoundedAbstraction` below is global and all-length, not length-restricted. -/
def PrecisionSchedule := ℕ → ℕ

/-- A **fixed** budget: `O(1)` bits, independent of length. This is the regime of
`BoundedAbstraction` and of every theorem in X20T/X21T/X22T-F1/F2. -/
def IsFixedBudget (p : PrecisionSchedule) : Prop := ∃ B, ∀ n, p n ≤ B

/-- A **growing** budget: unboundedly many bits at longer lengths. Merrill et al.'s
log precision `O(log n)` and poly precision `O(nᶜ)` are both growing. -/
def IsGrowingBudget (p : PrecisionSchedule) : Prop := ∀ B, ∃ n, B < p n

/-- `O(log n)` precision is a growing schedule, so Merrill et al.'s whole analysis
sits on the growing side of the split. -/
theorem log2_schedule_is_growing : IsGrowingBudget (fun n => Nat.log2 n) := by
  intro B
  refine ⟨2 ^ (B + 1), ?_⟩
  have h : B + 1 ≤ Nat.log2 (2 ^ (B + 1)) := by
    rw [Nat.le_log2 (by positivity)]
  exact Nat.lt_of_succ_le h

/-- Raising the budget never destroys a bounded abstraction. -/
def boundedAbstraction_mono {σ α β : Type*} {M : StreamModel σ α β} {n m : ℕ}
    (h : n ≤ m) (A : BoundedAbstraction M n) : BoundedAbstraction M m where
  q := fun s => Fin.castLE h (A.q s)
  step_congr := by
    intro s s' hs x
    have hs' : A.q s = A.q s' := (Fin.castLE_inj (hmn := h)).mp hs
    rw [A.step_congr s s' hs' x]
  out_congr := by
    intro s s' hs
    exact A.out_congr s s' ((Fin.castLE_inj (hmn := h)).mp hs)

/-- The collapse of an `n`-valued channel to an `n`-state model, with **no
surjectivity hypothesis**: states outside the image fall back to the initial state.
(X21T's `exists_fin_model_of_abstraction` assumes surjectivity; the schedule theorem
below cannot, because a fixed budget only bounds the number of *available* values.) -/
noncomputable def collapseFin {σ α β : Type*} {M : StreamModel σ α β} {n : ℕ}
    (A : BoundedAbstraction M n) : StreamModel (Fin n) α β := by
  classical
  exact
    { init := A.q M.init
      step := fun i x =>
        if h : ∃ s, A.q s = i then A.q (M.step (Classical.choose h) x) else A.q M.init
      out := fun i => if h : ∃ s, A.q s = i then M.out (Classical.choose h) else M.out M.init }

theorem collapseFin_step {σ α β : Type*} {M : StreamModel σ α β} {n : ℕ}
    (A : BoundedAbstraction M n) (s : σ) (x : α) :
    (collapseFin A).step (A.q s) x = A.q (M.step s x) := by
  have h : ∃ s', A.q s' = A.q s := ⟨s, rfl⟩
  simp only [collapseFin, dif_pos h]
  exact A.step_congr _ _ (Classical.choose_spec h) x

theorem collapseFin_out {σ α β : Type*} {M : StreamModel σ α β} {n : ℕ}
    (A : BoundedAbstraction M n) (s : σ) :
    (collapseFin A).out (A.q s) = M.out s := by
  have h : ∃ s', A.q s' = A.q s := ⟨s, rfl⟩
  simp only [collapseFin, dif_pos h]
  exact A.out_congr _ _ (Classical.choose_spec h)

/-- **Abstracting an `n`-valued channel gives an `n`-state model**, no surjectivity
required. -/
theorem exists_fin_model_of_abstraction' {σ α β : Type*} {M : StreamModel σ α β} {n : ℕ}
    (A : BoundedAbstraction M n) :
    ∃ M' : StreamModel (Fin n) α β, ∀ w, M'.run w = M.run w := by
  refine ⟨collapseFin A, fun w => ?_⟩
  have h := StreamModel.runFrom_sim (M := M) (M' := collapseFin A) A.q
    (fun s x => (collapseFin_step A s x).symm) (fun s => (collapseFin_out A s).symm) M.init w
  change (collapseFin A).runFrom (A.q M.init) w = M.runFrom M.init w
  exact h.symm

/-- **The fixed-budget regime.** A family of models whose state channels carry a
schedule-bounded number of values is, *uniformly in the length*, inside the `N`-state
automaton band for one fixed `N`. Both activation classes are included: the theorem
only sees `BoundedAbstraction`. -/
theorem fixed_budget_collapses_to_finite_state (p : PrecisionSchedule) (hp : IsFixedBudget p)
    {σ α β : Type*} (M : ℕ → StreamModel σ α β)
    (A : ∀ n, BoundedAbstraction (M n) (2 ^ p n)) :
    ∃ N, ∀ n, ∃ M' : StreamModel (Fin N) α β, ∀ w, M'.run w = (M n).run w := by
  obtain ⟨B, hB⟩ := hp
  refine ⟨2 ^ B, fun n => ?_⟩
  exact exists_fin_model_of_abstraction'
    (boundedAbstraction_mono (Nat.pow_le_pow_right (by norm_num) (hB n)) (A n))

/-- Impossible-global-abstraction corollary, not a Def.-10 schedule lower bound.
The premise gives an all-length global abstraction for every `n`; already `h 0`
contradicts `countModel_no_omega_finite_state`, regardless of the schedule. Even a
growing schedule cannot satisfy it. A length-restricted precision bridge is unproved. -/
theorem countModel_needs_growing_budget (p : PrecisionSchedule)
    (h : ∀ n, BoundedAbstraction countModel (2 ^ p n)) : IsGrowingBudget p := by
  intro B
  by_contra hcon
  push_neg at hcon
  exact countModel_no_omega_finite_state (2 ^ B)
    (boundedAbstraction_mono (Nat.pow_le_pow_right (by norm_num) (hcon 0)) (h 0))

/-- Regime skeleton: fixed-budget global abstractions give a uniform finite-state
bound; the counting implication has an impossible global premise; and `log₂` is
unbounded. This proves neither a length-restricted Def.-10 precision lower bound nor
that growing budget buys counting. The schedule bridge remains unproved. -/
theorem precision_regime_split :
    (∀ (p : PrecisionSchedule) (_hp : IsFixedBudget p) (_σ _α _β : Type*)
        (_M : ℕ → StreamModel _σ _α _β) (_A : ∀ n, BoundedAbstraction (_M n) (2 ^ p n)),
        ∃ N, ∀ n, ∃ M' : StreamModel (Fin N) _α _β, ∀ w, M'.run w = (_M n).run w) ∧
      (∀ (p : PrecisionSchedule) (_h : ∀ n, BoundedAbstraction countModel (2 ^ p n)),
        IsGrowingBudget p) ∧
      IsGrowingBudget (fun n => Nat.log2 n) :=
  ⟨fun p hp _σ _α _β M A => fixed_budget_collapses_to_finite_state p hp M A,
    fun p h => countModel_needs_growing_budget p h, log2_schedule_is_growing⟩

/-! ## 2. Mapping our arms onto Merrill et al.'s definitions -/

/-- Expressions with `k` free variables built from constants by `+`, `−`, `*`, `/`,
`exp`, `√` and `max`: Merrill et al. (2026), Def. 2 (arithmetic closed form) over
`ℝ`. Their Def. 2 is stated for a general semiring `K`; we only need `ℝ`. -/
inductive ArithClosedForm (k : ℕ) : Type
  | var (i : Fin k) : ArithClosedForm k
  | const (c : ℝ) : ArithClosedForm k
  | add (a b : ArithClosedForm k) : ArithClosedForm k
  | sub (a b : ArithClosedForm k) : ArithClosedForm k
  | mul (a b : ArithClosedForm k) : ArithClosedForm k
  | div (a b : ArithClosedForm k) : ArithClosedForm k
  | exp (a : ArithClosedForm k) : ArithClosedForm k
  | sqrt (a : ArithClosedForm k) : ArithClosedForm k
  | max (a b : ArithClosedForm k) : ArithClosedForm k
  | pow (a : ArithClosedForm k) (n : ℕ) : ArithClosedForm k

namespace ArithClosedForm

/-- Evaluation of a closed-form expression. -/
noncomputable def eval {k : ℕ} : ArithClosedForm k → (Fin k → ℝ) → ℝ
  | var i, v => v i
  | const c, _ => c
  | add a b, v => eval a v + eval b v
  | sub a b, v => eval a v - eval b v
  | mul a b, v => eval a v * eval b v
  | div a b, v => eval a v / eval b v
  | exp a, v => Real.exp (eval a v)
  | sqrt a, v => Real.sqrt (eval a v)
  | max a b, v => Max.max (eval a v) (eval b v)
  | pow a n, v => eval a v ^ n

/-- `tanh` as a closed form: `tanh x = (eˣ − e⁻ˣ)/(eˣ + e⁻ˣ)` with `e⁻ˣ = 1/eˣ`. -/
def tanhForm {k : ℕ} (t : ArithClosedForm k) : ArithClosedForm k :=
  let e := exp t
  let ei := div (const 1) e
  div (sub e ei) (add e ei)

/-- **`tanh` has an arithmetic closed form** in the sense of Merrill et al. Def. 2. -/
theorem eval_tanhForm {k : ℕ} (t : ArithClosedForm k) (v : Fin k → ℝ) :
    (tanhForm t).eval v = Real.tanh (t.eval v) := by
  simp only [tanhForm, eval]
  rw [Real.tanh_eq, Real.exp_neg, one_div]

end ArithClosedForm

/-- A scalar one-step map with a **closed-form step** (Merrill et al., Def. 3 at our
signature): one fixed `ArithClosedForm` expression in the glued `(token, state)`
coordinates. `k` token-encoding coordinates come first, the scalar state last. -/
def ScalarStepClosedForm {k : ℕ} (step : ℝ → (Fin k → ℝ) → ℝ) : Prop :=
  ∃ e : ArithClosedForm (k + 1),
    ∀ (s : ℝ) (u : Fin k → ℝ), step s u = e.eval (Fin.append u ![s])

/-- The scalar saturated step `s ↦ tanh (A·s + B)` has a closed-form step, with `A`
and `B` as the two token coordinates. -/
theorem tanhStep_scalarClosedForm :
    ScalarStepClosedForm (k := 2) (fun s u => Real.tanh (u 0 * s + u 1)) := by
  refine ⟨ArithClosedForm.tanhForm
    (ArithClosedForm.add (ArithClosedForm.mul (.var 0) (.var 2)) (.var 1)), ?_⟩
  intro s u
  rw [ArithClosedForm.eval_tanhForm]
  simp only [ArithClosedForm.eval]
  have h0 : Fin.append u ![s] 0 = u 0 := by
    have := Fin.append_left u ![s] (0 : Fin 2); simpa using this
  have h1 : Fin.append u ![s] 1 = u 1 := by
    have := Fin.append_left u ![s] (1 : Fin 2); simpa using this
  have h2 : Fin.append u ![s] 2 = s := by
    have := Fin.append_right u ![s] (0 : Fin 1); simpa using this
  rw [h0, h1, h2]

/-- **Mapping lemma (F1 side).** Our scalar saturated stream model
(`tanhStreamModel`) *is* a Merrill et al. Def. 3 nonlinear RNN, with the two-vector
`(A x, B x)` as the token encoding and `f(h, x) = tanh (A x · h + B x)` as the single
closed-form step over exact reals. Their complexity classification is cited only
under paper-compatible rational-approximation semantics; this shape lemma proves no
exact-real upper-bound transfer. -/
theorem tanhStreamModel_isClosedFormRNN {α β : Type*} (A B : α → ℝ) (ψ : ℝ → β) :
    ∃ step : ℝ → (Fin 2 → ℝ) → ℝ, ScalarStepClosedForm step ∧
      ∀ (s : ℝ) (x : α), (tanhStreamModel A B ψ).step s x = step s ![A x, B x] :=
  ⟨fun s u => Real.tanh (u 0 * s + u 1), tanhStep_scalarClosedForm, by
    intro s x
    simp [tanhStreamModel, tanhStep]⟩

/-- **DPLR shape** (Merrill et al. 2026, Def. 5): `A = D − k vᵀ` with `D` diagonal. -/
def IsDPLR {d : ℕ} (A : Fin d → Fin d → ℝ) : Prop :=
  ∃ (w k v : Fin d → ℝ), ∀ i j, A i j = (if i = j then w i else 0) - k i * v j

/-- **Mapping lemma (fla-gdn side).** Our gated-delta / fla-gdn step matrix
`diag (exp g) · (I − β k kᵀ)` — a multiplicative diagonal decay composed with the
DeltaNet rank-one correction — *is* their Def. 5 DPLR form, with
`D = diag (exp g)`, `k̃ = β ⊙ exp g ⊙ k` and `ṽ = k`. Differences from the paper's
DeltaNet parameterization: their `A_t = I − β_t k_t k_tᵀ` has `D_t = I` (no decay),
ours carries the gate `diag (exp g_t)`. Their Thm. 3 upper bound applies under
paper-compatible rational-approximation semantics only; no exact-real transfer
follows from this identity. Their sufficient four-layer WFA construction (Thm. 6)
is not a lower bound for a single configured head. -/
theorem gatedDelta_isDPLR {d : ℕ} (g k : Fin d → ℝ) (β : ℝ) :
    IsDPLR (fun i j => Real.exp (g i) * ((if i = j then 1 else 0) - β * k i * k j)) := by
  refine ⟨fun i => Real.exp (g i), fun i => β * Real.exp (g i) * k i, k, fun i j => ?_⟩
  simp only []
  by_cases h : i = j
  · subst h; rw [if_pos rfl, if_pos rfl]; ring
  · rw [if_neg h, if_neg h]; ring

/-- The `allow_neg_eigval` configuration sets `β = 2 · sigmoid z`, hence
`β ∈ (0,2)` — the range in which Grazzi et al. (2025) show fixed-precision DeltaNet
recognizes parity (`β ∈ (0,1)` does not), as cited by Merrill et al. §2.3. The
parity-recognition claim itself is theirs, not proved here. -/
theorem negEigval_beta_range (z : ℝ) : 2 * Real.sigmoid z ∈ Set.Ioo (0 : ℝ) 2 := by
  constructor
  · have := Real.sigmoid_pos z; linarith
  · have := Real.sigmoid_lt_one z; linarith

/-! ## 3. Our stack augmentation is a one-counter machine -/

/-- The one-counter form of `stackModel`: control phase plus an integer counter.
Push = increment, pop = decrement, decrementing at zero underflows. -/
def counterStep : StackPhase × ℕ → Bool → StackPhase × ℕ
  | (_, m), true => (StackPhase.pushing, m + 1)
  | (_, 0), false => (StackPhase.underflow, 0)
  | (_, m + 1), false => (StackPhase.popping, m)

/-- The one-counter machine corresponding to the stack model. -/
def counterModel : StreamModel (StackPhase × ℕ) Bool Bool where
  init := (StackPhase.pushing, 0)
  step := counterStep
  out := fun p => decide (p.1 = StackPhase.popping ∧ p.2 = 0)

theorem counterStep_depth (p : StackPhase × List Unit) (x : Bool) :
    ((stackStep p x).1, (stackStep p x).2.length) =
      counterStep (p.1, p.2.length) x := by
  obtain ⟨ph, L⟩ := p
  cases x
  · cases L with
    | nil => rfl
    | cons a L => rfl
  · rfl

theorem counterModel_out_depth (p : StackPhase × List Unit) :
    stackModel.out p = counterModel.out (p.1, p.2.length) := by
  obtain ⟨ph, L⟩ := p
  simp [stackModel, counterModel, List.length_eq_zero_iff]

/-- **Our stack augmentation is a one-counter machine.** The unary stack's contents
collapse to its depth, so `stackModel` is a counter machine in the sense of Fischer et
al. / Weiss et al. / Merrill et al. Lemma 3. The *simulation* of counter machines by a
real-state recurrent model is re-proved in `CounterMachine.lean`
(`CounterMachineSim.cmModel_fold`) as an additive-update `StreamModel` embedding.
Only `zeroMaskForm` has an arithmetic-expression correctness theorem; discrete
control, Boolean equality decisions, and arbitrary lookups remain in `cmRealStep`.
Complete closed-form assembly, including one counter and reset updates, is unproved. -/
theorem stackModel_run_eq_counterModel : ∀ w,
    stackModel.run w = counterModel.run w := by
  have hsim : ∀ w, stackModel.runFrom stackModel.init w
      = counterModel.runFrom (StackPhase.pushing, 0) w :=
    fun w => StreamModel.runFrom_sim (M := stackModel) (M' := counterModel)
      (fun p : StackPhase × List Unit => (p.1, p.2.length))
      (fun p x => counterStep_depth p x)
      (fun p => counterModel_out_depth p) stackModel.init w
  intro w
  simpa [StreamModel.run, counterModel] using hsim w

/-! ## 4. The cited half: conditional arm separation at growing precision -/

/-- **Arm separation at growing precision (conditional on the cited results).**

`PNC¹ ⊆ L`, and the paper's two arms are placed as follows at growing precision:

* `hDPLR` — every LRNN over `ℚ`, in particular every DPLR arm (Thm. 3; refined to
  `AC⁰[ENC¹]` at log precision by Thm. 4). Our `gatedDelta_isDPLR` puts the fla-gdn
  shape here only under paper-compatible rational-approximation semantics; the
  identity proves no exact-real upper-bound transfer.
* `hwit` — the nonlinear arm contains a language outside `PNC¹`: their Thm. 2 gives a
  log-precision nonlinear RNN solving an `L`-complete problem, which is outside
  `PNC¹` unless `PNC¹ = L`.

Conclusion: the nonlinear arm is **not** contained in the DPLR arm — the operator's
separation claim, located exactly on the precision-*growth* axis and conditional on
`PNC¹ ≠ L`. **Status after X22T pass 2:** this is deliberately kept as the *logical
skeleton* only. Its inputs are the class memberships, of which we now re-prove the
pieces — `CounterMachineSim.cmModel_fold` (exact additive-update embedding in a
`StreamModel`, not closed-form completeness) and `WeightedAutomata.wfaModel_readout`
(every WFA in an unrestricted LRNN layer) — but not circuit-class membership (O13)
or complete arithmetic transition assembly (O14). The paper's lower bounds are
ReLU-MLP constructions, not configured elementwise E97 activations. ReLU/softplus
configurations are candidate hosts requiring a realization proof; no softplus
construction is supplied, and the saturating `tanh` arm is not covered (O10). -/
theorem arm_separation_at_growing_precision
    (PNC1 DPLR Nonlin : Set (ℕ → Bool))
    (hDPLR : DPLR ⊆ PNC1)
    (hwit : ∃ L, L ∈ Nonlin ∧ L ∉ PNC1) :
    ¬ Nonlin ⊆ DPLR := by
  rintro hsub
  obtain ⟨L, hL, hnot⟩ := hwit
  exact hnot (hDPLR (hsub hL))

/-- The same separation read through the class chain: if `PNC¹ ⊆ L` and the nonlinear
arm reaches a language that is not in `PNC¹`, then no containment of the nonlinear arm
in the DPLR arm can hold. -/
theorem arm_separation_chain
    (PNC1 Lclass DPLR Nonlin : Set (ℕ → Bool))
    (hDPLR : DPLR ⊆ PNC1) (_hchain : PNC1 ⊆ Lclass)
    (hwit : ∃ L, L ∈ Nonlin ∧ L ∉ PNC1) :
    (¬ Nonlin ⊆ DPLR) ∧ (∃ L, L ∈ Nonlin ∧ L ∉ PNC1) :=
  ⟨arm_separation_at_growing_precision PNC1 DPLR Nonlin hDPLR hwit, hwit⟩

end PrecisionRegime
