/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.StreamingExpressivity

/-!
# Saturated-corner `TANH(n)` realization of every `n`-state automaton

X21T (`StreamingExpressivity.lean`) proved the collapse direction for both
activation classes and the realization direction only for `LIN`: every `n`-state
automaton has an exact *linear* realization of dimension `n` with one-hot `{0,1}`
reachable states (`moore_realized_by_onehot_linear`), and the `TANH` converse
(open item **O2**) was left open.

This file closes O2 in the reachable-realization sense, at dimension exactly `n`:

* `tanhCornerModel` — the **saturated-corner** `TANH(n)` realization. The code of
  automaton state `j` is the vector `enc j = τ · e_j` with `τ = tanh 1`; the
  per-token transition applies `s ↦ tanh (A(x) · s)` with the explicit corner
  matrix `cornerMatrix step x i j = if step j x = i then τ⁻¹ else 0`.
* `cornerMatrix_mul_enc` — the pre-activation of the code of `j` is *exactly* the
  `0/1` indicator vector of the successor state `step j x`. This is the whole
  content of the construction: the pre-activations stay at the two exact levels
  `0` and `1`, so `tanh` maps them exactly onto the two exact levels `0` and
  `τ = tanh 1`. Nothing is approximated and no precision hypothesis is needed.
* `tanh_cornerMatrix_mul_enc` — `tanh` lands exactly on the successor code,
  because `Real.tanh 0 = 0` and `Real.tanh 1 = τ` by definition.
* `moore_realized_by_tanh_corner` — **every `n`-state automaton is an exact
  `TANH(n)` model**: identical online output streams at every position of every
  word, one shared readout, horizon-free.
* `tanhCornerModel_reachable_enc`, `enc_entries`, `tanhCornerModel_reachable_bounded`
  — every reachable state is a corner of the cube `{0, τ}ⁿ` inside `(-1,1)ⁿ`: `n`
  states, each coordinate one of two values.
* `enc_diff_at`, `enc_robust_decode` — distinct corners are separated by the full
  level `τ`: a state cannot be within `τ / 2` coordinatewise of two corners. This
  is geometric uniqueness, not an implemented robust decoder; the readout below
  requires exact corner equality. No finite-precision dynamical guarantee follows.
* `finite_program_class_equality` — the finite-program answer: at dimension `n`,
  both activation classes realize *every* `n`-state automaton's online behaviour,
  so for indefinitely-running *finite* programs the two classes coincide at
  `d = n` in the reachable-realization sense.
* `no_ambient_abstraction_of_injective_readout` — the *ambient* version of that
  equality is impossible for either class: a distinguishing readout forces the
  quotient to be injective, so no finite ambient abstraction exists at any budget.

Scoping (identical to the `LIN` side, `X21T` correction 1): this is a
*sufficient* construction at dimension `n`. It supplies no bounded abstraction on
the ambient state space `Fin n → ℝ` (the reachable orbit is finite, the ambient
space is not), no dimension lower bound, and no claim about ambient
`BoundedAbstraction` for `TANH`. Exactness is exact-real-arithmetic exactness: the
pre-activations are the *rational* levels `0` and `1`, but the reachable code level
`τ = tanh 1` is irrational. A finite-precision implementation would approximate
parameters, states, and arithmetic; no error propagation or dynamical invariance
is proved. The geometric separation margin is `τ / 2 ≈ 0.381`, not a guarantee
that the exact-equality readout decodes perturbed states.
-/

namespace StreamingTanhRealization

open Function
open LinTanhExpressivity
open StreamingExpressivity

/-! ## The saturated level `τ = tanh 1` and the corner code -/

/-- The saturated-corner level: the value a coordinate takes when its pre-activation
is exactly `1`. `Real.tanh 1` is the largest level used anywhere in this file. -/
noncomputable def tau : ℝ := Real.tanh 1

theorem tau_eq_tanh_one : tau = Real.tanh 1 := rfl

theorem tau_pos : 0 < tau := Activation.tanh_pos_of_pos one_pos

theorem tau_ne_zero : tau ≠ 0 := tau_pos.ne'

theorem tau_abs_lt_one : |tau| < 1 := Activation.tanh_bounded 1

/-- The corner code of automaton state `j`: one coordinate at level `τ`, the others
at `0`. -/
noncomputable def enc {n : Nat} (j : Fin n) : Fin n → ℝ := fun i => if i = j then tau else 0

@[simp] theorem enc_self {n : Nat} (j : Fin n) : enc j j = tau := by
  simp [enc]

theorem enc_of_ne {n : Nat} {i j : Fin n} (h : i ≠ j) : enc j i = 0 := by
  simp [enc, h]

theorem enc_injective {n : Nat} : Injective (@enc n) := by
  intro j j' h
  have h1 : enc j' j = tau := by rw [← h]; exact enc_self j
  by_contra hne
  rw [enc_of_ne (i := j) (j := j') hne] at h1
  exact tau_ne_zero h1.symm

/-- Every coordinate of every code is `0` or `τ`: two levels, so one bit of code
information per coordinate at the code level. -/
theorem enc_entries {n : Nat} (j i : Fin n) : enc j i = 0 ∨ enc j i = tau := by
  by_cases h : i = j
  · exact Or.inr (by simp [enc, h])
  · exact Or.inl (by simp [enc, h])

theorem enc_abs_lt_one {n : Nat} (j i : Fin n) : |enc j i| < 1 := by
  rcases enc_entries j i with h | h
  · rw [h, abs_zero]; exact one_pos
  · rw [h]; exact tau_abs_lt_one

/-- Distinct corners differ by the full level `τ` in the coordinate where they
differ. -/
theorem enc_diff_at {n : Nat} {j j' : Fin n} (h : j ≠ j') :
    enc j j = tau ∧ enc j' j = 0 :=
  ⟨enc_self j, enc_of_ne h⟩

/-- Geometric corner separation: a state cannot be within `τ / 2` coordinatewise
of two distinct corners. Despite its name, this theorem does not implement a robust
decoder. `tanhCornerModel.out` requires exact corner equality and otherwise returns
the initial-state output, so a small perturbation of a noninitial corner need not
decode correctly. No finite-precision dynamical guarantee is claimed. -/
theorem enc_robust_decode {n : Nat} {j j' : Fin n} (s : Fin n → ℝ)
    (h : ∀ i, |s i - enc j i| < tau / 2) (h' : ∀ i, |s i - enc j' i| < tau / 2) :
    j = j' := by
  by_contra hne
  obtain ⟨hj, hj'⟩ := enc_diff_at hne
  have h1 : |s j - tau| < tau / 2 := by simpa [hj] using h j
  have h2 : |s j| < tau / 2 := by simpa [hj'] using h' j
  have h3 : tau ≤ |s j - tau| + |s j| := by
    have hsplit : tau = (tau - s j) + s j := by ring
    calc tau = |tau| := (abs_of_pos tau_pos).symm
      _ = |tau - s j + s j| := congrArg abs hsplit
      _ ≤ |tau - s j| + |s j| := abs_add_le _ _
      _ = |s j - tau| + |s j| := by rw [abs_sub_comm]
  linarith

/-! ## The saturated-corner transition -/

/-- The corner transition matrix for token `x`: column `j` is `τ⁻¹` times the
indicator of the successor state `step j x`. -/
noncomputable def cornerMatrix {n : Nat} {α : Type*} (step : Fin n → α → Fin n) (x : α) :
    Fin n → Fin n → ℝ :=
  fun i j => if step j x = i then tau⁻¹ else 0

/-- **The pre-activation of a corner code is exactly the `0/1` indicator of the
successor state.** Both levels are exact rationals: nothing is approximated. -/
theorem cornerMatrix_mul_enc_apply {n : Nat} {α : Type*} (step : Fin n → α → Fin n)
    (j i : Fin n) (x : α) :
    ∑ j', cornerMatrix step x i j' * enc j j' = (if step j x = i then (1 : ℝ) else 0) := by
  rw [Finset.sum_eq_single j]
  · rw [enc_self, cornerMatrix]
    by_cases h : step j x = i
    · rw [if_pos h, if_pos h, inv_mul_cancel₀ tau_ne_zero]
    · rw [if_neg h, if_neg h, zero_mul]
  · intro b _ hb
    rw [enc_of_ne (i := b) (j := j) hb, mul_zero]
  · intro h
    exact absurd (Finset.mem_univ j) h

/-- **The pre-activation of a corner code is exactly the `0/1` indicator of the
successor state**, as a vector identity. -/
theorem cornerMatrix_mul_enc {n : Nat} {α : Type*} (step : Fin n → α → Fin n) (j : Fin n)
    (x : α) :
    (fun i => ∑ j', cornerMatrix step x i j' * enc j j') =
      (fun i => if step j x = i then (1 : ℝ) else 0) :=
  funext fun i => cornerMatrix_mul_enc_apply step j i x

/-- **`tanh` lands exactly on the successor corner code**: `Real.tanh 0 = 0` and
`Real.tanh 1 = τ`, so the saturated nonlinearity is an exact corner map here. -/
theorem tanh_cornerMatrix_mul_enc {n : Nat} {α : Type*} (step : Fin n → α → Fin n)
    (j : Fin n) (x : α) :
    (fun i => Real.tanh (∑ j', cornerMatrix step x i j' * enc j j')) = enc (step j x) := by
  funext i
  rw [cornerMatrix_mul_enc_apply]
  by_cases h : step j x = i
  · rw [if_pos h, enc, if_pos h.symm]
    rfl
  · rw [if_neg h]
    simp only [enc, if_neg (Ne.symm h), Real.tanh_zero]

/-! ## The model and its orbit -/

/-- The saturated-corner `TANH(n)` realization of the `n`-state automaton
`step`/`init`/`outM`: state space `Fin n → ℝ`, the `tanh`-saturated corner step
above, and a shared readout requiring exact corner equality. Off the exact corners
it returns the initial-state output; it is not a robust decoder. -/
noncomputable def tanhCornerModel {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) : StreamModel (Fin n → ℝ) α β where
  init := enc init
  step := fun s x => fun i => Real.tanh (∑ j, cornerMatrix step x i j * s j)
  out := fun s => outM (if h : ∃ j, enc j = s then Classical.choose h else init)

theorem tanhCornerModel_step_enc {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) (j : Fin n) (x : α) :
    (tanhCornerModel step init outM).step (enc j) x = enc (step j x) :=
  tanh_cornerMatrix_mul_enc step j x

/-- The `TANH(n)` state after a word is the corner code of the automaton state. -/
theorem tanhCornerModel_fold {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) (s : Fin n) (w : List α) :
    w.foldl (tanhCornerModel step init outM).step (enc s) = enc (w.foldl step s) := by
  induction w generalizing s with
  | nil => rfl
  | cons x xs ih =>
      simp only [List.foldl_cons]
      rw [tanhCornerModel_step_enc]
      exact ih (step s x)

theorem tanhCornerModel_stateAfter {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) (w : List α) :
    (tanhCornerModel step init outM).stateAfter w =
      enc ((finModel step init outM).stateAfter w) :=
  tanhCornerModel_fold step init outM init w

/-- Every reachable state is an exact corner code: `n` reachable states. -/
theorem tanhCornerModel_reachable_enc {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) (w : List α) :
    ∃ j, (tanhCornerModel step init outM).stateAfter w = enc j :=
  ⟨_, tanhCornerModel_stateAfter step init outM w⟩

/-- Every reachable state lies in `(-1,1)ⁿ`: the bounded region that a fixed
resolution discretizes (compare `tanhStreamModel_state_bounded`). -/
theorem tanhCornerModel_reachable_bounded {n : Nat} {α β : Type*}
    (step : Fin n → α → Fin n) (init : Fin n) (outM : Fin n → β) (w : List α) (i : Fin n) :
    |(tanhCornerModel step init outM).stateAfter w i| < 1 := by
  obtain ⟨j, hj⟩ := tanhCornerModel_reachable_enc step init outM w
  rw [hj]
  exact enc_abs_lt_one j i

/-- **Every finite automaton is an exact `TANH(n)` model.** The online output
streams agree at every position of every word, with one shared readout and
reachable corner states. No ambient bounded abstraction and no dimension lower
bound is claimed. -/
theorem moore_realized_by_tanh_corner {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) :
    ∀ w, (tanhCornerModel step init outM).run w = (finModel step init outM).run w := by
  intro w
  have hsel : ∀ s : Fin n,
      Classical.choose (show ∃ j, enc j = enc s from ⟨s, rfl⟩) = s := by
    intro s
    have hs : enc (Classical.choose (show ∃ j, enc j = enc s from ⟨s, rfl⟩)) = enc s :=
      Classical.choose_spec (show ∃ j, enc j = enc s from ⟨s, rfl⟩)
    exact enc_injective hs
  have h := StreamModel.runFrom_sim (M := finModel step init outM)
    (M' := tanhCornerModel step init outM) (@enc n)
    (fun s x => (tanhCornerModel_step_enc step init outM s x).symm)
    (fun s => by
      change outM s = outM (if h : ∃ j, enc j = enc s then Classical.choose h else init)
      rw [dif_pos ⟨s, rfl⟩, hsel s])
    init w
  change (tanhCornerModel step init outM).runFrom (enc init) w
    = (finModel step init outM).runFrom init w
  exact h.symm

/-! ## The finite-program answer: both arms reach every `n`-state behaviour -/

/-- **Finite-program class equality at dimension `n`.** Every `n`-state automaton's
online behaviour — an indefinitely-running finite program with one output per
token — is realized exactly, with a single shared readout and no horizon
dependence, by
* a **linear** model of dimension `n` with `0/1` one-hot reachable states
  (`moore_realized_by_onehot_linear`), and
* a **`tanh`** model of dimension `n` with `{0, τ}` corner reachable states
  (`moore_realized_by_tanh_corner`).

So for the finite-program world the two activation classes coincide at `d = n` in
the reachable-realization sense. Neither conjunct supplies a bounded abstraction
on the ambient state space, and neither is a dimension lower bound; the strict
statements about *bounded-precision channels* stay with
`StreamingExpressivity.count_needs_unbounded_state` /
`balanced_needs_unbounded_state`. -/
theorem finite_program_class_equality (n : Nat) :
    (∀ (α β : Type*) (step : Fin n → α → Fin n) (init : Fin n) (outM : Fin n → β),
        ∃ M : StreamModel (Fin n → ℚ) α β,
          (∀ w, M.run w = (finModel step init outM).run w) ∧
          (∀ w, ∃ j, M.stateAfter w = onehot j)) ∧
    (∀ (α β : Type*) (step : Fin n → α → Fin n) (init : Fin n) (outM : Fin n → β),
        ∃ M : StreamModel (Fin n → ℝ) α β,
          (∀ w, M.run w = (finModel step init outM).run w) ∧
          (∀ w, ∃ j, M.stateAfter w = enc j)) := by
  refine ⟨fun α β step init outM =>
    ⟨finLinModel step init outM, moore_realized_by_onehot_linear step init outM,
      finLinModel_reachable_onehot step init outM⟩, fun α β step init outM =>
    ⟨tanhCornerModel step init outM, moore_realized_by_tanh_corner step init outM,
      tanhCornerModel_reachable_enc step init outM⟩⟩

/-! ## Why the equality is a reachable-realization statement, not an ambient one -/

/-- **An injective shared readout admits no bounded abstraction at all.** If `M.out`
is injective on an infinite state space, `BoundedAbstraction.out_congr` forces the
quotient `q` to be injective, contradicting finiteness. So no `TANH`- or
`LIN`-shaped model with a distinguishing readout has a finite *ambient* abstraction;
class equality at bounded precision can only be stated on the reachable orbit (as
`finite_program_class_equality` does) or conditionally on the abstraction hypothesis
(as `exists_fin_model_of_abstraction` and
`omega_class_activation_independent_at_bounded_precision` do). -/
theorem no_ambient_abstraction_of_injective_readout {σ α β : Type*} [Infinite σ]
    (M : StreamModel σ α β) (hout : Injective M.out) {n : Nat}
    (A : BoundedAbstraction M n) : False := by
  have hq : Injective A.q := fun s s' h => hout (A.out_congr s s' h)
  haveI : Finite σ := Finite.of_injective A.q hq
  exact not_finite σ

end StreamingTanhRealization
