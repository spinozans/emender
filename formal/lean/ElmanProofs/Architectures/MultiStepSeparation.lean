/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Architectures.M2RNNComparison
import ElmanProofs.Architectures.RecurrentResourceFormalism
import ElmanProofs.Activations.Lipschitz

/-!
# Multi-Step Resource Separation: Emender vs Fixed-Right Raw-Write M2RNN

This module extends the one-step resource separation in
`RecurrentResourceFormalism` to **two recurrent steps**.

## What this module proves

The existing one-step theorem `emender_m2rnn_one_step_resource_separation` shows that
for the witness `(lowerLeftState, mixedKey, 0)`, no fixed-right/raw-write
M2RNN-style resource (with row, column, or cell forget carry) can match
Emender's single recurrent update.

This module extends that to two steps. The construction:

* **Step 1 input:** `(mixedKey, 0)` — the existing one-step witness.
* **Step 2 input:** `(0, 0)` — zero key, zero value.

The key observation that makes the two-step proof clean is **row locality of
the fixed-right candidate path**. For any fixed-right M2RNN resource:

* If row 0 of the state matrix is the zero vector, then row 0 of `H · W` is
  also the zero vector (since right multiplication maps row 0 to a linear
  combination of row 0 of `H`).
* If additionally the raw write `k · vᵀ` is zero in row 0 (because either
  `k(0) = 0` or `v = 0`), then row 0 of the candidate `tanh(H W + k vᵀ)` is
  `tanh(0) = 0`.
* The row, column, and cell forget gates are then convex combinations of row 0
  of `H` and row 0 of `Z`, both zero, giving row 0 of the update equal to zero.

Conversely, Emender's expanded delta transition `(I − k kᵀ) H` does **not**
preserve the zero-row property of M2RNN: at the witness state and key, it
produces a nonzero entry in row 0. The second step (with `k = 0`) then just
applies `tanh` elementwise, preserving that nonzero entry.

The entry `(0, 0)` of the two-step trajectories is `tanh(tanh(−1))` for
Emender and `0` for every fixed-right/raw-write resource, witnessing the two-step
separation.

## Why this matters

The one-step theorem says a fixed-right raw-write rule cannot match Emender in
one step. By itself, this leaves open the possibility that two such rules
iterated might still match Emender iterated — for example, because composition could
"average out" the per-step gap. This module rules out that possibility on a
concrete two-step input sequence.

The two-step result is the natural composition extension; deeper multi-step
inseparability targets are documented in `formal/lean/LEAN_FRONTIER.md`.
-/

namespace MultiStepSeparation

open RecurrentResourceFormalism

/-! ## Two-step iteration helper -/

/-- One-step update type: maps `(state, key, value)` to a new state. -/
abbrev OneStepUpdate :=
  TwoMat → TwoVec → TwoVec → TwoMat

/-- Apply a one-step update twice with given inputs. -/
def twoStep (update : OneStepUpdate) (H : TwoMat) (k1 v1 k2 v2 : TwoVec) : TwoMat :=
  update (update H k1 v1) k2 v2

/-! ## Row 0 of `lowerLeftState`

`lowerLeftState` has row 0 equal to the zero vector. The fixed-right
raw-write resources preserve this when applied with `(mixedKey, 0)`. -/

theorem lowerLeftState_row0 (j : Fin 2) :
    lowerLeftState 0 j = 0 := by
  fin_cases j <;> simp [lowerLeftState, Matrix.of_apply]

/-! ## Step 1 — first row of `H₁_R` is zero for every M2RNN-style resource -/

/-- Row 0 of the M2RNN candidate `tanh(H · W + k · vᵀ)` is zero when
row 0 of `H` is zero and either `v = 0` or `k(0) = 0`.

In our witness we use `v = 0`, so the outer product `k · vᵀ` vanishes; the
right-multiplication `H · W` keeps row 0 zero by row locality. -/
theorem m2rnnCandidate_row0_zero_at_zero_row
    (W : TwoMat) (k : TwoVec) (H : TwoMat)
    (hrow : ∀ j, H 0 j = 0) (j : Fin 2) :
    M2RNNComparison.m2rnnCandidate W H k (0 : TwoVec) 0 j = 0 := by
  simp [M2RNNComparison.m2rnnCandidate, M2RNNComparison.matrixTanh,
    M2RNNComparison.matrixMap, M2RNNComparison.outerKV,
    Matrix.mul_apply, hrow]

/-- Row 0 of `m2rnnRowForgetUpdate2` is zero when row 0 of `H` is zero
and `v = 0`. -/
theorem m2rnnRowForgetUpdate2_row0_zero
    (W : TwoMat) (r : TwoVec) (k : TwoVec) (H : TwoMat)
    (hrow : ∀ j, H 0 j = 0) (j : Fin 2) :
    m2rnnRowForgetUpdate2 W r H k (0 : TwoVec) 0 j = 0 := by
  have hcand : M2RNNComparison.m2rnnCandidate W H k (0 : TwoVec) 0 j = 0 :=
    m2rnnCandidate_row0_zero_at_zero_row W k H hrow j
  simp [m2rnnRowForgetUpdate2, rowForgetCarry2, hrow, hcand]

/-- Row 0 of `m2rnnColumnForgetUpdate2` is zero when row 0 of `H` is zero
and `v = 0`. -/
theorem m2rnnColumnForgetUpdate2_row0_zero
    (W : TwoMat) (c : TwoVec) (k : TwoVec) (H : TwoMat)
    (hrow : ∀ j, H 0 j = 0) (j : Fin 2) :
    m2rnnColumnForgetUpdate2 W c H k (0 : TwoVec) 0 j = 0 := by
  have hcand : M2RNNComparison.m2rnnCandidate W H k (0 : TwoVec) 0 j = 0 :=
    m2rnnCandidate_row0_zero_at_zero_row W k H hrow j
  simp [m2rnnColumnForgetUpdate2, columnForgetCarry2, hrow, hcand]

/-- Row 0 of `m2rnnCellForgetUpdate2` is zero when row 0 of `H` is zero
and `v = 0`. -/
theorem m2rnnCellForgetUpdate2_row0_zero
    (W : TwoMat) (g : TwoMat) (k : TwoVec) (H : TwoMat)
    (hrow : ∀ j, H 0 j = 0) (j : Fin 2) :
    m2rnnCellForgetUpdate2 W g H k (0 : TwoVec) 0 j = 0 := by
  have hcand : M2RNNComparison.m2rnnCandidate W H k (0 : TwoVec) 0 j = 0 :=
    m2rnnCandidate_row0_zero_at_zero_row W k H hrow j
  simp [m2rnnCellForgetUpdate2, cellForgetCarry2, hrow, hcand]

/-- Generic zero-row preservation: for any fixed-right raw-write external-forget
resource, applying the update with `v = 0` to a state whose row 0 is zero
yields a state whose row 0 is also zero. -/
theorem FixedRightRawExternalForget2_preserves_zero_row
    (resource : FixedRightRawExternalForget2)
    (k : TwoVec) (H : TwoMat)
    (hrow : ∀ j, H 0 j = 0) (j : Fin 2) :
    resource.update H k (0 : TwoVec) 0 j = 0 := by
  cases resource with
  | row W r => exact m2rnnRowForgetUpdate2_row0_zero W r k H hrow j
  | column W c => exact m2rnnColumnForgetUpdate2_row0_zero W c k H hrow j
  | cell W g => exact m2rnnCellForgetUpdate2_row0_zero W g k H hrow j

/-! ## Step 2 entry (0, 0) is zero for the M2RNN resource family -/

/-- After two M2RNN-style steps with inputs `(k₁, 0)` and `(0, 0)` starting from
`lowerLeftState`, the entry `(0, 0)` is zero. -/
theorem twoStep_m2rnn_entry_zero
    (resource : FixedRightRawExternalForget2) (k1 : TwoVec) :
    twoStep resource.update lowerLeftState k1 (0 : TwoVec) (0 : TwoVec) (0 : TwoVec) 0 0
      = 0 := by
  have hrow0 : ∀ j, lowerLeftState 0 j = 0 := lowerLeftState_row0
  have hrow1 :
      ∀ j, resource.update lowerLeftState k1 (0 : TwoVec) 0 j = 0 := by
    intro j
    exact FixedRightRawExternalForget2_preserves_zero_row resource k1 lowerLeftState hrow0 j
  -- After step 2 (with k = 0, v = 0), row 0 of update is also zero.
  exact FixedRightRawExternalForget2_preserves_zero_row resource
    (0 : TwoVec) (resource.update lowerLeftState k1 (0 : TwoVec)) hrow1 0

/-! ## Step 2 entry (0, 0) is nonzero for NDM -/

/-- After one NDM step on `(lowerLeftState, mixedKey, 0)`, the state at entry
`(0, 0)` is `tanh(-1)`. -/
theorem emender_step1_entry_at_witness :
    M2RNNComparison.e88DeltaUpdateExpanded 1 lowerLeftState mixedKey (0 : TwoVec) 0 0
      = Real.tanh (-1) := by
  simp [M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
    M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
    M2RNNComparison.outerKV, lowerLeftState, mixedKey, Matrix.mul_apply]

/-- An NDM step with `k = 0` and `v = 0` reduces to elementwise `tanh`. -/
theorem emender_zero_input_step (H : TwoMat) (i j : Fin 2) :
    M2RNNComparison.e88DeltaUpdateExpanded 1 H (0 : TwoVec) (0 : TwoVec) i j
      = Real.tanh (H i j) := by
  simp [M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
    M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
    M2RNNComparison.outerKV, Matrix.mul_apply,
    Matrix.one_apply, Finset.sum_ite_eq]

/-- After two NDM steps on `(lowerLeftState, mixedKey, 0)` then `(0, 0)`, the
entry `(0, 0)` is `tanh(tanh(-1))`, which is strictly nonzero. -/
theorem emender_twoStep_entry_at_witness :
    twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
        lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec) 0 0
      = Real.tanh (Real.tanh (-1)) := by
  unfold twoStep
  rw [emender_zero_input_step]
  rw [emender_step1_entry_at_witness]

/-- `tanh(tanh(-1))` is nonzero. -/
theorem tanh_tanh_neg_one_ne_zero : Real.tanh (Real.tanh (-1)) ≠ 0 := by
  intro h
  have hinner : Real.tanh (-1) = 0 := by
    have hzero : Real.tanh (Real.tanh (-1)) = Real.tanh 0 := by
      simpa [Real.tanh_zero] using h
    exact Activation.tanh_injective hzero
  exact tanh_neg_one_ne_zero hinner

/-! ## Two-step separation theorem -/

/-- **Two-step resource separation (Milestone 1).**

NDM and every fixed-right raw-write M2RNN-style resource produce different
states after the two-token input sequence `[(mixedKey, 0), (0, 0)]` applied to
the witness state `lowerLeftState`.

This is the natural composition of the one-step separation. The first step is
the existing one-step witness; the second step injects a "free" elementwise
`tanh` for NDM while M2RNN's row-locality keeps the first row dead. The
distinguishing entry is `(0, 0)`: NDM produces `tanh(tanh(-1)) ≠ 0`, every
fixed-right raw-write resource produces exactly `0`.

This rules out the possibility that the one-step gap is washed out by
composition. The resource separation persists through (at least) two recurrent
steps. -/
theorem emender_m2rnn_two_step_separation :
    twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
        lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec)
      = Matrix.of (fun i j =>
          if i = 0 ∧ j = 0 then Real.tanh (Real.tanh (-1))
          else twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
                  lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec) i j)
    ∧ ∀ resource : FixedRightRawExternalForget2,
        twoStep resource.update
            lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec)
          ≠ twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
              lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec) := by
  refine ⟨?_, ?_⟩
  · ext i j
    by_cases h : i = 0 ∧ j = 0
    · rcases h with ⟨hi, hj⟩
      subst hi; subst hj
      simp [emender_twoStep_entry_at_witness]
    · simp [h]
  · intro resource hEq
    have hR : twoStep resource.update lowerLeftState mixedKey (0 : TwoVec)
        (0 : TwoVec) (0 : TwoVec) 0 0 = 0 :=
      twoStep_m2rnn_entry_zero resource mixedKey
    have hN : twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
        lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec) 0 0
        = Real.tanh (Real.tanh (-1)) :=
      emender_twoStep_entry_at_witness
    have hEqEntry :
        twoStep resource.update lowerLeftState mixedKey (0 : TwoVec)
            (0 : TwoVec) (0 : TwoVec) 0 0 =
          twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1)
            lowerLeftState mixedKey (0 : TwoVec) (0 : TwoVec) (0 : TwoVec) 0 0 := by
      exact congrFun (congrFun hEq 0) 0
    rw [hR, hN] at hEqEntry
    exact tanh_tanh_neg_one_ne_zero hEqEntry.symm

/-! ## Cleaner corollary form -/

/-- **Compact form of two-step separation.**

For every fixed-right raw-write M2RNN-style resource, there is a two-token
input sequence on which the composed two-step output differs from NDM's
composed two-step output.

The witnessing sequence is `[(mixedKey, 0), (0, 0)]`, applied to the witness
state `lowerLeftState`. -/
theorem emender_m2rnn_two_step_separation_exists :
    ∀ resource : FixedRightRawExternalForget2,
      ∃ (H0 : TwoMat) (k1 v1 k2 v2 : TwoVec),
        twoStep resource.update H0 k1 v1 k2 v2 ≠
          twoStep (M2RNNComparison.e88DeltaUpdateExpanded 1) H0 k1 v1 k2 v2 := by
  intro resource
  refine ⟨lowerLeftState, mixedKey, (0 : TwoVec), (0 : TwoVec), (0 : TwoVec), ?_⟩
  exact (emender_m2rnn_two_step_separation.2 resource)

/-! ## k-step separation

The argument above extends to k recurrent steps for every `k ≥ 1`. The
witnessing input sequence is the one-step witness `(mixedKey, 0)` followed by
`(k − 1)` "zero" steps `(0, 0)`.

**NDM trajectory.** After step 1, entry `(0, 0)` is `tanh(-1)`. Each subsequent
zero-input step with `k = 0` reduces NDM's transition to elementwise `tanh`
(since `(λ I − k kᵀ) = I` when `k = 0`), so the entry at `(0, 0)` after step
`k ≥ 1` is `tanh^k(-1)`, the `k`-fold composition `tanh` applied to `-1`. This
is nonzero by induction: `tanh(x) ≠ 0` whenever `x ≠ 0` (`tanh` is injective
and `tanh(0) = 0`).

**M2RNN trajectory.** Row 0 of the state stays the zero vector at every step,
by the same row-locality lemma that worked for the 2-step case. So entry
`(0, 0)` of the M2RNN trajectory is `0` at every step.
-/

/-- `n`-fold composition of `Real.tanh`. -/
noncomputable def tanhIter : ℕ → ℝ → ℝ
  | 0, x => x
  | n + 1, x => Real.tanh (tanhIter n x)

/-- `tanhIter n x` is zero iff `x` is zero (for any `n`). -/
theorem tanhIter_ne_zero_of_ne_zero (n : ℕ) (x : ℝ) (hx : x ≠ 0) :
    tanhIter n x ≠ 0 := by
  induction n with
  | zero => simpa [tanhIter]
  | succ n ih =>
    intro h
    have hzero : Real.tanh (tanhIter n x) = Real.tanh 0 := by
      simpa [tanhIter, Real.tanh_zero] using h
    have hcomp : tanhIter n x = 0 := Activation.tanh_injective hzero
    exact ih hcomp

/-- `tanhIter` commutes with prefix application of `tanh`: applying `tanhIter n`
to `tanh a` is the same as applying `tanh` to `tanhIter n a`. -/
theorem tanhIter_tanh_comm (n : ℕ) (a : ℝ) :
    tanhIter n (Real.tanh a) = Real.tanh (tanhIter n a) := by
  induction n with
  | zero => simp [tanhIter]
  | succ n ih => simp [tanhIter, ih]

/-- Apply `update` `n` times with inputs `(k₁, v₁)`, `(k₂, v₂)`, …. -/
def iterUpdate (update : OneStepUpdate) (H : TwoMat) :
    List (TwoVec × TwoVec) → TwoMat
  | [] => H
  | (k, v) :: rest => iterUpdate update (update H k v) rest

/-- The list of `n` "zero" steps `(0, 0)`. -/
def zeroSteps : ℕ → List (TwoVec × TwoVec)
  | 0 => []
  | n + 1 => ((0 : TwoVec), (0 : TwoVec)) :: zeroSteps n

/-- The k-step witness input: one witness step followed by `(k - 1)` zero
steps. Length is exactly `k` for every `k ≥ 1`. -/
def kStepWitnessInputs (k : ℕ) : List (TwoVec × TwoVec) :=
  (mixedKey, (0 : TwoVec)) :: zeroSteps (k - 1)

/-! ### M2RNN side: row 0 stays zero across every step -/

/-- Iterating any M2RNN-style resource on a list of zero-key/zero-value steps
preserves the zero-row-0 property. -/
theorem iterUpdate_zeroSteps_preserves_row0
    (resource : FixedRightRawExternalForget2)
    (H : TwoMat) (hrow : ∀ j, H 0 j = 0) (n : ℕ) (j : Fin 2) :
    iterUpdate resource.update H (zeroSteps n) 0 j = 0 := by
  induction n generalizing H with
  | zero => simpa [iterUpdate, zeroSteps] using hrow j
  | succ n ih =>
    simp only [zeroSteps, iterUpdate]
    apply ih
    intro j'
    exact FixedRightRawExternalForget2_preserves_zero_row resource
      (0 : TwoVec) H hrow j'

/-- Entry `(0, 0)` of the k-step M2RNN trajectory on the witness state is `0`
for every `k ≥ 1`. -/
theorem iterUpdate_m2rnn_kStep_entry_zero
    (resource : FixedRightRawExternalForget2) (k : ℕ) (hk : 1 ≤ k) :
    iterUpdate resource.update lowerLeftState (kStepWitnessInputs k) 0 0 = 0 := by
  unfold kStepWitnessInputs
  simp only [iterUpdate]
  have hrow1 : ∀ j, resource.update lowerLeftState mixedKey (0 : TwoVec) 0 j = 0 := by
    intro j
    exact FixedRightRawExternalForget2_preserves_zero_row resource
      mixedKey lowerLeftState lowerLeftState_row0 j
  exact iterUpdate_zeroSteps_preserves_row0 resource
    (resource.update lowerLeftState mixedKey (0 : TwoVec)) hrow1 (k - 1) 0

/-! ### NDM side: entry (0,0) is `tanhIter k (-1)` -/

/-- An NDM step with `(k = 0, v = 0)` applied to a matrix whose row 0 is
`(a, 0)` and rows ≥ 1 are zero produces a matrix whose row 0 is `(tanh a, 0)`
and rows ≥ 1 are zero. -/
theorem emender_zero_step_on_clean_state
    (a : ℝ) (i j : Fin 2) :
    M2RNNComparison.e88DeltaUpdateExpanded 1
        (Matrix.of (fun (r c : Fin 2) =>
          if r = 0 ∧ c = 0 then a else 0))
        (0 : TwoVec) (0 : TwoVec) i j
      = (if i = 0 ∧ j = 0 then Real.tanh a else 0) := by
  fin_cases i <;> fin_cases j <;>
    simp [M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap, M2RNNComparison.outerKV,
      Matrix.mul_apply, Matrix.one_apply,
      Matrix.of_apply, Real.tanh_zero]

/-- After one NDM step on the witness, the state at position `(i, j)` equals
`tanh(-1)` when `(i, j) = (0, 0)` and `0` otherwise. -/
theorem emender_step1_clean_form (i j : Fin 2) :
    M2RNNComparison.e88DeltaUpdateExpanded 1 lowerLeftState mixedKey (0 : TwoVec) i j
      = (if i = 0 ∧ j = 0 then Real.tanh (-1) else 0) := by
  fin_cases i <;> fin_cases j <;>
    simp [M2RNNComparison.e88DeltaUpdateExpanded, M2RNNComparison.e88DeltaTransition,
      M2RNNComparison.matrixTanh, M2RNNComparison.matrixMap,
      M2RNNComparison.outerKV, lowerLeftState, mixedKey, Matrix.mul_apply,
      Matrix.one_apply]

/-- After one NDM step on the witness, the full state matrix equals the
"clean" form: `(0, 0)` entry is `tanh(-1)`, all others zero. -/
theorem emender_step1_clean_matrix :
    M2RNNComparison.e88DeltaUpdateExpanded 1 lowerLeftState mixedKey (0 : TwoVec)
      = Matrix.of (fun (r c : Fin 2) =>
          if r = 0 ∧ c = 0 then Real.tanh (-1) else 0) := by
  ext i j
  rw [emender_step1_clean_form]
  by_cases h : i = 0 ∧ j = 0 <;> simp [h, Matrix.of_apply]

/-- Applying `n` zero-input NDM steps to a "clean" matrix whose `(0,0)` entry
is `a` and other entries are `0` produces another "clean" matrix whose `(0, 0)`
entry is `tanhIter n a`. -/
theorem emender_iter_zero_clean (n : ℕ) (a : ℝ) :
    iterUpdate (M2RNNComparison.e88DeltaUpdateExpanded 1)
        (Matrix.of (fun (r c : Fin 2) => if r = 0 ∧ c = 0 then a else 0))
        (zeroSteps n)
      = Matrix.of (fun (r c : Fin 2) =>
          if r = 0 ∧ c = 0 then tanhIter n a else 0) := by
  induction n generalizing a with
  | zero =>
    simp [iterUpdate, zeroSteps, tanhIter]
  | succ n ih =>
    simp only [zeroSteps, iterUpdate]
    have hstep :
        M2RNNComparison.e88DeltaUpdateExpanded 1
            (Matrix.of (fun (r c : Fin 2) => if r = 0 ∧ c = 0 then a else 0))
            (0 : TwoVec) (0 : TwoVec)
          = Matrix.of (fun (r c : Fin 2) =>
              if r = 0 ∧ c = 0 then Real.tanh a else 0) := by
      ext i j
      rw [emender_zero_step_on_clean_state]
      by_cases h : i = 0 ∧ j = 0 <;> simp [h, Matrix.of_apply]
    rw [hstep, ih]
    ext i j
    by_cases h : i = 0 ∧ j = 0
    · simp [h, Matrix.of_apply, tanhIter, tanhIter_tanh_comm]
    · simp [h, Matrix.of_apply]

/-- Entry `(0, 0)` of the k-step NDM trajectory on the witness is
`tanhIter k (-1)` for every `k ≥ 1`. -/
theorem iterUpdate_emender_kStep_entry
    (k : ℕ) (hk : 1 ≤ k) :
    iterUpdate (M2RNNComparison.e88DeltaUpdateExpanded 1)
        lowerLeftState (kStepWitnessInputs k) 0 0
      = tanhIter k (-1) := by
  unfold kStepWitnessInputs
  simp only [iterUpdate]
  rw [emender_step1_clean_matrix, emender_iter_zero_clean]
  simp only [Matrix.of_apply, and_self, if_true]
  -- After rewriting, goal: tanhIter (k - 1) (Real.tanh (-1)) = tanhIter k (-1)
  rw [tanhIter_tanh_comm]
  -- Goal: Real.tanh (tanhIter (k - 1) (-1)) = tanhIter k (-1)
  have heq : tanhIter k (-1) = Real.tanh (tanhIter (k - 1) (-1)) := by
    have : k = (k - 1) + 1 := by omega
    conv_lhs => rw [this]
    rfl
  exact heq.symm

/-- **k-step separation theorem (extension of Milestone 1).**

For every `k ≥ 1` and every fixed-right raw-write M2RNN-style resource, the
`k`-step trajectory of the resource on the witness input differs from NDM's
`k`-step trajectory on the same input. The distinguishing entry is `(0, 0)`:

* NDM produces `tanhIter k (-1)`, nonzero for every `k ≥ 0` (by injectivity).
* The resource produces exactly `0`, by row-0 zero preservation.

The witness input is `kStepWitnessInputs k = [(mixedKey, 0), (0, 0), …, (0, 0)]`
(length `k`) starting from `lowerLeftState`.

This shows the one-step separation does not wash out over composition: it
strictly persists for every finite length. The result generalizes
`emender_m2rnn_two_step_separation` and constitutes the cleanest Lean ceiling
reachable on the multi-step composition axis. -/
theorem emender_m2rnn_k_step_separation
    (k : ℕ) (hk : 1 ≤ k) (resource : FixedRightRawExternalForget2) :
    iterUpdate resource.update lowerLeftState (kStepWitnessInputs k) ≠
      iterUpdate (M2RNNComparison.e88DeltaUpdateExpanded 1)
        lowerLeftState (kStepWitnessInputs k) := by
  intro hEq
  have hEntry := congrFun (congrFun hEq 0) 0
  rw [iterUpdate_m2rnn_kStep_entry_zero resource k hk,
      iterUpdate_emender_kStep_entry k hk] at hEntry
  exact tanhIter_ne_zero_of_ne_zero k (-1) (by norm_num) hEntry.symm

/-- **k-step separation in existential form.** For every `k ≥ 1` and every
fixed-right raw-write resource, there is a `k`-token input sequence on which
the `k`-step composition differs from NDM. -/
theorem emender_m2rnn_k_step_separation_exists
    (k : ℕ) (hk : 1 ≤ k) (resource : FixedRightRawExternalForget2) :
    ∃ (H0 : TwoMat) (inputs : List (TwoVec × TwoVec)),
      inputs.length = k ∧
      iterUpdate resource.update H0 inputs ≠
        iterUpdate (M2RNNComparison.e88DeltaUpdateExpanded 1) H0 inputs := by
  refine ⟨lowerLeftState, kStepWitnessInputs k, ?_,
    emender_m2rnn_k_step_separation k hk resource⟩
  -- Length of kStepWitnessInputs k is k for k ≥ 1.
  unfold kStepWitnessInputs
  have hlen : (zeroSteps (k - 1)).length = k - 1 := by
    induction (k - 1) with
    | zero => simp [zeroSteps]
    | succ n ih => simp [zeroSteps, List.length_cons, ih]
  simp [List.length_cons, hlen]
  omega

/-! ## General-dimension k-step separation (K ≥ 2, V ≥ 1) — partial

The 2D k-step separation above is the cleanest concrete witness. This section
lifts the **M2RNN side** of the argument to arbitrary state dimensions
`K ≥ 2, V ≥ 1`, using the embedding scheme of
`emender_m2rnn_one_step_resource_separation_embeds`.

The M2RNN-side row-0 zero preservation lemmas generalize cleanly to KV; they
are stated and proved below. The full KV k-step separation theorem additionally
needs a clean-form tracking lemma for the NDM trajectory in KV dimensions —
that step is more intricate because the matrix multiplication in KV does not
reduce by `fin_cases` over a small index set, and proving entry `(i₀, j₀)` of
`(I − k kᵀ) · H` equals `−1` requires summing carefully over `Fin K`. The
proof structure exists in spirit (replicate `emender_step1_clean_matrix` with
explicit indices), but the in-Lean discharge runs into normalisation friction
that we leave as v17 work.

The artifacts that landed: M2RNN-side row-0 preservation in KV, which is the
heart of why the 2D extension argument lifts in principle.
-/

section KVLift

open M2RNNComparison

variable {K V : Nat} (hK : 1 < K) (hV : 0 < V)

/-- The two distinguished indices for the KV embedded witness. -/
@[reducible] def i0 (hK : 1 < K) : Fin K := ⟨0, Nat.lt_trans Nat.zero_lt_one hK⟩
@[reducible] def j0 (hV : 0 < V) : Fin V := ⟨0, hV⟩

/-- Row 0 of `embeddedLowerLeftState` is zero. -/
theorem embeddedLowerLeftState_row0
    (j : Fin V) :
    embeddedLowerLeftState hK hV (i0 hK) j = 0 := by
  simp [embeddedLowerLeftState, i0, Matrix.of_apply]

/-- Row 0 of the M2RNN candidate is zero in KV when row 0 of `H` is zero and
`v = 0`. -/
theorem m2rnnCandidate_row0_zero_at_zero_row_KV
    (W : Matrix (Fin V) (Fin V) Real) (k : Vec K)
    (H : MatState K V)
    (hrow : ∀ j, H (i0 hK) j = 0) (j : Fin V) :
    m2rnnCandidate W H k (0 : Vec V) (i0 hK) j = 0 := by
  simp [m2rnnCandidate, matrixTanh, matrixMap, outerKV,
    Matrix.mul_apply, hrow]

/-- Row 0 of `m2rnnRowForgetUpdateKV` is zero when row 0 of `H` is zero
and `v = 0`. -/
theorem m2rnnRowForgetUpdateKV_row0_zero
    (W : Matrix (Fin V) (Fin V) Real) (r : Vec K) (k : Vec K)
    (H : MatState K V)
    (hrow : ∀ j, H (i0 hK) j = 0) (j : Fin V) :
    m2rnnRowForgetUpdateKV W r H k (0 : Vec V) (i0 hK) j = 0 := by
  have hcand : m2rnnCandidate W H k (0 : Vec V) (i0 hK) j = 0 :=
    m2rnnCandidate_row0_zero_at_zero_row_KV hK W k H hrow j
  simp [m2rnnRowForgetUpdateKV, rowForgetCarryKV, hrow, hcand]

/-- Row 0 of `m2rnnColumnForgetUpdateKV` is zero when row 0 of `H` is zero
and `v = 0`. -/
theorem m2rnnColumnForgetUpdateKV_row0_zero
    (W : Matrix (Fin V) (Fin V) Real) (c : Vec V) (k : Vec K)
    (H : MatState K V)
    (hrow : ∀ j, H (i0 hK) j = 0) (j : Fin V) :
    m2rnnColumnForgetUpdateKV W c H k (0 : Vec V) (i0 hK) j = 0 := by
  have hcand : m2rnnCandidate W H k (0 : Vec V) (i0 hK) j = 0 :=
    m2rnnCandidate_row0_zero_at_zero_row_KV hK W k H hrow j
  simp [m2rnnColumnForgetUpdateKV, columnForgetCarryKV, hrow, hcand]

/-- Row 0 of `m2rnnCellForgetUpdateKV` is zero when row 0 of `H` is zero
and `v = 0`. -/
theorem m2rnnCellForgetUpdateKV_row0_zero
    (W : Matrix (Fin V) (Fin V) Real) (g : MatState K V) (k : Vec K)
    (H : MatState K V)
    (hrow : ∀ j, H (i0 hK) j = 0) (j : Fin V) :
    m2rnnCellForgetUpdateKV W g H k (0 : Vec V) (i0 hK) j = 0 := by
  have hcand : m2rnnCandidate W H k (0 : Vec V) (i0 hK) j = 0 :=
    m2rnnCandidate_row0_zero_at_zero_row_KV hK W k H hrow j
  simp [m2rnnCellForgetUpdateKV, cellForgetCarryKV, hrow, hcand]

/-- Generic zero-row preservation for the KV resource class. -/
theorem FixedRightRawExternalForgetKV_preserves_zero_row
    (resource : FixedRightRawExternalForgetKV K V)
    (k : Vec K) (H : MatState K V)
    (hrow : ∀ j, H (i0 hK) j = 0) (j : Fin V) :
    resource.update H k (0 : Vec V) (i0 hK) j = 0 := by
  cases resource with
  | row W r => exact m2rnnRowForgetUpdateKV_row0_zero hK W r k H hrow j
  | column W c => exact m2rnnColumnForgetUpdateKV_row0_zero hK W c k H hrow j
  | cell W g => exact m2rnnCellForgetUpdateKV_row0_zero hK W g k H hrow j

/-- The one-step update type for KV. -/
abbrev OneStepUpdateKV (K V : Nat) :=
  MatState K V → Vec K → Vec V → MatState K V

/-- Iterated KV update over a list of inputs. -/
def iterUpdateKV (update : OneStepUpdateKV K V) (H : MatState K V) :
    List (Vec K × Vec V) → MatState K V
  | [] => H
  | (k, v) :: rest => iterUpdateKV update (update H k v) rest

/-- List of `n` zero steps in KV. -/
def zeroStepsKV (K V : Nat) : ℕ → List (Vec K × Vec V)
  | 0 => []
  | n + 1 => ((0 : Vec K), (0 : Vec V)) :: zeroStepsKV K V n

/-- KV-dimensional k-step witness input: one witness step followed by `(k - 1)`
zero steps. -/
def kStepWitnessInputsKV (hK : 1 < K) (k : ℕ) : List (Vec K × Vec V) :=
  (embeddedMixedKey hK, (0 : Vec V)) :: zeroStepsKV K V (k - 1)

/-! ### KV M2RNN side: row 0 stays zero -/

/-- Iterating any KV M2RNN resource over zero-input steps preserves zero-row-0. -/
theorem iterUpdateKV_zeroSteps_preserves_row0
    (resource : FixedRightRawExternalForgetKV K V)
    (H : MatState K V) (hrow : ∀ j, H (i0 hK) j = 0) (n : ℕ) (j : Fin V) :
    iterUpdateKV resource.update H (zeroStepsKV K V n) (i0 hK) j = 0 := by
  induction n generalizing H with
  | zero => simpa [iterUpdateKV, zeroStepsKV] using hrow j
  | succ n ih =>
    simp only [zeroStepsKV, iterUpdateKV]
    apply ih
    intro j'
    exact FixedRightRawExternalForgetKV_preserves_zero_row hK resource
      (0 : Vec K) H hrow j'

/-- Entry `(i₀, j₀)` of the k-step KV M2RNN trajectory on the embedded witness
is `0` for every `k ≥ 1`. -/
theorem iterUpdateKV_m2rnn_kStep_entry_zero
    (resource : FixedRightRawExternalForgetKV K V) (k : ℕ) (hk : 1 ≤ k) :
    iterUpdateKV resource.update (embeddedLowerLeftState hK hV)
        (kStepWitnessInputsKV hK k) (i0 hK) (j0 hV) = 0 := by
  unfold kStepWitnessInputsKV
  simp only [iterUpdateKV]
  have hrow0 : ∀ j, embeddedLowerLeftState hK hV (i0 hK) j = 0 :=
    embeddedLowerLeftState_row0 hK hV
  have hrow1 : ∀ j, resource.update (embeddedLowerLeftState hK hV)
      (embeddedMixedKey hK) (0 : Vec V) (i0 hK) j = 0 := by
    intro j
    exact FixedRightRawExternalForgetKV_preserves_zero_row hK resource
      (embeddedMixedKey hK) (embeddedLowerLeftState hK hV) hrow0 j
  exact iterUpdateKV_zeroSteps_preserves_row0 hK resource
    (resource.update (embeddedLowerLeftState hK hV)
      (embeddedMixedKey hK) (0 : Vec V)) hrow1 (k - 1) (j0 hV)

/-! ### NDM-side k-step KV tracking — left to v17

Replicating `iterUpdate_emender_kStep_entry` in KV requires careful arithmetic on
`∑ l : Fin K, (I − k kᵀ) i l · H l j`. The exact value `−1` at entry
`(i₀, j₀)` after one NDM step depends on `embeddedMixedKey` having ones at
exactly two positions and zeros elsewhere, combined with
`embeddedLowerLeftState` having a single nonzero entry at `(1, j₀)`. The 2D
case discharges this via `fin_cases`; the KV case requires splitting the
`Fin K` summation manually.

We leave that mechanization to v17 and document the partial result here: the
M2RNN-side row-0 preservation in KV is the heart of the argument and lands
cleanly. See `LEAN_FRONTIER.md` for the roadmap to fully discharge this in KV.
-/

end KVLift

end MultiStepSeparation
