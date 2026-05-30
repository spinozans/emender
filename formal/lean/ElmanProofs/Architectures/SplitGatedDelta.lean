/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Architectures.M2RNNComparison

/-!
# Split-Gated Delta Updates

This module isolates the per-head algebra for the E97 split-gated delta update
and the matching GDN-2-style linear recurrent core. It proves only the shared
matrix algebra:

* direct E97 split-gated delta form equals its expanded transition form;
* all-one erase/write gates specialize E97 to the existing E88 delta update;
* GDN-2 and E97 use the same split-gated read/write core, with GDN-2 applying
  that core to a pre-decayed state.
-/

namespace SplitGatedDelta

open Matrix BigOperators
open M2RNNComparison

variable {K V : Nat}

/-! ## Pointwise Gates -/

/-- Pointwise product of two vectors. -/
def hadamard {n : Nat} (a b : Vec n) : Vec n :=
  fun i => a i * b i

/-- All-one gate vector. -/
def onesVec (n : Nat) : Vec n :=
  fun _ => 1

/-- An all-one left gate leaves a vector unchanged. -/
@[simp] theorem hadamard_onesVec_left {n : Nat} (v : Vec n) :
    hadamard (onesVec n) v = v := by
  funext i
  simp [hadamard, onesVec]

/-- An all-one right gate leaves a vector unchanged. -/
@[simp] theorem hadamard_onesVec_right {n : Nat} (v : Vec n) :
    hadamard v (onesVec n) = v := by
  funext i
  simp [hadamard, onesVec]

/-- E97 key-axis erase/read gate: `b * k`. -/
def e97ReadKey (k b : Vec K) : Vec K :=
  hadamard b k

/-- E97 value-axis write gate: `w * v`. -/
def e97WriteValue (w v : Vec V) : Vec V :=
  hadamard w v

/-! ## E97 Split-Gated Delta Core -/

/-- Split-gated E97 retrieval: `H^T (b * k)`. -/
def e97Retrieved
    (H : MatState K V) (k b : Vec K) : Vec V :=
  queryReadout H (e97ReadKey k b)

/-- Split-gated E97 delta:

`delta = (w * v) - H^T (b * k)`.
-/
def splitGatedDelta
    (H : MatState K V)
    (k b : Vec K)
    (w v : Vec V) :
    Vec V :=
  e97WriteValue w v - e97Retrieved H k b

/-- Expanded E97 left transition:

`lambda I - k (b * k)^T`.
-/
def splitGatedTransition
    (lambda : Real) (k b : Vec K) :
    Matrix (Fin K) (Fin K) Real :=
  lambda • (1 : Matrix (Fin K) (Fin K) Real) - outerKV k (e97ReadKey k b)

/-- Shared split-gated linear preactivation:

`lambda H + k ((w * v) - H^T (b * k))^T`.
-/
def splitGatedLinearCore
    (lambda : Real)
    (H : MatState K V)
    (k b : Vec K)
    (w v : Vec V) :
    MatState K V :=
  lambda • H + outerKV k (splitGatedDelta H k b w v)

/-- E97 linear preactivation before the elementwise state activation. -/
def e97LinearCore
    (lambda : Real)
    (H : MatState K V)
    (k b : Vec K)
    (w v : Vec V) :
    MatState K V :=
  splitGatedLinearCore lambda H k b w v

/-- E97 direct nonlinear update:

`tanh(lambda H + k ((w * v) - H^T (b * k))^T)`.
-/
noncomputable def e97UpdateDirect
    (lambda : Real)
    (H : MatState K V)
    (k b : Vec K)
    (w v : Vec V) :
    MatState K V :=
  matrixTanh (e97LinearCore lambda H k b w v)

/-- E97 expanded nonlinear update:

`tanh((lambda I - k (b * k)^T) H + k (w * v)^T)`.
-/
noncomputable def e97UpdateExpanded
    (lambda : Real)
    (H : MatState K V)
    (k b : Vec K)
    (w v : Vec V) :
    MatState K V :=
  matrixTanh (splitGatedTransition lambda k b * H + outerKV k (e97WriteValue w v))

/-! ## GDN-2-Style Linear Core -/

/-- GDN-2-style split-gated linear core.

GDN-2 applies the same erase/read and write algebra to a pre-decayed state
`D * H`, and does not apply the E97 elementwise state activation here.
-/
def gdn2LinearCore
    (D : Matrix (Fin K) (Fin K) Real)
    (H : MatState K V)
    (k b : Vec K)
    (w v : Vec V) :
    MatState K V :=
  splitGatedLinearCore 1 (D * H) k b w v

/-! ## Algebraic Theorems -/

/-- The direct E97 linear preactivation equals the expanded transition form. -/
theorem e97LinearCore_eq_expanded
    (lambda : Real) (H : MatState K V) (k b : Vec K) (w v : Vec V) :
    e97LinearCore lambda H k b w v =
      splitGatedTransition lambda k b * H + outerKV k (e97WriteValue w v) := by
  ext i j
  unfold e97LinearCore splitGatedLinearCore splitGatedTransition splitGatedDelta
    e97Retrieved e97ReadKey e97WriteValue hadamard
  simp [queryReadout, outerKV, Matrix.mul_apply, Matrix.mulVec, dotProduct]
  have h_id :
      (∑ x : Fin K, lambda * ((1 : Matrix (Fin K) (Fin K) Real) i x) * H x j) =
        lambda * H i j := by
    rw [Fintype.sum_eq_single i]
    · simp
    · intro x hx
      have hix : i ≠ x := by
        intro h
        exact hx h.symm
      simp [hix]
  have h_k :
      (∑ x : Fin K, k i * (b x * k x) * H x j) =
        k i * ∑ x : Fin K, H x j * (b x * k x) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro x _
    ring
  have h_sum :
      (∑ x : Fin K,
          (lambda * ((1 : Matrix (Fin K) (Fin K) Real) i x) -
              k i * (b x * k x)) * H x j) =
        lambda * H i j - k i * ∑ x : Fin K, H x j * (b x * k x) := by
    calc
      (∑ x : Fin K,
          (lambda * ((1 : Matrix (Fin K) (Fin K) Real) i x) -
              k i * (b x * k x)) * H x j)
          = ∑ x : Fin K,
              (lambda * ((1 : Matrix (Fin K) (Fin K) Real) i x) * H x j -
                k i * (b x * k x) * H x j) := by
            apply Finset.sum_congr rfl
            intro x _
            ring
      _ = (∑ x : Fin K,
              lambda * ((1 : Matrix (Fin K) (Fin K) Real) i x) * H x j) -
          (∑ x : Fin K, k i * (b x * k x) * H x j) := by
            rw [Finset.sum_sub_distrib]
      _ = lambda * H i j - k i * ∑ x : Fin K, H x j * (b x * k x) := by
            rw [h_id, h_k]
  rw [h_sum]
  ring

/-- Direct and expanded E97 split-gated nonlinear updates are algebraically equal. -/
theorem e97UpdateDirect_eq_expanded
    (lambda : Real) (H : MatState K V) (k b : Vec K) (w v : Vec V) :
    e97UpdateDirect lambda H k b w v =
      e97UpdateExpanded lambda H k b w v := by
  unfold e97UpdateDirect e97UpdateExpanded
  rw [e97LinearCore_eq_expanded]

/-- With all-one split gates, E97 direct form specializes to E88 direct form. -/
theorem e97_specializes_to_e88_all_one_gates_direct
    (lambda : Real) (H : MatState K V) (k : Vec K) (v : Vec V) :
    e97UpdateDirect lambda H k (onesVec K) (onesVec V) v =
      e88DeltaUpdateDirect lambda H k v := by
  simp [e97UpdateDirect, e97LinearCore, splitGatedLinearCore, splitGatedDelta,
    e97Retrieved, e97ReadKey, e97WriteValue, e88DeltaUpdateDirect, e88Delta]

/-- With all-one split gates, E97 expanded form specializes to E88 expanded form. -/
theorem e97_specializes_to_e88_all_one_gates_expanded
    (lambda : Real) (H : MatState K V) (k : Vec K) (v : Vec V) :
    e97UpdateExpanded lambda H k (onesVec K) (onesVec V) v =
      e88DeltaUpdateExpanded lambda H k v := by
  simp [e97UpdateExpanded, splitGatedTransition, e97ReadKey, e97WriteValue,
    e88DeltaUpdateExpanded, e88DeltaTransition]

/-- Constructive E88 inclusion: E97 expresses E88 by all-one gate specialization. -/
theorem e97_expresses_e88_by_specialization
    (lambda : Real) (H : MatState K V) (k : Vec K) (v : Vec V) :
    e97UpdateDirect lambda H k (onesVec K) (onesVec V) v =
        e88DeltaUpdateDirect lambda H k v ∧
      e97UpdateExpanded lambda H k (onesVec K) (onesVec V) v =
        e88DeltaUpdateExpanded lambda H k v := by
  constructor
  · exact e97_specializes_to_e88_all_one_gates_direct lambda H k v
  · exact e97_specializes_to_e88_all_one_gates_expanded lambda H k v

/-! ## Split Erase/Write Direction Witness -/

/-- Two-dimensional key/value vector used by the finite split-direction witness. -/
abbrev TwoVec := Vec 2

/-- Two-dimensional matrix state used by the finite split-direction witness. -/
abbrev TwoMat := Matrix (Fin 2) (Fin 2) Real

/-- Explicit split transition from independent write and erase/read directions:
`lambda I - writeDir eraseDir^T`. -/
def splitTransitionFromDirs
    (lambda : Real) (writeDir eraseDir : TwoVec) : TwoMat :=
  lambda • (1 : TwoMat) - outerKV writeDir eraseDir

/-- In two dimensions, parallel vectors are exactly the zero-determinant pairs. -/
def parallel2 (a b : TwoVec) : Prop :=
  a 0 * b 1 = a 1 * b 0

/-- E97's pointwise erase gate realizes the split transition whose erase/read
direction is `hadamard b k`. -/
theorem splitGatedTransition_eq_splitTransitionFromDirs
    (lambda : Real) (k b : TwoVec) :
    splitGatedTransition (K := 2) lambda k b =
      splitTransitionFromDirs lambda k (hadamard b k) := by
  rfl

/-- If a split transition equals an E88 coupled transition in two dimensions,
then the split write and erase/read directions must be parallel.

This is only a necessary collapse condition for the one-step transition factor;
it is not a broad impossibility theorem for all E88 behavior. -/
theorem e88_coupled_transition_forces_parallel_split_dirs
    (lambdaSplit lambdaE88 : Real) (writeDir eraseDir p : TwoVec) :
    splitTransitionFromDirs lambdaSplit writeDir eraseDir =
      e88DeltaTransition (K := 2) lambdaE88 p →
    parallel2 writeDir eraseDir := by
  intro h
  have h01 :
      -(writeDir 0 * eraseDir 1) = -(p 0 * p 1) := by
    have hentry := congrArg (fun M : TwoMat => M 0 1) h
    simpa [splitTransitionFromDirs, e88DeltaTransition, outerKV] using hentry
  have h10 :
      -(writeDir 1 * eraseDir 0) = -(p 1 * p 0) := by
    have hentry := congrArg (fun M : TwoMat => M 1 0) h
    simpa [splitTransitionFromDirs, e88DeltaTransition, outerKV] using hentry
  dsimp [parallel2]
  calc
    writeDir 0 * eraseDir 1 = p 0 * p 1 := by linarith
    _ = p 1 * p 0 := by ring
    _ = writeDir 1 * eraseDir 0 := by linarith

/-- Nonparallel split directions cannot be realized by any two-dimensional E88
coupled transition. The claim is deliberately finite and one-step: it compares
the transition factors `lambda I - u r^T` and `mu I - p p^T`. -/
theorem e88_cannot_realize_nonparallel_split_transition
    (lambdaSplit : Real) (writeDir eraseDir : TwoVec)
    (hnot : ¬ parallel2 writeDir eraseDir) :
    ¬ (∃ lambdaE88 : Real, ∃ p : TwoVec,
      splitTransitionFromDirs lambdaSplit writeDir eraseDir =
        e88DeltaTransition (K := 2) lambdaE88 p) := by
  intro h
  rcases h with ⟨lambdaE88, p, hEq⟩
  exact hnot
    (e88_coupled_transition_forces_parallel_split_dirs
      lambdaSplit lambdaE88 writeDir eraseDir p hEq)

/-- Concrete write direction `u = (1, 1)`. -/
def splitWitnessWriteDir : TwoVec :=
  fun _ => 1

/-- Concrete pointwise erase/read gate `b = (1, 0)`. -/
def splitWitnessEraseGate : TwoVec :=
  fun i => if i = 0 then 1 else 0

/-- Concrete erase/read direction `r = b * u = (1, 0)`. -/
def splitWitnessEraseDir : TwoVec :=
  hadamard splitWitnessEraseGate splitWitnessWriteDir

/-- The value payload is zero so the witness isolates the transition factor. -/
def splitWitnessValue : TwoVec :=
  0

/-- The concrete split witness has genuinely different write and erase/read
directions: `u = (1,1)` and `r = (1,0)` are not parallel. -/
theorem splitWitness_dirs_not_parallel :
    ¬ parallel2 splitWitnessWriteDir splitWitnessEraseDir := by
  intro h
  norm_num [parallel2, splitWitnessWriteDir, splitWitnessEraseDir,
    splitWitnessEraseGate, hadamard] at h

/-- E97 realizes the concrete split transition through its pointwise erase gate. -/
theorem e97_realizes_splitWitness_transition :
    splitGatedTransition (K := 2) 1
        splitWitnessWriteDir splitWitnessEraseGate =
      splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir := by
  rfl

/-- Entries of the concrete split transition:
`I - (1,1) (1,0)^T = [[0, 0], [-1, 1]]`. -/
theorem splitWitness_transition_entries :
    splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 0 0 = 0 ∧
      splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 0 1 = 0 ∧
      splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 1 0 = -1 ∧
      splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir 1 1 = 1 := by
  norm_num [splitTransitionFromDirs, splitWitnessWriteDir, splitWitnessEraseDir,
    splitWitnessEraseGate, hadamard, outerKV]

/-- No two-dimensional E88 coupled transition can realize the concrete
nonparallel split erase/write transition. -/
theorem e88_cannot_realize_splitWitness_transition :
    ¬ (∃ lambdaE88 : Real, ∃ p : TwoVec,
      splitTransitionFromDirs 1 splitWitnessWriteDir splitWitnessEraseDir =
        e88DeltaTransition (K := 2) lambdaE88 p) := by
  exact e88_cannot_realize_nonparallel_split_transition
    (lambdaSplit := 1)
    splitWitnessWriteDir
    splitWitnessEraseDir
    splitWitness_dirs_not_parallel

/-! ## 1x1 Strict Finite Witness -/

/-- Zero state for the 1x1 strict split-gate witness. -/
def strictWitnessZeroState : MatState 1 1 :=
  0

/-- Unit vector for the 1x1 strict split-gate witness. -/
def strictWitnessOneVec : Vec 1 :=
  fun _ => 1

/-- Write-gate vector equal to two for the 1x1 strict split-gate witness. -/
def strictWitnessTwoVec : Vec 1 :=
  fun _ => 2

/-- A concrete finite witness that E97 is not merely the all-one-gate E88
subfamily.

On the 1x1 zero state with unit key and value, setting the E97 write gate to
`2` produces `tanh 2`, while any E88/all-one-gate specialization on the same
state/input produces `tanh 1` (the decay scalar is irrelevant because the state
is zero). This proves a strict family extension for this concrete setting only;
it does not make any empirical efficiency claim. -/
theorem e97_split_gate_strict_witness_not_e88_all_one
    (lambdaE97 lambdaE88 : Real) :
    e97UpdateDirect lambdaE97 strictWitnessZeroState
        strictWitnessOneVec strictWitnessOneVec strictWitnessTwoVec strictWitnessOneVec ≠
      e88DeltaUpdateDirect lambdaE88 strictWitnessZeroState
        strictWitnessOneVec strictWitnessOneVec := by
  intro h
  have hentry := congrArg (fun M : MatState 1 1 => M 0 0) h
  simp [e97UpdateDirect, e97LinearCore, splitGatedLinearCore, splitGatedDelta,
    e97Retrieved, e97ReadKey, e97WriteValue, strictWitnessZeroState,
    strictWitnessOneVec, strictWitnessTwoVec, e88DeltaUpdateDirect, e88Delta,
    matrixTanh, matrixMap, outerKV, queryReadout, hadamard] at hentry
  have hpre : (2 : Real) = 1 := Activation.tanh_injective hentry
  norm_num at hpre

/-- GDN-2 applies the E97 split-gated linear core to the pre-decayed state. -/
theorem gdn2LinearCore_eq_e97LinearCore_on_decayed_state
    (D : Matrix (Fin K) (Fin K) Real)
    (H : MatState K V) (k b : Vec K) (w v : Vec V) :
    gdn2LinearCore D H k b w v =
      e97LinearCore 1 (D * H) k b w v := by
  rfl

/-- With identity decay, the GDN-2 linear core is exactly E97's unit-decay core. -/
theorem gdn2LinearCore_identity_decay_eq_e97LinearCore_one
    (H : MatState K V) (k b : Vec K) (w v : Vec V) :
    gdn2LinearCore (1 : Matrix (Fin K) (Fin K) Real) H k b w v =
      e97LinearCore 1 H k b w v := by
  simp [gdn2LinearCore, e97LinearCore]

/-- The GDN-2 linear core also has the expanded split-gated transition form. -/
theorem gdn2LinearCore_eq_expanded
    (D : Matrix (Fin K) (Fin K) Real)
    (H : MatState K V) (k b : Vec K) (w v : Vec V) :
    gdn2LinearCore D H k b w v =
      splitGatedTransition 1 k b * (D * H) + outerKV k (e97WriteValue w v) := by
  rw [gdn2LinearCore_eq_e97LinearCore_on_decayed_state,
    e97LinearCore_eq_expanded]

/-- E97 and GDN-2 share the same split-gated linear read/write core. -/
theorem e97_and_gdn2_share_split_gated_linear_core
    (lambda : Real)
    (D : Matrix (Fin K) (Fin K) Real)
    (H : MatState K V) (k b : Vec K) (w v : Vec V) :
    e97LinearCore lambda H k b w v =
        splitGatedLinearCore lambda H k b w v ∧
      gdn2LinearCore D H k b w v =
        splitGatedLinearCore 1 (D * H) k b w v := by
  constructor <;> rfl

end SplitGatedDelta
