/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
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
