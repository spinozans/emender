import ElmanProofs.Architectures.MemoryOperator

/-! # Diagonal affine summaries and sparse slot injections -/

namespace MemoryAlgebra

open BigOperators

variable {ι : Type*}

/-- A diagonal affine summary needs only two arrays, including after composition. -/
structure DiagonalOperator (ι : Type*) where
  scale : ι → ℝ
  shift : ι → ℝ

namespace DiagonalOperator

def toAffine (a : DiagonalOperator ι) : AffineOperator (ι → ℝ) where
  linear :=
    { toFun := fun s i => a.scale i * s i
      map_add' := by intros; ext i; exact mul_add _ _ _
      map_smul' := by
        intros; ext i
        simp only [Pi.smul_apply, smul_eq_mul, RingHom.id_apply]
        ring }
  injection := a.shift

@[simp] theorem toAffine_apply (a : DiagonalOperator ι) (s : ι → ℝ) (i : ι) :
    a.toAffine.apply s i = a.scale i * s i + a.shift i := rfl

def identity : DiagonalOperator ι := ⟨fun _ => 1, fun _ => 0⟩

def thenOp (a b : DiagonalOperator ι) : DiagonalOperator ι :=
  ⟨fun i => b.scale i * a.scale i, fun i => b.scale i * a.shift i + b.shift i⟩

/-- Diagonal closure, including the propagated injection. -/
theorem toAffine_thenOp (a b : DiagonalOperator ι) :
    (a.thenOp b).toAffine = a.toAffine * b.toAffine := by
  apply AffineOperator.ext
  · ext s i
    simp only [toAffine, thenOp, AffineOperator.linear_mul, LinearMap.comp_apply,
      LinearMap.coe_mk, AddHom.coe_mk]
    ring
  · rfl

theorem toAffine_identity : (identity : DiagonalOperator ι).toAffine = 1 := by
  apply AffineOperator.ext
  · ext s i
    change 1 * s i = s i
    exact one_mul _
  · rfl

def composeAll : List (DiagonalOperator ι) → DiagonalOperator ι
  | [] => identity
  | a :: as => a.thenOp (composeAll as)

theorem toAffine_composeAll (ops : List (DiagonalOperator ι)) :
    (composeAll ops).toAffine = AffineOperator.composeAll (ops.map toAffine) := by
  induction ops with
  | nil => exact toAffine_identity
  | cons a as ih => simp [composeAll, toAffine_thenOp, ih, AffineOperator.composeAll]

end DiagonalOperator

/-! Slot support concerns injections, not which states influence routing or reads. -/

variable {B : ℕ} {F : Type*} [Fintype F]

/-- One-hot slot outer product `e_j ⊗ delta`. -/
def slotInjection (j : Fin B) (delta : F → ℝ) : (Fin B × F) → ℝ :=
  fun q => if q.1 = j then delta q.2 else 0

omit [Fintype F] in
theorem slotInjection_eq_onehot (j : Fin B) (delta : F → ℝ) :
    slotInjection j delta = fun q => (if q.1 = j then 1 else 0) * delta q.2 := by
  funext q
  by_cases h : q.1 = j <;> simp [slotInjection, h]

/-- Exact (not merely potential) slot support. -/
noncomputable def slotSupport (v : (Fin B × F) → ℝ) : Finset (Fin B) :=
  Finset.univ.filter (fun i => ∃ f, v (i, f) ≠ 0)

theorem slotSupport_card_le (v : (Fin B × F) → ℝ) :
    (slotSupport v).card ≤ B := by
  exact (Finset.card_le_card (Finset.filter_subset _ _)).trans_eq (by simp)

theorem slotSupport_injection_subset (j : Fin B) (delta : F → ℝ) :
    slotSupport (slotInjection j delta) ⊆ {j} := by
  classical
  intro i hi
  obtain ⟨f, hf⟩ := (Finset.mem_filter.mp hi).2
  by_contra h
  have hij : i ≠ j := by simpa using h
  exact hf (by simp [slotInjection, hij])

theorem slotSupport_injection_card (j : Fin B) (delta : F → ℝ) :
    (slotSupport (slotInjection j delta)).card ≤ 1 := by
  exact (Finset.card_le_card (slotSupport_injection_subset j delta)).trans_eq (by simp)

/-- Activity-compressed reconstruction: omit every row outside a certified
activity set. Each remaining summand has one-slot / outer-product form. -/
theorem sparse_reconstruction (v : (Fin B × F) → ℝ) (active : Finset (Fin B))
    (h : slotSupport v ⊆ active) :
    v = ∑ j ∈ active, slotInjection j (fun f => v (j, f)) := by
  classical
  funext q
  rcases q with ⟨i, f⟩
  simp only [Finset.sum_apply, slotInjection]
  by_cases hi : i ∈ active
  · simp [hi]
  · have hz : v (i, f) = 0 := by
      by_contra hn
      exact hi (h (Finset.mem_filter.mpr ⟨Finset.mem_univ _, ⟨f, hn⟩⟩))
    simp [hi, hz]

/-- General slotwise decomposition of a diagonal step. -/
theorem diagonal_slot_decomposition (a : DiagonalOperator (Fin B × F))
    (s : (Fin B × F) → ℝ) :
    a.toAffine.apply s = (fun q => a.scale q * s q) +
      ∑ j ∈ slotSupport a.shift, slotInjection j (fun f => a.shift (j, f)) := by
  rw [← sparse_reconstruction a.shift (slotSupport a.shift) (fun _ h => h)]
  rfl

/-- Multiplying by a diagonal never spreads an injection into a new slot. -/
theorem diagonal_support_subset (scale v : (Fin B × F) → ℝ) :
    slotSupport (fun q => scale q * v q) ⊆ slotSupport v := by
  classical
  intro i hi
  obtain ⟨f, hf⟩ := (Finset.mem_filter.mp hi).2
  exact Finset.mem_filter.mpr ⟨Finset.mem_univ _,
    ⟨f, fun hz => hf (by simp [hz])⟩⟩

theorem support_add_subset (v w : (Fin B × F) → ℝ) :
    slotSupport (v + w) ⊆ slotSupport v ∪ slotSupport w := by
  classical
  intro i hi
  obtain ⟨f, hf⟩ := (Finset.mem_filter.mp hi).2
  by_cases hv : v (i, f) = 0
  · apply Finset.mem_union_right
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _,
      ⟨f, fun hw => hf (by simp [Pi.add_apply, hv, hw])⟩⟩
  · apply Finset.mem_union_left
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ _, ⟨f, hv⟩⟩

/-- Chunk composition grows injection activity only by union, with cancellation
and erasure allowed. -/
theorem diagonal_composition_support (a b : DiagonalOperator (Fin B × F)) :
    slotSupport (a.thenOp b).shift ⊆ slotSupport a.shift ∪ slotSupport b.shift := by
  intro i hi
  have hh := support_add_subset (fun q => b.scale q * a.shift q) b.shift hi
  rcases Finset.mem_union.mp hh with ha | hb
  · exact Finset.mem_union_left _ (diagonal_support_subset b.scale a.shift ha)
  · exact Finset.mem_union_right _ hb

end MemoryAlgebra
