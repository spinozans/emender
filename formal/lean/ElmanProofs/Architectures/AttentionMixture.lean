import ElmanProofs.Architectures.AttentionOperator

/-! # Additive attention: independent budgets, zero ablation, and measure addition -/

namespace MemoryAlgebra.Attention

open BigOperators

noncomputable section

variable {O U R V : Type*}

def mixtureMeasure [MeasurableSpace U] (ops : Finset O)
    (a : O → AttentionOperator U R V) : MeasureTheory.Measure U :=
  ∑ o ∈ ops, (a o).measure

theorem mixture_measure_mass [MeasurableSpace U] (ops : Finset O)
    (a : O → AttentionOperator U R V) :
    mixtureMeasure ops a Set.univ = ENNReal.ofReal (∑ o ∈ ops, (a o).outputMass) := by
  simp only [mixtureMeasure, MeasureTheory.Measure.finset_sum_apply,
    AttentionOperator.measure_univ]
  exact (ENNReal.ofReal_sum_of_nonneg (fun o _ =>
    Finset.sum_nonneg (fun j _ => (a o).coefficient_nonneg j))).symm

theorem mixture_mass_budget (ops : Finset O) (a : O → AttentionOperator U R V) :
    (∑ o ∈ ops, (a o).outputMass) ≤ ∑ o ∈ ops, (a o).alpha :=
  Finset.sum_le_sum (fun o _ => (a o).outputMass_le_alpha)

theorem independent_positive_budgets (a b : AttentionOperator U R V)
    (ha : a.normalization = .retained) (hb : b.normalization = .retained)
    (hma : a.retainedMass ≠ 0) (hmb : b.retainedMass ≠ 0) :
    a.outputMass + b.outputMass = a.alpha + b.alpha := by
  rw [a.retained_normalization_mass ha hma, b.retained_normalization_mass hb hmb]

variable [AddCommGroup V] [Module ℝ V]

def mixture (ops : Finset O) (a : O → AttentionOperator U R V) (v : U → V) : V :=
  ∑ o ∈ ops, (a o).readout v

def mixtureLinear (ops : Finset O) (a : O → AttentionOperator U R V) : (U → V) →ₗ[ℝ] V :=
  ∑ o ∈ ops, (a o).readoutLinear

/-- Distinct position spaces and value fields are also permitted. For measure
addition, embed these universes in their tagged disjoint union. -/
def heterogeneousMixtureLinear (ops : Finset O) (positions owners : O → Type*)
    (a : ∀ o, AttentionOperator (positions o) (owners o) V) :
    (∀ o, positions o → V) →ₗ[ℝ] V where
  toFun := fun v => ∑ o ∈ ops, (a o).readout (v o)
  map_add' := by
    intro x y
    change (∑ o ∈ ops, (a o).readoutLinear (x o + y o)) = _
    simp only [map_add, Finset.sum_add_distrib]
    rfl
  map_smul' := by
    intro c v
    change (∑ o ∈ ops, (a o).readoutLinear (c • v o)) = _
    simp only [map_smul, Finset.smul_sum]
    rfl

theorem heterogeneous_mixture_linear (ops : Finset O) (positions owners : O → Type*)
    (a : ∀ o, AttentionOperator (positions o) (owners o) V)
    (x y : ∀ o, positions o → V) (c : ℝ) :
    heterogeneousMixtureLinear ops positions owners a (c • x + y) =
      c • heterogeneousMixtureLinear ops positions owners a x +
      heterogeneousMixtureLinear ops positions owners a y := by
  rw [map_add, map_smul]

theorem mixture_linear (ops : Finset O) (a : O → AttentionOperator U R V)
    (x y : U → V) (c : ℝ) :
    mixture ops a (c • x + y) = c • mixture ops a x + mixture ops a y := by
  simp [mixture, AttentionOperator.readout_linear, Finset.sum_add_distrib, Finset.smul_sum]

theorem mixture_disjoint_union [DecidableEq O] (ops more : Finset O) (h : Disjoint ops more)
    (a : O → AttentionOperator U R V) (v : U → V) :
    mixture (ops ∪ more) a v = mixture ops a v + mixture more a v := by
  classical
  exact Finset.sum_union h

/-- Removing an alpha-zero branch preserves the entire additive readout. -/
theorem ablation_exact [DecidableEq O] (ops : Finset O) (a : O → AttentionOperator U R V)
    (o : O) (h : (a o).alpha = 0) (v : U → V) :
    mixture (ops.erase o) a v = mixture ops a v := by
  classical
  by_cases ho : o ∈ ops
  · have hh := Finset.sum_erase_add ops (fun p => (a p).readout v) ho
    simpa [mixture, (a o).alpha_zero_readout h] using hh
  · simp [Finset.erase_eq_of_notMem ho]

/-- Read-only additions commute on a common snapshot. Interleaved writes need
not commute; the combined scan preserves chronological order. -/
theorem attention_addition_commutes (a b : AttentionOperator U R V) (s : (U → V) × V) :
    b.step (a.step s) = a.step (b.step s) := by
  simp only [AttentionOperator.step]
  congr 1
  abel

theorem attention_product_readout (a b : AttentionOperator U R V) (s : (U → V) × V) :
    (a.embed * b.embed).apply s = (s.1, s.2 + (a.readout s.1 + b.readout s.1)) := by
  rw [AffineOperator.apply_mul, ← a.step_eq_embed, ← b.step_eq_embed]
  simp [AttentionOperator.step, add_assoc]

end

section NormBounds

variable {O U R V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem mixture_norm_budget (ops : Finset O) (a : O → AttentionOperator U R V)
    (v : U → V) (B : ℝ) (hB : 0 ≤ B)
    (hv : ∀ o ∈ ops, ∀ j ∈ (a o).candidates, ‖v j‖ ≤ B) :
    ‖mixture ops a v‖ ≤ (∑ o ∈ ops, (a o).alpha) * B := by
  calc
    _ ≤ ∑ o ∈ ops, ‖(a o).readout v‖ := norm_sum_le _ _
    _ ≤ ∑ o ∈ ops, (a o).alpha * B :=
      Finset.sum_le_sum (fun o ho => readout_norm_le (a o) v B hB (hv o ho))
    _ = _ := (Finset.sum_mul ..).symm

end NormBounds

end MemoryAlgebra.Attention
