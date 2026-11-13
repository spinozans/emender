import ElmanProofs.Architectures.MemoryOperator
import Mathlib.MeasureTheory.Measure.Dirac
import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Algebra.BigOperators.Field

/-!
# Scheduled attention as a finite atomic measure and a memory operator

`Neighborhood` is the complete eligible finite population, after all exclusions.
`AttentionOperator` adds a returned subset, normalization policy and fixed scale.
Exact aggregates are computed from the population, never from that subset.
The universe need not be finite: each query has finite support.

The linearity theorem varies **values only**. Eligibility, ownership, weights,
returned candidates, normalization policy/denominator, and alpha are fixed.
SELF/WORLD/STORE differ in universe and value source, not in this algebra.
-/

namespace MemoryAlgebra.Attention

open BigOperators

noncomputable section

variable {U R V : Type*}

/-- Eligible atoms and their unnormalized masses, before candidate truncation.
`owner` records the context region, corpus document, or store slot identity. -/
structure Neighborhood (U R : Type*) where
  eligible : Finset U
  weight : U → ℝ
  nonneg : ∀ j, 0 ≤ weight j
  owner : U → R

namespace Neighborhood

def mass (n : Neighborhood U R) : ℝ := ∑ j ∈ n.eligible, n.weight j

def supportCount (n : Neighborhood U R) : ℕ := n.eligible.card

def positiveSupport (n : Neighborhood U R) : Finset U :=
  n.eligible.filter (fun j => 0 < n.weight j)

def distinctOwners (n : Neighborhood U R) : Finset R := by
  classical
  exact n.eligible.image n.owner

/-- An actual measure, not merely a count or a top-k list. -/
def measure [MeasurableSpace U] (n : Neighborhood U R) : MeasureTheory.Measure U :=
  ∑ j ∈ n.eligible, ENNReal.ofReal (n.weight j) • MeasureTheory.Measure.dirac j

theorem mass_nonneg (n : Neighborhood U R) : 0 ≤ n.mass :=
  Finset.sum_nonneg (fun j _ => n.nonneg j)

theorem measure_univ [MeasurableSpace U] (n : Neighborhood U R) :
    n.measure Set.univ = ENNReal.ofReal n.mass := by
  simp only [measure, MeasureTheory.Measure.finset_sum_apply,
    MeasureTheory.Measure.smul_apply, MeasureTheory.Measure.dirac_apply_of_mem
      (Set.mem_univ _), smul_eq_mul, mul_one]
  exact (ENNReal.ofReal_sum_of_nonneg (fun j _ => n.nonneg j)).symm

theorem measure_finite [MeasurableSpace U] (n : Neighborhood U R) :
    n.measure Set.univ < ⊤ := by
  rw [measure_univ]
  exact ENNReal.ofReal_lt_top

theorem empty_aggregates (n : Neighborhood U R) (h : n.eligible = ∅) :
    n.mass = 0 ∧ n.supportCount = 0 ∧ n.distinctOwners = ∅ := by
  classical
  simp [mass, supportCount, distinctOwners, h]

end Neighborhood

inductive Normalization where
  /-- R2.4 as written: redistribute the entire budget over returned atoms. -/
  | retained
  /-- Keep each survivor's coefficient unchanged when truncating. -/
  | full
  deriving DecidableEq

/-- A completed decision, not a state-dependent search algorithm. -/
structure AttentionOperator (U R V : Type*) where
  population : Neighborhood U R
  candidates : Finset U
  eligible : candidates ⊆ population.eligible
  alpha : ℝ
  alpha_nonneg : 0 ≤ alpha
  normalization : Normalization

namespace AttentionOperator

def retainedMass (a : AttentionOperator U R V) : ℝ :=
  ∑ j ∈ a.candidates, a.population.weight j

def denominator (a : AttentionOperator U R V) : ℝ :=
  match a.normalization with
  | .retained => a.retainedMass
  | .full => a.population.mass

/-- Lean's total division gives zero at zero denominator. Theorems below
explicitly establish zero-mass and empty behavior, rather than adding noise. -/
def coefficient (a : AttentionOperator U R V) (j : U) : ℝ :=
  a.alpha * a.population.weight j / a.denominator

def outputMass (a : AttentionOperator U R V) : ℝ :=
  ∑ j ∈ a.candidates, a.coefficient j

def measure [MeasurableSpace U] (a : AttentionOperator U R V) : MeasureTheory.Measure U :=
  ∑ j ∈ a.candidates, ENNReal.ofReal (a.coefficient j) • MeasureTheory.Measure.dirac j

/-- Full aggregates travel with the answer, regardless of returned candidates. -/
def exactAggregates (a : AttentionOperator U R V) : ℝ × ℕ × ℕ × ℕ :=
  (a.population.mass, a.population.supportCount, a.population.positiveSupport.card,
    a.population.distinctOwners.card)

theorem aggregates_independent_of_truncation (a b : AttentionOperator U R V)
    (h : a.population = b.population) : a.exactAggregates = b.exactAggregates := by
  simp only [exactAggregates, h]

theorem retainedMass_nonneg (a : AttentionOperator U R V) : 0 ≤ a.retainedMass :=
  Finset.sum_nonneg (fun j _ => a.population.nonneg j)

theorem retainedMass_le_full (a : AttentionOperator U R V) :
    a.retainedMass ≤ a.population.mass :=
  Finset.sum_le_sum_of_subset_of_nonneg a.eligible (fun j _ _ => a.population.nonneg j)

theorem denominator_nonneg (a : AttentionOperator U R V) : 0 ≤ a.denominator := by
  cases h : a.normalization <;>
    simp only [denominator, h]
  · exact a.retainedMass_nonneg
  · exact a.population.mass_nonneg

theorem retainedMass_le_denominator (a : AttentionOperator U R V) :
    a.retainedMass ≤ a.denominator := by
  cases h : a.normalization <;> simp only [denominator, h]
  · exact le_rfl
  · exact a.retainedMass_le_full

theorem coefficient_nonneg (a : AttentionOperator U R V) (j : U) :
    0 ≤ a.coefficient j :=
  div_nonneg (mul_nonneg a.alpha_nonneg (a.population.nonneg j)) a.denominator_nonneg

theorem outputMass_formula (a : AttentionOperator U R V) :
    a.outputMass = a.alpha * a.retainedMass / a.denominator := by
  simp [outputMass, coefficient, retainedMass, Finset.sum_div, Finset.mul_sum]

theorem outputMass_le_alpha (a : AttentionOperator U R V) : a.outputMass ≤ a.alpha := by
  rw [outputMass_formula]
  by_cases h : a.denominator = 0
  · simpa [h] using a.alpha_nonneg
  · have hd := lt_of_le_of_ne a.denominator_nonneg (Ne.symm h)
    apply (div_le_iff₀ hd).mpr
    exact mul_le_mul_of_nonneg_left a.retainedMass_le_denominator a.alpha_nonneg

theorem retained_normalization_mass (a : AttentionOperator U R V)
    (hm : a.normalization = .retained) (h : a.retainedMass ≠ 0) :
    a.outputMass = a.alpha := by
  rw [outputMass_formula]
  simp [denominator, hm, h]

theorem zero_denominator (a : AttentionOperator U R V) (h : a.denominator = 0) :
    (∀ j, a.coefficient j = 0) ∧ a.outputMass = 0 := by
  simp [coefficient, outputMass, h]

theorem full_coefficients_truncation_invariant (a b : AttentionOperator U R V)
    (hp : a.population = b.population) (ha : a.alpha = b.alpha)
    (hna : a.normalization = .full) (hnb : b.normalization = .full) (j : U) :
    a.coefficient j = b.coefficient j := by
  simp only [coefficient, denominator, hna, hnb, hp, ha]

theorem full_truncation_mass_monotone (a b : AttentionOperator U R V)
    (hp : a.population = b.population) (ha : a.alpha = b.alpha)
    (hna : a.normalization = .full) (hnb : b.normalization = .full)
    (hc : a.candidates ⊆ b.candidates) : a.outputMass ≤ b.outputMass := by
  simp only [outputMass, a.full_coefficients_truncation_invariant b hp ha hna hnb]
  exact Finset.sum_le_sum_of_subset_of_nonneg hc (fun j _ _ => b.coefficient_nonneg j)

theorem measure_univ [MeasurableSpace U] (a : AttentionOperator U R V) :
    a.measure Set.univ = ENNReal.ofReal a.outputMass := by
  simp only [measure, MeasureTheory.Measure.finset_sum_apply,
    MeasureTheory.Measure.smul_apply, MeasureTheory.Measure.dirac_apply_of_mem
      (Set.mem_univ _), smul_eq_mul, mul_one]
  exact (ENNReal.ofReal_sum_of_nonneg (fun j _ => a.coefficient_nonneg j)).symm

theorem measure_finite [MeasurableSpace U] (a : AttentionOperator U R V) :
    a.measure Set.univ < ⊤ := by
  rw [measure_univ]
  exact ENNReal.ofReal_lt_top

theorem alpha_zero_measure [MeasurableSpace U] (a : AttentionOperator U R V)
    (h : a.alpha = 0) : a.measure = 0 := by
  simp [measure, coefficient, h]

variable [AddCommGroup V] [Module ℝ V]

/-- Integration of the value field against the returned finite atomic measure. -/
def readout (a : AttentionOperator U R V) (values : U → V) : V :=
  ∑ j ∈ a.candidates, a.coefficient j • values j

/-- Central theorem, packaged as a linear map; no assumption on the values. -/
def readoutLinear (a : AttentionOperator U R V) : (U → V) →ₗ[ℝ] V where
  toFun := a.readout
  map_add' := by intros; simp [readout, smul_add, Finset.sum_add_distrib]
  map_smul' := by
    intros c v
    simp only [readout, Pi.smul_apply, RingHom.id_apply, Finset.smul_sum]
    exact Finset.sum_congr rfl (fun j _ => smul_comm _ _ _)

theorem readout_linear (a : AttentionOperator U R V) (x y : U → V) (c : ℝ) :
    a.readout (c • x + y) = c • a.readout x + a.readout y := by
  change a.readoutLinear (c • x + y) = c • a.readoutLinear x + a.readoutLinear y
  rw [map_add, map_smul]

theorem readout_affine (a : AttentionOperator U R V) (x y : U → V) (c : ℝ) :
    a.readout (c • x + (1 - c) • y) =
      c • a.readout x + (1 - c) • a.readout y := by
  exact (a.readoutLinear : (U → V) →ₗ[ℝ] V).map_add _ _ |>.trans
    (congrArg₂ (· + ·) (map_smul a.readoutLinear c x) (map_smul a.readoutLinear (1-c) y))

/-- With the full denominator the readout decomposes into retained and omitted
terms exactly; no surviving atom is rescaled. -/
theorem full_truncation_decomposition [DecidableEq U] (a b : AttentionOperator U R V)
    (hp : a.population = b.population) (ha : a.alpha = b.alpha)
    (hna : a.normalization = .full) (hnb : b.normalization = .full)
    (hc : a.candidates ⊆ b.candidates) :
    b.readout = fun v => a.readout v +
      ∑ j ∈ b.candidates \ a.candidates, b.coefficient j • v j := by
  funext v
  simp only [readout, a.full_coefficients_truncation_invariant b hp ha hna hnb]
  rw [add_comm]
  exact (Finset.sum_sdiff hc).symm

theorem alpha_zero_readout (a : AttentionOperator U R V) (h : a.alpha = 0) (v : U → V) :
    a.readout v = 0 := by
  simp [readout, coefficient, h]

theorem empty_candidates (a : AttentionOperator U R V) (h : a.candidates = ∅)
    (v : U → V) : a.outputMass = 0 ∧ a.readout v = 0 := by
  simp [outputMass, readout, h]

theorem empty_support (a : AttentionOperator U R V) (h : a.population.eligible = ∅)
    (v : U → V) : a.population.mass = 0 ∧ a.outputMass = 0 ∧ a.readout v = 0 := by
  have hc : a.candidates = ∅ := Finset.subset_empty.mp (h ▸ a.eligible)
  exact ⟨(a.population.empty_aggregates h).1, a.empty_candidates hc v⟩

theorem zero_mass_readout (a : AttentionOperator U R V) (h : a.retainedMass = 0)
    (v : U → V) : a.outputMass = 0 ∧ a.readout v = 0 := by
  have hw : ∀ j ∈ a.candidates, a.population.weight j = 0 :=
    (Finset.sum_eq_zero_iff_of_nonneg (fun j _ => a.population.nonneg j)).mp h
  constructor
  · simp [outputMass_formula, h]
  · exact Finset.sum_eq_zero (fun j hj => by simp [coefficient, hw j hj])

/-- Lift a read into an endomorphism: keep values, add the result to a residual. -/
def step (a : AttentionOperator U R V) (s : (U → V) × V) : (U → V) × V :=
  (s.1, s.2 + a.readout s.1)

def embed (a : AttentionOperator U R V) : AffineOperator ((U → V) × V) where
  linear :=
    { toFun := a.step
      map_add' := by
        intro x y
        apply Prod.ext
        · rfl
        · change x.2 + y.2 + a.readoutLinear (x.1 + y.1) =
            (x.2 + a.readoutLinear x.1) + (y.2 + a.readoutLinear y.1)
          rw [map_add]; abel
      map_smul' := by
        intro c x
        apply Prod.ext
        · rfl
        · change c • x.2 + a.readoutLinear (c • x.1) =
            c • (x.2 + a.readoutLinear x.1)
          rw [map_smul, smul_add] }
  injection := 0

theorem step_eq_embed (a : AttentionOperator U R V) (s : (U → V) × V) :
    a.step s = a.embed.apply s := by
  simp [embed, AffineOperator.apply]

instance : MemoryOperator (AttentionOperator U R V) ((U → V) × V) where
  step := step
  embed := embed
  correct := step_eq_embed

theorem alpha_zero_noop (a : AttentionOperator U R V) (h : a.alpha = 0)
    (s : (U → V) × V) : a.step s = s := by
  simp [step, a.alpha_zero_readout h]

theorem alpha_zero_identity (a : AttentionOperator U R V) (h : a.alpha = 0) :
    a.embed = (1 : AffineOperator ((U → V) × V)) := by
  apply AffineOperator.ext
  · apply LinearMap.ext
    intro s
    exact a.alpha_zero_noop h s
  · rfl

theorem fixed_schedule_scan (ops : List (AttentionOperator U R V)) :
    ScanCompatibleRelativeTo (step (V := V)) ops :=
  memoryOperator_scanCompatible (S := (U → V) × V) ops

end AttentionOperator

namespace AttentionOperator

theorem readout_one_eq_outputMass (a : AttentionOperator U R ℝ) :
    a.readout (fun _ => 1) = a.outputMass := by
  simp [readout, outputMass]

/-- The converse to zero ablation needs positive retained mass and a value
space capable of observing it. Here scalar values suffice. -/
theorem alpha_zero_iff_all_readouts_zero (a : AttentionOperator U R ℝ)
    (hm : 0 < a.retainedMass) : a.alpha = 0 ↔ ∀ v, a.readout v = 0 := by
  constructor
  · exact a.alpha_zero_readout
  · intro h
    have hz := h (fun _ => 1)
    rw [readout_one_eq_outputMass, outputMass_formula] at hz
    have hd : a.denominator ≠ 0 :=
      ne_of_gt (lt_of_lt_of_le hm a.retainedMass_le_denominator)
    have hp : a.alpha * a.retainedMass = 0 := (div_eq_zero_iff.mp hz).resolve_right hd
    exact (mul_eq_zero.mp hp).resolve_right (ne_of_gt hm)

end AttentionOperator

section Bounds

variable [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- A truncation-independent budget, conditional on bounded values and alpha.
It does not assert equality of readouts or preservation of survivor weights. -/
theorem readout_norm_le (a : AttentionOperator U R V) (v : U → V) (B : ℝ)
    (hB : 0 ≤ B) (hv : ∀ j ∈ a.candidates, ‖v j‖ ≤ B) :
    ‖a.readout v‖ ≤ a.alpha * B := by
  calc
    ‖a.readout v‖ ≤ ∑ j ∈ a.candidates, ‖a.coefficient j • v j‖ := norm_sum_le _ _
    _ = ∑ j ∈ a.candidates, a.coefficient j * ‖v j‖ := by
      apply Finset.sum_congr rfl
      intro j _
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (a.coefficient_nonneg j)]
    _ ≤ ∑ j ∈ a.candidates, a.coefficient j * B :=
      Finset.sum_le_sum (fun j hj => mul_le_mul_of_nonneg_left (hv j hj) (a.coefficient_nonneg j))
    _ = a.outputMass * B := (Finset.sum_mul ..).symm
    _ ≤ a.alpha * B := mul_le_mul_of_nonneg_right a.outputMass_le_alpha hB

end Bounds

end

end MemoryAlgebra.Attention
