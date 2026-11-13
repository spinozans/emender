import ElmanProofs.Architectures.AttentionScan

/-!
# Named attention obstructions, with fixed-schedule counterclaims distinguished

Joint normalization and a learned alpha do NOT by themselves destroy linearity
in values. Recomputing either from the scanned state can do so. Candidate
normalization does sharpen survivors under truncation. These are exact real
counterexamples, not numerical experiments.
-/

namespace MemoryAlgebra.Attention.Obstructions

open BigOperators

noncomputable section

/-- Two eligible atoms with equal raw mass and distinct owners. -/
def pairPopulation : Neighborhood Bool Bool :=
  ⟨Finset.univ, fun _ => 1, fun _ => by norm_num, id⟩

def pairRead (candidates : Finset Bool) (mode : Normalization) :
    AttentionOperator Bool Bool ℝ :=
  ⟨pairPopulation, candidates, Finset.subset_univ _, 1, by norm_num, mode⟩

theorem singleton_read (x : ℝ) :
    (pairRead {true} .retained).readout (fun _ => x) = x := by
  norm_num [AttentionOperator.readout, AttentionOperator.coefficient,
    AttentionOperator.denominator, AttentionOperator.retainedMass, pairRead, pairPopulation]

/-- R2.4's candidate denominator doubles the surviving atom, despite exact
metadata being unchanged. This refutes "truncation can never sharpen". -/
theorem truncation_sharpens_survivor :
    (pairRead Finset.univ .retained).coefficient true = 1/2 ∧
    (pairRead {true} .retained).coefficient true = 1 ∧
    (pairRead Finset.univ .retained).exactAggregates =
      (pairRead {true} .retained).exactAggregates := by
  norm_num [AttentionOperator.coefficient, AttentionOperator.denominator,
    AttentionOperator.retainedMass, pairRead, pairPopulation, Fintype.sum_bool,
    AttentionOperator.exactAggregates]

/-- Even with the same alpha and total output mass, the output norm can double. -/
theorem truncation_changes_readout_scale :
    (pairRead Finset.univ .retained).readout (fun j => if j then 2 else 0) = 1 ∧
    (pairRead {true} .retained).readout (fun j => if j then 2 else 0) = 2 := by
  norm_num [AttentionOperator.readout, AttentionOperator.coefficient,
    AttentionOperator.denominator, AttentionOperator.retainedMass,
    pairRead, pairPopulation, Fintype.sum_bool]

/-- Full-population normalization preserves survivor coefficients but loses
omitted mass. Preserving both the whole budget and survivors is impossible here. -/
theorem full_normalization_loses_omitted_mass :
    (pairRead Finset.univ .full).outputMass = 1 ∧
    (pairRead {true} .full).outputMass = 1/2 ∧
    (pairRead {true} .full).coefficient true = 1/2 := by
  norm_num [AttentionOperator.outputMass, AttentionOperator.coefficient,
    AttentionOperator.denominator, Neighborhood.mass, pairRead, pairPopulation,
    Fintype.sum_bool]

theorem full_normalization_readout :
    (pairRead Finset.univ .full).readout (fun j => if j then 2 else 0) = 1 ∧
    (pairRead {true} .full).readout (fun j => if j then 2 else 0) = 1 := by
  norm_num [AttentionOperator.readout, AttentionOperator.coefficient,
    AttentionOperator.denominator, Neighborhood.mass, pairRead, pairPopulation,
    Fintype.sum_bool]

theorem no_preserved_mass_and_survivor (survivor : ℝ)
    (unchanged : survivor = (pairRead Finset.univ .retained).coefficient true) :
    survivor ≠ (pairRead Finset.univ .retained).outputMass := by
  rw [unchanged, truncation_sharpens_survivor.1]
  norm_num [AttentionOperator.outputMass, AttentionOperator.coefficient,
    AttentionOperator.denominator, AttentionOperator.retainedMass,
    pairRead, pairPopulation, Fintype.sum_bool]

/-- Empty returned support is a no-op even for positive alpha: the converse
"no-op implies alpha=0" needs nondegeneracy hypotheses. -/
theorem positive_alpha_can_be_noop :
    (pairRead ∅ .retained).alpha = 1 ∧
    ∀ s : (Bool → ℝ) × ℝ, (pairRead ∅ .retained).step s = s := by
  constructor
  · rfl
  · intro s
    simp [AttentionOperator.step, AttentionOperator.readout, pairRead]

theorem cancellation_with_positive_alpha :
    (pairRead Finset.univ .retained).alpha = 1 ∧
    (pairRead Finset.univ .retained).readout (fun j => if j then 1 else -1) = 0 := by
  norm_num [AttentionOperator.readout, AttentionOperator.coefficient,
    AttentionOperator.denominator, AttentionOperator.retainedMass,
    pairRead, pairPopulation, Fintype.sum_bool]

theorem no_bound_without_bounded_values (M : ℝ) :
    ∃ v : Bool → ℝ, M < ‖(pairRead {true} .retained).readout v‖ := by
  refine ⟨fun _ => |M| + 1, ?_⟩
  rw [singleton_read, Real.norm_eq_abs, abs_of_pos (by positivity)]
  linarith [le_abs_self M]

/-- Generic joint normalization is linear when its weights are held fixed.
This refutes the unconditional assertion "joint normalization is not affine". -/
def fixedJointLinear {U V : Type*} [AddCommGroup V] [Module ℝ V]
    (support : Finset U) (weight : U → ℝ) : (U → V) →ₗ[ℝ] V where
  toFun := fun v => ∑ j ∈ support, (weight j / ∑ k ∈ support, weight k) • v j
  map_add' := by intros; simp [smul_add, Finset.sum_add_distrib]
  map_smul' := by
    intros c v
    simp only [Pi.smul_apply, RingHom.id_apply, Finset.smul_sum]
    exact Finset.sum_congr rfl (fun j _ => smul_comm _ _ _)

theorem fixed_joint_normalization_linear {U V : Type*} [AddCommGroup V] [Module ℝ V]
    (support : Finset U) (weight : U → ℝ) (x y : U → V) (c : ℝ) :
    fixedJointLinear support weight (c • x + y) =
      c • fixedJointLinear support weight x + fixedJointLinear support weight y := by
  rw [map_add, map_smul]

/-- Holding a learned number fixed in the schedule preserves linearity. -/
theorem fixed_learned_alpha_linear {U R V : Type*} [AddCommGroup V] [Module ℝ V]
    (a : AttentionOperator U R V) (x y : U → V) (c : ℝ) :
    a.readout (c • x + y) = c • a.readout x + a.readout y := a.readout_linear x y c

/-- Joint normalization can starve an operator even though it remains linear
in values. Independent normalization preserves each nonempty branch's budget. -/
theorem joint_normalization_starvation :
    (1 : ℝ) / (1 + 99) = 1/100 ∧ (1 : ℝ) / (1 + 99) < 1/2 ∧
    (1 : ℝ) / 1 = 1 := by norm_num

def strengthRead (x : ℝ) : ℝ := x * x / (x + 1)

/-- Literal normalized attention with raw weights (1,x) and values (0,x).
All weights and values in the later obstruction lie in [0,1]. -/
def strengthSchedule (x : ℝ) (hx : 0 ≤ x) : AttentionOperator Bool Bool ℝ where
  population :=
    ⟨Finset.univ, fun j => if j then x else 1,
      fun j => by cases j <;> simp [hx], id⟩
  candidates := Finset.univ
  eligible := Finset.Subset.refl _
  alpha := 1
  alpha_nonneg := by norm_num
  normalization := .retained

theorem strength_read_is_attention (x : ℝ) (hx : 0 ≤ x) :
    (strengthSchedule x hx).readout (fun j => if j then x else 0) = strengthRead x := by
  simp [AttentionOperator.readout, AttentionOperator.coefficient,
    AttentionOperator.denominator, AttentionOperator.retainedMass,
    strengthSchedule, strengthRead]
  ring

theorem state_dependent_weights_witness :
    strengthRead 1 + strengthRead 0 ≠ strengthRead (1/2) + strengthRead (1/2) := by
  norm_num [strengthRead]

/-- Freezing the support alone is insufficient for ATTEND_STORE when its
normalization uses strengths from the same numeric state being scanned. -/
theorem state_dependent_weights_not_affine :
    ¬ ∃ a : AffineOperator ℝ, ∀ x, 0 ≤ x → x ≤ 1 → a.apply x = strengthRead x := by
  rintro ⟨a, ha⟩
  have h := a.affine_identity (1/2) (1/2)
  rw [show (1/2 : ℝ) + 1/2 = 1 by norm_num,
    ha 1 (by norm_num) (by norm_num), ha 0 (by norm_num) (by norm_num),
    ha (1/2) (by norm_num) (by norm_num)] at h
  exact state_dependent_weights_witness h

def adaptiveAlpha (x : ℝ) : ℝ :=
  x * (pairRead {true} .retained).readout (fun _ => x)

theorem adaptive_alpha_formula (x : ℝ) : adaptiveAlpha x = x * x := by
  rw [adaptiveAlpha, singleton_read]

/-- Alpha in [0,1], recomputed from the value state, already breaks affinity.
An alpha fixed in the schedule is covered by fixed_learned_alpha_linear. -/
theorem state_dependent_alpha_not_affine :
    ¬ ∃ a : AffineOperator ℝ, ∀ x, 0 ≤ x → x ≤ 1 → a.apply x = adaptiveAlpha x := by
  rintro ⟨a, ha⟩
  have h := a.affine_identity (1/2) (1/2)
  rw [show (1/2 : ℝ) + 1/2 = 1 by norm_num,
    ha 1 (by norm_num) (by norm_num), ha 0 (by norm_num) (by norm_num),
    ha (1/2) (by norm_num) (by norm_num)] at h
  norm_num [adaptive_alpha_formula] at h

def routedSupportRead (x : ℝ) : ℝ :=
  (pairRead (if 0 ≤ x then {true} else {false}) .retained).readout
    (fun j => if j then x else -x)

theorem routed_support_formula (x : ℝ) : routedSupportRead x = |x| := by
  by_cases h : 0 ≤ x
  · simp [routedSupportRead, h, AttentionOperator.readout, AttentionOperator.coefficient,
      AttentionOperator.denominator, AttentionOperator.retainedMass,
      pairRead, pairPopulation, abs_of_nonneg h]
  · simp [routedSupportRead, h, AttentionOperator.readout, AttentionOperator.coefficient,
      AttentionOperator.denominator, AttentionOperator.retainedMass,
      pairRead, pairPopulation, abs_of_neg (lt_of_not_ge h)]

theorem state_dependent_support_not_affine :
    ¬ ∃ a : AffineOperator ℝ, ∀ x, a.apply x = routedSupportRead x := by
  rintro ⟨a, ha⟩
  have h := a.affine_identity (-1) 1
  simp only [ha, routed_support_formula] at h
  norm_num at h

/-- Readout linearity does not make a nonlinear value producer linear. -/
theorem nonlinear_value_source_not_affine :
    ¬ ∃ a : AffineOperator ℝ, ∀ x,
      a.apply x = (pairRead {true} .retained).readout (fun _ => x * x) := by
  rintro ⟨a, ha⟩
  have h := a.affine_identity 1 1
  simp only [ha, singleton_read] at h
  norm_num at h

/-- A single fixed read moves information between coordinates. It cannot be
represented by the old diagonal family on those same coordinates. -/
def readShear (s : Bool → ℝ) : Bool → ℝ :=
  fun j => if j then s true else s false + s true

theorem read_not_diagonal :
    ¬ ∃ d : DiagonalOperator Bool, ∀ s, d.toAffine.apply s = readShear s := by
  rintro ⟨d, hd⟩
  have h0 := congrFun (hd 0) false
  have h1 := congrFun (hd (fun j => if j then 1 else 0)) false
  simp [DiagonalOperator.toAffine_apply, readShear] at h0 h1
  linarith

/-- For every cap B, a sum of B+1 singleton reads cannot be represented by
B atoms on the original value coordinates. This is a support-size obstruction,
not a claim that finite-dimensional affine maps lack closure. -/
theorem bounded_support_not_closed (B : ℕ) :
    ¬ ∃ (support : Finset (Fin (B+1))) (weight : Fin (B+1) → ℝ),
      support.card ≤ B ∧ ∀ v : Fin (B+1) → ℝ,
        (∑ j ∈ support, weight j * v j) = ∑ j, v j := by
  rintro ⟨support, weight, hcard, heq⟩
  have hall : ∀ j : Fin (B+1), j ∈ support := by
    intro j
    by_contra hj
    have h := heq (fun k => if k = j then 1 else 0)
    simp [hj] at h
  have hs : support = Finset.univ := Finset.eq_univ_of_forall hall
  simp [hs] at hcard

/-- A nonnegative atomic read functional represented with at most B atoms. -/
def HasAtomicBudget {U : Type*} (B : ℕ) (f : (U → ℝ) → ℝ) : Prop :=
  ∃ (support : Finset U) (weight : U → ℝ), support.card ≤ B ∧
    (∀ j, 0 ≤ weight j) ∧ ∀ v, f v = ∑ j ∈ support, weight j * v j

theorem zero_budget_iff {U : Type*} (f : (U → ℝ) → ℝ) :
    HasAtomicBudget 0 f ↔ f = 0 := by
  constructor
  · rintro ⟨s, w, hc, _, heq⟩
    have hs : s = ∅ := Finset.card_eq_zero.mp (Nat.eq_zero_of_le_zero hc)
    funext v
    simpa [hs] using heq v
  · intro h
    refine ⟨∅, fun _ => 0, by simp, fun _ => le_rfl, ?_⟩
    simp [h]

theorem zero_budget_add_closed {U : Type*} (f g : (U → ℝ) → ℝ)
    (hf : HasAtomicBudget 0 f) (hg : HasAtomicBudget 0 g) : HasAtomicBudget 0 (f + g) := by
  rw [zero_budget_iff] at hf hg ⊢
  simp [hf, hg]

/-- Actual failure of binary closure for every positive support budget. The
zero-budget family is closed, so its exclusion is essential here. -/
theorem positive_budget_not_additively_closed (B : ℕ) (hB : 0 < B) :
    ∃ f g : (Fin (B+1) → ℝ) → ℝ,
      HasAtomicBudget B f ∧ HasAtomicBudget B g ∧ ¬ HasAtomicBudget B (f + g) := by
  let support : Finset (Fin (B+1)) := Finset.univ.erase 0
  refine ⟨fun v => ∑ j ∈ support, v j, fun v => v 0, ?_, ?_, ?_⟩
  · refine ⟨support, fun _ => 1, ?_, fun _ => by norm_num, ?_⟩
    · simp [support]
    · intro v; simp
  · refine ⟨{0}, fun _ => 1, ?_, fun _ => by norm_num, ?_⟩
    · simpa using hB
    · intro v; simp
  · rintro ⟨c, w, hc, _, heq⟩
    apply bounded_support_not_closed B
    refine ⟨c, w, hc, fun v => ?_⟩
    rw [← heq]
    exact Finset.sum_erase_add Finset.univ v (Finset.mem_univ 0)

/-- No function of the original store alone can recover arbitrary omitted
external values. Enlarging the state or supplying constant payloads is required. -/
theorem omitted_values_obstruction {S : Type*} [Nonempty S] :
    ¬ ∃ f : S → ℝ, ∀ (s : S) (v : ℝ),
      f s = (pairRead {true} .retained).readout (fun _ => v) := by
  rintro ⟨f, hf⟩
  obtain ⟨s⟩ := ‹Nonempty S›
  have h0 := hf s 0
  have h1 := hf s 1
  rw [singleton_read] at h0 h1
  linarith

end

end MemoryAlgebra.Attention.Obstructions
