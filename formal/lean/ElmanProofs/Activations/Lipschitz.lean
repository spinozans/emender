/-
Copyright (c) 2024 Elman Ablation Ladder Project. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Elman Ablation Ladder Team
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Topology.Order.Basic

/-!
# Lipschitz Properties of Activation Functions

This file proves Lipschitz constants for common activation functions used in RNNs.
The Lipschitz constant is critical for contraction analysis.

## Main Results

* `tanh_lipschitz`: tanh is 1-Lipschitz
* `sigmoid_lipschitz`: sigmoid is 1/4-Lipschitz
* `relu_lipschitz`: ReLU is 1-Lipschitz

## Implications for RNNs

If an activation σ is L-Lipschitz and ‖R_h‖ < 1/L, then the RNN is a contraction.
For tanh (L=1), we need ‖R_h‖ < 1.
For sigmoid (L=1/4), we need ‖R_h‖ < 4.

-/

namespace Activation

open Real

/-- Bounded activation functions have bounded outputs.
    tanh = sinh/cosh ∈ (-1, 1) since cosh² - sinh² = 1 and cosh > 0.

    Proof: From cosh² - sinh² = 1 and cosh > 0, we have sinh² < cosh²,
    hence |sinh| < cosh, so |tanh| = |sinh|/cosh < 1. -/
theorem tanh_bounded (x : ℝ) : |Real.tanh x| < 1 := by
  rw [Real.tanh_eq_sinh_div_cosh]
  rw [abs_div]
  have hcosh_pos : 0 < Real.cosh x := Real.cosh_pos x
  rw [div_lt_one (abs_pos.mpr hcosh_pos.ne')]
  rw [abs_of_pos hcosh_pos]
  -- From cosh² - sinh² = 1 and cosh > 0, sinh² < cosh²
  have h := Real.cosh_sq_sub_sinh_sq x
  have hsinh_sq : Real.sinh x ^ 2 < Real.cosh x ^ 2 := by linarith [sq_nonneg (Real.cosh x)]
  -- |sinh x| < cosh x follows from sinh² < cosh²
  rw [abs_lt]
  constructor
  · -- -cosh x < sinh x
    nlinarith [sq_abs (Real.sinh x), sq_abs (Real.cosh x)]
  · -- sinh x < cosh x
    nlinarith [sq_abs (Real.sinh x), sq_abs (Real.cosh x)]

/-- tanh is differentiable everywhere (composition of exp functions). -/
theorem differentiable_tanh : Differentiable ℝ Real.tanh := by
  have h_eq : Real.tanh = Real.sinh / Real.cosh := by
    ext y
    exact Real.tanh_eq_sinh_div_cosh y
  rw [h_eq]
  exact Real.differentiable_sinh.div Real.differentiable_cosh (fun x => (Real.cosh_pos x).ne')

/-- The derivative of tanh is 1 - tanh².
    Proof: tanh = sinh/cosh, so by quotient rule:
    tanh' = (sinh' · cosh - sinh · cosh') / cosh²
          = (cosh · cosh - sinh · sinh) / cosh²
          = (cosh² - sinh²) / cosh²
          = 1 / cosh² = sech² = 1 - tanh² -/
theorem deriv_tanh (x : ℝ) : deriv Real.tanh x = 1 - (Real.tanh x)^2 := by
  have h_eq : Real.tanh = Real.sinh / Real.cosh := by
    ext y
    exact Real.tanh_eq_sinh_div_cosh y
  conv_lhs => rw [h_eq]
  conv_rhs => rw [Real.tanh_eq_sinh_div_cosh]
  have hcosh_pos : Real.cosh x ≠ 0 := (Real.cosh_pos x).ne'
  simp only [deriv_div Real.differentiable_sinh.differentiableAt
      Real.differentiable_cosh.differentiableAt hcosh_pos]
  rw [Real.deriv_sinh, Real.deriv_cosh]
  -- (cosh * cosh - sinh * sinh) / cosh² = 1 - sinh²/cosh²
  have h := Real.cosh_sq_sub_sinh_sq x
  field_simp [h]

/-- The derivative of tanh is bounded by 1.
    Proof: tanh'(x) = 1 - tanh²(x) ∈ (0, 1] since |tanh x| < 1. -/
theorem tanh_deriv_bound (x : ℝ) : |deriv Real.tanh x| ≤ 1 := by
  rw [deriv_tanh]
  have h := tanh_bounded x
  -- |tanh x| < 1 implies tanh² < 1
  have h_sq : (Real.tanh x)^2 < 1 := by
    rw [sq_lt_one_iff_abs_lt_one]
    exact h
  have h_nonneg : 0 ≤ 1 - (Real.tanh x)^2 := by linarith
  rw [abs_of_nonneg h_nonneg]
  have h_sq_nonneg : 0 ≤ (Real.tanh x)^2 := sq_nonneg _
  linarith

/-- The derivative of tanh is positive (used for strict monotonicity) -/
theorem tanh_deriv_pos (x : ℝ) : 0 < deriv Real.tanh x := by
  rw [deriv_tanh]
  have h := tanh_bounded x
  have h_sq : (Real.tanh x)^2 < 1 := by
    rw [sq_lt_one_iff_abs_lt_one]
    exact h
  linarith

/-- Tanh is strictly monotone -/
theorem tanh_strictMono : StrictMono Real.tanh :=
  strictMono_of_deriv_pos tanh_deriv_pos

/-- Tanh is injective -/
theorem tanh_injective : Function.Injective Real.tanh :=
  tanh_strictMono.injective

/-- The derivative of tanh is strictly less than 1 for x ≠ 0.
    Proof: tanh'(x) = 1 - tanh²(x). Since tanh(x) ≠ 0 when x ≠ 0 (tanh is injective
    with tanh(0) = 0), we have tanh²(x) > 0, hence tanh'(x) < 1. -/
theorem tanh_deriv_lt_one_of_ne_zero (x : ℝ) (hx : x ≠ 0) : |deriv Real.tanh x| < 1 := by
  rw [deriv_tanh]
  -- Since tanh is strictly monotone and tanh(0) = 0, tanh(x) ≠ 0 for x ≠ 0
  have h_tanh_ne : Real.tanh x ≠ 0 := by
    intro h_eq
    have h_zero : Real.tanh 0 = 0 := Real.tanh_zero
    rw [← h_zero] at h_eq
    exact hx (tanh_injective h_eq)
  -- tanh²(x) > 0 since tanh(x) ≠ 0
  have h_sq_pos : 0 < (Real.tanh x)^2 := sq_pos_of_ne_zero h_tanh_ne
  -- deriv = 1 - tanh² is positive and strictly less than 1
  have h_deriv_pos : 0 < 1 - (Real.tanh x)^2 := by
    have h := tanh_bounded x
    have h_sq : (Real.tanh x)^2 < 1 := by rw [sq_lt_one_iff_abs_lt_one]; exact h
    linarith
  rw [abs_of_pos h_deriv_pos]
  linarith

/-- tanh(x) → 1 as x → +∞.
    Proof: tanh(x) = (e^x - e^{-x})/(e^x + e^{-x}) = (1 - e^{-2x})/(1 + e^{-2x}) → 1
    as e^{-2x} → 0 when x → ∞. -/
theorem tendsto_tanh_atTop : Filter.Tendsto Real.tanh Filter.atTop (nhds 1) := by
  -- Strategy: show tanh = (1 - e^{-2x})/(1 + e^{-2x}) and use that e^{-2x} → 0
  -- First, e^{-2x} → 0 as x → ∞
  have h_exp_neg2 : Filter.Tendsto (fun x => Real.exp (-(2 * x))) Filter.atTop (nhds 0) := by
    rw [Real.tendsto_exp_comp_nhds_zero]
    -- Need: -(2*x) → -∞ as x → ∞
    have h1 : Filter.Tendsto (fun x : ℝ => 2 * x) Filter.atTop Filter.atTop :=
      Filter.Tendsto.const_mul_atTop (by norm_num : (0 : ℝ) < 2) Filter.tendsto_id
    exact Filter.tendsto_neg_atTop_atBot.comp h1
  -- (1 - e^{-2x}) → 1 and (1 + e^{-2x}) → 1
  have h_num : Filter.Tendsto (fun x => 1 - Real.exp (-(2 * x))) Filter.atTop (nhds 1) := by
    convert (tendsto_const_nhds (x := (1 : ℝ))).sub h_exp_neg2 using 1
    simp
  have h_denom : Filter.Tendsto (fun x => 1 + Real.exp (-(2 * x))) Filter.atTop (nhds 1) := by
    convert (tendsto_const_nhds (x := (1 : ℝ))).add h_exp_neg2 using 1
    simp
  -- So the ratio → 1/1 = 1
  have h_ratio : Filter.Tendsto (fun x => (1 - Real.exp (-(2 * x))) / (1 + Real.exp (-(2 * x))))
      Filter.atTop (nhds 1) := by
    convert h_num.div h_denom (by norm_num : (1 : ℝ) ≠ 0) using 1
    simp
  -- Now show tanh equals this expression
  -- Goal: tanh x = (1 - e^{-2x})/(1 + e^{-2x})
  refine h_ratio.congr (fun x => ?_)
  -- tanh = sinh/cosh where sinh = (e^x - e^{-x})/2, cosh = (e^x + e^{-x})/2
  rw [Real.tanh_eq_sinh_div_cosh, Real.sinh_eq, Real.cosh_eq]
  have h_exp_pos : 0 < Real.exp x := Real.exp_pos x
  have h_exp_neg : Real.exp (-x) = (Real.exp x)⁻¹ := Real.exp_neg x
  have hne : Real.exp x ≠ 0 := ne_of_gt h_exp_pos
  -- Compute exp(-(2*x)) in terms of exp(x)^{-2}
  have h_exp_neg_2x : Real.exp (-(2*x)) = (Real.exp x)⁻¹ * (Real.exp x)⁻¹ := by
    have h1 : -(2*x) = (-x) + (-x) := by ring
    simp only [h1, Real.exp_add, Real.exp_neg]
  have h_cosh_ne : Real.exp x + (Real.exp x)⁻¹ ≠ 0 := by
    have h2 : 0 < (Real.exp x)⁻¹ := inv_pos.mpr h_exp_pos
    linarith
  have h_denom_ne : 1 + Real.exp (-(2*x)) ≠ 0 := by
    have := Real.exp_pos (-(2*x))
    linarith
  -- Transform from (1 - e^{-2x})/(1 + e^{-2x}) to sinh/cosh
  symm
  rw [h_exp_neg]
  calc (Real.exp x - (Real.exp x)⁻¹) / 2 / ((Real.exp x + (Real.exp x)⁻¹) / 2)
      = (Real.exp x - (Real.exp x)⁻¹) / (Real.exp x + (Real.exp x)⁻¹) := by field_simp
    _ = (1 - (Real.exp x)⁻¹ * (Real.exp x)⁻¹) / (1 + (Real.exp x)⁻¹ * (Real.exp x)⁻¹) := by
        field_simp [hne, h_cosh_ne]
    _ = (1 - Real.exp (-(2*x))) / (1 + Real.exp (-(2*x))) := by rw [h_exp_neg_2x]

/-- For any ε > 0, there exists c > 0 such that |x| > c implies |tanh'(x)| < ε.
    This is the saturation property: tanh derivative vanishes at infinity.
    Proof: As |x| → ∞, |tanh(x)| → 1, so tanh²(x) → 1, hence 1 - tanh²(x) → 0. -/
theorem tanh_saturation (ε : ℝ) (hε : 0 < ε) :
    ∃ c : ℝ, 0 < c ∧ ∀ x : ℝ, c < |x| → |deriv Real.tanh x| < ε := by
  -- tanh(x) → 1 as x → +∞ and tanh(x) → -1 as x → -∞
  -- So tanh²(x) → 1 and deriv tanh x = 1 - tanh²(x) → 0
  have h_tendsto : Filter.Tendsto (fun x => 1 - (Real.tanh x)^2) Filter.atTop (nhds 0) := by
    have h1 : Filter.Tendsto Real.tanh Filter.atTop (nhds 1) := tendsto_tanh_atTop
    have h2 : Filter.Tendsto (fun x => (Real.tanh x)^2) Filter.atTop (nhds (1^2)) :=
      h1.pow 2
    simp only [one_pow] at h2
    have h3 : Filter.Tendsto (fun x => 1 - (Real.tanh x)^2) Filter.atTop (nhds (1 - 1)) :=
      tendsto_const_nhds.sub h2
    simp only [sub_self] at h3
    exact h3
  -- Get c from the ε-δ characterization of limit
  rw [Metric.tendsto_atTop] at h_tendsto
  obtain ⟨N, hN⟩ := h_tendsto ε hε
  -- We need c such that both x > c and x < -c give small derivative
  -- For x > N, we have |1 - tanh²(x) - 0| < ε
  -- For x < -N, use oddness: tanh(-x) = -tanh(x), so tanh²(-x) = tanh²(x)
  -- So we just need |x| > N
  use max N 1
  constructor
  · exact lt_max_of_lt_right one_pos
  · intro x hx
    rw [deriv_tanh]
    -- We have |x| > max N 1 ≥ N, so either x > N or x < -N
    have hxN : N < |x| := lt_of_le_of_lt (le_max_left N 1) hx
    -- Use that 1 - tanh² is even: depends only on |x|
    have h_even : (Real.tanh x)^2 = (Real.tanh |x|)^2 := by
      rcases abs_cases x with ⟨habs, _⟩ | ⟨habs, _⟩
      · rw [habs]
      · rw [habs, Real.tanh_neg, neg_sq]
    rw [h_even]
    -- Now |x| > N, so we can apply hN
    have hN_applied := hN |x| (le_of_lt hxN)
    simp only [Real.dist_eq, sub_zero] at hN_applied
    -- |1 - tanh²(|x|)| < ε
    -- Since tanh²(|x|) < 1, we have 1 - tanh²(|x|) > 0
    have h_pos : 0 < 1 - (Real.tanh |x|)^2 := by
      have h := tanh_bounded |x|
      have h_sq : (Real.tanh |x|)^2 < 1 := by rw [sq_lt_one_iff_abs_lt_one]; exact h
      linarith
    rw [abs_of_pos h_pos] at hN_applied ⊢
    exact hN_applied

/-- tanh is positive for positive inputs -/
theorem tanh_pos_of_pos {x : ℝ} (hx : 0 < x) : 0 < Real.tanh x := by
  have := tanh_strictMono hx
  rwa [Real.tanh_zero] at this

/-- tanh is negative for negative inputs -/
theorem tanh_neg_of_neg {x : ℝ} (hx : x < 0) : Real.tanh x < 0 := by
  have := tanh_strictMono hx
  rwa [Real.tanh_zero] at this

/-- When |x| ≥ δ > 0, the derivative is uniformly bounded below 1.
    Specifically, |tanh'(x)| ≤ 1 - tanh²(δ) < 1.
    This is the key to proving gradient vanishing. -/
theorem tanh_deriv_uniform_bound (δ : ℝ) (hδ : 0 < δ) :
    ∃ r : ℝ, r < 1 ∧ 0 < r ∧ ∀ x : ℝ, δ ≤ |x| → |deriv Real.tanh x| ≤ r := by
  use 1 - (Real.tanh δ)^2
  have h_tanh_δ_ne : Real.tanh δ ≠ 0 := by
    intro h_eq
    have h_zero : Real.tanh 0 = 0 := Real.tanh_zero
    rw [← h_zero] at h_eq
    have := tanh_injective h_eq
    linarith
  have h_sq_pos : 0 < (Real.tanh δ)^2 := sq_pos_of_ne_zero h_tanh_δ_ne
  have h_tanh_bnd := tanh_bounded δ
  have h_sq_lt_one : (Real.tanh δ)^2 < 1 := by rw [sq_lt_one_iff_abs_lt_one]; exact h_tanh_bnd
  constructor
  · linarith
  constructor
  · linarith
  · intro x hx
    rw [deriv_tanh]
    have h_tanh_x := tanh_bounded x
    have h_sq_x : (Real.tanh x)^2 < 1 := by rw [sq_lt_one_iff_abs_lt_one]; exact h_tanh_x
    have h_pos : 0 < 1 - (Real.tanh x)^2 := by linarith
    rw [abs_of_pos h_pos]
    -- Need: 1 - tanh²(x) ≤ 1 - tanh²(δ)
    -- i.e., tanh²(δ) ≤ tanh²(x)
    -- Since |x| ≥ δ > 0 and tanh is odd and increasing, |tanh(x)| ≥ |tanh(δ)|
    have h_mono : |Real.tanh x| ≥ |Real.tanh δ| := by
      -- tanh is odd: tanh(-y) = -tanh(y)
      -- tanh is strictly increasing
      -- So |tanh(x)| = tanh(|x|) since tanh is odd
      -- Since |x| ≥ δ > 0, we have tanh(|x|) ≥ tanh(δ)
      have h_abs_eq : |Real.tanh x| = Real.tanh |x| := by
        by_cases hx_neg : x < 0
        · have h_neg_x_pos : 0 < -x := neg_pos.mpr hx_neg
          have h_tanh_neg : Real.tanh x < 0 := tanh_neg_of_neg hx_neg
          rw [abs_of_neg hx_neg]
          rw [abs_of_neg h_tanh_neg]
          rw [Real.tanh_neg]
        · push_neg at hx_neg
          rw [abs_of_nonneg hx_neg]
          by_cases hx_zero : x = 0
          · simp [hx_zero, Real.tanh_zero]
          · have hx_pos : 0 < x := lt_of_le_of_ne hx_neg (Ne.symm hx_zero)
            rw [abs_of_pos (tanh_pos_of_pos hx_pos)]
      have h_δ_eq : |Real.tanh δ| = Real.tanh δ := abs_of_pos (tanh_pos_of_pos hδ)
      rw [h_abs_eq, h_δ_eq]
      exact tanh_strictMono.monotone hx
    -- |tanh(x)|² ≥ |tanh(δ)|²
    have h_sq_mono : (Real.tanh δ)^2 ≤ (Real.tanh x)^2 := by
      have h1 : 0 ≤ |Real.tanh δ| := abs_nonneg _
      have h2 : |Real.tanh δ| ≤ |Real.tanh x| := h_mono
      have := sq_le_sq' (by linarith) h2
      simp only [sq_abs] at this
      exact this
    linarith

/-- Product of T tanh derivatives with inputs bounded away from zero tends to zero.
    If |x_t| ≥ δ > 0 for all t, then ∏_{t=0}^{T-1} |tanh'(x_t)| ≤ r^T → 0 as T → ∞
    where r = 1 - tanh²(δ) < 1. This is why deep tanh networks have vanishing gradients. -/
theorem deep_tanh_gradient_vanishing (δ : ℝ) (hδ : 0 < δ) :
    ∃ r : ℝ, r < 1 ∧ 0 < r ∧
    ∀ (T : ℕ) (x : Fin T → ℝ), (∀ t, δ ≤ |x t|) →
    (∏ t : Fin T, |deriv Real.tanh (x t)|) ≤ r ^ T := by
  obtain ⟨r, hr_lt_one, hr_pos, hr_bound⟩ := tanh_deriv_uniform_bound δ hδ
  use r, hr_lt_one, hr_pos
  intro T x hx
  calc ∏ t : Fin T, |deriv Real.tanh (x t)|
      ≤ ∏ _t : Fin T, r := Finset.prod_le_prod
          (fun t _ => abs_nonneg _)
          (fun t _ => hr_bound (x t) (hx t))
    _ = r ^ T := by rw [Finset.prod_const, Finset.card_fin]

/-- tanh is 1-Lipschitz: |tanh(x) - tanh(y)| ≤ |x - y|.
    Proof: tanh' = 1 - tanh² ∈ (0, 1], so by MVT |tanh x - tanh y| ≤ |x - y|. -/
theorem tanh_lipschitz : LipschitzWith 1 Real.tanh := by
  apply lipschitzWith_of_nnnorm_deriv_le differentiable_tanh
  intro x
  rw [← NNReal.coe_le_coe, NNReal.coe_one, coe_nnnorm]
  exact tanh_deriv_bound x

/-- sigmoid(x) = 1 / (1 + exp(-x)). -/
noncomputable def sigmoid (x : ℝ) : ℝ := 1 / (1 + exp (-x))

/-- sigmoid is bounded in (0, 1). -/
theorem sigmoid_bounded (x : ℝ) : 0 < sigmoid x ∧ sigmoid x < 1 := by
  constructor
  · simp only [sigmoid, one_div]
    apply inv_pos.mpr
    linarith [exp_pos (-x)]
  · simp only [sigmoid, one_div]
    have h : 1 < 1 + exp (-x) := by linarith [exp_pos (-x)]
    exact inv_lt_one_of_one_lt₀ h

/-- sigmoid is continuous everywhere. -/
theorem continuous_sigmoid : Continuous sigmoid := by
  unfold sigmoid
  exact continuous_one.div (continuous_one.add (continuous_exp.comp continuous_neg))
    fun x => by linarith [exp_pos (-x)]

/-- Helper: exp ∘ (-id) is differentiable. -/
theorem differentiableAt_exp_neg (x : ℝ) : DifferentiableAt ℝ (fun y => exp (-y)) x := by
  exact Real.differentiable_exp.differentiableAt.comp x differentiable_neg.differentiableAt

/-- sigmoid is differentiable everywhere.
    Proof: sigmoid = 1 / (1 + exp(-x)) is a quotient of differentiable functions.
    The denominator 1 + exp(-x) > 1 > 0 is never zero. -/
theorem differentiable_sigmoid : Differentiable ℝ sigmoid := by
  intro x
  unfold sigmoid
  have h_denom_ne : 1 + exp (-x) ≠ 0 := by linarith [exp_pos (-x)]
  have h_num : DifferentiableAt ℝ (fun _ : ℝ => (1 : ℝ)) x := differentiableAt_const 1
  have h_exp_neg : DifferentiableAt ℝ (fun y => exp (-y)) x := differentiableAt_exp_neg x
  have h_denom : DifferentiableAt ℝ (fun y => 1 + exp (-y)) x :=
    (differentiableAt_const 1).add h_exp_neg
  exact DifferentiableAt.div h_num h_denom h_denom_ne

/-- The derivative of sigmoid equals sigmoid(x) * (1 - sigmoid(x)).
    Proof via quotient rule on 1 / (1 + exp(-x)). -/
theorem deriv_sigmoid (x : ℝ) : deriv sigmoid x = sigmoid x * (1 - sigmoid x) := by
  unfold sigmoid
  have h_denom_ne : 1 + exp (-x) ≠ 0 := by linarith [exp_pos (-x)]
  have h_denom_pos : 0 < 1 + exp (-x) := by linarith [exp_pos (-x)]
  have h_num : DifferentiableAt ℝ (fun _ : ℝ => (1 : ℝ)) x := differentiableAt_const 1
  have h_exp_neg : DifferentiableAt ℝ (fun y => exp (-y)) x := differentiableAt_exp_neg x
  have h_denom : DifferentiableAt ℝ (fun y => 1 + exp (-y)) x :=
    (differentiableAt_const 1).add h_exp_neg
  simp only [deriv_fun_div h_num h_denom h_denom_ne]
  simp only [deriv_const, zero_mul, zero_sub]
  -- deriv of 1 + exp(-y) = deriv of exp(-y) = -exp(-x)
  have h_denom_deriv : deriv (fun y => 1 + exp (-y)) x = -exp (-x) := by
    have h2 : (fun y => 1 + exp (-y)) = (fun y => (1 : ℝ) + exp (-y)) := rfl
    rw [h2, deriv_const_add]
    -- deriv of exp(-y) at x = exp(-x) * (-1) = -exp(-x)
    -- Use HasDerivAt for the chain rule
    have h_neg : HasDerivAt Neg.neg (-1 : ℝ) x := hasDerivAt_neg x
    have h_exp : HasDerivAt exp (exp (-x)) (-x) := Real.hasDerivAt_exp (-x)
    have h_comp : HasDerivAt (fun y => exp (-y)) (exp (-x) * (-1)) x :=
      h_exp.comp x h_neg
    simp only [mul_neg_one] at h_comp
    exact h_comp.deriv
  rw [h_denom_deriv]
  -- Simplify: -(1 * (-exp(-x))) / (1 + exp(-x))^2 = exp(-x) / (1 + exp(-x))^2
  field_simp
  ring

/-- The derivative of sigmoid is bounded by 1/4.
    Since sigmoid(x) ∈ (0, 1), the product sigmoid(x)(1 - sigmoid(x)) is maximized
    when sigmoid(x) = 1/2, giving maximum derivative value 1/4.

    Proof: sigmoid'(x) = sigmoid(x) · (1 - sigmoid(x)).
    For s ∈ (0,1), s(1-s) ≤ 1/4 by AM-GM: s(1-s) ≤ ((s + (1-s))/2)² = 1/4. -/
theorem sigmoid_deriv_bound (x : ℝ) : |deriv sigmoid x| ≤ 1/4 := by
  rw [deriv_sigmoid]
  have hs := sigmoid_bounded x
  set s := sigmoid x with hs_def
  have h_prod_nonneg : 0 ≤ s * (1 - s) := mul_nonneg (le_of_lt hs.1) (by linarith [hs.2])
  have h_prod_bound : s * (1 - s) ≤ 1/4 := by nlinarith [hs.1, hs.2, sq_nonneg (s - 1/2)]
  rw [abs_of_nonneg h_prod_nonneg]
  exact h_prod_bound

/-- sigmoid is 1/4-Lipschitz.
    Proof: By the mean value theorem, for any x, y with x ≠ y,
    there exists c between x and y such that
    sigmoid(x) - sigmoid(y) = sigmoid'(c) * (x - y).
    Since |sigmoid'(c)| ≤ 1/4, we get |sigmoid(x) - sigmoid(y)| ≤ 1/4 * |x - y|. -/
theorem sigmoid_lipschitz : ∀ x y : ℝ, |sigmoid x - sigmoid y| ≤ (1/4) * |x - y| := by
  intro x y
  by_cases hxy : x = y
  · simp [hxy]
  · by_cases h : x < y
    · -- Case: x < y, apply MVT
      have hab : x < y := h
      have hcont : ContinuousOn sigmoid (Set.Icc x y) :=
        continuous_sigmoid.continuousOn
      have hdiff : DifferentiableOn ℝ sigmoid (Set.Ioo x y) :=
        (differentiable_sigmoid).differentiableOn
      obtain ⟨c, _hc_mem, hc_deriv⟩ :=
        exists_deriv_eq_slope sigmoid hab hcont hdiff
      have mvt_eq : sigmoid y - sigmoid x = deriv sigmoid c * (y - x) := by
        have h_ne : y - x ≠ 0 := by linarith
        field_simp [h_ne] at hc_deriv
        field_simp
        exact hc_deriv.symm
      calc |sigmoid x - sigmoid y|
          = |-(sigmoid y - sigmoid x)| := by simp [neg_sub]
        _ = |sigmoid y - sigmoid x| := abs_neg _
        _ = |deriv sigmoid c * (y - x)| := by rw [mvt_eq]
        _ = |deriv sigmoid c| * |y - x| := abs_mul _ _
        _ ≤ (1/4) * |y - x| := mul_le_mul_of_nonneg_right (sigmoid_deriv_bound c) (abs_nonneg _)
        _ = (1/4) * |x - y| := by rw [abs_sub_comm x y]
    · -- Case: y < x (since x ≠ y and ¬(x < y))
      push_neg at h
      have hyx : y < x := Ne.lt_of_le (Ne.symm hxy) h
      have hab : y < x := hyx
      have hcont : ContinuousOn sigmoid (Set.Icc y x) :=
        continuous_sigmoid.continuousOn
      have hdiff : DifferentiableOn ℝ sigmoid (Set.Ioo y x) :=
        (differentiable_sigmoid).differentiableOn
      obtain ⟨c, _hc_mem, hc_deriv⟩ :=
        exists_deriv_eq_slope sigmoid hab hcont hdiff
      have mvt_eq : sigmoid x - sigmoid y = deriv sigmoid c * (x - y) := by
        have h_ne : x - y ≠ 0 := by linarith
        field_simp [h_ne] at hc_deriv
        field_simp
        exact hc_deriv.symm
      calc |sigmoid x - sigmoid y|
          = |deriv sigmoid c * (x - y)| := by rw [mvt_eq]
        _ = |deriv sigmoid c| * |x - y| := abs_mul _ _
        _ ≤ (1/4) * |x - y| := mul_le_mul_of_nonneg_right (sigmoid_deriv_bound c) (abs_nonneg _)

/-- ReLU is 1-Lipschitz. -/
def relu (x : ℝ) : ℝ := max 0 x

theorem relu_lipschitz : LipschitzWith 1 relu := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  simp only [relu, Real.dist_eq, NNReal.coe_one, one_mul]
  -- |max 0 x - max 0 y| ≤ |x - y|
  -- Case analysis on signs of x and y
  rcases le_or_gt x 0 with hx | hx <;> rcases le_or_gt y 0 with hy | hy
  · -- x ≤ 0, y ≤ 0: max 0 x = 0, max 0 y = 0
    simp only [max_eq_left hx, max_eq_left hy, sub_self, abs_zero, abs_nonneg]
  · -- x ≤ 0, y > 0: max 0 x = 0, max 0 y = y
    simp only [max_eq_left hx, max_eq_right (le_of_lt hy)]
    rw [zero_sub, abs_neg, abs_of_pos hy]
    have : y ≤ |x - y| := by
      rw [abs_sub_comm]
      calc y = y - x + x := by ring
        _ ≤ y - x + 0 := by linarith
        _ = y - x := by ring
        _ ≤ |y - x| := le_abs_self _
    exact this
  · -- x > 0, y ≤ 0: max 0 x = x, max 0 y = 0
    simp only [max_eq_right (le_of_lt hx), max_eq_left hy, sub_zero]
    rw [abs_of_pos hx]
    calc x = x - y + y := by ring
      _ ≤ x - y + 0 := by linarith
      _ = x - y := by ring
      _ ≤ |x - y| := le_abs_self _
  · -- x > 0, y > 0: max 0 x = x, max 0 y = y
    simp only [max_eq_right (le_of_lt hx), max_eq_right (le_of_lt hy)]
    exact le_refl _

/-- SiLU(x) = x * sigmoid(x). -/
noncomputable def silu (x : ℝ) : ℝ := x * sigmoid x

/-- Monotonicity of relu. -/
theorem relu_monotone : Monotone relu := by
  intro x y hxy
  simp only [relu]
  exact max_le_max le_rfl hxy

end Activation
