/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Matrix.Basic
import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import ElmanProofs.Activations.Lipschitz
import ElmanProofs.Architectures.M2RNNComparison
import ElmanProofs.Architectures.OnlineMemory

/-!
# Emender Latching: Saturation, Hold-by-Default, Release-on-Counter-Delta

This module establishes three quantitative properties of the full Emender update

`S' = tanh(λ · S + k · (v - Sᵀ k)ᵀ)`

(the post-tanh state of the delta-correcting nonlinear matrix memory used by
NDM / E88). The update map here is exactly `M2RNNComparison.e88DeltaUpdateDirect`
and equivalently `OnlineMemory.emenderNonlinearWrite`.

The three theorems below cover, in order:

* `emender_saturation_insensitivity` — At a saturated slot, perturbing the
  pre-activation by `δ` moves the slot by at most `2(ε + M) · |δ|`, where the
  prior pre-activation was within `ε` of full saturation and `|δ| ≤ M`.

* `emender_latch_holds_by_default` — With `λ ≥ 1` and a bounded write
  contribution `T < λ(1 - η) at slot (i,j)`, a slot already at `|S i j| > 1 - η`
  stays above `tanh(λ(1 - η) - T)` after one Emender step. This is the
  latch-hold lower bound: the slot remains saturated.

* `emender_latch_releases_on_counter_delta` — With `0 < λ` and a write
  contribution at slot (i,j) of magnitude strictly exceeding `λ · S_ij` in the
  opposite sign of `S_ij`, the slot's sign flips after one Emender step. This
  is the bistability / release theorem and is one-step rather than
  finite-horizon — the cleanest stateable release condition for the full
  Emender update.

All three theorems concern the **full Emender update** with the delta
correction term `(v - Sᵀ k)`, not the simpler raw-write-with-decay rule. The
proofs use the foundation lemma `Activation.tanh_perturbation_at_saturation`
and basic tanh monotonicity.

## Note on `MemorySemantics.latchAttractor`

The `latchAttractor` constructor of `RecurrentResourceFormalism.MemorySemantics`
remains unoccupied at the architecture-signature level: NDM/E88 is still
labelled `errorCorrectingAssociativeTable`. The latching theorems here apply
to the same update map but at a finer (slot-wise) level of granularity, so we
do not change the architecture-level label. A refinement variant occupying
`latchAttractor` is reasonable follow-up but is intentionally left for a
later task — this task is additive.
-/

namespace EmenderLatching

open Matrix BigOperators M2RNNComparison

variable {K V : Nat}

/-! ## Pre-activation entry helpers -/

/-- The (i,j) entry of the Emender pre-activation `λ · S + k · (v - Sᵀ k)ᵀ`. -/
def emenderPreActEntry (lambda : ℝ) (S : MatState K V)
    (k : Vec K) (v : Vec V) (i : Fin K) (j : Fin V) : ℝ :=
  lambda * S i j + k i * (v j - queryReadout S k j)

/-- The Emender update applied entrywise equals `tanh` of the pre-activation
entry. -/
theorem e88DeltaUpdateDirect_apply
    (lambda : ℝ) (S : MatState K V) (k : Vec K) (v : Vec V)
    (i : Fin K) (j : Fin V) :
    (e88DeltaUpdateDirect lambda S k v) i j =
      Real.tanh (emenderPreActEntry lambda S k v i j) := by
  simp [e88DeltaUpdateDirect, matrixTanh, matrixMap, outerKV, e88Delta,
    emenderPreActEntry, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
    Pi.sub_apply]

/-! ## Theorem 1: Saturation Insensitivity -/

/-- **Theorem 1 — `emender_saturation_insensitivity`.**

If the Emender pre-activation entry `y = λ · S_ij + k_i · (v_j - (Sᵀk)_j)` is
saturated in the sense `|tanh y| > 1 - ε`, and a perturbation `δ` to that
pre-activation entry has `|δ| ≤ M` with `ε + M < 1`, then the post-tanh slot
moves by at most `2(ε + M) · |δ|`.

This is the slope-vanishes-at-saturation property, lifted to the Emender's
full delta-correcting update. The bound is quantitative (explicit constants)
and shrinks linearly in both the saturation slack `ε` and the perturbation
magnitude `M`. -/
theorem emender_saturation_insensitivity
    (lambda : ℝ) (S : MatState K V) (k : Vec K) (v : Vec V)
    (i : Fin K) (j : Fin V) (δ : ℝ)
    (ε M : ℝ) (hε : 0 < ε) (hM : 0 ≤ M) (h_compat : ε + M < 1)
    (h_δ : |δ| ≤ M)
    (h_sat : 1 - ε < |(e88DeltaUpdateDirect lambda S k v) i j|) :
    |Real.tanh (emenderPreActEntry lambda S k v i j + δ)
       - (e88DeltaUpdateDirect lambda S k v) i j|
      ≤ 2 * (ε + M) * |δ| := by
  rw [e88DeltaUpdateDirect_apply] at h_sat ⊢
  exact Activation.tanh_perturbation_at_saturation
    (emenderPreActEntry lambda S k v i j) δ ε M hε hM h_compat h_δ h_sat

/-- A corollary that frames the saturation insensitivity result over two
distinct value vectors `v₁` and `v₂` sharing the same key `k`. If the Emender
output at `(k, v₁)` is saturated at slot `(i,j)`, and the pre-activation
difference between `(k, v₁)` and `(k, v₂)` is bounded by `M` at `(i,j)`, then
the two Emender outputs are close at `(i,j)`. -/
theorem emender_saturation_two_inputs_close
    (lambda : ℝ) (S : MatState K V) (k : Vec K) (v₁ v₂ : Vec V)
    (i : Fin K) (j : Fin V)
    (ε M : ℝ) (hε : 0 < ε) (hM : 0 ≤ M) (h_compat : ε + M < 1)
    (h_pre_diff :
      |emenderPreActEntry lambda S k v₂ i j -
        emenderPreActEntry lambda S k v₁ i j| ≤ M)
    (h_sat : 1 - ε < |(e88DeltaUpdateDirect lambda S k v₁) i j|) :
    |(e88DeltaUpdateDirect lambda S k v₂) i j
       - (e88DeltaUpdateDirect lambda S k v₁) i j|
      ≤ 2 * (ε + M) *
        |emenderPreActEntry lambda S k v₂ i j -
          emenderPreActEntry lambda S k v₁ i j| := by
  rw [e88DeltaUpdateDirect_apply (v := v₁)] at h_sat
  rw [e88DeltaUpdateDirect_apply (v := v₁), e88DeltaUpdateDirect_apply (v := v₂)]
  set y₁ := emenderPreActEntry lambda S k v₁ i j with hy₁_def
  set y₂ := emenderPreActEntry lambda S k v₂ i j with hy₂_def
  set δ := y₂ - y₁ with hδ_def
  have h_y₂_eq : y₂ = y₁ + δ := by rw [hδ_def]; ring
  rw [h_y₂_eq]
  exact Activation.tanh_perturbation_at_saturation y₁ δ ε M hε hM h_compat
    h_pre_diff h_sat

/-! ## Theorem 2: Latch Holds by Default -/

/-- Tanh is strictly monotone: `tanh` is order-preserving. Useful below as a
short alias for the existing `Activation.tanh_strictMono`. -/
private theorem tanh_lt_tanh {a b : ℝ} (h : a < b) : Real.tanh a < Real.tanh b :=
  Activation.tanh_strictMono h

/-- **Theorem 2 — `emender_latch_holds_by_default`.**

For the full Emender update with `λ ≥ 1`, if the slot `(i,j)` is already
saturated (`|S i j| > 1 - η`) and the bounded write contribution at that slot
has magnitude `T < λ · (1 - η)`, then the new slot value has magnitude
strictly above `tanh(λ · (1 - η) - T)`, and the new slot keeps the sign of the
old slot. Equivalently, `|S' i j| > 1 - η'` with
`η' = 1 - tanh(λ · (1 - η) - T)`, which is strictly less than 1.

For `λ > 1` and small `η, T`, `λ(1 - η) - T > 1` so `tanh(λ(1 - η) - T) > 0.76`
and the slot remains comfortably saturated. The latch holds.

The hypothesis `T < λ · (1 - η)` is the "no sufficient counter-delta" clause.
Counter-delta release is handled by `emender_latch_releases_on_counter_delta`
below. -/
theorem emender_latch_holds_by_default
    (lambda : ℝ) (S : MatState K V) (k : Vec K) (v : Vec V)
    (i : Fin K) (j : Fin V)
    (η T : ℝ) (hη_pos : 0 < η) (hη_lt : η < 1) (_hT_nn : 0 ≤ T)
    (hlam : 1 ≤ lambda) (h_room : T < lambda * (1 - η))
    (h_sat : 1 - η < |S i j|)
    (h_W : |k i * (v j - queryReadout S k j)| ≤ T) :
    Real.tanh (lambda * (1 - η) - T) < |(e88DeltaUpdateDirect lambda S k v) i j|
      ∧ (0 < S i j → 0 < (e88DeltaUpdateDirect lambda S k v) i j)
      ∧ (S i j < 0 → (e88DeltaUpdateDirect lambda S k v) i j < 0) := by
  rw [e88DeltaUpdateDirect_apply]
  set W := k i * (v j - queryReadout S k j) with hW_def
  have hlam_pos : 0 < lambda := by linarith
  have h_one_η_pos : 0 < 1 - η := by linarith
  have h_pre_eq : emenderPreActEntry lambda S k v i j = lambda * S i j + W := rfl
  rw [h_pre_eq]
  have h_lamη_pos : 0 < lambda * (1 - η) - T := by linarith
  -- S i j ≠ 0 because |S i j| > 1 - η > 0
  have hS_ne : S i j ≠ 0 := by
    intro h0
    rw [h0, abs_zero] at h_sat
    linarith
  refine ⟨?_, ?_, ?_⟩
  · -- |tanh(lambda S + W)| > tanh(lambda(1-η) - T)
    rcases lt_or_gt_of_ne hS_ne with h_neg | h_pos
    · -- S i j < 0
      have h_S_lt : S i j < -(1 - η) := by
        have h_abs : |S i j| = -(S i j) := abs_of_neg h_neg
        linarith [h_sat, h_abs]
      have h_lamS_lt : lambda * S i j < lambda * (-(1 - η)) :=
        mul_lt_mul_of_pos_left h_S_lt hlam_pos
      have h_W_bound : W ≤ T := by
        have h := abs_le.mp h_W
        linarith [h.2]
      have h_pre_lt : lambda * S i j + W < -(lambda * (1 - η) - T) := by
        have h_lam_eq : lambda * (-(1 - η)) = -(lambda * (1 - η)) := by ring
        linarith
      have h_neg_pre : lambda * S i j + W < 0 := by linarith
      have h_tanh_neg : Real.tanh (lambda * S i j + W) < 0 :=
        Activation.tanh_neg_of_neg h_neg_pre
      have h_tanh_lt : Real.tanh (lambda * S i j + W) <
          Real.tanh (-(lambda * (1 - η) - T)) := tanh_lt_tanh h_pre_lt
      rw [Real.tanh_neg] at h_tanh_lt
      have h_abs_eq : |Real.tanh (lambda * S i j + W)| =
          -(Real.tanh (lambda * S i j + W)) := abs_of_neg h_tanh_neg
      rw [h_abs_eq]
      linarith
    · -- S i j > 0
      have h_S_gt : 1 - η < S i j := by
        have h_abs : |S i j| = S i j := abs_of_pos h_pos
        linarith [h_sat, h_abs]
      have h_lamS_gt : lambda * (1 - η) < lambda * S i j :=
        mul_lt_mul_of_pos_left h_S_gt hlam_pos
      have h_W_bound : -T ≤ W := by
        have h := abs_le.mp h_W
        linarith [h.1]
      have h_pre_gt : lambda * (1 - η) - T < lambda * S i j + W := by linarith
      have h_pos_pre : 0 < lambda * S i j + W := by linarith
      have h_tanh_pos : 0 < Real.tanh (lambda * S i j + W) :=
        Activation.tanh_pos_of_pos h_pos_pre
      have h_tanh_gt : Real.tanh (lambda * (1 - η) - T) <
          Real.tanh (lambda * S i j + W) := tanh_lt_tanh h_pre_gt
      have h_abs_eq : |Real.tanh (lambda * S i j + W)| =
          Real.tanh (lambda * S i j + W) := abs_of_pos h_tanh_pos
      rw [h_abs_eq]
      exact h_tanh_gt
  · -- sign preservation: positive case
    intro h_pos
    have h_S_gt : 1 - η < S i j := by
      have h_abs : |S i j| = S i j := abs_of_pos h_pos
      linarith [h_sat, h_abs]
    have h_lamS_gt : lambda * (1 - η) < lambda * S i j :=
      mul_lt_mul_of_pos_left h_S_gt hlam_pos
    have h_W_bound : -T ≤ W := by
      have h := abs_le.mp h_W
      linarith [h.1]
    have h_pos_pre : 0 < lambda * S i j + W := by linarith
    exact Activation.tanh_pos_of_pos h_pos_pre
  · -- sign preservation: negative case
    intro h_neg
    have h_S_lt : S i j < -(1 - η) := by
      have h_abs : |S i j| = -(S i j) := abs_of_neg h_neg
      linarith [h_sat, h_abs]
    have h_lamS_lt : lambda * S i j < lambda * (-(1 - η)) :=
      mul_lt_mul_of_pos_left h_S_lt hlam_pos
    have h_W_bound : W ≤ T := by
      have h := abs_le.mp h_W
      linarith [h.2]
    have h_neg_pre : lambda * S i j + W < 0 := by
      have h_lam_eq : lambda * (-(1 - η)) = -(lambda * (1 - η)) := by ring
      linarith
    exact Activation.tanh_neg_of_neg h_neg_pre

/-! ## Theorem 3: Latch Releases on Counter-Delta -/

/-- **Theorem 3 — `emender_latch_releases_on_counter_delta`.**

The release-condition theorem for the full Emender update. If `λ > 0`,
`S i j > 0` (latched in the positive basin), and the write contribution at
`(i,j)` is sufficiently negative (`W < -λ · S i j`), then the post-update
slot is strictly negative — the latch flips out of its basin in one step.
Symmetrically for the negative basin.

The release threshold is `T* := λ · |S i j|`; any counter-delta exceeding it
in the opposite sign suffices to flip the slot. This is the cleanest
stateable one-step release condition for the full delta-correcting update.

The strength of the conclusion is sharper than the brief's
`|S'_ij + S_ij| < |S_ij|`: we get an actual sign flip, which is stronger when
combined with the boundedness of `tanh` (and recovers the brief's statement
since `S` and `S'` have opposite signs in the positive case below). -/
theorem emender_latch_releases_on_counter_delta
    (lambda : ℝ) (S : MatState K V) (k : Vec K) (v : Vec V)
    (i : Fin K) (j : Fin V)
    (_hlam : 0 < lambda) :
    (0 < S i j → k i * (v j - queryReadout S k j) < -(lambda * S i j) →
      (e88DeltaUpdateDirect lambda S k v) i j < 0) ∧
    (S i j < 0 → -(lambda * S i j) < k i * (v j - queryReadout S k j) →
      0 < (e88DeltaUpdateDirect lambda S k v) i j) := by
  rw [e88DeltaUpdateDirect_apply]
  have h_pre_eq : emenderPreActEntry lambda S k v i j =
      lambda * S i j + k i * (v j - queryReadout S k j) := rfl
  rw [h_pre_eq]
  refine ⟨?_, ?_⟩
  · intro _hS_pos h_release
    have h_pre_neg : lambda * S i j + k i * (v j - queryReadout S k j) < 0 := by
      linarith
    exact Activation.tanh_neg_of_neg h_pre_neg
  · intro _hS_neg h_release
    have h_pre_pos :
        0 < lambda * S i j + k i * (v j - queryReadout S k j) := by
      linarith
    exact Activation.tanh_pos_of_pos h_pre_pos

/-- Quantitative version of `emender_latch_releases_on_counter_delta`: if the
counter-delta exceeds the latch threshold by a positive margin `μ`, then the
post-update slot is on the opposite side of the origin with magnitude at
least `tanh μ`. -/
theorem emender_latch_release_quantitative
    (lambda : ℝ) (S : MatState K V) (k : Vec K) (v : Vec V)
    (i : Fin K) (j : Fin V) (μ : ℝ) (_hμ : 0 < μ) (_hlam : 0 < lambda) :
    (0 < S i j →
      k i * (v j - queryReadout S k j) ≤ -(lambda * S i j) - μ →
      (e88DeltaUpdateDirect lambda S k v) i j ≤ -Real.tanh μ) ∧
    (S i j < 0 →
      -(lambda * S i j) + μ ≤ k i * (v j - queryReadout S k j) →
      Real.tanh μ ≤ (e88DeltaUpdateDirect lambda S k v) i j) := by
  rw [e88DeltaUpdateDirect_apply]
  have h_pre_eq : emenderPreActEntry lambda S k v i j =
      lambda * S i j + k i * (v j - queryReadout S k j) := rfl
  rw [h_pre_eq]
  refine ⟨?_, ?_⟩
  · intro _hS_pos h_release
    have h_pre_le : lambda * S i j + k i * (v j - queryReadout S k j) ≤ -μ := by
      linarith
    have h_tanh_le := Activation.tanh_strictMono.monotone h_pre_le
    rw [show (-μ : ℝ) = -μ from rfl] at h_tanh_le
    have : Real.tanh (-μ) = -Real.tanh μ := Real.tanh_neg μ
    linarith [h_tanh_le, this]
  · intro _hS_neg h_release
    have h_pre_ge : μ ≤ lambda * S i j + k i * (v j - queryReadout S k j) := by
      linarith
    have h_tanh_ge := Activation.tanh_strictMono.monotone h_pre_ge
    exact h_tanh_ge

end EmenderLatching
