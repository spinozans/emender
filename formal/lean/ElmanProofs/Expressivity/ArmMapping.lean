/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.PrecisionRegime

/-!
# R1 — Arm mapping, re-proved: our recurrences instantiate Merrill et al.'s definitions

`Why Are Linear RNNs More Parallelizable?` (Merrill, Jiang, Li, Lin, Sabharwal; ICML
2026, PMLR 306:88230–88260) classifies RNN parameterizations by circuit complexity.
X22T's first pass *cited* the mapping of our two arms onto their definitions (Defs. 2, 3,
5). This file **re-proves** it, so neither their statement, nor our transcription, nor the
mapping is taken on trust. Every claim below is a Lean theorem about explicit formulas.

## What is proved here

* `lrnnStep` — their **Def. 4** LRNN head, one step `S_t = A_t S_{t-1} + b_t`.
* `gdnStep` — **our** fla-gdn / gated-delta recurrence, transcribed from the kernel
  (`fla/ops/gated_delta_rule/fused_recurrent.py:136` decay, `:146` delta, `:150` write).
* `gdnStep_eq_lrnnStep` — **it *is* a Def. 4 LRNN step**, with
  `A_t = diag (exp g)(I − β k kᵀ)` and the rank-one write `b_t = β ⊙ k ⊗ v`.
* `gdnMatrix_zero_decay`, `gdnStep_zero_decay_eq_deltaNet` — with **zero decay** the step
  matrix *is* their DeltaNet parameterization `A_t = I − β_t k_t k_tᵀ`, and the recurrence
  *is* their DeltaNet head update.
* `deltaNetMatrix_isDPLR`, `gdnMatrix_isDPLR`, `gdnMatrixPerChannel_isDPLR`,
  `diagM_isDPLR` — the step matrix is **Def. 5 DPLR** `D − k vᵀ` in the zero-decay,
  scalar-decay and per-channel-decay forms.
* `expTaylorForm`, `eval_expTaylorForm`, `expTaylor_two_ne_exp` — the nonlinear activation
  is closed-form arithmetic, *including* the Taylor-truncation reading of their datatype
  (Def. 1), and the truncation is provably not `Real.exp`.
* `negEigval_beta_range` (imported) — `allow_neg_eigval` gives `β ∈ (0,2)`.

## Divergences from their definitions — flagged, not smoothed

1. **Their Def. 4 says `b_t(x_t) ∈ K^d` (a vector); their own instances use a `d × d`
   rank-one matrix write.** DeltaNet is stated as `b_t = v_t k_tᵀ` and RWKV-7 as
   `b_t = v_t k̃_tᵀ`, both matrices. Our `gdnStep` matches the *instances*, and we
   formalize `lrnnStep` with a matrix additive term plus `lrnnStepVec` for the literal
   Def. 4 (column-constant `b`), so the discrepancy is visible in the code.
2. **Their DeltaNet has `D_t = I`; ours has `D_t = diag (exp g_t)`.** Our arm is the gated
   variant (RWKV-7-shaped), not plain DeltaNet. `gdnStep_zero_decay_eq_deltaNet` isolates
   the exact special case where the two agree.
3. **`β` range.** Their DeltaNet has `β_t ∈ (0,1)`; with `allow_neg_eigval` ours is
   `β_t = 2·σ(z) ∈ (0,2)`. DPLR-ness is *insensitive* to the range (any real `β` gives a
   DPLR matrix), so their Thm. 3 upper bound is insensitive to that range under
   paper-compatible rational-approximation semantics (exact-real transfer unproved);
   the range matters only for the parity-recognition result of Grazzi et al., which we do **not** re-prove.
4. **Per-channel decay flips the orientation of the rank-one term.** With a per-row decay
   the step matrix is `D − k (D k)ᵀ`, not `D(I − β k kᵀ)`. Both are DPLR
   (`gdnMatrixPerChannel_isDPLR`), but they are *different matrices*, so any claim about
   "the DPLR arm" must fix a decay placement. The kernel supports both
   (`USE_GK`/`USE_GV`), which the paper's single `DeltaNet` formula does not model.
5. **Exact `exp` vs Taylor-truncated `exp`.** Their datatype uses finite Taylor
   approximations (Def. 1); we use `Real.exp`. `expTaylor_two_ne_exp` proves a
   discrepancy only, not computational-class strictness or a transfer theorem. Cited
   upper bounds require paper-compatible rational-approximation semantics.
-/

namespace ArmMapping

open Function
open LinTanhExpressivity
open PrecisionRegime

/-! ## Hand-rolled linear algebra (same style as `StreamingExpressivity`) -/

/-- Matrix–matrix product. -/
def matMul {d : ℕ} (A B : Fin d → Fin d → ℝ) : Fin d → Fin d → ℝ :=
  fun i j => ∑ k, A i k * B k j

/-- Matrix–vector product. -/
def matVec {d : ℕ} (A : Fin d → Fin d → ℝ) (v : Fin d → ℝ) : Fin d → ℝ :=
  fun i => ∑ j, A i j * v j

/-- Rank-one outer product. -/
def outer {d : ℕ} (u v : Fin d → ℝ) : Fin d → Fin d → ℝ := fun i j => u i * v j

/-- Diagonal matrix. -/
def diagM {d : ℕ} (w : Fin d → ℝ) : Fin d → Fin d → ℝ := fun i j => if i = j then w i else 0

/-- Identity matrix. -/
def oneM {d : ℕ} : Fin d → Fin d → ℝ := fun i j => if i = j then 1 else 0

theorem matMul_diagM_left {d : ℕ} (w : Fin d → ℝ) (B : Fin d → Fin d → ℝ) (i j : Fin d) :
    matMul (diagM w) B i j = w i * B i j := by
  rw [matMul, Finset.sum_eq_single i]
  · rw [diagM]; simp
  · intro b _ hb; rw [diagM, if_neg (Ne.symm hb), zero_mul]
  · intro h; exact absurd (Finset.mem_univ i) h

/-- Sum of a DPLR row against a column: the algebraic core of every identity in this file. -/
theorem sum_diag_minus {d : ℕ} (c β : ℝ) (k : Fin d → ℝ) (S : Fin d → ℝ) (i : Fin d) :
    (∑ m, c * ((if i = m then (1 : ℝ) else 0) - β * k i * k m) * S m)
      = c * S i - c * β * k i * ∑ m, k m * S m := by
  have hterm : ∀ m, c * ((if i = m then (1 : ℝ) else 0) - β * k i * k m) * S m
      = c * (if i = m then (1 : ℝ) else 0) * S m - (c * β * k i) * (k m * S m) := by
    intro m; by_cases h : i = m <;> simp [h] <;> ring
  simp only [hterm, Finset.sum_sub_distrib]
  congr 1
  · rw [Finset.sum_eq_single i]
    · simp
    · intro b _ hb; simp [show ¬ i = b from fun hh => hb hh.symm]
    · intro h; exact absurd (Finset.mem_univ i) h
  · rw [Finset.mul_sum]

/-! ## Merrill et al. Def. 4: the LRNN head step -/

/-- **Merrill et al. (2026), Def. 4** — one LRNN head step `S_t = A_t S_{t-1} + b_t`,
with the additive term as a matrix (their instances' rank-one write; divergence 1). -/
def lrnnStep {d : ℕ} (A b S : Fin d → Fin d → ℝ) : Fin d → Fin d → ℝ :=
  fun i j => matVec A (fun m => S m j) i + b i j

/-- The **literal** Def. 4 form, `b_t ∈ K^d` broadcast down the columns. Kept so that the
difference from the instances is visible rather than hidden. -/
def lrnnStepVec {d : ℕ} (A : Fin d → Fin d → ℝ) (b : Fin d → ℝ) (S : Fin d → Fin d → ℝ) :
    Fin d → Fin d → ℝ :=
  fun i j => matVec A (fun m => S m j) i + b i

/-! ## Our fla-gdn recurrence, transcribed from the kernel -/

/-- **Our gated-delta / fla-gdn recurrence** (`fla/ops/gated_delta_rule/fused_recurrent.py`,
`STATE_V_FIRST=False`), with scalar per-head decay `g`:

`S ← diag (exp g) · S`, then `delta_j = β (v_j − Σ_m (exp g · S_mj) k_m)`, then
`S_ij ← exp g · S_ij + k_i · delta_j`.

Rows index the key dimension, columns the value dimension. -/
noncomputable def gdnStep {d : ℕ} (g β : ℝ) (k v : Fin d → ℝ) (S : Fin d → Fin d → ℝ) :
    Fin d → Fin d → ℝ :=
  fun i j => Real.exp g * S i j + β * k i * (v j - ∑ m, (Real.exp g * S m j) * k m)

/-- **The fla-gdn step matrix**: `diag (exp g) · (I − β k kᵀ)`. -/
noncomputable def gdnMatrix {d : ℕ} (g β : ℝ) (k : Fin d → ℝ) : Fin d → Fin d → ℝ :=
  matMul (diagM (fun _ => Real.exp g)) (fun i j => oneM i j - β * k i * k j)

/-- **The DeltaNet step matrix** (Merrill et al. §2.3): `I − β_t k_t k_tᵀ`. -/
def deltaNetMatrix {d : ℕ} (β : ℝ) (k : Fin d → ℝ) : Fin d → Fin d → ℝ :=
  fun i j => oneM i j - β * k i * k j

theorem gdnMatrix_apply {d : ℕ} (g β : ℝ) (k : Fin d → ℝ) (i m : Fin d) :
    gdnMatrix g β k i m
      = Real.exp g * ((if i = m then (1 : ℝ) else 0) - β * k i * k m) := by
  rw [gdnMatrix, matMul_diagM_left]
  simp [oneM]

/-- **Our gated-delta step *is* a Def. 4 LRNN step**, with `A_t = diag (exp g)(I − β k kᵀ)`
and the rank-one write `b_t = β ⊙ k ⊗ v`. -/
theorem gdnStep_eq_lrnnStep {d : ℕ} (g β : ℝ) (k v : Fin d → ℝ) (S : Fin d → Fin d → ℝ) :
    gdnStep g β k v S = lrnnStep (gdnMatrix g β k) (fun i j => β * (k i * v j)) S := by
  funext i j
  simp only [gdnStep, lrnnStep, matVec, gdnMatrix_apply]
  rw [sum_diag_minus]
  have hs : (∑ m, (Real.exp g * S m j) * k m) = Real.exp g * ∑ m, k m * S m j := by
    rw [Finset.mul_sum]
    exact Finset.sum_congr rfl fun m _ => by ring
  rw [hs]
  ring

/-- The step matrix is the decayed DeltaNet matrix. -/
theorem gdnMatrix_eq_decayed_deltaNet {d : ℕ} (g β : ℝ) (k : Fin d → ℝ) :
    gdnMatrix g β k = matMul (diagM (fun _ => Real.exp g)) (deltaNetMatrix β k) := rfl

/-- **Zero decay recovers DeltaNet exactly**: the step matrix is `I − β k kᵀ`. -/
theorem gdnMatrix_zero_decay {d : ℕ} (β : ℝ) (k : Fin d → ℝ) :
    gdnMatrix 0 β k = deltaNetMatrix β k := by
  funext i j
  rw [gdnMatrix, matMul_diagM_left]
  simp [deltaNetMatrix, oneM]

/-- The zero-decay gated-delta recurrence *is* the DeltaNet head update. -/
theorem gdnStep_zero_decay_eq_deltaNet {d : ℕ} (β : ℝ) (k v : Fin d → ℝ)
    (S : Fin d → Fin d → ℝ) :
    gdnStep 0 β k v S
      = lrnnStep (deltaNetMatrix β k) (fun i j => β * (k i * v j)) S := by
  rw [gdnStep_eq_lrnnStep, gdnMatrix_zero_decay]

/-! ## Def. 5 DPLR: the step matrices are diagonal-plus-rank-one -/

theorem deltaNetMatrix_isDPLR {d : ℕ} (β : ℝ) (k : Fin d → ℝ) :
    IsDPLR (deltaNetMatrix β k) :=
  ⟨fun _ => 1, fun i => β * k i, k, fun i j => by simp only [deltaNetMatrix, oneM]⟩

theorem gdnMatrix_isDPLR {d : ℕ} (g β : ℝ) (k : Fin d → ℝ) :
    IsDPLR (gdnMatrix g β k) := by
  refine ⟨fun _ => Real.exp g, fun i => β * Real.exp g * k i, k, fun i j => ?_⟩
  rw [gdnMatrix_apply]
  by_cases h : i = j
  · subst h; rw [if_pos rfl, if_pos rfl]; ring
  · rw [if_neg h, if_neg h]; ring

/-- **Per-channel decay, the other orientation** (`USE_GK`, line 141: the decay is applied
per *row* before the delta, so the rank-one term becomes `k (D k)ᵀ`). -/
noncomputable def gdnMatrixPerChannel {d : ℕ} (gk : Fin d → ℝ) (β : ℝ) (k : Fin d → ℝ) :
    Fin d → Fin d → ℝ :=
  fun i j => Real.exp (gk i) * (if i = j then 1 else 0) - β * k i * (Real.exp (gk j) * k j)

theorem gdnMatrixPerChannel_isDPLR {d : ℕ} (gk : Fin d → ℝ) (β : ℝ) (k : Fin d → ℝ) :
    IsDPLR (gdnMatrixPerChannel gk β k) :=
  ⟨fun i => Real.exp (gk i), fun i => β * k i, fun j => Real.exp (gk j) * k j,
    fun i j => by by_cases h : i = j <;> simp [gdnMatrixPerChannel, h]⟩

/-- A **diagonal** matrix is DPLR (the rank-zero case), so every diagonal gate is inside
Def. 5. "Is DPLR" is therefore a weak condition: the paper's `TC⁰` classification of
S4/Mamba uses a stronger diagonal-plus-time-invariance property, not DPLR alone. -/
theorem diagM_isDPLR {d : ℕ} (w : Fin d → ℝ) : IsDPLR (diagM w) :=
  ⟨w, 0, 0, fun i j => by simp [diagM]⟩

/-! ## Closed forms, including their Taylor-truncated datatype -/

/-- `exp` as their datatype evaluates it (Def. 1): an `N`-term Taylor truncation at the
origin, `Σ_{i<N} xⁱ/i!`, built from `+`, `*`, `/` and powers only. -/
def expTaylorForm {k : ℕ} : ℕ → ArithClosedForm k → ArithClosedForm k
  | 0, _ => ArithClosedForm.const 0
  | N + 1, t => ArithClosedForm.add (expTaylorForm N t)
      (ArithClosedForm.div (ArithClosedForm.pow t N) (ArithClosedForm.const (N.factorial : ℝ)))

/-- The truncated exponential evaluates to the Taylor polynomial. -/
theorem eval_expTaylorForm {k : ℕ} (N : ℕ) (t : ArithClosedForm k) (v : Fin k → ℝ) :
    (expTaylorForm N t).eval v = ∑ i ∈ Finset.range N, (t.eval v) ^ i / (i.factorial : ℝ) := by
  induction N with
  | zero => simp [expTaylorForm, ArithClosedForm.eval]
  | succ N ih => simp only [expTaylorForm, ArithClosedForm.eval, ih, Finset.sum_range_succ]

/-- **The exactness gap, stated rather than hidden.** Our Lean models use exact `Real.exp`;
their datatype uses a finite Taylor truncation. The two are not the same function — already
the two-term truncation `1 + x` disagrees with `exp` at `x = 1` (`2 < e`). This
proves a discrepancy only, not computational-class strictness or a transfer theorem.
The paper's upper bounds require paper-compatible rational-approximation semantics;
no exact-real transfer is established here. -/
theorem expTaylor_two_ne_exp :
    Real.exp 1 ≠ (expTaylorForm 2 (ArithClosedForm.var (0 : Fin 1))).eval (fun _ => (1 : ℝ)) := by
  have h : (2 : ℝ) < Real.exp 1 := by
    have h1 := Real.add_one_lt_exp (x := 1) one_ne_zero
    norm_num at h1 ⊢
    exact h1
  have htaylor : (expTaylorForm 2 (ArithClosedForm.var (0 : Fin 1))).eval (fun _ => (1 : ℝ)) = 2 := by
    norm_num [expTaylorForm, ArithClosedForm.eval, Finset.sum_range_succ]
  rw [htaylor]
  exact ne_of_gt h

end ArmMapping
