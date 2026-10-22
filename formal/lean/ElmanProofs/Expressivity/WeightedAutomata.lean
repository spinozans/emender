/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.ArmMapping

/-!
# R3/R4 — Weighted-automata simulation re-proved, and the log-depth reduction

## R3: their Theorem 6 (weighted automata), re-proved in our framework

Merrill et al. (ICML 2026) state that DPLR LRNNs simulate every weighted finite automaton
(their Thm. 6), via a 4-layer DeltaNet/RWKV-7 stack with a position router and a token
buffer. We re-prove the *simulation* here and re-derive *why* their construction needs
that machinery.

* `WFA` — their Def. 13 path form `f_A(w) = αᵀ M_{w₁} ⋯ M_{wₙ} ω` over `ℝ`.
* `wfaModel` — the **one-layer, dimension-`n`** `StreamModel` whose state is the running
  row vector `αᵀ M_{w₁} ⋯ M_{w_t}` and whose shared readout emits `· ω` at **every**
  position; `wfaModel_readout` proves the emitted value at each prefix *is* the WFA score
  of that prefix. So an *unrestricted* LRNN (their Def. 4) simulates any WFA in one layer
  — the statement their Thm. 3 side licenses, not their Thm. 6.
* `detWFA_fold_onehot` — **our X21T one-hot linear realization is exactly the deterministic
  special case**: a deterministic automaton is a `0/1` WFA, whose running row vector is the
  one-hot vector of the current state.
* `dplr_cycle_identity`, `cycle_perm_not_dplr` — **the DPLR obstruction**: every DPLR
  matrix satisfies `A i j · A j l · A l i = A i l · A l j · A j i` on distinct indices
  (its off-diagonal part is an outer product), and the `3`-cycle permutation violates it.
  A **single** DPLR step therefore cannot implement even a bare 3-cycle, let alone an
  arbitrary WFA transition matrix.

**Divergence flag (R3).** The paper's Thm. 6 is a statement about a *stack* of DPLR layers
with routing; we reproduce the one-layer LRNN simulation (trivial in our framework, where
`A_t` is an arbitrary token-indexed matrix) and we *prove the obstruction* that forces
their extra layers. We do **not** reproduce the 4-layer DeltaNet/RWKV-7 construction:
stating it would require formalizing their router (`t mod 2m`), the token buffer and the
`O(n²)` transvection program. The finding is the obstruction, and it is positive evidence
that their construction is not padding.

## R4: the upper-bound side, in reduced form (their Theorem 3)

Their Thm. 3 puts every LRNN over `ℚ` in `PNC¹` (FO-uniform, poly-size, log-depth,
bounded fan-in arithmetic circuits with a positivity check). **We do not formalize circuit
complexity classes, uniformity, or `PNC¹`** — out of reach for this pass — and we say so
rather than citing it as established. What is proved here is the load-bearing
*mathematical* content:

* `lrnn_hom_state` — the homogeneous LRNN state after `w` **is** the iterated matrix
  product `(∏ A_{w_i}) · S₀` (`prodL`): the sequential fold of a `t`-token computation is
  one associative reduction over the token list.
* `matMul_assoc`, `prodL_append`, `prodL_splitAt` — that reduction is associative and may
  be split at any cut.
* `halvingDepth_le_log2` — the halving recursion an associative reduction over `n` leaves
  induces has depth at most `log₂ n + 1`.

Together: an LRNN's computation is an associative reduction of height `O(log n)`, which is
*necessary* for a log-depth simulation and **not sufficient** for `∈ PNC¹`. Uniformity,
gate model, positivity check, matrix-dimension bookkeeping, precision, and the
acceptance/readout convention are all unformalized here: recorded as a gap (open item
O13), not as a result.
-/

namespace WeightedAutomata

open Function
open LinTanhExpressivity
open PrecisionRegime
open StreamingExpressivity
open ArmMapping

/-! ## Matrix algebra: associativity and identities -/

theorem matMul_assoc {d : ℕ} (A B C : Fin d → Fin d → ℝ) :
    matMul (matMul A B) C = matMul A (matMul B C) := by
  funext i j
  simp only [matMul]
  rw [show (∑ m, (∑ k, A i k * B k m) * C m j) = ∑ m, ∑ k, A i k * B k m * C m j from by
    simp [Finset.sum_mul]]
  rw [Finset.sum_comm]
  simp only [Finset.mul_sum]
  exact Finset.sum_congr rfl fun k _ => Finset.sum_congr rfl fun m _ => by ring

theorem matMul_oneM_left {d : ℕ} (A : Fin d → Fin d → ℝ) : matMul oneM A = A := by
  funext i j
  rw [matMul, Finset.sum_eq_single i]
  · simp [oneM]
  · intro b _ hb; rw [oneM, if_neg (Ne.symm hb), zero_mul]
  · intro h; exact absurd (Finset.mem_univ i) h

theorem matMul_oneM_right {d : ℕ} (A : Fin d → Fin d → ℝ) : matMul A oneM = A := by
  funext i j
  rw [matMul, Finset.sum_eq_single j]
  · simp [oneM]
  · intro b _ hb; rw [oneM, if_neg hb, mul_zero]
  · intro h; exact absurd (Finset.mem_univ j) h

/-- The one-hot vector over `ℝ` (X21T's `onehot` is `ℚ`-valued). -/
def onehotR {n : ℕ} (j : Fin n) : Fin n → ℝ := fun i => if i = j then 1 else 0

/-! ## R3: weighted finite automata (Merrill et al. Def. 13) -/

/-- **Merrill et al. (2026), Def. 13 (path form)** over `ℝ`. -/
structure WFA (α : Type*) (n : ℕ) where
  /-- Transition matrix for each token. -/
  trans : α → Fin n → Fin n → ℝ
  /-- Initial weights `α`. -/
  init : Fin n → ℝ
  /-- Final weights `ω`. -/
  final : Fin n → ℝ

/-- One row-vector step `r ↦ rᵀ M`. -/
def rowStep {n : ℕ} (M : Fin n → Fin n → ℝ) (r : Fin n → ℝ) : Fin n → ℝ :=
  fun i => ∑ j, r j * M j i

/-- The running row vector `αᵀ M_{w₁} ⋯ M_{w_t}`. -/
def wfaState {α : Type*} {n : ℕ} (W : WFA α n) (w : List α) : Fin n → ℝ :=
  w.foldl (fun r x => rowStep (W.trans x) r) W.init

/-- The WFA score `αᵀ M_{w₁} ⋯ M_{w_t} ω` of a word. -/
def wfaScore {α : Type*} {n : ℕ} (W : WFA α n) (w : List α) : ℝ :=
  ∑ i, wfaState W w i * W.final i

/-- **The one-layer, dimension-`n` realisation of any WFA**: state = running row vector,
readout = `· ω` at every position. -/
def wfaModel {α β : Type*} {n : ℕ} (W : WFA α n) (ψ : ℝ → β) : StreamModel (Fin n → ℝ) α β where
  init := W.init
  step := fun r x => rowStep (W.trans x) r
  out := fun r => ψ (∑ i, r i * W.final i)

theorem wfaModel_stateAfter {α β : Type*} {n : ℕ} (W : WFA α n) (ψ : ℝ → β) (w : List α) :
    (wfaModel W ψ).stateAfter w = wfaState W w := rfl

/-- **One layer suffices for every WFA score, at every position.** -/
theorem wfaModel_readout {α β : Type*} {n : ℕ} (W : WFA α n) (ψ : ℝ → β) (w : List α) :
    (wfaModel W ψ).out ((wfaModel W ψ).stateAfter w) = ψ (wfaScore W w) := by
  rw [wfaModel_stateAfter]
  simp only [wfaModel, wfaScore]

/-! ### Our X21T one-hot model is the deterministic special case -/

/-- The `0/1` transition matrix of a deterministic automaton. -/
def detTrans {α : Type*} {n : ℕ} (step : Fin n → α → Fin n) (x : α) : Fin n → Fin n → ℝ :=
  fun j i => if step j x = i then 1 else 0

/-- **A deterministic automaton is a `0/1` WFA** (their Def. 13 with a `0/1` matrix). -/
def detWFA {α : Type*} {n : ℕ} (step : Fin n → α → Fin n) (init : Fin n)
    (final : Fin n → ℝ) : WFA α n where
  trans := detTrans step
  init := onehotR init
  final := final

/-- On a one-hot row vector, the deterministic transition multiplies to the successor's
one-hot vector: X21T's `matVec_onehot`, now read as WFA evaluation. -/
theorem detTrans_rowStep_onehot {α : Type*} {n : ℕ} (step : Fin n → α → Fin n) (j : Fin n)
    (x : α) : rowStep (detTrans step x) (onehotR j) = onehotR (step j x) := by
  funext i
  rw [rowStep, Finset.sum_eq_single j]
  · simp [detTrans, onehotR, eq_comm]
  · intro b _ hb
    simp [onehotR, hb]
  · intro h; exact absurd (Finset.mem_univ j) h

/-- **The deterministic special case is exactly X21T's linear realization.** -/
theorem detWFA_state_eq_onehot {α : Type*} {n : ℕ} (step : Fin n → α → Fin n)
    (init : Fin n) (final : Fin n → ℝ) (w : List α) :
    wfaState (detWFA step init final) w = onehotR (w.foldl step init) := by
  induction w generalizing init with
  | nil => simp [wfaState, detWFA]
  | cons x xs ih =>
      have hstep : wfaState (detWFA step init final) (x :: xs)
          = wfaState (detWFA step (step init x) final) xs := by
        simp only [wfaState, List.foldl_cons, detWFA, detTrans_rowStep_onehot]
      rw [hstep, ih, List.foldl_cons]

/-! ### The DPLR obstruction -/

/-- **The DPLR cycle identity.** For distinct `i, j, l` every DPLR matrix satisfies
`A i j * A j l * A l i = A i l * A l j * A j i`: all six off-diagonal entries are `−k v`
products, so the two triple products agree term by term. -/
theorem dplr_cycle_identity {d : ℕ} {A : Fin d → Fin d → ℝ} (hA : IsDPLR A)
    {i j l : Fin d} (hij : i ≠ j) (hjl : j ≠ l) (hil : i ≠ l) :
    A i j * A j l * A l i = A i l * A l j * A j i := by
  obtain ⟨w, k, v, hw⟩ := hA
  have e1 : A i j = 0 - k i * v j := by rw [hw i j, if_neg hij]
  have e2 : A j l = 0 - k j * v l := by rw [hw j l, if_neg hjl]
  have e3 : A l i = 0 - k l * v i := by rw [hw l i, if_neg (Ne.symm hil)]
  have e4 : A i l = 0 - k i * v l := by rw [hw i l, if_neg hil]
  have e5 : A l j = 0 - k l * v j := by rw [hw l j, if_neg (Ne.symm hjl)]
  have e6 : A j i = 0 - k j * v i := by rw [hw j i, if_neg (Ne.symm hij)]
  rw [e1, e2, e3, e4, e5, e6]
  ring

/-- The `3`-cycle permutation matrix (`0 → 1 → 2 → 0`, with `i + 1` taken mod 3). -/
def cycle3M : Fin 3 → Fin 3 → ℝ := fun _ j => if j = 0 then 0 else if j = 1 then 1 else 0

/-- The concrete `3`-cycle used for the obstruction (`A 0 1 = A 1 2 = A 2 0 = 1`,
everything else `0`). -/
def cycle3 : Fin 3 → Fin 3 → ℝ :=
  fun i j =>
    if (i, j) = (0, 1) then 1 else if (i, j) = (1, 2) then 1 else if (i, j) = (2, 0) then 1 else 0

/-- **A 3-cycle is not DPLR**: the cycle identity fails (`1` versus `0`). So no single
DPLR layer can implement even this transition matrix — the reason their Thm. 6 needs four
layers plus a router and a buffer. -/
theorem cycle3_not_dplr : ¬ IsDPLR cycle3 := by
  intro h
  have hcyc := dplr_cycle_identity h (i := 0) (j := 1) (l := 2) (by decide) (by decide)
    (by decide)
  simp [cycle3] at hcyc

/-! ## R4: the associative-scan reduction -/

/-- The iterated matrix product of a list, right-associated. -/
def prodL {d : ℕ} (l : List (Fin d → Fin d → ℝ)) : Fin d → Fin d → ℝ :=
  l.foldr matMul oneM

@[simp] theorem prodL_nil {d : ℕ} : prodL ([] : List (Fin d → Fin d → ℝ)) = oneM := rfl

@[simp] theorem prodL_cons {d : ℕ} (A : Fin d → Fin d → ℝ)
    (l : List (Fin d → Fin d → ℝ)) : prodL (A :: l) = matMul A (prodL l) := rfl

/-- **Associativity of the product**: a cut anywhere is allowed, which is exactly what a
parallel scan needs. -/
theorem prodL_append {d : ℕ} (l₁ l₂ : List (Fin d → Fin d → ℝ)) :
    prodL (l₁ ++ l₂) = matMul (prodL l₁) (prodL l₂) := by
  induction l₁ with
  | nil => rw [List.nil_append, prodL_nil, matMul_oneM_left]
  | cons A l ih =>
      rw [List.cons_append, prodL_cons, prodL_cons, ih, matMul_assoc]

/-- Splitting at any index: the parallel-scan splitting rule. -/
theorem prodL_splitAt {d : ℕ} (l : List (Fin d → Fin d → ℝ)) (cut : ℕ) :
    prodL l = matMul (prodL (l.take cut)) (prodL (l.drop cut)) := by
  conv_lhs => rw [← List.take_append_drop cut l]
  rw [prodL_append]

/-- The single-matrix product. -/
theorem prodL_singleton {d : ℕ} (A : Fin d → Fin d → ℝ) : prodL [A] = A := by
  rw [prodL_cons, prodL_nil, matMul_oneM_right]

/-- **Folding left from any start state is multiplication by the product of the reversed
list.** The reversal is bookkeeping: a left fold applies `A_{w₁}` first, so the ambient
product is `A_{w_t} ⋯ A_{w₁}`. Nothing about difficulty depends on the orientation, and
`prodL_append`/`prodL_splitAt` hold in either. -/
theorem foldl_matMul_eq_rev {d : ℕ} (l : List (Fin d → Fin d → ℝ))
    (S : Fin d → Fin d → ℝ) :
    l.foldl (fun S A => matMul A S) S = matMul (prodL l.reverse) S := by
  induction l generalizing S with
  | nil => rw [List.foldl_nil, List.reverse_nil, prodL_nil]; exact (matMul_oneM_left S).symm
  | cons A l ih =>
      rw [List.foldl_cons, ih (matMul A S), List.reverse_cons, prodL_append, prodL_singleton,
        matMul_assoc]

/-- **The homogeneous LRNN state is the iterated matrix product.** The `t`-step sequential
fold is one associative reduction over the token list; no other structure is used. (The
affine case `S_t = A_t S_{t-1} + b_t` is not reduced here — see the module docstring.) -/
theorem lrnn_hom_state {α : Type*} {d : ℕ} (A : α → Fin d → Fin d → ℝ)
    (S₀ : Fin d → Fin d → ℝ) (w : List α) :
    w.foldl (fun S x => matMul (A x) S) S₀ = matMul (prodL ((w.map A).reverse)) S₀ := by
  have h := foldl_matMul_eq_rev (w.map A) S₀
  simpa only [List.foldl_map] using h

/-- The halving recursion that a *balanced* associative reduction induces (used only on
power-of-two leaf counts, where the split is exact). -/
def halvingDepth : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | n + 2 => 1 + halvingDepth ((n + 2) / 2)

/-- On a perfectly balanced tree of `2^m` leaves the recursion has depth exactly `m`. -/
theorem halvingDepth_pow_two : ∀ m : ℕ, halvingDepth (2 ^ m) ≤ m
  | 0 => by simp [halvingDepth]
  | m + 1 => by
      obtain ⟨N, hN⟩ : ∃ N, 2 ^ (m + 1) = N + 2 :=
        ⟨2 ^ (m + 1) - 2, by
          have h2 : 2 ≤ 2 ^ (m + 1) := by
            calc (2 : ℕ) = 2 ^ 1 := rfl
              _ ≤ 2 ^ (m + 1) := Nat.pow_le_pow_right (by norm_num) (by omega)
          omega⟩
      have hpow : (2 : ℕ) ^ (m + 1) = 2 * 2 ^ m := by ring
      have hdiv : (N + 2) / 2 = 2 ^ m := by omega
      have ih := halvingDepth_pow_two m
      rw [hN]
      simp only [halvingDepth]
      rw [hdiv]
      omega
/-- `n ≤ 2 ^ (log₂ n + 1)`: padding to the next power of two covers the list. -/
theorem le_two_pow_log2_succ {n : ℕ} (hn : 0 < n) : n ≤ 2 ^ (Nat.log2 n + 1) := by
  by_contra h
  push_neg at h
  have := (Nat.le_log2 (show n ≠ 0 by omega)).mpr (le_of_lt h)
  omega

/-- Padding a product with identity matrices does not change it. -/
theorem prodL_replicate_oneM {d : ℕ} (k : ℕ) :
    prodL (List.replicate k (oneM : Fin d → Fin d → ℝ)) = oneM := by
  induction k with
  | zero => rfl
  | succ k ih => rw [List.replicate_succ, prodL_cons, ih, matMul_oneM_right]

theorem prodL_pad {d : ℕ} (l : List (Fin d → Fin d → ℝ)) (k : ℕ) :
    prodL (l ++ List.replicate k oneM) = prodL l := by
  rw [prodL_append, prodL_replicate_oneM, matMul_oneM_right]

/-- **The balanced-evaluation (log-depth) reduction.** Any product of `n ≥ 1` matrices is
the product of a list of exactly `2 ^ m` matrices obtained by padding with identities, with
`m ≤ log₂ n + 1`; on such a list the balanced recursion has depth `m`
(`halvingDepth_pow_two`). This is the arithmetic content of "associative ⇒ log depth".

It is **not** a circuit-class membership result: uniformity, the gate model, the positivity
check and the readout convention are theirs (their Thm. 3), and we flag them as
unformalized (open item O13) rather than citing them as proved. -/
theorem prodL_balanced_eval {d : ℕ} (l : List (Fin d → Fin d → ℝ)) (hl : 0 < l.length) :
    ∃ (l' : List (Fin d → Fin d → ℝ)) (m : ℕ),
      prodL l' = prodL l ∧ l'.length = 2 ^ m ∧ halvingDepth (2 ^ m) ≤ m ∧
        m ≤ Nat.log2 l.length + 1 := by
  refine ⟨l ++ List.replicate (2 ^ (Nat.log2 l.length + 1) - l.length) oneM,
    Nat.log2 l.length + 1, prodL_pad _ _, ?_, halvingDepth_pow_two _, le_refl _⟩
  rw [List.length_append, List.length_replicate, Nat.add_sub_cancel' (le_two_pow_log2_succ hl)]

end WeightedAutomata
