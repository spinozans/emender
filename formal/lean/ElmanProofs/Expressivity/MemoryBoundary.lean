/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.OmegaStreamExpressivity

/-!
# The memory boundary: what no bounded-precision channel can do, and what a stack buys

X21T proved (`StreamingExpressivity.balanced_needs_unbounded_state`) that the
`aᵏbᵏ` discrimination `SeparatesBalanced` is inconsistent with *any* `n`-valued
state channel — activation-free and dimension-free. This file draws the two
consequences that make that a boundary statement rather than a negative result.

## (a) The native impossibility corollary

* `context_free_discrimination_impossible` — **no bounded-precision channel
  computes the context-free `aᵏbᵏ` discrimination**: for every `StreamModel` of
  either activation at any dimension, `SeparatesBalanced M yes no` with `yes ≠ no`
  implies `NoBoundedAbstraction M`, i.e. no `n`-valued quotient exists at any
  budget `n`.
* `tanh_family_context_free_impossible` — the same corollary instantiated at the
  concrete scalar `TANH` family (`tanhStreamModel`): **no activation escapes it**.

`SeparatesBalanced` demands the classification at the end of `aᵏbᵏ` for *every* `k`,
so it already ranges over indefinitely many input lengths. The Lean statement here
is the finite-word one; the `ω` reading follows because an `ω`-program's state at
position `2k` of a stream beginning `aᵏbᵏ` *is* the finite-word fold
(`StreamModel.omegaState`), so a bounded-precision `ω`-program reproducing these
verdicts at every block boundary would exhibit such an `M`. (No separate `ω`
statement is formalized.)

## (b) The positive direction: unbounded-depth stack memory

`stackModel` augments a finite control state with a **stack** (`List Unit`:
unbounded depth, one cell symbol). The step is a *differentiable-stack-style*
push/pop head driven by the input token (`stackPush` on `a`, `stackPop` on `b`),
read through the shared online readout.

* `stackModel_separates_balanced` — the augmented class **does** compute the
  discrimination: accept exactly at the end of `aᵏbᵏ`, reject `aᵏ'bᵏ` for
  `k' ≠ k`.
* `stackModel_no_finite_abstraction` — the augmented model has no bounded-precision
  channel at any budget, exactly as (a) demands; its memory is unbounded.
* `stackModel_cells_unary` — every reachable stack is `List.replicate m ()`: the
  **per-cell alphabet is one symbol** and the control phase has three values. So
  the construction's per-cell precision is trivial and *all* of the unbounded
  information is in the **depth**, not in the cell values.
* `memory_boundary_dichotomy` — the pair, side by side: bounded-precision channels
  (either activation) cannot compute `SeparatesBalanced`, while the
  stack-augmented channel does.

Scoping: this is a construction **inside the formal framework**. It is *not* a
claim about the current `HybridLadderLM`: the production model has no native stack
or tape, so nothing here says the Emender series as currently built computes
`aᵏbᵏ`. The architecture-level statement is: hosting this behaviour needs an added
differentiable stack/tape memory (soft push/pop head with a pointer, the discrete
idealization proved here); the approximation properties of the *soft* version are
not proved. What is proved is that activation is not the obstruction — memory depth
is. -/

namespace MemoryBoundary

open LinTanhExpressivity
open StreamingExpressivity

/-! ## (a) Native impossibility, consolidated -/

/-- **No bounded-precision channel computes the context-free `aᵏbᵏ`
discrimination.** `SeparatesBalanced` is inconsistent with any `n`-valued state
channel, at every dimension, for every activation, at every budget. -/
theorem context_free_discrimination_impossible {σ β : Type*} (M : StreamModel σ Bool β)
    {yes no : β} (hs : SeparatesBalanced M yes no) (hne : yes ≠ no) :
    NoBoundedAbstraction M :=
  fun n A => balanced_needs_unbounded_state M hs hne n A

/-- The `TANH` family is no exception: a scalar saturated model that separated
`aᵏbᵏ` would need unbounded state. -/
theorem tanh_family_context_free_impossible (A B : Bool → ℝ) (ψ : ℝ → β)
    {yes no : β} (hs : SeparatesBalanced (tanhStreamModel A B ψ) yes no) (hne : yes ≠ no) :
    NoBoundedAbstraction (tanhStreamModel A B ψ) :=
  context_free_discrimination_impossible _ hs hne

/-! ## (b) The stack-augmented class -/

/-- Control phase: reading the `a`-block, reading the `b`-block, or underflowed. -/
inductive StackPhase
  | pushing
  | popping
  | underflow
  deriving DecidableEq, Repr

/-- Push the single cell symbol and enter the `a`-reading phase. -/
def stackPush : StackPhase × List Unit → StackPhase × List Unit
  | (_, L) => (StackPhase.pushing, () :: L)

/-- Pop one cell; an empty stack signals underflow. -/
def stackPop : StackPhase × List Unit → StackPhase × List Unit
  | (_, []) => (StackPhase.underflow, [])
  | (_, _ :: L) => (StackPhase.popping, L)

/-- The token-driven push/pop head: `true` = `a` pushes, `false` = `b` pops. -/
def stackStep : StackPhase × List Unit → Bool → StackPhase × List Unit
  | p, true => stackPush p
  | p, false => stackPop p

/-- **The stack-augmented stream model**: three-valued control plus an
unbounded-depth unary stack, shared online readout `popping ∧ stack empty`. -/
def stackModel : StreamModel (StackPhase × List Unit) Bool Bool where
  init := (StackPhase.pushing, [])
  step := stackStep
  out := fun p => decide (p.1 = StackPhase.popping ∧ p.2 = [])

theorem fold_push (k : ℕ) (p : StackPhase × List Unit) :
    (List.replicate k true).foldl stackStep p =
      (if k = 0 then p.1 else StackPhase.pushing, List.replicate k () ++ p.2) := by
  induction k generalizing p with
  | zero => simp
  | succ k ih =>
      rw [List.replicate_succ, List.foldl_cons]
      have hstep : stackStep p true = (StackPhase.pushing, () :: p.2) := rfl
      rw [hstep, ih]
      simp [List.replicate_succ', List.append_assoc]

theorem fold_push_init (k : ℕ) :
    (List.replicate k true).foldl stackStep (StackPhase.pushing, []) =
      (StackPhase.pushing, List.replicate k ()) := by
  have h := fold_push k (StackPhase.pushing, [])
  simpa using h

/-- Pops from the underflow state stay in the underflow state. -/
theorem fold_pop_underflow (j : ℕ) :
    (List.replicate j false).foldl stackStep (StackPhase.underflow, []) =
      (StackPhase.underflow, []) := by
  induction j with
  | zero => rfl
  | succ j ih =>
      rw [List.replicate_succ, List.foldl_cons]
      have hstep : stackStep (StackPhase.underflow, []) false = (StackPhase.underflow, []) := rfl
      rw [hstep, ih]

/-- **The pop chain on a unary stack.** Popping `j` times from a stack of depth `m`
in phase `p`: depth `m - j` in phase `popping` while `j ≤ m`, and the underflow
state once `j > m`. -/
theorem pop_chain (j m : ℕ) (p : StackPhase) :
    (List.replicate j false).foldl stackStep (p, List.replicate m ()) =
      if j ≤ m then (if j = 0 then p else StackPhase.popping, List.replicate (m - j) ())
        else (StackPhase.underflow, []) := by
  induction j generalizing p m with
  | zero => simp
  | succ j ih =>
      rw [List.replicate_succ, List.foldl_cons]
      rcases m with _ | m
      · have hstep : stackStep (p, List.replicate 0 ()) false =
            (StackPhase.underflow, []) := rfl
        rw [hstep, fold_pop_underflow, if_neg (by omega : ¬ j + 1 ≤ 0)]
      · have hstep : stackStep (p, List.replicate (m + 1) ()) false =
            (StackPhase.popping, List.replicate m ()) := by
          simp [stackStep, stackPop, List.replicate_succ]
        rw [hstep, ih (p := StackPhase.popping) (m := m)]
        by_cases hjm : j ≤ m
        · rw [if_pos hjm, if_pos (by omega : j + 1 ≤ m + 1),
              if_neg (Nat.succ_ne_zero j)]
          have hcomp : (if j = 0 then StackPhase.popping else StackPhase.popping)
              = StackPhase.popping := by split <;> rfl
          rw [hcomp, show m + 1 - (j + 1) = m - j by omega]
        · rw [if_neg hjm, if_neg (by omega : ¬ j + 1 ≤ m + 1)]

/-- **The word `aᵏ' bᵏ` read by the stack model**: depth `k' - k` in the `b`-phase,
or the underflow state when there were fewer `a`s than `b`s. -/
theorem fold_blocks (k k' : ℕ) :
    (List.replicate k' true ++ List.replicate k false).foldl stackStep (StackPhase.pushing, []) =
      if k ≤ k' then (if k = 0 then StackPhase.pushing else StackPhase.popping,
          List.replicate (k' - k) ())
        else (StackPhase.underflow, []) := by
  rw [List.foldl_append, fold_push_init]
  exact pop_chain k k' StackPhase.pushing

/-- **The stack-augmented class computes the context-free discrimination**: accept
exactly at the end of `aᵏbᵏ`, reject `aᵏ'bᵏ` whenever `k' ≠ k`. -/
theorem stackModel_separates_balanced : SeparatesBalanced stackModel true false where
  hits := by
    intro k hk
    have hk0 : k ≠ 0 := by omega
    have h : ((List.replicate k true ++ List.replicate k false).foldl stackStep
        (StackPhase.pushing, [])).1 = StackPhase.popping ∧
        ((List.replicate k true ++ List.replicate k false).foldl stackStep
        (StackPhase.pushing, [])).2 = [] := by
      rw [fold_blocks k k, if_pos (le_refl k), if_neg hk0, Nat.sub_self]
      exact ⟨rfl, rfl⟩
    simp only [StreamModel.stateAfter, stackModel]
    rw [decide_eq_true_eq]
    exact h
  misses := by
    intro k k' _ _ hne
    have hk0 : k ≠ 0 := by omega
    have h : ¬ (((List.replicate k' true ++ List.replicate k false).foldl stackStep
        (StackPhase.pushing, [])).1 = StackPhase.popping ∧
        ((List.replicate k' true ++ List.replicate k false).foldl stackStep
        (StackPhase.pushing, [])).2 = []) := by
      by_cases hle : k ≤ k'
      · have hsub : k' - k ≠ 0 := by omega
        rw [fold_blocks k k', if_pos hle, if_neg hk0]
        intro hcon
        apply hsub
        have hlen := congrArg List.length hcon.2
        simpa using hlen
      · rw [fold_blocks k k', if_neg hle]
        intro hcon
        exact StackPhase.noConfusion hcon.1
    simp only [StreamModel.stateAfter, stackModel]
    rw [decide_eq_false_iff_not]
    exact h

/-- **The stack model has no bounded-precision channel** — precisely because it
succeeds where (a) says no bounded-precision channel can. -/
theorem stackModel_no_finite_abstraction : NoBoundedAbstraction stackModel :=
  context_free_discrimination_impossible stackModel stackModel_separates_balanced (by simp)

/-- **Every reachable stack is unary and every cell value is `() `**: the cell
alphabet has one symbol, so all of the unbounded information is in the *depth*. The
control phase is three-valued. -/
theorem stackModel_cells_unary (w : List Bool) :
    ∃ (p : StackPhase) (m : ℕ), stackModel.stateAfter w = (p, List.replicate m ()) := by
  have hfold : ∀ (v : List Bool) (p : StackPhase) (m : ℕ),
      ∃ (p' : StackPhase) (m' : ℕ),
        v.foldl stackStep (p, List.replicate m ()) = (p', List.replicate m' ()) := by
    intro v
    induction v with
    | nil => intro p m; exact ⟨p, m, rfl⟩
    | cons x xs ih =>
        intro p m
        rw [List.foldl_cons]
        cases x with
        | true =>
            have hstep : stackStep (p, List.replicate m ()) true =
                (StackPhase.pushing, List.replicate (m + 1) ()) := by
              simp [stackStep, stackPush, List.replicate_succ]
            rw [hstep]
            exact ih StackPhase.pushing (m + 1)
        | false =>
            rcases m with _ | m
            · have hstep : stackStep (p, List.replicate 0 ()) false =
                  (StackPhase.underflow, []) := rfl
              rw [hstep]
              exact ih StackPhase.underflow 0
            · have hstep : stackStep (p, List.replicate (m + 1) ()) false =
                  (StackPhase.popping, List.replicate m ()) := by
                simp [stackStep, stackPop, List.replicate_succ]
              rw [hstep]
              exact ih StackPhase.popping m
  exact hfold w StackPhase.pushing 0

/-! ## The dichotomy -/

/-- **The memory boundary.** Side by side:

* every bounded-precision channel — either activation, any dimension, any budget —
  fails the context-free `aᵏbᵏ` discrimination (`SeparatesBalanced` implies
  `NoBoundedAbstraction`), and
* the unbounded-depth stack-augmented channel computes it
  (`stackModel_separates_balanced`).

So the dividing line for indefinitely-running programs is memory (unbounded
structural state), not activation and not dimension. -/
theorem memory_boundary_dichotomy :
    (∀ (σ β : Type*) (M : StreamModel σ Bool β) (yes no : β),
        SeparatesBalanced M yes no → yes ≠ no → NoBoundedAbstraction M) ∧
      SeparatesBalanced stackModel true false :=
  ⟨fun _ _ M _ _ hs hne => context_free_discrimination_impossible M hs hne,
    stackModel_separates_balanced⟩

end MemoryBoundary
