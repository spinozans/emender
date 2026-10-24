/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.MemoryBoundary

/-!
# X28T — Tape memory: the constructive counterpart of the both-impossible theorems

`count_needs_unbounded_state` and `balanced_needs_unbounded_state`
(X21T, `StreamingExpressivity.lean`) are the *both-impossible* theorems: no state
channel carrying at most `n` values — any dimension, any activation, any readout —
can report `n + 1` distinct counts or compute the `aᵏbᵏ` discrimination
`SeparatesBalanced` (consolidated in `MemoryBoundary.context_free_discrimination_impossible`).
They bound the **total** state: `BoundedAbstraction M n` constrains the whole
state space to `n` values.

This file exhibits the precise constructive counterpart: a **tape/stack memory
model** whose state is a finite bounded-precision control plus an **unbounded
slot count**, with **bounded-precision slots**. Exactly the axis the
impossibility theorems do not bound — the number of slots — is the one that
grows, and that axis alone buys both counting and the full `{aᵏbᵏ}` decision.

## The model

`TapePhase × List Bool`: a three-valued control (`pushing`, `popping`,
`garbage`) and a tape of one-bit `Bool` slots at unbounded depth. `tapeStep`:
`a` (`true`) appends one slot; `b` (`false`) consumes one slot; an `a` after the
`b`-phase began, or a `b` with no slot to consume, falls into the `garbage` sink.
The readout accepts exactly "never malformed, tape empty at the end".

## What is proved (all sorry-free)

* `fold_pushTape`, `fold_garbage`, `fold_popping_sees_a`, `fold_popChain`,
  `fold_trichotomy` — the fold engines: pushing `k` slots, the garbage sink, the
  pop chain, and the trichotomy every fold from a non-`garbage` phase lands in.
* `tapeCount_slots`, `tapeCount_exceeds_bound`, `tapeCount_twin_countModel` —
  **(a) unbounded counting**: after `aᵏ` the slot count is exactly `k`, it
  exceeds every fixed bound, and it realizes the exact count that `countModel`
  reports online — the constructive twin of `count_needs_unbounded_state`.
* `isBalancedAB`, `isBalancedAB_iff`, `tapeModel_decides_balanced`,
  `tapeModel_accepts_iff` — **(b) the full language decision**: a constructive
  decider for `{aᵏbᵏ : k ≥ 0}` and the theorem that `tapeModel` computes it on
  **all** inputs.
* `tapeModel_rejects_leading_b`, `tapeModel_rejects_count_mismatch`,
  `tapeModel_rejects_trailing_a` — explicit rejection of each malformed shape:
  `b` before any `a`, mismatched counts, and any `a` after the `b`-phase began.
* `tapePhase_card`, `tapeModel_cells_unary` — the bounded-precision side: the
  control is a three-valued finite type, and every reachable tape is unary (the
  per-slot alphabet is the single symbol `true`; all unbounded information is in
  the slot *count*).
* `tapeModel_separates_balanced`, `tapeModel_no_finite_abstraction` — the tape
  model computes `SeparatesBalanced`, so it has **no** bounded-precision total
  state channel: it lives exactly where the both-impossible theorems say a
  bounded-precision channel cannot.
* `tape_memory_pair` — the pair, side by side: bounded per-step precision in
  every component, unbounded slot count, the full `{aᵏbᵏ}` decision, and
  `NoBoundedAbstraction`.

## Scoping

* This is a construction inside the formal framework, **not** a claim about
  `HybridLadderLM` (which has no native tape or stack): it is the discrete
  idealization of the soft push/pop head that would host it. No approximation
  property of a soft tape is proved.
* Unlike X22T's `stackModel_separates_balanced` (block *discrimination* only —
  that witness also accepts `abab`, X22TR condition 6), `tapeModel` decides the
  language on all inputs; the added `garbage` control phase is what upgrades
  discrimination to recognition.
* Per-slot precision here is bounded and degenerate (a unary reachable
  alphabet); no claim is made about tapes with informative slot contents.
* The acceptance predicate and every proof in this file are constructive:
  `isBalancedAB` is a Bool-valued decision procedure, no `Classical` shortcut is
  taken for it, and the axiom audit shows at most `propext`,
  `Classical.choice`, `Quot.sound` (as used by the imported framework).
-/

namespace TapeMemory

open StreamingExpressivity
open MemoryBoundary

/-! ## The tape machine -/

/-- Control phase: reading the `a`-block (`pushing`), reading the `b`-block
(`popping`), or malformed (`garbage`: `b` before any `a`, `b` with no slot to
consume, or `a` after the `b`-phase began). A three-valued finite control. -/
inductive TapePhase
  | pushing
  | popping
  | garbage
  deriving DecidableEq, Fintype, Repr

/-- The control is a three-valued finite type: the control's per-step precision
is ⌈log₂ 3⌉ bits. -/
theorem tapePhase_card : Fintype.card TapePhase = 3 := by decide

/-- The token-driven tape head. `a` (`true`) **appends one slot** (`true`: one
bit of per-slot precision, stated here explicitly in the model); `b` (`false`)
**consumes one slot**. Malformed input falls into the `garbage` sink and stays
there. -/
def tapeStep : TapePhase × List Bool → Bool → TapePhase × List Bool
  | (TapePhase.pushing, L), true => (TapePhase.pushing, true :: L)
  | (TapePhase.pushing, []), false => (TapePhase.garbage, [])
  | (TapePhase.pushing, _ :: L), false => (TapePhase.popping, L)
  | (TapePhase.popping, []), false => (TapePhase.garbage, [])
  | (TapePhase.popping, _ :: L), false => (TapePhase.popping, L)
  | (TapePhase.popping, L), true => (TapePhase.garbage, L)
  | (TapePhase.garbage, L), _ => (TapePhase.garbage, L)

/-- **The tape-augmented stream model**: three-valued control plus an
unbounded-depth tape of one-bit slots. The readout accepts exactly "never
malformed, tape empty at the end" — `aᵏbᵏ` with `k ≥ 0`, including the empty
word (zero slots appended, zero consumed, empty at the end). -/
def tapeModel : StreamModel (TapePhase × List Bool) Bool Bool where
  init := (TapePhase.pushing, [])
  step := tapeStep
  out := fun p => decide (p.1 ≠ TapePhase.garbage ∧ p.2 = [])

/-- The language `{aᵏbᵏ : k ≥ 0}` as a word predicate, with `a = true` and
`b = false`. -/
def BalancedAB (w : List Bool) : Prop :=
  ∃ k, w = List.replicate k true ++ List.replicate k false

/-- **The constructive decision procedure for the shape `aᵏbᵏ`**: split the word
at its first `b`, require the remainder to be all `b`s, and require the two
counts to match. Bool-valued, no `Classical` shortcut. -/
def isBalancedAB (w : List Bool) : Bool :=
  let as := w.takeWhile (fun x => x)
  let bs := w.dropWhile (fun x => x)
  bs.all (fun x => !x) && decide (as.length = bs.length)

/-! ## The fold engines -/

/-- Appending one more `true` after a homogeneous pushed block is the same
as consing it before the block. -/
theorem replicate_true_cons (k : ℕ) (L : List Bool) :
    List.replicate k true ++ (true :: L) = true :: (List.replicate k true ++ L) := by
  induction k generalizing L with
  | zero => simp
  | succ k ih =>
      rw [show List.replicate (k + 1) true = true :: List.replicate k true from
        List.replicate_succ, List.cons_append]
      exact congrArg (List.cons true) (ih L)

/-- Pushing `k` slots onto any tape. -/
theorem fold_pushTape (k : ℕ) (L : List Bool) :
    (List.replicate k true).foldl tapeStep (TapePhase.pushing, L)
      = (TapePhase.pushing, List.replicate k true ++ L) := by
  induction k generalizing L with
  | zero => simp
  | succ k ih =>
      rw [show List.replicate (k + 1) true = true :: List.replicate k true from
          List.replicate_succ, List.foldl_cons]
      have hstep : tapeStep (TapePhase.pushing, L) true
          = (TapePhase.pushing, true :: L) := rfl
      rw [hstep, ih (true :: L), replicate_true_cons]
      exact congrArg (Prod.mk TapePhase.pushing) List.cons_append.symm

/-- After reading `aᵏ` from the initial configuration the machine holds exactly
`k` slots in the `pushing` phase. -/
theorem stateAfter_push (k : ℕ) :
    tapeModel.stateAfter (List.replicate k true)
      = (TapePhase.pushing, List.replicate k true) := by
  simpa [StreamModel.stateAfter, tapeModel] using fold_pushTape k []

/-- The `garbage` sink absorbs every word. -/
theorem fold_garbage (w : List Bool) (L : List Bool) :
    w.foldl tapeStep (TapePhase.garbage, L) = (TapePhase.garbage, L) := by
  induction w generalizing L with
  | nil => rfl
  | cons x xs ih =>
      rw [List.foldl_cons]
      have hstep : tapeStep (TapePhase.garbage, L) x = (TapePhase.garbage, L) := rfl
      rw [hstep]
      exact ih _

/-- Any word beginning with `a`, folded from the `popping` phase, lands in the
`garbage` sink. -/
theorem fold_popping_sees_a (tail : List Bool) (L : List Bool) :
    (true :: tail).foldl tapeStep (TapePhase.popping, L)
      = (TapePhase.garbage, L) := by
  rw [List.foldl_cons,
    show tapeStep (TapePhase.popping, L) true
        = (TapePhase.garbage, L) from by cases L <;> rfl]
  exact fold_garbage tail L

/-- Popping an empty tape underflows, from either live phase. -/
theorem step_pop_nil {p : TapePhase} (hp : p ≠ TapePhase.garbage) :
    tapeStep (p, []) false = (TapePhase.garbage, []) := by
  cases p with
  | pushing => rfl
  | popping => rfl
  | garbage => exact absurd rfl hp

/-- Popping a nonempty tape consumes one slot, from either live phase. -/
theorem step_pop_cons {p : TapePhase} (b : Bool) (L : List Bool)
    (hp : p ≠ TapePhase.garbage) :
    tapeStep (p, b :: L) false = (TapePhase.popping, L) := by
  cases p with
  | pushing => rfl
  | popping => rfl
  | garbage => exact absurd rfl hp

/-- **The pop chain.** Popping `j` times from `m` slots: exactly `m − j` slots
remain in the `popping` phase while `j ≤ m`; underflow (`j > m`) lands in the
`garbage` sink. -/
theorem fold_popChain (j m : ℕ) (p : TapePhase) (hp : p ≠ TapePhase.garbage) :
    (List.replicate j false).foldl tapeStep (p, List.replicate m true) =
      if j ≤ m then (if j = 0 then p else TapePhase.popping, List.replicate (m - j) true)
        else (TapePhase.garbage, []) := by
  induction j generalizing m p with
  | zero => simp
  | succ j ih =>
      rcases m with _ | m'
      · rw [show List.replicate (j + 1) false = false :: List.replicate j false from
            List.replicate_succ, List.foldl_cons, List.replicate_zero,
          step_pop_nil hp, fold_garbage]
        simp
      · rw [show List.replicate (j + 1) false = false :: List.replicate j false from
            List.replicate_succ, List.foldl_cons,
            show List.replicate (m' + 1) true = true :: List.replicate m' true from
              List.replicate_succ,
          step_pop_cons true (List.replicate m' true) hp, ih m' TapePhase.popping (by simp)]
        by_cases hjm : j ≤ m'
        · rw [if_pos hjm, if_pos (by omega : j + 1 ≤ m' + 1), if_neg (Nat.succ_ne_zero j)]
          have hj0 : (if j = 0 then TapePhase.popping else TapePhase.popping)
              = TapePhase.popping := by split <;> rfl
          rw [hj0, show m' + 1 - (j + 1) = m' - j from by omega]
        · rw [if_neg hjm, if_neg (by omega : ¬ j + 1 ≤ m' + 1)]

/-- **The fold trichotomy.** From any live phase with `m` slots, after any word
the machine is (i) in the `garbage` sink, (ii) holding a nonempty tape, or
(iii) exactly empty and live, on a word of the form `aⁱ b^(i+m)`. This is the
engine behind the all-inputs decision theorem. -/
theorem fold_trichotomy : ∀ (w : List Bool) (p : TapePhase) (m : ℕ),
    p ≠ TapePhase.garbage →
    (w.foldl tapeStep (p, List.replicate m true)).1 = TapePhase.garbage ∨
    (w.foldl tapeStep (p, List.replicate m true)).2 ≠ [] ∨
    ∃ i, w = List.replicate i true ++ List.replicate (i + m) false ∧
      (w.foldl tapeStep (p, List.replicate m true)).1 ≠ TapePhase.garbage ∧
      (w.foldl tapeStep (p, List.replicate m true)).2 = [] := by
  intro w
  induction w with
  | nil =>
      intro p m hp
      rcases m with _ | m
      · refine Or.inr (Or.inr ⟨0, by simp, ?_, rfl⟩)
        exact hp
      · exact Or.inr (Or.inl (by simp))
  | cons x xs ih =>
      intro p m hp
      cases x with
      | true =>
          cases p with
          | pushing =>
              have hs : (true :: xs).foldl tapeStep (TapePhase.pushing,
                    List.replicate m true)
                  = xs.foldl tapeStep (TapePhase.pushing, List.replicate (m + 1) true) := by
                rw [List.foldl_cons,
                  show tapeStep (TapePhase.pushing, List.replicate m true) true
                      = (TapePhase.pushing, List.replicate (m + 1) true) from rfl]
              rw [hs]
              rcases ih TapePhase.pushing (m + 1) (by simp) with h1 | h2 | ⟨i, hw, hph, hte⟩
              · exact Or.inl h1
              · exact Or.inr (Or.inl h2)
              · refine Or.inr (Or.inr ⟨i + 1, ?_, hph, hte⟩)
                rw [hw, show i + (m + 1) = i + m + 1 from by omega,
                  show i + 1 + m = i + m + 1 from by omega]
                simp [List.replicate_succ, List.cons_append]
          | popping =>
              rw [fold_popping_sees_a xs (List.replicate m true)]
              exact Or.inl rfl
          | garbage =>
              exact absurd rfl hp
      | false =>
          rcases m with _ | m
          · -- `b` before any slot was pushed: underflow
            have hs : (false :: xs).foldl tapeStep (p, List.replicate 0 true)
                = (TapePhase.garbage, []) := by
              rw [List.foldl_cons, List.replicate_zero, step_pop_nil hp, fold_garbage]
            rw [hs]
            exact Or.inl rfl
          · -- `b` consumes one slot and enters (stays in) the popping phase
            have hs : (false :: xs).foldl tapeStep (p, List.replicate (m + 1) true)
                = xs.foldl tapeStep (TapePhase.popping, List.replicate m true) := by
              rw [List.foldl_cons,
                show List.replicate (m + 1) true = true :: List.replicate m true from
                  List.replicate_succ,
                step_pop_cons true (List.replicate m true) hp]
            rw [hs]
            rcases ih TapePhase.popping m (by simp) with h1 | h2 | ⟨i, hw, hph, hte⟩
            · exact Or.inl h1
            · exact Or.inr (Or.inl h2)
            · rcases i with _ | i
              · refine Or.inr (Or.inr ⟨0, ?_, hph, hte⟩)
                rw [hw]
                simp [List.replicate_succ]
              · exfalso
                -- the tail began with an `a` after the popping phase: garbage
                have htail : xs = true :: (List.replicate i true
                    ++ List.replicate (i + 1 + m) false) := by
                  rw [hw, List.replicate_succ, List.cons_append]
                rw [htail, fold_popping_sees_a] at hph
                exact hph rfl

/-! ## (a) Unbounded counting — the constructive twin -/

/-- **After `aᵏ` the tape holds exactly `k` slots.** -/
theorem tapeCount_slots (k : ℕ) :
    (tapeModel.stateAfter (List.replicate k true)).2.length = k := by
  have h : tapeModel.stateAfter (List.replicate k true)
      = (TapePhase.pushing, List.replicate k true ++ []) := by
    simpa using fold_pushTape k []
  rw [h]
  simp

/-- **The slot count exceeds every fixed bound**: for every `B` there is a word
(namely `a^(B+1)`) after which the machine holds more than `B` slots. This is
the constructive twin of `count_needs_unbounded_state`: the exact count that
no bounded-precision total state can carry is carried, exactly, by the slot
count. -/
theorem tapeCount_exceeds_bound (B : ℕ) :
    ∃ w : List Bool, B < (tapeModel.stateAfter w).2.length := by
  refine ⟨List.replicate (B + 1) true, ?_⟩
  rw [tapeCount_slots]
  omega

/-- **The slot count realizes the exact count** that `countModel` reports online
(`countModel_out`) and that `count_needs_unbounded_state` shows no bounded
total state can: the tape's unbounded slot count is the counting channel. -/
theorem tapeCount_twin_countModel (k : ℕ) :
    countModel.out (countModel.stateAfter (List.replicate k true))
      = (tapeModel.stateAfter (List.replicate k true)).2.length := by
  rw [countModel_out, tapeCount_slots]

/-! ## (b) The full `{aᵏbᵏ}` decision on all inputs -/

/-- The machine's configuration after the balanced word `aᵏbᵏ`: exactly empty,
in the popping phase for `k ≥ 1` and still pushing for `k = 0`. -/
theorem stateAfter_balanced (k : ℕ) :
    tapeModel.stateAfter (List.replicate k true ++ List.replicate k false)
      = (if k = 0 then TapePhase.pushing else TapePhase.popping, []) := by
  simp only [StreamModel.stateAfter, tapeModel, List.foldl_append, fold_pushTape,
    List.append_nil]
  rw [fold_popChain k k TapePhase.pushing (by simp),
    if_pos (le_refl k), show k - k = 0 from Nat.sub_self k, List.replicate_zero]

/-- The balanced words are accepted. -/
theorem tapeModel_accepts_balanced (k : ℕ) :
    tapeModel.out (tapeModel.stateAfter (List.replicate k true ++ List.replicate k false))
      = true := by
  rw [stateAfter_balanced]
  split <;> simp [tapeModel]

/-- `takeWhile` keeps the whole `a`-block of `aᵏbᵏ`. -/
theorem takeWhile_balanced (k : ℕ) :
    (List.replicate k true ++ List.replicate k false).takeWhile (fun x => x)
      = List.replicate k true := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [show List.replicate (k + 1) true = true :: List.replicate k true from
          List.replicate_succ, List.cons_append]
      simp [List.takeWhile]

/-- `dropWhile` skips the whole `a`-block of `aᵏbᵏ`. -/
theorem dropWhile_push (k : ℕ) :
    (List.replicate k true ++ List.replicate k false).dropWhile (fun x => x)
      = List.replicate k false := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [show List.replicate (k + 1) true = true :: List.replicate k true from
          List.replicate_succ, List.cons_append]
      simp [List.dropWhile]

/-- `takeWhile` stops at the first `b`. -/
theorem takeWhile_replicate_false (k : ℕ) :
    (List.replicate k false).takeWhile (fun x => x) = [] := by
  cases k with
  | zero => simp
  | succ k =>
      rw [show List.replicate (k + 1) false = false :: List.replicate k false from
          List.replicate_succ]
      simp [List.takeWhile]

/-- `dropWhile` preserves the `b`-block: the predicate fails at the first `b`,
so the whole block is returned unchanged. -/
theorem dropWhile_replicate_false (k : ℕ) :
    (List.replicate k false).dropWhile (fun x => x) = List.replicate k false := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [show List.replicate (k + 1) false = false :: List.replicate k false from
          List.replicate_succ]
      simp [List.dropWhile]

/-- The constructive decider decides its own specification. -/
theorem isBalancedAB_iff (w : List Bool) :
    isBalancedAB w = true ↔ BalancedAB w := by
  constructor
  · intro h
    simp only [isBalancedAB, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
    obtain ⟨hall, hlen⟩ := h
    have hAs : w.takeWhile (fun x => x)
        = List.replicate (w.takeWhile (fun x => x)).length true :=
      (List.eq_replicate_length).mpr (fun x hx => by
        simpa using List.mem_takeWhile_imp hx)
    have hBs : w.dropWhile (fun x => x)
        = List.replicate (w.dropWhile (fun x => x)).length false :=
      (List.eq_replicate_length).mpr (fun x hx => by simpa using hall x hx)
    refine ⟨(w.takeWhile (fun x => x)).length, ?_⟩
    calc w = w.takeWhile (fun x => x) ++ w.dropWhile (fun x => x) :=
          (List.takeWhile_append_dropWhile).symm
      _ = List.replicate (w.takeWhile (fun x => x)).length true
            ++ w.dropWhile (fun x => x) := by nth_rw 1 [hAs]
      _ = List.replicate (w.takeWhile (fun x => x)).length true
            ++ List.replicate (w.dropWhile (fun x => x)).length false := by
          nth_rw 1 [hBs]
      _ = List.replicate (w.takeWhile (fun x => x)).length true
            ++ List.replicate (w.takeWhile (fun x => x)).length false := by rw [← hlen]
  · intro ⟨k, hk⟩
    subst hk
    simp only [isBalancedAB]
    rw [takeWhile_balanced, dropWhile_push]
    simp

/-- **The tape model decides `aᵏbᵏ` on all inputs.** Its readout after any word
is exactly the constructive shape decision `isBalancedAB` — the accepted words
are precisely the `BalancedAB` words, and every malformed input (leading `b`,
mismatched counts, `a` after the `b`-phase) is rejected. -/
theorem tapeModel_decides_balanced (w : List Bool) :
    tapeModel.out (tapeModel.stateAfter w) = isBalancedAB w := by
  by_cases hout : tapeModel.out (tapeModel.stateAfter w) = true
  · have hstate : (tapeModel.stateAfter w).1 ≠ TapePhase.garbage ∧
        (tapeModel.stateAfter w).2 = [] := by
      simpa [tapeModel] using hout
    rcases fold_trichotomy w TapePhase.pushing 0 (by simp) with hg | hne | ⟨i, hw, hph, hte⟩
    · exact absurd hg hstate.1
    · exact absurd hne (by
        simpa [StreamModel.stateAfter, tapeModel] using hstate.2)
    · rw [hout, (isBalancedAB_iff w).mpr ⟨i, by rw [hw]; simp⟩]
  · have hf : tapeModel.out (tapeModel.stateAfter w) = false := by
      cases h : tapeModel.out (tapeModel.stateAfter w) with
      | true => exact absurd h hout
      | false => rfl
    by_cases hb : isBalancedAB w = true
    · exfalso
      obtain ⟨k, hk⟩ := (isBalancedAB_iff w).mp hb
      rw [hk] at hout
      exact hout (tapeModel_accepts_balanced k)
    · rw [hf]
      simpa using hb

/-- Acceptance is exactly membership in `{aᵏbᵏ : k ≥ 0}`. -/
theorem tapeModel_accepts_iff (w : List Bool) :
    tapeModel.out (tapeModel.stateAfter w) = true ↔ BalancedAB w := by
  rw [tapeModel_decides_balanced w]
  exact isBalancedAB_iff w

/-- **Malformed input, `b` before any `a`: always rejected.** The very first
token underflows the empty tape. -/
theorem tapeModel_rejects_leading_b (w : List Bool) :
    tapeModel.out (tapeModel.stateAfter (false :: w)) = false := by
  have hs : tapeModel.stateAfter (false :: w) = (TapePhase.garbage, []) := by
    simp only [StreamModel.stateAfter, tapeModel, List.foldl_cons]
    have hstep : tapeStep (TapePhase.pushing, ([] : List Bool)) false
        = (TapePhase.garbage, []) := rfl
    rw [hstep, fold_garbage]
  rw [hs]
  simp [tapeModel]

/-- **Malformed input, mismatched counts: `aᵏ'bᵏ` with `k ≠ k'` is always
rejected** — a leftover slot when `k < k'`, underflow when `k > k'`. -/
theorem tapeModel_rejects_count_mismatch (k k' : ℕ) (h : k ≠ k') :
    tapeModel.out (tapeModel.stateAfter (List.replicate k' true ++ List.replicate k false))
      = false := by
  simp only [StreamModel.stateAfter, tapeModel, List.foldl_append, fold_pushTape,
    List.append_nil]
  rw [fold_popChain k k' TapePhase.pushing (by simp)]
  by_cases hle : k ≤ k'
  · rw [if_pos hle]
    by_cases hk : k = 0
    · subst hk
      rw [Nat.sub_zero]
      have hk' : k' ≠ 0 := Ne.symm h
      simp [hk']
    · rw [if_neg hk]
      have hne' : List.replicate (k' - k) true ≠ [] := by
        intro hnil
        rw [List.replicate_eq_nil_iff] at hnil
        omega
      simp [hne']
  · rw [if_neg hle]
    simp

/-- **Malformed input, trailing garbage: any `a` after the `b`-phase began is
always rejected** (`j ≥ 1` `b`s read, then an `a`, then anything). -/
theorem tapeModel_rejects_trailing_a (k j : ℕ) (hj : 1 ≤ j) (v : List Bool) :
    tapeModel.out (tapeModel.stateAfter
      (List.replicate k true ++ List.replicate j false ++ (true :: v))) = false := by
  simp only [StreamModel.stateAfter, tapeModel, List.foldl_append, fold_pushTape,
    List.append_nil]
  rw [fold_popChain j k TapePhase.pushing (by simp)]
  by_cases hle : j ≤ k
  · rw [if_pos hle, if_neg (by omega : ¬ j = 0), List.foldl_cons,
      show tapeStep (TapePhase.popping, List.replicate (k - j) true) true
          = (TapePhase.garbage, List.replicate (k - j) true) from by
        generalize hn : k - j = n
        cases n <;> rfl,
      fold_garbage]
    simp
  · rw [if_neg hle, fold_garbage]
    simp

/-! ## Bounded per-step precision, unbounded slot count -/

/-- **Every reachable tape is unary**: each slot holds the single symbol `true`,
so the per-slot alphabet is one value (zero bits per slot) and all the
unbounded information is in the slot *count*. -/
theorem tapeModel_cells_unary (w : List Bool) :
    ∃ (p : TapePhase) (m : ℕ), tapeModel.stateAfter w = (p, List.replicate m true) := by
  have hfold : ∀ (v : List Bool) (p : TapePhase) (m : ℕ),
      ∃ (p' : TapePhase) (m' : ℕ),
        v.foldl tapeStep (p, List.replicate m true) = (p', List.replicate m' true) := by
    intro v
    induction v with
    | nil => intro p m; exact ⟨p, m, rfl⟩
    | cons x xs ih =>
        intro p m
        cases x with
        | true =>
            cases p with
            | pushing =>
                rw [List.foldl_cons,
                  show tapeStep (TapePhase.pushing, List.replicate m true) true
                      = (TapePhase.pushing, List.replicate (m + 1) true) from rfl]
                exact ih TapePhase.pushing (m + 1)
            | popping =>
                rw [List.foldl_cons,
                  show tapeStep (TapePhase.popping, List.replicate m true) true
                      = (TapePhase.garbage, List.replicate m true) from by
                    cases m <;> rfl]
                exact ⟨TapePhase.garbage, m, fold_garbage xs _⟩
            | garbage =>
                rw [List.foldl_cons,
                  show tapeStep (TapePhase.garbage, List.replicate m true) true
                      = (TapePhase.garbage, List.replicate m true) from rfl]
                exact ⟨TapePhase.garbage, m, fold_garbage xs _⟩
        | false =>
            cases p with
            | pushing =>
                cases m with
                | zero =>
                    rw [List.foldl_cons,
                      show tapeStep (TapePhase.pushing, List.replicate 0 true) false
                          = (TapePhase.garbage, List.replicate 0 true) from rfl]
                    exact ⟨TapePhase.garbage, 0, fold_garbage xs _⟩
                | succ m =>
                    rw [List.foldl_cons,
                      show tapeStep (TapePhase.pushing, List.replicate (m + 1) true) false
                          = (TapePhase.popping, List.replicate m true) from rfl]
                    exact ih TapePhase.popping m
            | popping =>
                cases m with
                | zero =>
                    rw [List.foldl_cons,
                      show tapeStep (TapePhase.popping, List.replicate 0 true) false
                          = (TapePhase.garbage, List.replicate 0 true) from rfl]
                    exact ⟨TapePhase.garbage, 0, fold_garbage xs _⟩
                | succ m =>
                    rw [List.foldl_cons,
                      show tapeStep (TapePhase.popping, List.replicate (m + 1) true) false
                          = (TapePhase.popping, List.replicate m true) from rfl]
                    exact ih TapePhase.popping m
            | garbage =>
                rw [List.foldl_cons,
                  show tapeStep (TapePhase.garbage, List.replicate m true) false
                      = (TapePhase.garbage, List.replicate m true) from rfl]
                exact ⟨TapePhase.garbage, m, fold_garbage xs _⟩
  exact hfold w TapePhase.pushing 0

/-! ## The pair -/

/-- The tape model computes the `aᵏbᵏ` discrimination, hence — by the
both-impossible theorem — cannot have a bounded-precision total state channel. -/
theorem tapeModel_separates_balanced : SeparatesBalanced tapeModel true false where
  hits := by
    intro k _
    exact tapeModel_accepts_balanced k
  misses := by
    intro k k' _ _ hne
    exact tapeModel_rejects_count_mismatch k k' (Ne.symm hne)

/-- **The tape model has no bounded-precision total state channel** — precisely
because it succeeds where `balanced_needs_unbounded_state` says no
bounded-precision channel can. Its unbounded component is the slot count. -/
theorem tapeModel_no_finite_abstraction : NoBoundedAbstraction tapeModel :=
  context_free_discrimination_impossible tapeModel tapeModel_separates_balanced (by simp)

/-- **The tape-memory pair.** Side by side:

* **per-step state is bounded-precision in every component** — a three-valued
  control (`tapePhase_card`) and a one-symbol slot alphabet
  (`tapeModel_cells_unary`): no slot ever carries more than the single symbol
  `true`;
* **the slot count is unbounded** (`tapeCount_exceeds_bound`) — exactly the
  axis the both-impossible theorems do not bound;
* **it decides `{aᵏbᵏ}` on all inputs** (`tapeModel_decides_balanced`) — what
  `balanced_needs_unbounded_state` denies every bounded-precision channel;
* **it admits no bounded-precision total state channel**
  (`NoBoundedAbstraction`) — the slot count, not per-slot or control precision,
  is the resource.

What unbounded slot *count* buys over bounded total state is exactly what the
impossibility theorems deny the native channels. -/
theorem tape_memory_pair :
    (∀ w, ∃ (p : TapePhase) (m : ℕ), tapeModel.stateAfter w = (p, List.replicate m true)) ∧
    (∀ B, ∃ w : List Bool, B < (tapeModel.stateAfter w).2.length) ∧
    (∀ w, tapeModel.out (tapeModel.stateAfter w) = isBalancedAB w) ∧
    NoBoundedAbstraction tapeModel :=
  ⟨fun w => tapeModel_cells_unary w, fun B => tapeCount_exceeds_bound B,
    fun w => tapeModel_decides_balanced w, tapeModel_no_finite_abstraction⟩

end TapeMemory
