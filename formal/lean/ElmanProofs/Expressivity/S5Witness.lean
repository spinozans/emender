/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Fintype.Perm
import Mathlib.GroupTheory.Solvable

/-!
# S5 Witness Scaffold

This module records the checked core of the corrected complexity story:

* A fixed-width, finite-precision online recurrence has a finite state space.
* The S3 permutation tracker is the solvable-control ladder.
* The S5 permutation tracker is the smallest symmetric-group witness with a
  non-solvable transition group.

The external complexity facts used in the paper narrative are not reproved
here: regular languages are in NC1, Barrington's theorem relates S5 programs
to NC1, and the relevant linear-scan lower bounds should be cited from the
state-tracking/SSM literature. This file keeps the trusted Lean surface on the
group-theoretic witness and the fixed-state formal vocabulary.
-/

namespace S5Witness

/-- A fixed-precision online recognizer: one finite state updated once per
input symbol. This is the formal object behind the "finite state ceiling" for
fixed-width, finite-precision recurrent recognizers. -/
structure FixedPrecisionOnlineRecognizer (Alphabet : Type*) where
  State : Type*
  [stateFintype : Fintype State]
  [stateDecidableEq : DecidableEq State]
  init : State
  step : State -> Alphabet -> State
  accept : State -> Bool

attribute [instance] FixedPrecisionOnlineRecognizer.stateFintype
attribute [instance] FixedPrecisionOnlineRecognizer.stateDecidableEq

/-- Run an online recognizer left-to-right over a finite input word. -/
def FixedPrecisionOnlineRecognizer.run
    {Alphabet : Type*} (M : FixedPrecisionOnlineRecognizer Alphabet)
    (w : List Alphabet) : M.State :=
  w.foldl M.step M.init

/-- Language accepted by an online recognizer. -/
def FixedPrecisionOnlineRecognizer.accepts
    {Alphabet : Type*} (M : FixedPrecisionOnlineRecognizer Alphabet)
    (w : List Alphabet) : Prop :=
  M.accept (M.run w) = true

/-- The state space of a fixed-precision online recognizer is finite by
construction. This is the Lean-side version of the fixed-width ceiling:
such a recognizer is a DFA-shaped object. -/
theorem fixed_precision_state_space_finite
    {Alphabet : Type*} (M : FixedPrecisionOnlineRecognizer Alphabet) :
    Nonempty (Fintype M.State) :=
  ⟨inferInstance⟩

/-- The S3 permutation group, used as the solvable control task. -/
abbrev S3 := Equiv.Perm (Fin 3)

/-- The S5 permutation group, used as the non-solvable witness task. -/
abbrev S5 := Equiv.Perm (Fin 5)

/-- S3 tracking has six possible states. -/
theorem s3_state_count : Fintype.card S3 = 6 := by
  change Fintype.card (Equiv.Perm (Fin 3)) = 6
  rw [Fintype.card_perm]
  decide

/-- S5 tracking has 120 possible states. -/
theorem s5_state_count : Fintype.card S5 = 120 := by
  change Fintype.card (Equiv.Perm (Fin 5)) = 120
  rw [Fintype.card_perm]
  decide

/-- S5 is non-solvable. This is the checked group-theoretic reason it is the
right witness instead of parity or modular counting, which live in solvable
groups. -/
theorem s5_not_solvable : ¬ IsSolvable S5 := by
  exact Equiv.Perm.fin_5_not_solvable

/-- Witness tiers used by the empirical suite. -/
inductive PermutationWitness where
  | s3Control
  | s5NonSolvable
  deriving Repr, DecidableEq

/-- State count for each permutation witness tier. -/
def witnessStateCount : PermutationWitness -> Nat
  | .s3Control => 6
  | .s5NonSolvable => 120

theorem s5_witness_state_count :
    witnessStateCount .s5NonSolvable = 120 := by
  rfl

theorem s3_witness_state_count :
    witnessStateCount .s3Control = 6 := by
  rfl

end S5Witness
