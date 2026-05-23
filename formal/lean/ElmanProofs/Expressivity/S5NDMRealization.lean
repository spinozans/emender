/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import Mathlib.Data.Fintype.Prod
import ElmanProofs.Expressivity.S5Tracker

/-!
# Finite Lookup Realization for the S5 Tracker

This module proves a deliberately conservative positive result: every finite
fixed-precision recognizer has an exact finite associative transition table.

This is the Lean-checked discrete idealization behind an NDM/E88-style
"read key, retrieve transition, update state" story. It does **not** claim a
trained real-valued network learns the table, and it is not a lower bound for
linear or scan-compatible models.
-/

namespace S5NDMRealization

/-- A finite associative lookup memory from keys to values. -/
structure LookupMemory (Key Value : Type*) where
  read : Key -> Value

/-- Transition-table key for a recognizer: current state plus input symbol. -/
abbrev TransitionKey
    {Alphabet : Type*}
    (M : S5Witness.FixedPrecisionOnlineRecognizer Alphabet) :=
  M.State × Alphabet

/-- The exact transition table of a recognizer. -/
def exactTransitionMemory
    {Alphabet : Type*}
    (M : S5Witness.FixedPrecisionOnlineRecognizer Alphabet) :
    LookupMemory (TransitionKey M) M.State where
  read key := M.step key.1 key.2

/-- One table-driven update. -/
def lookupStep
    {Alphabet : Type*}
    {M : S5Witness.FixedPrecisionOnlineRecognizer Alphabet}
    (mem : LookupMemory (TransitionKey M) M.State)
    (s : M.State) (a : Alphabet) : M.State :=
  mem.read (s, a)

/-- Run a recognizer using an explicit transition table. -/
def lookupRun
    {Alphabet : Type*}
    (M : S5Witness.FixedPrecisionOnlineRecognizer Alphabet)
    (mem : LookupMemory (TransitionKey M) M.State)
    (w : List Alphabet) : M.State :=
  w.foldl (lookupStep mem) M.init

/-- The exact transition memory agrees with the recognizer's one-step update. -/
theorem exactTransitionMemory_step
    {Alphabet : Type*}
    (M : S5Witness.FixedPrecisionOnlineRecognizer Alphabet)
    (s : M.State) (a : Alphabet) :
    lookupStep (exactTransitionMemory M) s a = M.step s a := by
  rfl

/-- The exact transition memory simulates the recognizer on every word. -/
theorem exactTransitionMemory_run
    {Alphabet : Type*}
    (M : S5Witness.FixedPrecisionOnlineRecognizer Alphabet)
    (w : List Alphabet) :
    lookupRun M (exactTransitionMemory M) w = M.run w := by
  induction w generalizing M with
  | nil =>
      rfl
  | cons a as ih =>
      change lookupRun
          { M with init := M.step M.init a }
          (exactTransitionMemory { M with init := M.step M.init a }) as =
        S5Witness.FixedPrecisionOnlineRecognizer.run
          { M with init := M.step M.init a } as
      exact ih { M with init := M.step M.init a }

/-- If the alphabet is finite, the transition-table key space is finite. -/
theorem transition_key_space_finite
    {Alphabet : Type*} [Fintype Alphabet]
    (M : S5Witness.FixedPrecisionOnlineRecognizer Alphabet) :
    Nonempty (Fintype (TransitionKey M)) :=
  letI := M.stateFintype
  ⟨inferInstance⟩

/-- The S5 tracker's exact transition memory. -/
def s5TransitionMemory :
    LookupMemory
      (TransitionKey S5Tracker.recognizer)
      S5Tracker.recognizer.State :=
  exactTransitionMemory S5Tracker.recognizer

/-- The S5 transition table has `120 * 4 = 480` state/input keys. -/
theorem s5_transition_key_count :
    Fintype.card (S5Tracker.State × S5Tracker.AdjacentGenerator) = 480 := by
  rw [Fintype.card_prod]
  rw [S5Witness.s5_state_count, S5Tracker.adjacent_generator_count]

/-- The S5 transition memory agrees with the explicit S5 tracker step. -/
theorem s5TransitionMemory_step
    (s : S5Tracker.State) (g : S5Tracker.AdjacentGenerator) :
    lookupStep s5TransitionMemory s g = S5Tracker.step s g := by
  rfl

/-- The S5 transition memory simulates the explicit S5 tracker on every input
word. -/
theorem s5TransitionMemory_run
    (w : List S5Tracker.AdjacentGenerator) :
    lookupRun S5Tracker.recognizer s5TransitionMemory w =
      S5Tracker.run w := by
  change lookupRun S5Tracker.recognizer
    (exactTransitionMemory S5Tracker.recognizer) w = S5Tracker.run w
  rw [exactTransitionMemory_run]
  rfl

end S5NDMRealization
