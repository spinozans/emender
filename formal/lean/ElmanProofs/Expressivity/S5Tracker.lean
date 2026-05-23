/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.S5Witness

/-!
# Explicit S5 Permutation Tracker

This module turns the S5 witness into the concrete online task used by the
experiment code:

* tokens are the four adjacent transposition generators `(0 1)`, `(1 2)`,
  `(2 3)`, and `(3 4)`;
* the hidden state is the current element of `S5`;
* each token updates the state by right-composition with its transposition.

The Python task in
`/home/erikg/elman/experiments/expressivity_tasks/tasks/s5_permutation.py`
stores a permutation as a tuple of images and updates it by swapping adjacent
entries. The `pythonApplySwap_eq_step_tuple` theorem below proves that this is
the same transition as the Lean permutation update.
-/

namespace S5Tracker

abbrev State := S5Witness.S5

/-- The four adjacent transposition tokens for S5. -/
inductive AdjacentGenerator where
  | s01
  | s12
  | s23
  | s34
  deriving Repr, DecidableEq

instance : Fintype AdjacentGenerator where
  elems := {AdjacentGenerator.s01, AdjacentGenerator.s12,
    AdjacentGenerator.s23, AdjacentGenerator.s34}
  complete := by
    intro g
    cases g <;> simp

/-- S5 has four adjacent-generator input symbols. -/
theorem adjacent_generator_count :
    Fintype.card AdjacentGenerator = 4 := by
  decide

/-- Interpret an input token as the corresponding adjacent transposition. -/
def transposition : AdjacentGenerator -> State
  | .s01 => Equiv.swap (0 : Fin 5) (1 : Fin 5)
  | .s12 => Equiv.swap (1 : Fin 5) (2 : Fin 5)
  | .s23 => Equiv.swap (2 : Fin 5) (3 : Fin 5)
  | .s34 => Equiv.swap (3 : Fin 5) (4 : Fin 5)

/-- The S5 tracker step: compose the current permutation with the token on the
right. For a tuple-of-images representation this swaps adjacent entries. -/
def step (p : State) (g : AdjacentGenerator) : State :=
  p * transposition g

@[simp]
theorem step_apply (p : State) (g : AdjacentGenerator) (i : Fin 5) :
    step p g i = p (transposition g i) := by
  simp [step, Equiv.Perm.mul_def]

/-- Run the tracker from an arbitrary starting state. -/
def runFrom (p : State) (w : List AdjacentGenerator) : State :=
  w.foldl step p

/-- Run the tracker from the identity state. -/
def run (w : List AdjacentGenerator) : State :=
  runFrom 1 w

/-- The concrete S5 tracker as a fixed-precision online recognizer. Its
acceptor is intentionally trivial: this task is state tracking, not a Boolean
language decision problem. -/
def recognizer : S5Witness.FixedPrecisionOnlineRecognizer AdjacentGenerator where
  State := State
  init := 1
  step := step
  accept := fun _ => true

/-- The explicit tracker has exactly the 120 S5 states. -/
theorem recognizer_state_count :
    Fintype.card recognizer.State = 120 :=
  S5Witness.s5_state_count

/-- The recognizer run is the same fold used by the task-specific definition. -/
theorem recognizer_run_eq (w : List AdjacentGenerator) :
    recognizer.run w = run w := by
  rfl

@[simp]
theorem run_nil :
    run [] = 1 := by
  rfl

@[simp]
theorem runFrom_nil (p : State) :
    runFrom p [] = p := by
  rfl

/-- Running from `p` composes `p` with the product of the input word. -/
theorem runFrom_eq_mul_run (p : State) (w : List AdjacentGenerator) :
    runFrom p w = p * run w := by
  induction w generalizing p with
  | nil =>
      simp [runFrom, run]
  | cons g gs ih =>
      change runFrom (p * transposition g) gs =
        p * runFrom (transposition g) gs
      rw [ih (p * transposition g), ih (transposition g)]
      simp [mul_assoc]

/-- Concatenating input words composes their tracked S5 products. -/
theorem run_append (u v : List AdjacentGenerator) :
    run (u ++ v) = run u * run v := by
  calc
    run (u ++ v) = runFrom (run u) v := by
      simp [run, runFrom]
    _ = run u * run v := runFrom_eq_mul_run (run u) v

/-- Appending one token composes by that token's adjacent transposition. -/
theorem run_concat (w : List AdjacentGenerator) (g : AdjacentGenerator) :
    run (w ++ [g]) = run w * transposition g := by
  simpa [run] using (show runFrom 1 (w ++ [g]) = run w * transposition g by
    calc
      runFrom 1 (w ++ [g]) = runFrom (run w) [g] := by
        simp [run, runFrom]
      _ = run w * transposition g := by
        rfl)

/-! ## Bridge to the Python Tuple Semantics -/

/-- Python represents an S5 permutation as a length-five tuple of images. -/
abbrev PythonTuple := Fin 5 -> Fin 5

/-- Tuple update matching the Python `_apply_swap`: swap entries at positions
`a` and `b`, leaving all other entries unchanged. -/
def tupleSwap (a b : Fin 5) (t : PythonTuple) : PythonTuple :=
  fun i => if i = a then t b else if i = b then t a else t i

/-- The Python task's four input-token updates. -/
def pythonApplySwap (t : PythonTuple) : AdjacentGenerator -> PythonTuple
  | .s01 => tupleSwap 0 1 t
  | .s12 => tupleSwap 1 2 t
  | .s23 => tupleSwap 2 3 t
  | .s34 => tupleSwap 3 4 t

/-- Convert a Lean permutation to the tuple-of-images representation used by
the executable task. -/
def toPythonTuple (p : State) : PythonTuple :=
  p

/-- One Python tuple update is exactly one Lean S5 tracker step. -/
theorem pythonApplySwap_eq_step_tuple (p : State) (g : AdjacentGenerator) :
    pythonApplySwap (toPythonTuple p) g = toPythonTuple (step p g) := by
  funext i
  cases g <;>
    simp [pythonApplySwap, tupleSwap, toPythonTuple, step, transposition,
      Equiv.Perm.mul_def, Equiv.swap_apply_def]
    <;> split_ifs <;> rfl

/-- Run the Python-style tuple update from an arbitrary tuple. -/
def pythonRunFrom (t : PythonTuple) (w : List AdjacentGenerator) : PythonTuple :=
  w.foldl pythonApplySwap t

/-- Python-style tuple execution agrees with Lean tracker execution from every
S5 state. -/
theorem pythonRunFrom_eq_tracker_tuple
    (p : State) (w : List AdjacentGenerator) :
    pythonRunFrom (toPythonTuple p) w = toPythonTuple (runFrom p w) := by
  induction w generalizing p with
  | nil =>
      rfl
  | cons g gs ih =>
      change pythonRunFrom (pythonApplySwap (toPythonTuple p) g) gs =
        toPythonTuple (runFrom (step p g) gs)
      rw [pythonApplySwap_eq_step_tuple]
      exact ih (step p g)

/-- The identity tuple used by the Python generator. -/
def identityTuple : PythonTuple :=
  fun i => i

/-- Python-style tuple execution from the identity agrees with the S5 tracker. -/
theorem pythonRun_eq_tracker_tuple (w : List AdjacentGenerator) :
    pythonRunFrom identityTuple w = toPythonTuple (run w) := by
  simpa [identityTuple, run] using
    pythonRunFrom_eq_tracker_tuple (1 : State) w

/-- Running supervised targets: after each token, emit the current state. This
is the mathematical object behind the Python `mode = "running"` target array. -/
def runningTargetsFrom : State -> List AdjacentGenerator -> List State
  | _, [] => []
  | p, g :: gs =>
      let p' := step p g
      p' :: runningTargetsFrom p' gs

/-- Running targets from the identity state. -/
def runningTargets (w : List AdjacentGenerator) : List State :=
  runningTargetsFrom 1 w

/-- The running target sequence has one target per input token. -/
theorem runningTargetsFrom_length
    (p : State) (w : List AdjacentGenerator) :
    (runningTargetsFrom p w).length = w.length := by
  induction w generalizing p with
  | nil =>
      rfl
  | cons g gs ih =>
      simp [runningTargetsFrom, ih]

/-- The public running-target sequence has one target per input token. -/
theorem runningTargets_length (w : List AdjacentGenerator) :
    (runningTargets w).length = w.length := by
  exact runningTargetsFrom_length 1 w

/-- The first running target is the state after the first token. -/
theorem runningTargets_cons
    (g : AdjacentGenerator) (gs : List AdjacentGenerator) :
    runningTargets (g :: gs) =
      step 1 g :: runningTargetsFrom (step 1 g) gs := by
  rfl

end S5Tracker
