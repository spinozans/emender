/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Architectures.SplitGatedDelta
import ElmanProofs.Expressivity.E1HDefinition

/-!
# Compositional Depth Duality: E97 composes in time, GDN pays in depth

The E97/E88 mixed-key delta-correcting update and the GDN/gated-delta linear core
sit next to each other in `SplitGatedDelta` as one-step relations. This module
formalizes the *time* axis of the resulting duality, over those actual
definitions:

* `e97_compositional_depth_grows_in_time`: the E97/E88 state after `T` steps is
  the `T`-fold left composition of the one-step nonlinear update map. Each
  additional time step adds exactly one more application of the update and
  therefore one more elementwise `tanh` layer. The per-layer compositional depth
  is `T`, growing in sequence length, independent of network depth.
* `gdn_state_channel_depth_bounded_in_time`: for the GDN/gated-delta family,
  *conditional on the per-token gates/values/decay*, the state channel map after
  any number of steps is affine in the initial state — `H_T = A * H_0 + C` for
  fixed `A`, `C`. The state channel introduces no nonlinear temporal
  composition, so its nonlinear compositional depth through time is constant
  (in fact `0`) in `T`; any depth growth must be paid in the per-token functions
  (gates/values/MLP).

Neither theorem pins down empirical behavior: they make the structural
depth-versus-time trade-off precise for the two actual one-step definitions.

Both claims are stated over the concrete one-step maps `SplitGatedDelta.e97UpdateDirect`
and `SplitGatedDelta.gdn2LinearCore`; the E97 step map is literally a `matrixTanh`
composed with a state-affine preactivation, while the GDN step map is state-affine
with no nonlinearity at all.
-/

namespace CompositionalDepthDuality

open Matrix BigOperators
open M2RNNComparison

variable {K V : Nat}

/-! ## Generic left-to-right composition of self-maps -/

/-- Left-to-right composition of a list of self-maps, applied to a starting
point: the first map in the list acts first. -/
def composeAll {α : Type*} : List (α → α) → α → α
  | [], x => x
  | f :: fs, x => composeAll fs (f x)

/-- `List.foldl` is left-to-right composition: a fold over `xs` starting from
`init` equals the composition of the induced self-maps applied to `init`. -/
theorem foldl_eq_composeAll {α β : Type*} (f : α → β → α) (xs : List β) (init : α) :
    xs.foldl f init = composeAll (xs.map (fun x => fun s => f s x)) init := by
  induction xs generalizing init with
  | nil => rfl
  | cons x xs ih => exact ih (f init x)

/-! ## E97 / E88 mixed-key delta family: depth grows in time -/

/-- The per-token arguments of one E97/E88 mixed-key delta-correcting step:
key, key-axis erase/read gate, value-axis write gate, and value. -/
structure E97Step (K V : Nat) where
  k : Vec K
  b : Vec K
  w : Vec V
  v : Vec V

/-- One E97/E88 mixed-key delta-correcting update applied with the step's
per-token arguments, as a self-map of the matrix state. -/
noncomputable def e97Step (lambda : Real) (p : E97Step K V) (H : MatState K V) :
    MatState K V :=
  SplitGatedDelta.e97UpdateDirect lambda H p.k p.b p.w p.v

/-- The E97/E88 update map is the elementwise `tanh` nonlinearity composed with
a state-affine preactivation. This is the nonlinearity whose per-step stacking
`e97_compositional_depth_grows_in_time` counts. -/
theorem e97Step_eq_tanh_comp (lambda : Real) (p : E97Step K V) (H : MatState K V) :
    e97Step lambda p H =
      matrixTanh (SplitGatedDelta.e97LinearCore lambda H p.k p.b p.w p.v) :=
  rfl

/-- The self-map on the state space induced by a single E97/E88 step. -/
noncomputable def e97StepMap (lambda : Real) (p : E97Step K V) :
    MatState K V → MatState K V :=
  fun H => e97Step lambda p H

/-- The E97/E88 state after processing a list of steps in time order. -/
noncomputable def e97StateAfter (lambda : Real) :
    List (E97Step K V) → MatState K V → MatState K V
  | [], H => H
  | p :: ps, H => e97StateAfter lambda ps (e97Step lambda p H)

/-- **E97/E88 state after `T` steps is the `T`-fold composition of the one-step
nonlinear update map.** Proved by induction over the actual per-step definitions:
the fold that builds the state is exactly the left-to-right composition of the
induced update maps. -/
theorem e97StateAfter_eq_compose (lambda : Real) (steps : List (E97Step K V))
    (H : MatState K V) :
    e97StateAfter lambda steps H =
      composeAll (steps.map (e97StepMap lambda)) H := by
  induction steps generalizing H with
  | nil => rfl
  | cons p ps ih => exact ih (e97Step lambda p H)

/-- Appending steps composes: the state after `s` then `t` is the state after `t`
started from the state after `s`. -/
theorem e97StateAfter_append (lambda : Real) (s t : List (E97Step K V))
    (H : MatState K V) :
    e97StateAfter lambda (s ++ t) H =
      e97StateAfter lambda t (e97StateAfter lambda s H) := by
  induction s generalizing H with
  | nil => rfl
  | cons p ps ih => exact ih (e97Step lambda p H)

/-- **E97/E88 compositional depth grows in time.**

Every additional time step applies exactly one more instance of the nonlinear
update map on top of the current state. Hence, for a length-`T` step schedule,
the state is the `T`-fold left composition of the update map (see
`e97StateAfter_eq_compose`), and the per-layer compositional depth equals the
sequence length `T`, independent of network depth. -/
theorem e97_compositional_depth_grows_in_time (lambda : Real)
    (steps : List (E97Step K V)) (p : E97Step K V) (H : MatState K V) :
    e97StateAfter lambda (steps ++ [p]) H =
      e97Step lambda p (e97StateAfter lambda steps H) := by
  rw [e97StateAfter_append]
  rfl

/-! ## E1H sibling: the existing temporal-depth theorem, strengthened

`E1H.e1h_temporal_depth` records the depth claim by comparing two `let`-bound
numbers. The theorem below proves the same content structurally: the E1H state
after `T` steps is the `T`-fold composition of the per-head nonlinear update map.
-/

/-- The per-head E1H self-map induced by one input token. -/
noncomputable def e1hStepMap {d : Nat} (head : E1H.E1HHead d d) (x : Fin d → ℝ) :
    E1H.E1HHeadState d → E1H.E1HHeadState d :=
  fun s => E1H.e1hHeadUpdate head s x

/-- **E1H state after `T` steps is the `T`-fold composition of the Elman update
map.** This strengthens `E1H.e1h_temporal_depth` from a `let`-bound comparison
to a structural induction over the actual `e1hHeadStateAfterT` fold. -/
theorem e1hStateAfterT_eq_compose {d : Nat} (head : E1H.E1HHead d d) (T : Nat)
    (inputs : Fin T → (Fin d → ℝ)) (init : E1H.E1HHeadState d) :
    E1H.e1hHeadStateAfterT head T inputs init =
      composeAll ((List.ofFn inputs).map (e1hStepMap head)) init := by
  rw [E1H.e1hHeadStateAfterT]
  exact foldl_eq_composeAll
    (fun s x => E1H.e1hHeadUpdate head s x) (List.ofFn inputs) init

/-! ## GDN / gated-delta family: state-channel depth bounded in time -/

/-- The per-token arguments of one GDN/gated-delta step: the pre-decay matrix
(typically `D * H` folded forward), key, key-axis erase/read gate, value-axis
write gate, and value. The decay matrix is kept explicit so the statement is
honest about conditioning on the gates. -/
structure GdnStepParams (K V : Nat) where
  decay : Matrix (Fin K) (Fin K) Real
  k : Vec K
  b : Vec K
  w : Vec V
  v : Vec V

/-- One GDN/gated-delta state update applied with the step's per-token
arguments. This is the concrete `SplitGatedDelta.gdn2LinearCore`, which applies
the split-gated erase/write core to the pre-decayed state and applies no
elementwise state nonlinearity. -/
def gdnStep (p : GdnStepParams K V) (H : MatState K V) : MatState K V :=
  SplitGatedDelta.gdn2LinearCore p.decay H p.k p.b p.w p.v

/-- A single GDN step is affine in the state: a fixed left transition times the
state plus a fixed input-write term. The state channel enters only linearly. -/
theorem gdnStep_eq_affine (p : GdnStepParams K V) (H : MatState K V) :
    gdnStep p H =
      (SplitGatedDelta.splitGatedTransition 1 p.k p.b * p.decay) * H
        + outerKV p.k (SplitGatedDelta.e97WriteValue p.w p.v) := by
  rw [gdnStep, SplitGatedDelta.gdn2LinearCore_eq_expanded, ← Matrix.mul_assoc]

/-- The GDN/gated-delta state after processing a list of steps in time order. -/
def gdnStateFold : List (GdnStepParams K V) → MatState K V → MatState K V
  | [], H => H
  | p :: ps, H => gdnStateFold ps (gdnStep p H)

/-- **GDN state channel depth is bounded (in fact constant) in time.**

For any finite schedule of GDN/gated-delta steps, the map from the initial state
to the final state is affine: there are a fixed matrix product `A` and a fixed
accumulated write `C` with `H_T = A * H_0 + C` for every initial state. Because
the state channel is affine at every step, no nonlinear temporal composition
accumulates; the nonlinear compositional depth of the state channel is constant
in `T` (zero here), unlike the E97/E88 case. All depth growth in a GDN stack must
therefore be paid in the per-token functions (gates/values/MLP), not in time.

The statement is conditional on the gate/decay/value schedule `steps`, as it must
be: `A` and `C` are built from those per-token parameters. -/
theorem gdn_state_channel_depth_bounded_in_time (steps : List (GdnStepParams K V)) :
    ∃ A : Matrix (Fin K) (Fin K) Real, ∃ C : MatState K V,
      ∀ init : MatState K V, gdnStateFold steps init = A * init + C := by
  induction steps with
  | nil =>
      refine ⟨1, 0, ?_⟩
      intro init
      simp [gdnStateFold]
  | cons p ps ih =>
      rcases ih with ⟨A, C, hAC⟩
      refine ⟨A * SplitGatedDelta.splitGatedTransition 1 p.k p.b * p.decay,
        A * outerKV p.k (SplitGatedDelta.e97WriteValue p.w p.v) + C, ?_⟩
      intro init
      calc
        gdnStateFold (p :: ps) init
            = gdnStateFold ps (gdnStep p init) := rfl
        _ = A * gdnStep p init + C := hAC (gdnStep p init)
        _ = A * ((SplitGatedDelta.splitGatedTransition 1 p.k p.b * p.decay) * init
                + outerKV p.k (SplitGatedDelta.e97WriteValue p.w p.v)) + C := by
              rw [gdnStep_eq_affine]
        _ = A * ((SplitGatedDelta.splitGatedTransition 1 p.k p.b * p.decay) * init)
              + A * outerKV p.k (SplitGatedDelta.e97WriteValue p.w p.v) + C := by
              rw [Matrix.mul_add]
        _ = (A * SplitGatedDelta.splitGatedTransition 1 p.k p.b * p.decay) * init
              + (A * outerKV p.k (SplitGatedDelta.e97WriteValue p.w p.v) + C) := by
              rw [← Matrix.mul_assoc A
                    (SplitGatedDelta.splitGatedTransition 1 p.k p.b * p.decay) init]
              rw [← Matrix.mul_assoc A (SplitGatedDelta.splitGatedTransition 1 p.k p.b) p.decay]
              abel

/-- The affine state channel introduces no nonlinear temporal mixing: the
second difference of the `T`-step state map vanishes for every pair of initial
states. Affine maps satisfy this identity; any map that applies a nonlinearity
inside the state path generically would not. Together with
`gdn_state_channel_depth_bounded_in_time` this certifies that GDN's temporal
state path has constant nonlinear depth. -/
theorem gdn_state_channel_affine_identity (steps : List (GdnStepParams K V))
    (H₁ H₂ : MatState K V) :
    gdnStateFold steps (H₁ + H₂) + gdnStateFold steps 0 =
      gdnStateFold steps H₁ + gdnStateFold steps H₂ := by
  obtain ⟨A, C, hAC⟩ := gdn_state_channel_depth_bounded_in_time steps
  rw [hAC (H₁ + H₂), hAC 0, hAC H₁, hAC H₂, Matrix.mul_add, Matrix.mul_zero, zero_add]
  abel

end CompositionalDepthDuality
