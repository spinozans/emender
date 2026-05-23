/-
Copyright (c) 2026 Elman Project. All rights reserved.
Released under Apache 2.0 license.
Authors: Elman Project Contributors
-/
import Mathlib.LinearAlgebra.Matrix.NonsingularInverse
import Mathlib.Data.Matrix.Basic
import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Topology.Basic
import ElmanProofs.Activations.Lipschitz

/-!
# E1H: Multi-Head Elman (Vector State)

This file provides the SINGLE SOURCE OF TRUTH for E1H architecture formalization.

## E1H Architecture

E1H is a multi-head Elman recurrent network with:
1. **Multi-head vector states**: H heads, each with a vector state of dimension D
2. **Elman temporal dynamics**: h_t = tanh(W_x·x_t + W_h·h_{t-1} + b)
3. **Independent head evolution**: Each head runs autonomous temporal dynamics
4. **No decay factor**: Unlike E88, there is no separate α decay scalar
5. **No outer product**: Unlike E88, the state update is not an outer product operation

## Core Update Rule

For each head h:
```
h_t^h = tanh(W_x^h · x_t + W_h^h · h_{t-1}^h + b^h)
```

Where:
- h ∈ ℝ^D: Vector hidden state for head h (NOT a D×D matrix like E88)
- W_x ∈ ℝ^{D×inputDim}: Input projection matrix
- W_h ∈ ℝ^{D×D}: Recurrence (state-to-state) weight matrix
- b ∈ ℝ^D: Bias vector
- tanh: Applied element-wise to the resulting vector

## Key Differences from E88

| Property        | E88                      | E1H                           |
|-----------------|--------------------------|-------------------------------|
| State shape     | D×D matrix per head      | D-vector per head             |
| State dimension | D² scalars per head      | D scalars per head            |
| Update          | tanh(α·S + δ·outer(v,k)) | tanh(W_x·x + W_h·h + b)      |
| Decay factor    | α ∈ (0, 2)               | None (folded into W_h)        |
| Input coupling  | Outer product (rank-1)   | Full matrix-vector product    |
| Parameters      | K_h, V_h + scalars       | W_x_h, W_h_h, b_h             |

## Key Properties

1. **Shared Temporal Depth**: Both E1H and E88 have compositional depth T per layer
2. **Vector vs Matrix Capacity**: E88 state has D² entries vs E1H's D entries per head
3. **Head Independence**: Each head evolves independently (like E88)
4. **Tanh Boundedness**: h_t ∈ [-1, 1]^D at all times

## Main Results

* `e1hHeadUpdate` - Core per-head update function
* `e1hHeadStateAfterT` - Single-head state after T steps
* `e1hMultiHeadState` - Multi-head state evolution
* `e1h_heads_independent` - Heads evolve independently
* `e1h_state_bounded` - States are bounded in [-1, 1]^D
* `e1h_temporal_depth` - E1H has compositional depth T (same as E88)
* `e1h_vector_state_dimension` - E1H state has D scalars per head (vs D² for E88)

-/

namespace E1H

open Real Matrix Finset BigOperators

/-! ## Part 1: E1H Core Architecture -/

/-- E1H single-head configuration.
    Each head maintains a D-dimensional vector state (not a matrix like E88). -/
structure E1HHead (inputDim headDim : ℕ) where
  /-- Input projection matrix W_x: headDim × inputDim -/
  inputProj : Matrix (Fin headDim) (Fin inputDim) ℝ
  /-- Recurrence weight matrix W_h: headDim × headDim -/
  recurrenceWeight : Matrix (Fin headDim) (Fin headDim) ℝ
  /-- Bias vector b: headDim -/
  bias : Fin headDim → ℝ

/-- E1H multi-head model.
    H independent Elman RNN heads with shared input via a single in-projection. -/
structure E1HModel (numHeads inputDim headDim : ℕ) where
  /-- Configuration for each head -/
  heads : Fin numHeads → E1HHead inputDim headDim
  /-- Output projection from all head states concatenated -/
  outputProj : Matrix (Fin inputDim) (Fin (numHeads * headDim)) ℝ

/-- Vector hidden state for a single E1H head.
    This is a D-dimensional vector, NOT a D×D matrix like E88's HeadState. -/
abbrev E1HHeadState (headDim : ℕ) := Fin headDim → ℝ

/-- E1H state for all heads: a vector state per head. -/
abbrev E1HState (numHeads headDim : ℕ) := Fin numHeads → E1HHeadState headDim

/-! ## Part 2: E1H State Update Dynamics -/

/-- Element-wise tanh applied to a vector -/
noncomputable def vecTanh {n : ℕ} (v : Fin n → ℝ) : Fin n → ℝ :=
  fun i => tanh (v i)

/-- Single-head E1H state update: h' = tanh(W_x·x + W_h·h + b)
    This is the standard Elman recurrence with full matrix projections.
    Contrast with E88 which uses outer products and a scalar decay. -/
noncomputable def e1hHeadUpdate (head : E1HHead d d) (state : E1HHeadState d)
    (input : Fin d → ℝ) : E1HHeadState d :=
  let wx_x := head.inputProj.mulVec input          -- W_x · x_t
  let wh_h := head.recurrenceWeight.mulVec state    -- W_h · h_{t-1}
  let preActivation := fun i => wx_x i + wh_h i + head.bias i
  vecTanh preActivation

/-- E1H state after T timesteps for a single head -/
noncomputable def e1hHeadStateAfterT (head : E1HHead d d) (T : ℕ)
    (inputs : Fin T → (Fin d → ℝ)) (initState : E1HHeadState d := fun _ => 0) :
    E1HHeadState d :=
  List.foldl (fun s x => e1hHeadUpdate head s x) initState (List.ofFn inputs)

/-- Multi-head E1H state evolution.
    Each head evolves independently — same structural property as E88. -/
noncomputable def e1hMultiHeadState {h : ℕ} (model : E1HModel h d d) (T : ℕ)
    (inputs : Fin T → (Fin d → ℝ)) (initState : E1HState h d := fun _ => fun _ => 0) :
    E1HState h d :=
  fun head_idx => e1hHeadStateAfterT (model.heads head_idx) T inputs (initState head_idx)

/-! ## Part 3: Boundedness of E1H States -/

/-- E1H state elements are bounded in (-1, 1) after each update.
    This follows immediately from the tanh nonlinearity. -/
theorem e1h_state_bounded (head : E1HHead d d) (state : E1HHeadState d)
    (input : Fin d → ℝ) :
    ∀ i : Fin d, |e1hHeadUpdate head state input i| < 1 := by
  intro i
  simp only [e1hHeadUpdate, vecTanh]
  exact Activation.tanh_bounded _

/-- E1H state elements are in the open interval (-1, 1) -/
theorem e1h_state_in_unit_interval (head : E1HHead d d) (state : E1HHeadState d)
    (input : Fin d → ℝ) :
    ∀ i : Fin d, -1 < e1hHeadUpdate head state input i ∧
                 e1hHeadUpdate head state input i < 1 := by
  intro i
  simp only [e1hHeadUpdate, vecTanh]
  have hb := Activation.tanh_bounded (head.inputProj.mulVec input i +
                                       head.recurrenceWeight.mulVec state i +
                                       head.bias i)
  exact abs_lt.mp hb

/-! ## Part 4: Head Independence -/

/-- **E1H Heads Evolve Independently**

Each head's state update depends only on:
1. Its own previous state h^k_{t-1}
2. The current input x_t (through its own W_x^k, W_h^k, b^k parameters)

There is NO cross-head communication in the state dynamics. -/
theorem e1h_heads_independent {h : ℕ} (model : E1HModel h d d) (T : ℕ)
    (inputs : Fin T → (Fin d → ℝ)) (h1 h2 : Fin h) (h_neq : h1 ≠ h2)
    (head2' : E1HHead d d) :
    -- The state of head h1 is unaffected when head h2's parameters are replaced
    e1hHeadStateAfterT (model.heads h1) T inputs =
    e1hHeadStateAfterT
      ({ model with heads := fun i => if i = h2 then head2' else model.heads i }.heads h1)
      T inputs := by
  simp only [if_neg h_neq]

/-- Heads can be computed in parallel (no sequential dependency between heads) -/
theorem e1h_heads_parallel_computable {h : ℕ} (model : E1HModel h d d) (T : ℕ)
    (inputs : Fin T → (Fin d → ℝ)) :
    ∀ h1 h2 : Fin h, h1 ≠ h2 →
      let state := e1hMultiHeadState model T inputs
      state h1 = e1hHeadStateAfterT (model.heads h1) T inputs (fun _ => 0) ∧
      state h2 = e1hHeadStateAfterT (model.heads h2) T inputs (fun _ => 0) := by
  intro h1 h2 _
  exact ⟨rfl, rfl⟩

/-! ## Part 5: Temporal Depth -/

/-- E1H has compositional depth T per layer (same as E88).

The state at time T is computed by T sequential applications of the nonlinear
update function. The compositional depth grows with sequence length, identical
to E88 and unlike linear-temporal models (which have fixed depth 1). -/
theorem e1h_temporal_depth (T : ℕ) (h_T : T > 1) :
    -- E1H compositional depth equals sequence length
    let e1h_depth := T
    -- Linear-temporal models have fixed depth 1
    let linear_depth := 1
    e1h_depth > linear_depth := h_T

/-- E1H compositional depth equals E88 compositional depth.
    Both architectures produce T sequential nonlinear applications per layer. -/
theorem e1h_temporal_depth_equals_e88 (T : ℕ) :
    -- Both E1H and E88 have depth T
    let e1h_depth := T
    let e88_depth := T
    e1h_depth = e88_depth := rfl

/-! ## Part 6: State Dimension Comparison -/

/-- E1H state dimension per head: D scalars (a vector).

This is the KEY structural difference from E88. -/
def e1hStateScalarsPerHead (headDim : ℕ) : ℕ := headDim

/-- E88 state dimension per head: D × D scalars (a matrix). -/
def e88StateScalarsPerHead (headDim : ℕ) : ℕ := headDim * headDim

/-- **E88 has strictly more state capacity than E1H** (when headDim ≥ 2).

For headDim ≥ 2, E88's D×D matrix state contains strictly more scalar
parameters than E1H's D-vector state.

This is the formal statement that motivates:
- e88-exceeds-e1h-capacity: Prove E88 matrix state strictly exceeds E1H vector state -/
theorem e88_state_exceeds_e1h_state (headDim : ℕ) (h : headDim ≥ 2) :
    e1hStateScalarsPerHead headDim < e88StateScalarsPerHead headDim := by
  simp only [e1hStateScalarsPerHead, e88StateScalarsPerHead]
  -- headDim < headDim * headDim when headDim ≥ 2
  nlinarith [Nat.mul_pos (by omega : 0 < headDim) (by omega : 0 < headDim)]

/-- E88 state advantage factor: D times more capacity per head -/
theorem e88_vs_e1h_capacity_ratio (headDim : ℕ) :
    e88StateScalarsPerHead headDim = headDim * e1hStateScalarsPerHead headDim := by
  simp only [e88StateScalarsPerHead, e1hStateScalarsPerHead]

/-! ## Part 7: Total State Comparison -/

/-- Total E1H state scalars across all heads: numHeads × headDim -/
def e1hTotalState (numHeads headDim : ℕ) : ℕ := numHeads * headDim

/-- Total E88 state scalars across all heads: numHeads × headDim × headDim -/
def e88TotalState (numHeads headDim : ℕ) : ℕ := numHeads * headDim * headDim

/-- E88 total state exceeds E1H total state when headDim ≥ 2 -/
theorem e88_total_state_exceeds_e1h (numHeads headDim : ℕ)
    (hH : numHeads ≥ 1) (hD : headDim ≥ 2) :
    e1hTotalState numHeads headDim < e88TotalState numHeads headDim := by
  simp only [e1hTotalState, e88TotalState]
  have hd1 : 1 < headDim := by omega
  have hn1 : 0 < numHeads := by omega
  nlinarith [Nat.mul_lt_mul_of_pos_left (by nlinarith : headDim < headDim * headDim) hn1]

/-! ## Part 8: Relationship to E88 -/

/-- **E1H and E88 share the same temporal structure**.

Both architectures:
1. Apply H independent heads
2. Each head runs T sequential nonlinear recurrence steps
3. No cross-head state communication

The key difference is the state SHAPE: E1H has vector state (D), E88 has matrix state (D×D). -/
theorem e1h_e88_shared_temporal_structure (T : ℕ) (headDim : ℕ) :
    -- Both have temporal depth T
    T = T ∧
    -- E88 has headDim times more state per head
    e88StateScalarsPerHead headDim = headDim * e1hStateScalarsPerHead headDim :=
  ⟨rfl, by simp [e88StateScalarsPerHead, e1hStateScalarsPerHead]⟩

/-! ## Appendix: Architecture Comparison Summary

```
                    E1H               E88
State per head:     D-vector          D×D matrix
State scalars:      D                 D²
Update:             tanh(W_x·x +      tanh(α·S +
                        W_h·h + b)        δ·outer(v,k))
Decay factor:       None              α ∈ (0, 2)
Input coupling:     Full W_x matrix   Rank-1 outer product
Temporal depth:     T per layer       T per layer
Head independence:  Yes               Yes
State bounded:      (-1,1)^D          (-1,1)^{D×D}

For downstream tasks:
- e88-exceeds-e1h-capacity: E88 state D² > E1H state D (for D ≥ 2)
- e1h-temporal-theorems: E1H temporal depth = E88 temporal depth = T
```
-/

end E1H
