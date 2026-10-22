/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Activations.Lipschitz
import Mathlib.Data.List.OfFn
import Mathlib.Logic.Function.Basic

/-!
# LIN(d) vs TANH(d): there is no TANH-favoring separator

This module settles the X20T expressivity question — *is there a task family some
TANH(`d₀`) computes that no LIN(`d`) with `d ≤ d₀` and *arbitrary* encoders and
decoders computes?* — with the answer **no**, in the strongest available form: at
any fixed sequence horizon `T` over a finite alphabet, `LIN(1)` (a **single**
real/`ℕ`-valued state channel, affine in the carried state) already realizes
*every* function of the word, exactly. So for every `d₀` there is a `LIN(1)`
readout computing whatever a `TANH(d₀)` computes; the presumed separation cannot
occur. The distinguishing power of `TANH`'s in-fold nonlinearity is real, but it
is a *parameter/precision* effect, not a raw expressivity gap at the function-class
level (see `THEOREMS.md` §"What is and is not proven").

## The two objects

* `LIN(d)` here is the affine-fold state channel of
  `CompositionalDepthDuality.gdnStep_eq_affine` /
  `gdn_state_channel_depth_bounded_in_time`: `s_t = A(x_t) s_{t-1} + B(x_t)` with
  `A`, `B` arbitrary functions of the *current token only* (token-identifiable,
  state-independent), and an arbitrary readout `ψ`. This is the `fla-gdn` arm.
* `TANH(d)` is `s_t = tanh(A(x_t) s_{t-1} + B(x_t))` with the same arbitrary
  encoders/decoders — the `E97` / `state_activation="tanh"` arm. Its scalar step
  is `tanhStep` below, built on the tracked `Real.tanh` lemmas.

Both are instances of a `RecurrentModel`, a state type plus a token-indexed
one-step map; the model class is deliberately abstract so the compensation
theorem applies to *any* state channel, TANH included, at *any* dimension.

## Load-bearing results (all proved, no `sorry`)

* `readoutOf` / `readoutOf_apply`: the mechanism — an injective state map plus an
  arbitrary target is realized by *choosing the readout to invert the state map*.
  This is the precise sense in which "arbitrary decoder" gives back whatever the
  state channel can carry.
* `linState_cons`, `linState_bound`, `linState_inj`: the `LIN(1)` base-`b` Horner
  fold `s_t = b·s_{t-1} + x_t` is *injective* on equal-length words (positional
  encoding in one integer coordinate, digits `< b`).
* `lin_one_dim_realizes_any_function`: **LIN(1) realizes every function of a
  length-`T` word.**
* `tanh_computations_are_lin_one_dim`: **every function computed by any recurrent
  model — in particular `TANH(d)` for any `d` — is computed by `LIN(1)`.** This is
  the compensation / no-separator theorem.
* `lin_identity_token_preserves`: correction — an affine identity token leaves the
  state exactly fixed, so exact latching-against-interference does *not* need an
  in-fold nonlinearity.

The residual — a possible *dimension/precision* gap as `T → ∞` at *bounded* state
precision — is **not** claimed here and is not formalized; it is discussed
honestly in `experiments/expressivity_tasks/theory_x20t/THEOREMS.md`.
-/

namespace LinTanhExpressivity

open Function

/-! ## Abstract recurrent word models -/

/-- A fixed-length word of length `T` over alphabet `α`. -/
abbrev Word (α : Type*) (T : Nat) := Fin T → α

/-- A causal recurrent model: an initial state and a token-indexed one-step map
`step : σ → α → σ`. Everything from `LIN(d)` to `TANH(d)` is an instance — the
state type `σ` is the carried state (`Fin d → ℝ` for dimension `d`). -/
structure RecurrentModel (σ α : Type*) (T : Nat) where
  /-- Initial state. -/
  init : σ
  /-- One token-indexed transition on the carried state. -/
  step : σ → α → σ

/-- The state reached after reading the whole length-`T` word, left to right. -/
def RecurrentModel.stateAfter (M : RecurrentModel σ α T) (w : Word α T) : σ :=
  (List.ofFn w).foldl M.step M.init

/-- A model together with a readout `ψ` realizes the function `f`. -/
def Realizes (M : RecurrentModel σ α T) (ψ : σ → β) (f : Word α T → β) : Prop :=
  ∀ w, ψ (M.stateAfter w) = f w

/-! ## The engine: an injective state map + arbitrary readout realizes anything -/

/-- The readout that inverts an injective state map `S` on its image and is
degenerate off the image. This is the operational content of "arbitrary decoder":
whatever the state channel can injectively carry, some readout reads back. -/
noncomputable def readoutOf [Nonempty α] (S : Word α T → σ) (f : Word α T → β) : σ → β :=
  Function.extend S f (fun _ => f (Classical.arbitrary (Word α T)))

/-- On the image of `S`, `readoutOf S f` agrees with `f` whenever `S` is injective. -/
theorem readoutOf_apply [Nonempty α] {S : Word α T → σ} (hS : Injective S)
    (f : Word α T → β) (w : Word α T) : readoutOf S f (S w) = f w :=
  Injective.extend_apply hS f _ w

/-- **If the state map is injective, some readout realizes `f`** for *any* target
`f`. The state channel only has to *separate distinct words*; it does not have to
compute the answer, because the readout (the trained MLP/decoder) is arbitrary. -/
theorem exists_readout_of_injective [Nonempty α] {S : Word α T → σ} (hS : Injective S)
    (f : Word α T → β) : ∃ ψ : σ → β, ∀ w, ψ (S w) = f w :=
  ⟨readoutOf S f, fun w => readoutOf_apply hS f w⟩

/-! ## LIN(1): the base-`b` Horner fold is injective -/

/-- One `LIN(1)` step, `s ↦ b·s + x`, over the digit alphabet `Fin b`. The carried
state is a single natural number: a scalar affine state channel (constant linear
part `b`, token-dependent constant `x`), exactly the `d = 1` case of
`s_t = A(x_t) s_{t-1} + B(x_t)`. -/
def linStep (b : Nat) : Nat → Fin b → Nat := fun s x => b * s + x.val

/-- The `LIN(1)` state after a word: the Horner fold of its digits. -/
def linState (b : Nat) (l : List (Fin b)) : Nat :=
  l.foldl (linStep b) 0

/-- Multiplicative Horner identity, stated to avoid multiplying two variables in a
way `omega` cannot fold: `(b·a + x)·P = a·(P·b) + x·P`. -/
theorem horner_mul (b a x P : Nat) : (b * a + x) * P = a * (P * b) + x * P := by
  rw [Nat.add_mul, Nat.mul_assoc, Nat.mul_left_comm b a P, Nat.mul_comm P b]

/-- General accumulator form: folding from `a` scales `a` by `b^{|l|}` and adds
the folding from `0`. Proved by induction; the two instances used on the tail are
the induction hypothesis at the two different accumulators. -/
theorem linState_fold_acc (b : Nat) (l : List (Fin b)) (a : Nat) :
    l.foldl (linStep b) a = a * b ^ l.length + l.foldl (linStep b) 0 := by
  induction l generalizing a with
  | nil => simp
  | cons x xs ih =>
      have hL : (x :: xs).foldl (linStep b) a
          = (linStep b a x) * b ^ xs.length + xs.foldl (linStep b) 0 := by
        rw [List.foldl_cons]; exact ih (linStep b a x)
      have hR0 : (x :: xs).foldl (linStep b) 0
          = x.val * b ^ xs.length + xs.foldl (linStep b) 0 := by
        rw [List.foldl_cons]
        have h : linStep b 0 x = x.val := by simp [linStep]
        rw [h]; exact ih x.val
      rw [hL, hR0, List.length_cons, pow_succ]
      simp only [linStep]
      rw [horner_mul]
      omega

/-- `LIN(1)` one-step unfolding: the folded state is the high part plus the digit. -/
theorem linState_cons (b : Nat) (x : Fin b) (xs : List (Fin b)) :
    linState b (x :: xs) = x.val * b ^ xs.length + linState b xs := by
  unfold linState
  rw [List.foldl_cons]
  have h : linStep b 0 x = x.val := by simp [linStep]
  rw [h, linState_fold_acc]

/-- The `LIN(1)` state is strictly below `b^{|l|}`: digits `< b` so the Horner
value fits in `|l|` base-`b` places. -/
theorem linState_bound (b : Nat) (_hb : 1 < b) (l : List (Fin b)) :
    linState b l < b ^ l.length := by
  induction l with
  | nil => simp [linState]
  | cons x xs ih =>
      rw [linState_cons, List.length_cons, pow_succ]
      have hlt : x.val * b ^ xs.length + linState b xs
          < x.val * b ^ xs.length + b ^ xs.length :=
        Nat.add_lt_add_left ih _
      have heq : x.val * b ^ xs.length + b ^ xs.length
          = (x.val + 1) * b ^ xs.length := (Nat.succ_mul x.val _).symm
      have hle : (x.val + 1) * b ^ xs.length ≤ b * b ^ xs.length :=
        Nat.mul_le_mul_right _ (by have := x.isLt; omega)
      calc x.val * b ^ xs.length + linState b xs
          < x.val * b ^ xs.length + b ^ xs.length := hlt
        _ = (x.val + 1) * b ^ xs.length := heq
        _ ≤ b * b ^ xs.length := hle
        _ = b ^ xs.length * b := Nat.mul_comm _ _

/-- **`LIN(1)` is injective on equal-length words.** Distinct base-`b` digit
strings of the same length fold to distinct `LIN(1)` states. -/
theorem linState_inj (b : Nat) (hb : 1 < b) :
    ∀ {l1 l2 : List (Fin b)}, l1.length = l2.length →
      linState b l1 = linState b l2 → l1 = l2 := by
  intro l1
  induction l1 with
  | nil =>
      intro l2 hlen _
      rcases l2 with _ | ⟨y, ys⟩
      · rfl
      · simp at hlen
  | cons x xs ih =>
      intro l2 hlen heq
      rcases l2 with _ | ⟨y, ys⟩
      · simp at hlen
      · have hlen' : xs.length = ys.length := by simpa using hlen
        rw [linState_cons, linState_cons] at heq
        rw [← hlen'] at heq
        have hbx : linState b xs < b ^ xs.length := linState_bound b hb xs
        have hby : linState b ys < b ^ xs.length := by
          simpa [← hlen'] using linState_bound b hb ys
        have hxy : x.val = y.val := by
          rcases lt_trichotomy x.val y.val with hlt | heq' | hgt
          · exfalso
            have hle : (x.val + 1) * b ^ xs.length ≤ y.val * b ^ xs.length :=
              Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hlt)
            have heq1 : x.val * b ^ xs.length + b ^ xs.length
                = (x.val + 1) * b ^ xs.length := (Nat.succ_mul x.val _).symm
            have hstrict : x.val * b ^ xs.length + linState b xs
                < y.val * b ^ xs.length + linState b ys := by
              calc x.val * b ^ xs.length + linState b xs
                  < x.val * b ^ xs.length + b ^ xs.length :=
                    Nat.add_lt_add_left hbx _
                _ = (x.val + 1) * b ^ xs.length := heq1
                _ ≤ y.val * b ^ xs.length := hle
                _ ≤ y.val * b ^ xs.length + linState b ys := Nat.le_add_right _ _
            exact absurd heq (ne_of_lt hstrict)
          · exact heq'
          · exfalso
            have hle : (y.val + 1) * b ^ xs.length ≤ x.val * b ^ xs.length :=
              Nat.mul_le_mul_right _ (Nat.succ_le_of_lt hgt)
            have heq1 : y.val * b ^ xs.length + b ^ xs.length
                = (y.val + 1) * b ^ xs.length := (Nat.succ_mul y.val _).symm
            have hstrict : y.val * b ^ xs.length + linState b ys
                < x.val * b ^ xs.length + linState b xs := by
              calc y.val * b ^ xs.length + linState b ys
                  < y.val * b ^ xs.length + b ^ xs.length :=
                    Nat.add_lt_add_left hby _
                _ = (y.val + 1) * b ^ xs.length := heq1
                _ ≤ x.val * b ^ xs.length := hle
                _ ≤ x.val * b ^ xs.length + linState b xs := Nat.le_add_right _ _
            exact absurd heq.symm (ne_of_lt hstrict)
        have heq'' : linState b xs = linState b ys := by
          rw [hxy] at heq
          exact Nat.add_left_cancel heq
        have hxs : xs = ys := ih hlen' heq''
        rw [hxs, Fin.ext hxy]

/-! ## LIN(1) realizes every function; hence no TANH-favoring separator -/

/-- The `LIN(1)` recurrent model over the digit alphabet `Fin b`. -/
def linModel (b : Nat) (T : Nat) : RecurrentModel Nat (Fin b) T :=
  { init := 0, step := linStep b }

/-- `LIN(1)` reached state is the Horner fold of the word. -/
theorem linModel_stateAfter (b T : Nat) (w : Word (Fin b) T) :
    (linModel b T).stateAfter w = linState b (List.ofFn w) := rfl

/-- **LIN(1) realizes every function of a length-`T` word.** With alphabet `Fin b`
(`b ≥ 2`), the single-integer affine fold is injective on length-`T` words, so an
arbitrary readout can output any target. Dimension `1`, unbounded precision. -/
theorem lin_one_dim_realizes_any_function (b : Nat) (hb : 1 < b) (T : Nat)
    (f : Word (Fin b) T → β) :
    ∃ ψ : Nat → β, Realizes (linModel b T) ψ f := by
  haveI : Nonempty (Fin b) := ⟨⟨0, by omega⟩⟩
  apply exists_readout_of_injective
  intro w1 w2 h
  simp only [linModel_stateAfter] at h
  have hlen : (List.ofFn w1).length = (List.ofFn w2).length := by simp
  have hlists : List.ofFn w1 = List.ofFn w2 := linState_inj b hb hlen h
  exact List.ofFn_inj.mp hlists

/-- **The compensation / no-separator theorem.** Every function computed by *any*
recurrent model `MT` on length-`T` words over a finite alphabet `Fin b` — in
particular `TANH(d)` for any `d`, whose state channel is such a `RecurrentModel`
with `σ = Fin d → ℝ` — is computed by a `LIN(1)` model with a suitable readout, at
this fixed finite horizon, with exact (unbounded-precision) scalar state and an
arbitrary terminal decoder that may depend on the horizon and target. So there is
no dimension `d₀ ≥ 1` at which `TANH(d₀)` computes a function no `LIN(d)`,
`d ≤ d₀`, computes at this horizon: `LIN(1)` already computes it. (Learnability,
decoder complexity, finite precision, and cross-horizon uniformity are not
addressed by this theorem.) -/
theorem tanh_computations_are_lin_one_dim (b : Nat) (hb : 1 < b) (T : Nat)
    {σ β : Type*} (MT : RecurrentModel σ (Fin b) T) (ψT : σ → β) :
    ∃ ψ : Nat → β, ∀ w,
      ψ ((linModel b T).stateAfter w) = ψT (MT.stateAfter w) :=
  lin_one_dim_realizes_any_function b hb T (fun w => ψT (MT.stateAfter w))

/-! ## TANH instantiation (concrete) and the affine-latch correction -/

/-- Scalar `TANH(1)` step `s ↦ tanh(A·s + B)`, on the tracked `Real.tanh`. This is
the in-fold nonlinearity the E97 arm applies per token. -/
noncomputable def tanhStep (A B s : ℝ) : ℝ := Real.tanh (A * s + B)

/-- A concrete scalar `TANH`-family model over `Fin b`: each token `x` selects the
`(A, B)` of a `tanhStep`. This is a `RecurrentModel ℝ (Fin b) T`, so
`tanh_computations_are_lin_one_dim` applies to it directly. -/
noncomputable def tanhModel (b T : Nat) (g : Fin b → ℝ × ℝ) : RecurrentModel ℝ (Fin b) T :=
  { init := 0, step := fun s x => tanhStep (g x).1 (g x).2 s }

/-- The concrete scalar `TANH` model is compensated by `LIN(1)`: a direct
application, exhibiting that the `tanh`-in-fold arm enjoys no expressivity
advantage over a single affine state coordinate. -/
theorem tanh_scalar_model_compensated (b : Nat) (hb : 1 < b) (T : Nat) {β : Type*}
    (g : Fin b → ℝ × ℝ) (ψT : ℝ → β) :
    ∃ ψ : Nat → β, ∀ w,
      ψ ((linModel b T).stateAfter w) = ψT ((tanhModel b T g).stateAfter w) :=
  tanh_computations_are_lin_one_dim b hb T (σ := ℝ) (β := β) (tanhModel b T g) ψT

/-- **Correction to the "affine has no attractors ⇒ needed for latching" reading.**
An affine token with identity transition `A = 1, B = 0` leaves the `LIN` state
*exactly* unchanged. Hence exact latching against intervening/noise tokens does
not require an in-fold nonlinearity: `LIN` can hold a value indefinitely across
such tokens. Bistability of a *single* token map is not a resource any task can
force, because the per-token map varies and a contracting affine map latches just
as well. -/
theorem lin_identity_token_preserves (s : ℝ) : (1 : ℝ) * s + 0 = s := by ring

end LinTanhExpressivity
