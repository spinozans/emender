/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.StreamingTanhRealization

/-!
# The runs-forever object: `ω`-stream transduction at bounded precision

X20T/X21T/X22T-F1 study *words*: `StreamModel.run` emits one output per token of a
finite input. An **indefinitely-running program** consumes an infinite input stream
and emits one output per token, forever. This file formalizes that object and ports
the X21T abstraction collapse to it.

## The formalism

An `ω`-stream over `α` is just `x : ℕ → α`. A `StreamModel M` acts on it through
*`StreamModel.omegaRun`*: the state after the first `t` tokens is the same fold
`M.step` used for words (`omegaState`), and the output at time `t` is the *shared*
readout of the state after `t + 1` tokens:

`M.omegaRun x t = M.out (M.omegaState x (t + 1))`.

This is a **total deterministic sequential (Moore) transducer over `ω`-streams**:
every input stream produces an infinite output stream, productivity is definitional
(the step is total, there is no halting state and no acceptance condition), and it
is causal/causal-online — the output at time `t` depends only on the prefix of
length `t + 1`. We deliberately do *not* use an `ω`-acceptance condition (Büchi,
Muller, parity). Justification: an indefinitely running program's observable
behaviour is its output stream; an acceptance condition is an extra predicate on
that stream, and `omegaRun_acceptance_invariant` below shows any such predicate is
carried along unchanged by every simulation result here, so nothing is lost by
stating the theorems acceptance-free. This also matches the finite-word development:
`StreamModel.run` is the same fold with the same readout.

## Load-bearing results (all proved, no `sorry`)

* `StreamModel.omegaState_succ` — one state update per token, forever.
* `omegaRun_sim` — **the `ω` simulation engine**: a state map commuting with the
  transitions and respecting the readout (and the initial state) gives *identical
  infinite output streams*.
* `omegaRun_collapse` / `exists_fin_omega_model_of_abstraction` — **the abstraction
  collapse over `ω`-streams**: a bounded-precision (`n`-valued, surjective) channel
  over `ω`-streams is simulated exactly by an `n`-state `ω`-transducer. Same
  hypothesis, same conclusion as the finite-word collapse.
* `omega_class_activation_independent_at_bounded_precision` — **the operator claim, at bounded
  precision**: (a) any model of *either* activation class satisfying the bounded
  abstraction hypothesis has finite-state `ω`-behaviour; (b) every `n`-state
  `ω`-behaviour is realized *exactly*, at dimension `n` with a shared readout, by
  the `LIN(n)` one-hot construction *and* by the `TANH(n)` saturated-corner
  construction. So under the bounded-precision hypothesis the `ω`-program class is
  activation-independent: the dividing line is how much information the memory
  channel carries, not which activation carries it.
* `omegaRun_needs_unbounded_state` — activation-free and dimension-free: a model
  whose `ω`-outputs are pairwise distinct cannot have an `n`-valued channel.
* `countModel_omegaRun`, `countModel_no_omega_finite_state` — exact `LIN(1)`
  arithmetic counts forever (`1, 2, 3, …` on the all-true stream) and therefore
  admits *no* finite state channel at any budget.

## What survives and what bends

Survives: at bounded precision the `ω`-program class is the finite-state class for
both activations, and both realizations are exact at `d = n`. Bends: (i) the upper
bound is *conditional* on the abstraction hypothesis — it is not proved that an
`ω`-transducer built from `tanh` or linear recurrences must have bounded precision;
(ii) the sandwich is a *reachable-realization* equality, not a proof that the
ambient state spaces admit finite abstractions; (iii) with **unbounded** precision
the `ω` classes are not classified here: exact linear arithmetic provably exceeds
finite-state (`countModel_no_omega_finite_state`), and no corresponding exact
`TANH` counting statement is proved, so `ω`-equality at unbounded precision is
**open**. The precision/memory axis, not activation, is what the proved theorems
separate.
-/

namespace StreamingExpressivity

open Function
open StreamingTanhRealization

namespace StreamModel

variable {σ α β : Type*}

/-- The first `t` tokens of an `ω`-stream. -/
def omegaPrefix (x : ℕ → α) : ℕ → List α
  | 0 => []
  | t + 1 => omegaPrefix x t ++ [x t]

@[simp] theorem omegaPrefix_zero (x : ℕ → α) : omegaPrefix x 0 = [] := rfl

@[simp] theorem omegaPrefix_succ (x : ℕ → α) (t : ℕ) :
    omegaPrefix x (t + 1) = omegaPrefix x t ++ [x t] := rfl

/-- A constant `ω`-stream prefixes to a constant list. -/
theorem omegaPrefix_const {α : Type*} (a : α) : ∀ t : ℕ,
    omegaPrefix (fun _ => a) t = List.replicate t a
  | 0 => rfl
  | t + 1 => by
      rw [omegaPrefix_succ, omegaPrefix_const a t, List.replicate_succ']

/-- The state after the first `t` tokens of an `ω`-stream: the same fold `M.step`
used for finite words. -/
def omegaState (M : StreamModel σ α β) (x : ℕ → α) (t : ℕ) : σ :=
  (omegaPrefix x t).foldl M.step M.init

@[simp] theorem omegaState_zero (M : StreamModel σ α β) (x : ℕ → α) :
    M.omegaState x 0 = M.init := rfl

/-- One state update per token, forever. -/
theorem omegaState_succ (M : StreamModel σ α β) (x : ℕ → α) (t : ℕ) :
    M.omegaState x (t + 1) = M.step (M.omegaState x t) (x t) := by
  simp [omegaState, omegaPrefix, List.foldl_append]

/-- **The infinite online output stream**: the shared readout applied after every
token of an infinite input, one output per token forever. -/
def omegaRun (M : StreamModel σ α β) (x : ℕ → α) : ℕ → β :=
  fun t => M.out (M.omegaState x (t + 1))

end StreamModel

open StreamModel

variable {σ α β : Type*}

/-- **The `ω` simulation engine.** A state map respecting the initial state, the
transitions and the readout produces *identical infinite output streams*. -/
theorem omegaRun_sim {τ : Type*} (M : StreamModel σ α β) (M' : StreamModel τ α β) (g : σ → τ)
    (hinit : g M.init = M'.init) (hstep : ∀ s x, g (M.step s x) = M'.step (g s) x)
    (hout : ∀ s, M.out s = M'.out (g s)) (x : ℕ → α) :
    M.omegaRun x = M'.omegaRun x := by
  have hstate : ∀ t, g (M.omegaState x t) = M'.omegaState x t := by
    intro t
    induction t with
    | zero => simpa using hinit
    | succ t ih => rw [M.omegaState_succ, M'.omegaState_succ, hstep, ih]
  funext t
  simp only [StreamModel.omegaRun]
  rw [← hstate (t + 1), hout]

/-- Any `ω`-acceptance condition — a predicate on the infinite output stream — is
invariant under stream equality. This is why nothing is lost by stating the results
for the acceptance-free total transducer: an added Büchi/Muller/parity condition is
carried along by every simulation result below. -/
theorem omegaRun_acceptance_invariant {P : (ℕ → β) → Prop} {M M' : StreamModel σ α β}
    {x : ℕ → α} (h : M.omegaRun x = M'.omegaRun x) :
    P (M.omegaRun x) ↔ P (M'.omegaRun x) := by
  rw [h]

/-! ## The abstraction collapse over `ω`-streams -/

/-- **Abstraction collapse over `ω`-streams.** A bounded-precision channel with a
surjective quotient over infinite input streams is simulated exactly by an
`n`-state `ω`-transducer: same infinite output stream. -/
theorem omegaRun_collapse {n : Nat} {M : StreamModel σ α β} (A : BoundedAbstraction M n)
    (hq : Surjective A.q) (x : ℕ → α) :
    (collapse A hq).omegaRun x = M.omegaRun x :=
  (omegaRun_sim M (collapse A hq) A.q rfl
    (fun s y => (collapse.step_apply A hq s y).symm)
    (fun s => (collapse.out_apply A hq s).symm) x).symm

/-- **Bounded precision over `ω`-streams is finite-state**: an `n`-valued channel
is simulated by an `n`-state `ω`-transducer, for either activation, at any
dimension. -/
theorem exists_fin_omega_model_of_abstraction {n : Nat} {M : StreamModel σ α β}
    (A : BoundedAbstraction M n) (hq : Surjective A.q) :
    ∃ M' : StreamModel (Fin n) α β, ∀ x t, M'.omegaRun x t = M.omegaRun x t :=
  ⟨collapse A hq, fun x t => congrFun (omegaRun_collapse A hq x) t⟩

/-! ## The operator claim at bounded precision: activation independence -/

/-- **The `ω`-program class is activation-independent at bounded precision.**
For every `n`:

* **Upper bound (either activation, any dimension).** Any `ω`-transducer whose state
  channel is `n`-valued with a surjective quotient — `LIN`- or `TANH`-shaped — has
  exactly the `ω`-behaviour of an `n`-state transducer.
* **Exact realization (both arms, dimension `n`).** Every `n`-state
  `ω`-behaviour is realized exactly, with one shared readout and no horizon or
  length dependence, by the `LIN(n)` one-hot model (`moore_realized_by_onehot_linear`)
  *and* by the `TANH(n)` saturated-corner model (`moore_realized_by_tanh_corner`).

So at bounded precision the two activation classes coincide on the `ω`-program
class: the operative diviser is the information the state channel carries, not the
activation. Neither conjunct supplies an ambient bounded abstraction (the upper
bound *assumes* one) and neither is a dimension lower bound. -/
theorem omega_class_activation_independent_at_bounded_precision (n : Nat) :
    (∀ (σ α β : Type*) (M : StreamModel σ α β) (A : BoundedAbstraction M n)
        (_hq : Surjective A.q) (x : ℕ → α),
        ∃ M' : StreamModel (Fin n) α β, ∀ t, M'.omegaRun x t = M.omegaRun x t) ∧
    (∀ (α β : Type*) (step : Fin n → α → Fin n) (init : Fin n) (outM : Fin n → β),
        (∃ M : StreamModel (Fin n → ℚ) α β,
            ∀ x t, M.omegaRun x t = (finModel step init outM).omegaRun x t) ∧
        (∃ M : StreamModel (Fin n → ℝ) α β,
            ∀ x t, M.omegaRun x t = (finModel step init outM).omegaRun x t)) := by
  refine ⟨?_, ?_⟩
  · intro σ α β M A hq x
    obtain ⟨M', hM'⟩ := exists_fin_omega_model_of_abstraction A hq
    exact ⟨M', hM' x⟩
  · intro α β step init outM
    constructor
    · -- the one-hot linear realization, transferred to `ω`-streams
      have hsel : ∀ s : Fin n,
          Classical.choose (show ∃ j, onehot j = onehot s from ⟨s, rfl⟩) = s := by
        intro s
        have hs : onehot (Classical.choose (show ∃ j, onehot j = onehot s from ⟨s, rfl⟩))
            = onehot s := Classical.choose_spec (show ∃ j, onehot j = onehot s from ⟨s, rfl⟩)
        exact onehot_injective hs
      refine ⟨finLinModel step init outM, fun x t => ?_⟩
      have h := omegaRun_sim (finModel step init outM) (finLinModel step init outM) (@onehot n)
        rfl (fun s y => (matVec_onehot step s y).symm)
        (fun s => by
          change outM s = outM (if h : ∃ j, onehot j = onehot s then Classical.choose h else init)
          rw [dif_pos ⟨s, rfl⟩, hsel s]) x
      exact (congrFun h t).symm
    · -- the saturated-corner `tanh` realization, transferred to `ω`-streams
      have hsel : ∀ s : Fin n,
          Classical.choose (show ∃ j, enc j = enc s from ⟨s, rfl⟩) = s := by
        intro s
        have hs : enc (Classical.choose (show ∃ j, enc j = enc s from ⟨s, rfl⟩)) = enc s :=
          Classical.choose_spec (show ∃ j, enc j = enc s from ⟨s, rfl⟩)
        exact enc_injective hs
      refine ⟨tanhCornerModel step init outM, fun x t => ?_⟩
      have h := omegaRun_sim (finModel step init outM) (tanhCornerModel step init outM) (@enc n)
        rfl (fun s y => (tanhCornerModel_step_enc step init outM s y).symm)
        (fun s => by
          change outM s = outM (if h : ∃ j, enc j = enc s then Classical.choose h else init)
          rw [dif_pos ⟨s, rfl⟩, hsel s]) x
      exact (congrFun h t).symm

/-! ## Unbounded precision over `ω`-streams still escapes finite state -/

/-- **No bounded-precision channel at any budget.** The precision boundary is the
absence of a finite state channel; `BoundedAbstraction` is data, not a proposition,
so this is a `Prop`-valued wrapper for "no `n`-valued channel exists". -/
def NoBoundedAbstraction {σ α β : Type*} (M : StreamModel σ α β) : Prop :=
  ∀ n, BoundedAbstraction M n → False

/-- **Distinct `ω`-outputs need unbounded state.** A model that emits `n + 1`
pairwise distinct outputs along one infinite input cannot have an `n`-valued state
channel. Activation-free and dimension-free: it quantifies over an arbitrary
`StreamModel`. -/
theorem omegaRun_needs_unbounded_state {σ α β : Type*} (M : StreamModel σ α β)
    (x : ℕ → α) (outv : ℕ → β) (hM : ∀ t, M.omegaRun x t = outv t)
    (hinj : Function.Injective outv) (n : Nat) (A : BoundedAbstraction M n) : False := by
  have hinjq : Function.Injective
      fun i : Fin (n + 1) => A.q (M.omegaState x ((i : Nat) + 1)) := by
    intro i j hij
    have h1 : M.omegaRun x (i : Nat) = M.omegaRun x (j : Nat) := A.out_congr _ _ hij
    rw [hM, hM] at h1
    exact Fin.ext (hinj h1)
  have hle : n + 1 ≤ n := by simpa using Fintype.card_le_of_injective _ hinjq
  omega

/-- Exact `LIN(1)` arithmetic counts forever: on the all-true `ω`-stream the online
readout reports `1, 2, 3, …`. -/
theorem countModel_omegaRun (t : ℕ) : countModel.omegaRun (fun _ => true) t = t + 1 := by
  have h : countModel.omegaState (fun _ => true) (t + 1) = t + 1 := by
    have h1 : countModel.omegaState (fun _ => true) (t + 1) = countModel.init + (t + 1) := by
      rw [StreamModel.omegaState, StreamModel.omegaPrefix_const]
      exact countModel_fold countModel.init (t + 1)
    simpa [countModel] using h1
  rw [StreamModel.omegaRun, h]
  rfl

/-- **Exact precision exceeds finite state over `ω`-streams**: the counting
transducer admits no `n`-valued state channel at any budget `n`. -/
theorem countModel_no_omega_finite_state : NoBoundedAbstraction countModel :=
  fun n A => omegaRun_needs_unbounded_state countModel (fun _ => true) (fun t => t + 1)
    countModel_omegaRun (fun _ _ h => Nat.succ_injective h) n A

/-- **The precision boundary over `ω`-streams, explicitly**: there is an
indefinitely-running `LIN(1)` program with no finite-state channel, at any budget.
This is a precision statement, not an activation separator: the same collapse
(`exists_fin_omega_model_of_abstraction`) covers `LIN` and `TANH` alike. -/
theorem exists_omega_model_outside_finite_state :
    ∃ M : StreamModel ℕ Bool ℕ, NoBoundedAbstraction M :=
  ⟨countModel, countModel_no_omega_finite_state⟩

end StreamingExpressivity
