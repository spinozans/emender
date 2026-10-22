/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.LinTanhExpressivity
import Mathlib.Data.Fintype.Pigeonhole
import Mathlib.Data.Set.Card

/-!
# Streaming expressivity at bounded precision and unbounded horizon

X20T (`LinTanhExpressivity.lean`) proved that at every *fixed* horizon `T`,
`LIN(1)` with exact (unbounded-precision) scalar state and an arbitrary
T-dependent terminal readout realizes every function of the length-`T` word, so
no TANH-favouring separator exists at fixed `T`. That development isolated three
escapes for the *unbounded* horizon: (E1) precision, (E2) one decoder shared by
all lengths, (E3) online readout at every position. This file formalizes those
three escapes as objects and proves what is provable about them.

## The objects

`StreamModel σ α β` is a recurrent model with a **shared online readout**: one
`out : σ → β` applied after every token, so the model emits one output per input
symbol on words of *any* length. `LIN(d)` and `TANH(d)` are both instances; the
class-level statements below quantify over the abstract `StreamModel`, so they
apply to either activation at *any* dimension at once.

Bounded precision is formalized *semantically*, as a bound on how many values the
state channel can carry (`BoundedAbstraction`): a quotient `q : σ → Fin n`
through which all future behaviour and the readout factor. `B` bits of state
precision means `n = 2 ^ B`; the formalization deliberately does not model
rounding modes, only the information the channel carries.

## Load-bearing results (all proved, no `sorry`)

* `StreamingExpressivity.StreamModel.runFrom_sim` — the simulation engine: a
  state-level correspondence plus readout agreement gives identical online output
  streams. Every realization statement below is an instance.
* `exists_fin_model_of_abstraction` — **collapse**: a model whose state channel
  carries `n` values is simulated *exactly* (same online output at every position
  of every word) by a model with `n` states (`Fin n`), i.e. by a deterministic
  finite automaton with output. So at bounded precision both `LIN` and `TANH` are
  finite-state, whatever the dimension.
* `moore_realized_by_onehot_linear`, `finLinModel_reachable_onehot`,
  `onehot_entries` — **the converse at dimension `n`**: every `n`-state automaton
  is realized exactly by a *linear* state channel `s ↦ A(x) s` of dimension `n`
  whose reachable states are the one-hot vectors, i.e. coordinates in `{0, 1}` —
  one bit each. Precision is not the binding resource; dimension is.
* `bounded_precision_class_characterization` — the two together: at budget `n` the
  bounded-precision class *is* the class of `n`-state automata.
* `countModel_out` / `count_needs_unbounded_state`, `balanced_needs_unbounded_state`
  — **what bounded precision cannot do at any dimension or activation**: exact
  `LIN(1)` counts (`s ↦ s + 1`), while the counting task and the `aᵏ bᵏ`
  discrimination admit *no* state channel with finitely many values. This is the
  precision-collapse boundary in miniature.
* `lin_one_dim_no_three_cycle`, `tanh_one_dim_no_three_cycle` — at dimension 1
  *both* activations are stuck below a 3-state automaton (one-dimensional steps
  are monotone, so they have no 3-cycle); `rot3Model_three_cycle` exhibits the
  2-dimensional **linear** channel that does cycle three distinct states. The
  stratum is dimension, not activation.
* `tanhStreamModel_state_bounded` — a scalar `TANH(1)` state is confined to
  `(-1, 1)`, the bounded region that at fixed resolution is a finite value set.
* `tanh_one_dim_two_fixed_points` — a correction: scalar `tanh` can latch two
  values *exactly*, so "`tanh` cannot hold a value" is not a separator.

The literature anchors this development leans on are labelled verified-vs-secondary
in `experiments/expressivity_tasks/theory_x21t/THEOREMS.md`.
-/

namespace StreamingExpressivity

open Function
open LinTanhExpressivity

/-! ## Streaming models with a shared online readout -/

/-- A streaming recurrent model: a state channel (`init`, `step`) and a **single
shared online readout** `out : σ → β` applied after every token. `LIN(d)`,
`TANH(d)` and finite automata with output are all instances. -/
structure StreamModel (σ α β : Type*) where
  /-- Initial state. -/
  init : σ
  /-- Token-indexed one-step transition on the carried state. -/
  step : σ → α → σ
  /-- The shared per-position readout (the trained decoder). -/
  out : σ → β

namespace StreamModel

variable {σ α β : Type*}

/-- The state after reading a word — the fold shared by all prefixes. -/
def stateAfter (M : StreamModel σ α β) (w : List α) : σ :=
  w.foldl M.step M.init

variable (M : StreamModel σ α β)

/-- Outputs emitted while reading `w` from state `s`: exactly one per token. This
is the online/streaming readout: the same `M.out` is used at every position and
at every length. -/
def runFrom : σ → List α → List β
  | _, [] => []
  | s, x :: xs => M.out (M.step s x) :: runFrom (M.step s x) xs

/-- The full online output stream of `M` on the word `w`. -/
def run (w : List α) : List β := M.runFrom M.init w

@[simp] theorem runFrom_nil (s : σ) : M.runFrom s [] = [] := rfl

@[simp] theorem runFrom_cons (s : σ) (x : α) (xs : List α) :
    M.runFrom s (x :: xs) = M.out (M.step s x) :: M.runFrom (M.step s x) xs := rfl

theorem length_runFrom (s : σ) (w : List α) : (M.runFrom s w).length = w.length := by
  induction w generalizing s with
  | nil => rfl
  | cons x xs ih => simp [runFrom_cons, ih]

theorem length_run (w : List α) : (M.run w).length = w.length := M.length_runFrom M.init w

theorem stateAfter_append (u v : List α) :
    M.stateAfter (u ++ v) = v.foldl M.step (M.stateAfter u) := by
  simp [stateAfter, List.foldl_append]

/-- **The simulation engine.** If a state map `g` carries one model's states into
another's, commuting with the transitions and respecting the readout, then the two
models emit *identical online output streams* from corresponding states. Every
realization statement in this file is an instance. -/
theorem runFrom_sim {τ : Type*} (M' : StreamModel τ α β) (g : σ → τ)
    (hstep : ∀ s x, g (M.step s x) = M'.step (g s) x)
    (hout : ∀ s, M.out s = M'.out (g s)) (s : σ) (w : List α) :
    M.runFrom s w = M'.runFrom (g s) w := by
  induction w generalizing s with
  | nil => rfl
  | cons x xs ih =>
      have htail := ih (M.step s x)
      simp only [runFrom_cons, hstep, hout] at htail ⊢
      exact congrArg _ htail

end StreamModel

/-! ## Bounded precision: the state channel carries at most `n` values -/

/-- **Bounded precision at budget `n`.** The state channel is `n`-valued: a
quotient `q : σ → Fin n` determines both the next quotient state and the emitted
output. `B` bits of state precision is the case `n = 2 ^ B` (a state quantized to
`2^B` values, or confined to a bounded region at fixed resolution, always admits
such a `q`); the definition records the information the channel carries and
deliberately does not model rounding modes. -/
structure BoundedAbstraction {σ α β : Type*} (M : StreamModel σ α β) (n : Nat) where
  /-- The quotient through which the state channel factors. -/
  q : σ → Fin n
  /-- The transition is well defined on `q`-classes. -/
  step_congr : ∀ s s' : σ, q s = q s' → ∀ x, q (M.step s x) = q (M.step s' x)
  /-- The shared readout is well defined on `q`-classes. -/
  out_congr : ∀ s s' : σ, q s = q s' → M.out s = M.out s'

namespace BoundedAbstraction

variable {σ α β : Type*} {M : StreamModel σ α β} {n : Nat}

/-- Two states with the same `q`-value stay indistinguishable under any suffix. -/
theorem fold_congr (A : BoundedAbstraction M n) {s s' : σ} (h : A.q s = A.q s')
    (w : List α) : A.q (w.foldl M.step s) = A.q (w.foldl M.step s') := by
  induction w generalizing s s' with
  | nil => exact h
  | cons x xs ih => exact ih (A.step_congr s s' h x)

end BoundedAbstraction

/-! ## Collapse: an `n`-valued state channel is an `n`-state automaton -/

/-- The `n`-valued state channel of `A`, read as a finite automaton with output:
states `Fin n`, the transition taken through a representative of each class, and
the readout pushed to the quotient. -/
noncomputable def collapse {σ α β : Type*} {M : StreamModel σ α β} {n : Nat}
    (A : BoundedAbstraction M n) (hq : Surjective A.q) : StreamModel (Fin n) α β where
  init := A.q M.init
  step := fun i x => A.q (M.step (Classical.choose (hq i)) x)
  out := fun i => M.out (Classical.choose (hq i))

namespace collapse

variable {σ α β : Type*} {M : StreamModel σ α β} {n : Nat}

theorem step_apply (A : BoundedAbstraction M n) (hq : Surjective A.q) (s : σ) (x : α) :
    (collapse A hq).step (A.q s) x = A.q (M.step s x) := by
  change A.q (M.step (Classical.choose (hq (A.q s))) x) = A.q (M.step s x)
  exact A.step_congr _ _ (Classical.choose_spec (hq (A.q s))) x

theorem out_apply (A : BoundedAbstraction M n) (hq : Surjective A.q) (s : σ) :
    (collapse A hq).out (A.q s) = M.out s :=
  A.out_congr _ _ (Classical.choose_spec (hq (A.q s)))

end collapse

/-- **Collapse theorem.** A model whose state channel carries only `n` values is
simulated *exactly* — same online output at every position of every word — by a
model with `n` finite states. At bounded precision `LIN(d)` and `TANH(d)` are
therefore both finite-state; the dimension `d` no longer separates anything at the
class level. -/
theorem exists_fin_model_of_abstraction {σ α β : Type*} {M : StreamModel σ α β} {n : Nat}
    (A : BoundedAbstraction M n) (hq : Surjective A.q) :
    ∃ M' : StreamModel (Fin n) α β, ∀ w, M'.run w = M.run w := by
  refine ⟨collapse A hq, fun w => ?_⟩
  have h := StreamModel.runFrom_sim (M := M) (M' := collapse A hq) A.q
    (fun s x => (collapse.step_apply A hq s x).symm) (fun s => (collapse.out_apply A hq s).symm)
    M.init w
  change (collapse A hq).runFrom (A.q M.init) w = M.runFrom M.init w
  exact h.symm

/-! ## Converse at dimension `n`: every finite automaton is an exact one-hot
linear state channel with one-bit coordinates -/

/-- The one-hot state vector at index `j`. -/
def onehot {n : Nat} (j : Fin n) : Fin n → ℚ := fun i => if i = j then 1 else 0

theorem onehot_injective {n : Nat} : Injective (@onehot n) := by
  intro j j' h
  have h1 : (if j = j' then (1 : ℚ) else 0) = 1 := by
    simpa [onehot] using congrFun h j
  by_cases hjj : j = j'
  · exact hjj
  · rw [if_neg hjj] at h1
    exact absurd h1 (by norm_num)

/-- Every coordinate of a one-hot state is `0` or `1`: one bit each. -/
theorem onehot_entries {n : Nat} (j i : Fin n) : onehot j i = 0 ∨ onehot j i = 1 := by
  by_cases h : i = j
  · exact Or.inr (by simp [onehot, h])
  · exact Or.inl (by simp [onehot, h])

/-- The one-hot states are exactly `n` of them. -/
theorem onehot_range_ncard {n : Nat} : (Set.range (@onehot n)).ncard = n := by
  rw [Set.ncard_range_of_injective onehot_injective, Nat.card_fin]

/-- Matrix–vector product: one `LIN` step with token-dependent matrix `A x`. -/
def matVec {n : Nat} (A : Fin n → Fin n → ℚ) (s : Fin n → ℚ) : Fin n → ℚ :=
  fun i => ∑ j, A i j * s j

/-- The `0/1` matrix of a transition sends one-hot vectors to one-hot vectors. This
is the whole content of "a linear state channel can implement a finite automaton at
finite precision": the entries never leave `{0, 1}`. -/
theorem matVec_onehot {n : Nat} {α : Type*} (step : Fin n → α → Fin n) (j : Fin n) (x : α) :
    matVec (fun i j => if step j x = i then (1 : ℚ) else 0) (onehot j) = onehot (step j x) := by
  funext i
  rw [matVec, Finset.sum_eq_single j]
  · by_cases h : step j x = i
    · simp [onehot, h]
    · have h2 : ¬ i = step j x := fun hh => h hh.symm
      simp [onehot, h, h2]
  · intro b _ hb
    simp [onehot, hb]
  · intro h
    exact absurd (Finset.mem_univ j) h

/-- The `n`-state automaton `step`/`init`/`outM` realized as a *linear* model of
dimension `n`: state `s ↦ A(x) s` with the `0/1` matrix above, no bias, and the
shared readout reading the one-hot index back. -/
noncomputable def finLinModel {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) : StreamModel (Fin n → ℚ) α β where
  init := onehot init
  step := fun s x => matVec (fun i j => if step j x = i then (1 : ℚ) else 0) s
  out := fun s => outM (if h : ∃ j, onehot j = s then Classical.choose h else init)

/-- The finite model that `finLinModel` linearizes. -/
abbrev finModel {n : Nat} {α β : Type*} (step : Fin n → α → Fin n) (init : Fin n)
    (outM : Fin n → β) : StreamModel (Fin n) α β := ⟨init, step, outM⟩

/-- `finLinModel`'s state channel is the one-hot embedding of the automaton state. -/
theorem finLinModel_fold {n : Nat} {α β : Type*} (step : Fin n → α → Fin n) (init : Fin n)
    (outM : Fin n → β) (s : Fin n) (w : List α) :
    w.foldl (finLinModel step init outM).step (onehot s) = onehot (w.foldl step s) := by
  induction w generalizing s with
  | nil => rfl
  | cons x xs ih =>
      simp only [List.foldl_cons]
      rw [show (finLinModel step init outM).step (onehot s) x = onehot (step s x) from
        matVec_onehot step s x]
      exact ih (step s x)

theorem finLinModel_stateAfter {n : Nat} {α β : Type*} (step : Fin n → α → Fin n) (init : Fin n)
    (outM : Fin n → β) (w : List α) :
    (finLinModel step init outM).stateAfter w = onehot ((finModel step init outM).stateAfter w) :=
  finLinModel_fold step init outM init w

/-- Every reachable state of the linear model is one-hot — `n` values, one bit per
coordinate. -/
theorem finLinModel_reachable_onehot {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) (w : List α) :
    ∃ j, (finLinModel step init outM).stateAfter w = onehot j :=
  ⟨_, finLinModel_stateAfter step init outM w⟩

/-- **Every finite automaton is an exact linear model at dimension `n`.** The
online output streams agree at every position of every word, with a single shared
readout. So the whole of the `n`-state bounded-precision class is reached by
`LIN(n)` — and by anything at least as expressive. -/
theorem moore_realized_by_onehot_linear {n : Nat} {α β : Type*} (step : Fin n → α → Fin n)
    (init : Fin n) (outM : Fin n → β) :
    ∀ w, (finLinModel step init outM).run w = (finModel step init outM).run w := by
  intro w
  have hsel : ∀ s : Fin n, Classical.choose (show ∃ j, onehot j = onehot s from ⟨s, rfl⟩) = s := by
    intro s
    have hs : onehot (Classical.choose (show ∃ j, onehot j = onehot s from ⟨s, rfl⟩)) = onehot s :=
      Classical.choose_spec (show ∃ j, onehot j = onehot s from ⟨s, rfl⟩)
    exact onehot_injective hs
  have h := StreamModel.runFrom_sim (M := finModel step init outM)
    (M' := finLinModel step init outM) (@onehot n)
    (fun s x => (matVec_onehot step s x).symm)
    (fun s => by
      change outM s = outM (if h : ∃ j, onehot j = onehot s then Classical.choose h else init)
      rw [dif_pos ⟨s, rfl⟩, hsel s])
    init w
  change (finLinModel step init outM).runFrom (onehot init) w
    = (finModel step init outM).runFrom init w
  exact h.symm

/-- **Characterization at budget `n`.** The online stream functions computed by
models whose state channel carries at most `n` values — the bounded-precision
`LIN(d,B)` and `TANH(d,B)` classes with `2 ^ B = n`, at *any* dimension — are
exactly the functions computed by `n`-state finite automata; and the linear side
realizes all of them at dimension `n` with one-bit coordinates. -/
theorem bounded_precision_class_characterization (n : Nat) :
    (∀ (σ α β : Type*) (M : StreamModel σ α β) (A : BoundedAbstraction M n)
        (_hq : Surjective A.q), ∃ M' : StreamModel (Fin n) α β, ∀ w, M'.run w = M.run w) ∧
    (∀ (α β : Type*) (step : Fin n → α → Fin n) (init : Fin n) (outM : Fin n → β),
        ∃ M : StreamModel (Fin n → ℚ) α β,
          (∀ w, M.run w = (finModel step init outM).run w) ∧
          (∀ w, ∃ j, M.stateAfter w = onehot j)) := by
  refine ⟨fun σ α β M A hq => exists_fin_model_of_abstraction A hq, fun α β step init outM => ?_⟩
  exact ⟨finLinModel step init outM, moore_realized_by_onehot_linear step init outM,
    finLinModel_reachable_onehot step init outM⟩


/-! ## What bounded precision cannot do at any dimension: counting

The two theorems of this section are **activation-free**: they quantify over an
arbitrary `StreamModel`, so they hold for `LIN(d)` and `TANH(d)` at every
dimension and every precision budget. They are the formal content of the
precision-collapse boundary: exact linear recurrences count (`countModel_out`),
bounded-precision ones cannot. -/

/-- `LIN(1)` over `ℕ` that counts the `true` tokens: token maps `s ↦ s + 1` and
`s ↦ s`, both affine with `A = 1`, exact unbounded-precision state. -/
def countModel : StreamModel ℕ Bool ℕ where
  init := 0
  step := fun s x => if x then s + 1 else s
  out := fun s => s

/-- The fold of the count model adds one per `true` token. -/
theorem countModel_fold (s k : ℕ) :
    (List.replicate k true).foldl countModel.step s = s + k := by
  induction k generalizing s with
  | zero => simp
  | succ k ih =>
      rw [List.replicate_succ, List.foldl_cons]
      have h1 : countModel.step s true = s + 1 := by simp [countModel]
      rw [h1, ih (s + 1)]
      omega

/-- The exact count is online-readable at every prefix length. -/
theorem countModel_out (k : ℕ) :
    countModel.out (countModel.stateAfter (List.replicate k true)) = k := by
  have h : countModel.stateAfter (List.replicate k true)
      = (List.replicate k true).foldl countModel.step 0 := rfl
  rw [h, countModel_fold]
  simp [countModel]

/-- **Bounded precision cannot count.** A model whose online readout reports
`n + 1` distinct counts on the prefixes `a⁰, a¹, …, aⁿ` cannot have an `n`-valued
state channel. No dimension, no activation, no readout helps: the counting
information is simply not in the channel. -/
theorem count_needs_unbounded_state {σ β : Type*} (M : StreamModel σ Bool β)
    (outv : ℕ → β) (hM : ∀ k, M.out (M.stateAfter (List.replicate k true)) = outv k)
    (hinj : Function.Injective outv) (n : Nat) (A : BoundedAbstraction M n) : False := by
  have hinjq : Function.Injective
      fun i : Fin (n + 1) => A.q (M.stateAfter (List.replicate (i : Nat) true)) := by
    intro i j hij
    have h1 : M.out (M.stateAfter (List.replicate (i : Nat) true)) =
        M.out (M.stateAfter (List.replicate (j : Nat) true)) := A.out_congr _ _ hij
    rw [hM, hM] at h1
    exact Fin.ext (hinj h1)
  have hle : n + 1 ≤ n := by simpa using Fintype.card_le_of_injective _ hinjq
  omega

/-- The `aᵏbᵏ` discrimination demands of the balanced-block streaming task — the
end-of-word readout requirements of "accept exactly the prefixes of the form
`aᵏbᵏ`": the state at the end of `aᵏbᵏ` must read `yes`, and at the end of
`aᵏ'bᵏ` (`k' ≠ k`) must read `no`. These are necessary conditions for computing
that task, and they are inconsistent with any finite state channel. -/
structure SeparatesBalanced {σ β : Type*} (M : StreamModel σ Bool β) (yes no : β) : Prop where
  /-- The balanced words are accepted. -/
  hits : ∀ k, 1 ≤ k →
    M.out (M.stateAfter (List.replicate k true ++ List.replicate k false)) = yes
  /-- A wrong number of `b`s is rejected. -/
  misses : ∀ k k', 1 ≤ k → 1 ≤ k' → k' ≠ k →
    M.out (M.stateAfter (List.replicate k' true ++ List.replicate k false)) = no

/-- **Bounded precision cannot count-and-compare.** Any state channel that
separates `aᵏbᵏ` from `aᵏ'bᵏ` needs an unbounded number of values, whatever the
dimension, activation, or readout: pairwise different `k` demand pairwise
distinguishable states and the channel only has `n`. -/
theorem balanced_needs_unbounded_state {σ β : Type*} (M : StreamModel σ Bool β)
    {yes no : β} (hs : SeparatesBalanced M yes no) (hne : yes ≠ no) (n : Nat)
    (A : BoundedAbstraction M n) : False := by
  have hinjq : Function.Injective
      fun i : Fin (n + 1) => A.q (M.stateAfter (List.replicate ((i : Nat) + 1) true)) := by
    intro i j hij
    by_contra hneij
    have hk : 1 ≤ (i : Nat) + 1 := Nat.succ_le_succ (Nat.zero_le _)
    have hk' : 1 ≤ (j : Nat) + 1 := Nat.succ_le_succ (Nat.zero_le _)
    have hkk' : (j : Nat) + 1 ≠ (i : Nat) + 1 := by
      intro h
      exact hneij (Fin.ext (by omega))
    have hout : M.out (M.stateAfter (List.replicate ((i : Nat) + 1) true ++
          List.replicate ((i : Nat) + 1) false)) =
        M.out (M.stateAfter (List.replicate ((j : Nat) + 1) true ++
          List.replicate ((i : Nat) + 1) false)) := by
      apply A.out_congr
      rw [M.stateAfter_append, M.stateAfter_append]
      exact A.fold_congr hij (List.replicate ((i : Nat) + 1) false)
    rw [hs.hits _ hk, hs.misses _ _ hk hk' hkk'] at hout
    exact hne hout
  have hle : n + 1 ≤ n := by simpa using Fintype.card_le_of_injective _ hinjq
  omega

/-! ## The stratum is dimension, not activation

A one-dimensional state channel moves its single coordinate by a map of the form
`s ↦ a·s + b` (linear) or `s ↦ tanh (a·s + b)` (saturated). Both are monotone,
antitone, or constant — and **no such map has a 3-cycle**. So at dimension 1 both
classes are strictly weaker than a 3-state automaton; a two-dimensional *linear*
channel already cycles three states (`rot3Model_three_cycle`). -/

/-- **No weakly monotone or weakly antitone self-map of `ℝ` has a 3-cycle.** Every
one-dimensional state step studied below — `s ↦ a·s + b` and `s ↦ tanh (a·s + b)`
— is of this form, and such a map can only have fixed points and 2-cycles. -/
theorem no_three_cycle_of_mono_or_antitone (f : ℝ → ℝ)
    (hmono : (∀ x y, x < y → f x ≤ f y) ∨ (∀ x y, x < y → f y ≤ f x))
    {v0 v1 v2 : ℝ} (h01 : f v0 = v1) (h12 : f v1 = v2) (h20 : f v2 = v0)
    (hd01 : v0 ≠ v1) (hd12 : v1 ≠ v2) (hd20 : v2 ≠ v0) : False := by
  rcases hmono with hinc | hdec
  · rcases lt_trichotomy v0 v1 with h | h | h
    · have h1 : v1 ≤ v2 := by simpa [h01, h12] using hinc v0 v1 h
      have h1' : v1 < v2 := lt_of_le_of_ne h1 hd12
      have h2 : v2 ≤ v0 := by simpa [h12, h20] using hinc v1 v2 h1'
      linarith
    · exact hd01 h
    · have h1 : v2 ≤ v1 := by simpa [h12, h01] using hinc v1 v0 h
      have h1' : v2 < v1 := lt_of_le_of_ne h1 (Ne.symm hd12)
      have h2 : v0 ≤ v2 := by simpa [h20, h12] using hinc v2 v1 h1'
      have h2' : v0 < v2 := lt_of_le_of_ne h2 (Ne.symm hd20)
      linarith
  · rcases lt_trichotomy v0 v1 with h | h | h
    · have h1 : v2 ≤ v1 := by simpa [h12, h01] using hdec v0 v1 h
      have h1' : v2 < v1 := lt_of_le_of_ne h1 (Ne.symm hd12)
      have h2 : v2 ≤ v0 := by simpa [h20, h12] using hdec v2 v1 h1'
      have h2' : v2 < v0 := lt_of_le_of_ne h2 hd20
      have h3 : v1 ≤ v0 := by simpa [h01, h20] using hdec v2 v0 h2'
      linarith
    · exact hd01 h
    · have h1 : v1 ≤ v2 := by simpa [h01, h12] using hdec v1 v0 h
      have h1' : v1 < v2 := lt_of_le_of_ne h1 hd12
      have h2 : v0 ≤ v2 := by simpa [h12, h20] using hdec v1 v2 h1'
      have h2' : v0 < v2 := lt_of_le_of_ne h2 (Ne.symm hd20)
      have h3 : v0 ≤ v1 := by simpa [h20, h01] using hdec v0 v2 h2'
      have h3' : v0 < v1 := lt_of_le_of_ne h3 hd01
      linarith

/-- A one-dimensional affine map `s ↦ a·s + b` is weakly monotone or weakly
antitone (constant when `a = 0`). -/
theorem affine_mono_or_antitone (a b : ℝ) :
    (∀ x y : ℝ, x < y → a * x + b ≤ a * y + b) ∨
      (∀ x y : ℝ, x < y → a * y + b ≤ a * x + b) := by
  rcases lt_trichotomy a 0 with ha | ha | ha
  · exact Or.inr fun x y h => by nlinarith
  · subst ha
    exact Or.inl fun x y _ => by simp
  · exact Or.inl fun x y h => by nlinarith

/-- A scalar saturated map `s ↦ tanh (a·s + b)` is weakly monotone or weakly
antitone (constant when `a = 0`, since `tanh` is strictly monotone). -/
theorem tanhStep_mono_or_antitone (a b : ℝ) :
    (∀ x y : ℝ, x < y → tanhStep a b x ≤ tanhStep a b y) ∨
      (∀ x y : ℝ, x < y → tanhStep a b y ≤ tanhStep a b x) := by
  have hkey : ∀ x y : ℝ, tanhStep a b x < tanhStep a b y ↔ a * x + b < a * y + b :=
    fun x y => Activation.tanh_strictMono.lt_iff_lt
  rcases lt_trichotomy a 0 with ha | ha | ha
  · refine Or.inr fun x y h => ?_
    have : a * y + b < a * x + b := by nlinarith
    exact le_of_lt ((hkey y x).mpr this)
  · subst ha
    exact Or.inl fun x y _ => by simp [tanhStep]
  · refine Or.inl fun x y h => ?_
    have : a * x + b < a * y + b := by nlinarith
    exact le_of_lt ((hkey x y).mpr this)

/-- **`LIN(1)` cannot cycle three states**: no one-dimensional affine state
channel `s ↦ a·s + b` has a 3-cycle, at any precision and with any readout. -/
theorem lin_one_dim_no_three_cycle (a b v0 v1 v2 : ℝ)
    (h01 : a * v0 + b = v1) (h12 : a * v1 + b = v2) (h20 : a * v2 + b = v0)
    (hd01 : v0 ≠ v1) (hd12 : v1 ≠ v2) (hd20 : v2 ≠ v0) : False :=
  no_three_cycle_of_mono_or_antitone (fun s => a * s + b) (affine_mono_or_antitone a b)
    h01 h12 h20 hd01 hd12 hd20

/-- **`TANH(1)` cannot cycle three states either**: saturation is monotone, so the
scalar saturated channel has only fixed points and 2-cycles. -/
theorem tanh_one_dim_no_three_cycle (a b v0 v1 v2 : ℝ)
    (h01 : tanhStep a b v0 = v1) (h12 : tanhStep a b v1 = v2) (h20 : tanhStep a b v2 = v0)
    (hd01 : v0 ≠ v1) (hd12 : v1 ≠ v2) (hd20 : v2 ≠ v0) : False :=
  no_three_cycle_of_mono_or_antitone (tanhStep a b) (tanhStep_mono_or_antitone a b)
    h01 h12 h20 hd01 hd12 hd20

/-- The 2-dimensional **linear** channel `(x, y) ↦ (-y, x - y)`; its matrix has
characteristic polynomial `x² + x + 1`, so it has order three. -/
def rot3Pair (p : ℚ × ℚ) : ℚ × ℚ := (-p.2, p.1 - p.2)

/-- Order three: `T³ = id`. -/
theorem rot3Pair_cube (p : ℚ × ℚ) : rot3Pair (rot3Pair (rot3Pair p)) = p := by
  obtain ⟨x, y⟩ := p
  simp [rot3Pair]
  ring_nf

/-- A two-dimensional linear state channel with a single token that cycles three
distinct states: dimension 2 clears the stratum dimension 1 provably cannot. -/
def rot3Model : StreamModel (ℚ × ℚ) Unit (ℚ × ℚ) where
  init := (1, 0)
  step := fun s _ => rot3Pair s
  out := id

/-- **Dimension 2 suffices for a 3-cycle** — the three states are explicitly
distinct, so this needs just two bits of state description. -/
theorem rot3Model_three_cycle :
    rot3Model.stateAfter [(), (), ()] = (1, 0) ∧
      rot3Model.stateAfter [()] = (0, 1) ∧
      rot3Model.stateAfter [(), ()] = (-1, -1) := by
  refine ⟨?_, ?_, ?_⟩ <;> simp [rot3Model, StreamModel.stateAfter, rot3Pair]

/-! ## The `TANH` side lands in its bounded region -/

/-- Scalar `TANH(1)` as a *streaming* model: one real coordinate, `tanh` in the
fold, one shared readout. -/
noncomputable def tanhStreamModel {α β : Type*} (A B : α → ℝ) (ψ : ℝ → β) :
    StreamModel ℝ α β where
  init := 0
  step := fun s x => tanhStep (A x) (B x) s
  out := ψ

/-- Folding `tanh` steps over a nonempty suffix ends at a `tanh` value. -/
private theorem fold_tanh {α : Type*} (A B : α → ℝ) :
    ∀ (s : ℝ) (l : List α), l ≠ [] →
      ∃ c, l.foldl (fun s x => tanhStep (A x) (B x) s) s = Real.tanh c
  | _, [], h => absurd rfl h
  | s, [x], _ => ⟨A x * s + B x, rfl⟩
  | s, x :: y :: ys, _ =>
      fold_tanh A B (tanhStep (A x) (B x) s) (y :: ys) (by simp)

/-- **A scalar `TANH(1)` state is confined to `(-1, 1)`** after any nonempty word:
the bounded region that a fixed-resolution representation turns into a finite value
set (`BoundedAbstraction`). This is the concrete `TANH`-side hypothesis, in contrast
to the unbounded-range `LIN(1)` state. -/
theorem tanhStreamModel_state_bounded {α β : Type*} (A B : α → ℝ) (ψ : ℝ → β)
    (w : List α) (hw : w ≠ []) :
    |(tanhStreamModel A B ψ).stateAfter w| < 1 := by
  obtain ⟨c, hc⟩ := fold_tanh A B 0 w hw
  have h : (tanhStreamModel A B ψ).stateAfter w = Real.tanh c := by
    simpa [StreamModel.stateAfter, tanhStreamModel] using hc
  rw [h]
  exact Activation.tanh_bounded c

/-! ## A correction: a scalar saturated channel *can* latch several values exactly

The tempting separator "linear can hold a value, `tanh` cannot" fails: `s ↦ tanh(a·s)`
can have several exact fixed points, so a scalar `TANH(1)` model can latch values
exactly, as a linear one can. -/

/-- `TANH(1)` has two exact fixed points at once (`0` and `tanh 1`): perfect
latching of two values needs no affine identity token. -/
theorem tanh_one_dim_two_fixed_points :
    ∃ a b : ℝ, ∃ t : ℝ, t ≠ 0 ∧ tanhStep a b 0 = 0 ∧ tanhStep a b t = t := by
  have ht : 0 < Real.tanh 1 := Activation.tanh_pos_of_pos zero_lt_one
  refine ⟨(Real.tanh 1)⁻¹, 0, Real.tanh 1, ht.ne', ?_, ?_⟩
  · simp [tanhStep, Real.tanh_zero]
  · change Real.tanh ((Real.tanh 1)⁻¹ * Real.tanh 1 + 0) = Real.tanh 1
    rw [inv_mul_cancel₀ ht.ne', add_zero]

end StreamingExpressivity
