/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Expressivity.CompositionalDepthDuality
import ElmanProofs.Architectures.S5Inseparability

/-!
# Routing bridge: a GDN-family tracker realizer must carry S5 per token

`CompositionalDepthDuality` proved the *time* axis of the E97/GDN duality over
the concrete one-step maps: E97 applies an elementwise `tanh` at every time
step (compositional depth grows with sequence length), while the GDN/gated-delta
state channel is affine in the carried state — `H_T = A * H_0 + C` with `A`, `C`
fixed by the per-token gate/decay/value schedule
(`gdn_state_channel_depth_bounded_in_time`).

This module turns that structural fact into a statement about *task behavior*:
what a GDN-family model would have to do to realize the S5 adjacent-transposition
tracker (`S5Tracker.run`), the canonical non-solvable word problem used
throughout the paper.

## What is proved here

Fix a per-token GDN parameter schedule (the model's gate/value/decay functions
evaluated on each of the four adjacent-generator tokens) and a decoder from the
carried matrix state to `S5`.

* `gdnRun` is the state produced by running the schedule over a word from the
  zero state.
* `GdnRealizesS5` packages a schedule and a decoder that recovers
  `S5Tracker.run` on every word, and whose decoder *separates the reachable
  states* (a satisfiable injectivity-on-the-image condition; it is the only
  thing that can be asked of a decoder into the 120-element `S5` from the
  uncountable matrix state space).

The headline `gdn_s5_realization_requires_per_token_action` then states that any
such realizer induces a function `F : S5 → MatState` with

* `gdnRun params w = F (S5Tracker.run w)` for every word — i.e. the word-level
  state map **factors through the S5 element**; the per-token parameter family
  realizes a well-defined action of the S5 transition monoid on the state;
* the action law: running the schedule for a suffix `v` from `F g` returns
  `F (g * run v)` — the per-token fold implements *right multiplication* by
  `run v` on the reachable image;
* every such right-multiplication action is an **affine** map on the state
  (`A * · + C`), inherited from `gdn_state_channel_depth_bounded_in_time`.

`gdn_s5_constant_parameters_cannot_realize` is the complementary negative
result: a schedule whose per-token parameters do not depend on the token cannot
realize the tracker at all, because two length-2 words with the same parameter
sequence but different S5 products collapse to the same state.

## What is *not* proved here — read this before citing

These theorems are a **routing** statement, not a circuit-depth lower bound.

* We do **not** prove that the per-token gate/value network requires any
  particular depth, size, or nonlinearity. The bridge shows only *where* the S5
  state-dependence must live: since the state channel introduces no nonlinear
  temporal composition (`gdnStateFold_affine_compose`: composing two word-actions
  returns an affine map with no additional nonlinear layer), and since
  token-independent parameters are impossible
  (`gdn_s5_constant_parameters_cannot_realize`), the entire S5 action must be
  produced *per token* by the gate/value functions.
* The affinity results hold for **state-independent** schedules — the per-token
  parameters are a function of the token alone. This is exactly the regime D1's
  `gdn_state_channel_depth_bounded_in_time` covers. If the gates are allowed to
  read the carried state, the recurrence is no longer affine in the state and
  the affinity argument does not apply; that state-dependent regime is *not*
  modeled here.
* Because each token is one of four discrete symbols, an input-only per-token
  gate function can in principle be a plain lookup / affine map over a one-hot
  token encoding, so nothing above forces the per-token network to be deep. The
  precise cost is documented in `formal/lean/GAPS-D2.md`.

Consequently the honest paper claim is a routing claim: "a GDN-family model can
match the tracker only by carrying the S5 transition-monoid action in its
per-token functions; its recurrent state channel is provably affine (zero
nonlinear temporal depth), so none of the S5 state-tracking can be delegated to
the recurrence." It is **not** the claim "the per-token network must have depth
≥ k".
-/

namespace S5GDNBridge

open Matrix BigOperators
open M2RNNComparison
open CompositionalDepthDuality

variable {K V : Nat}

/-! ## Per-token GDN schedules and their runs -/

/-- A per-token GDN parameter schedule for the S5 tracker: the GDN one-step
gate/value/decay parameters produced for each of the four adjacent-generator
tokens. This is the *state-independent* regime: the parameters are a function of
the token alone. -/
abbrev GdnSchedule (K V : Nat) :=
  S5Tracker.AdjacentGenerator → GdnStepParams K V

/-- The GDN matrix state produced by running a per-token schedule over a word,
starting from the zero state. -/
def gdnRun (params : GdnSchedule K V) (w : List S5Tracker.AdjacentGenerator) :
    MatState K V :=
  gdnStateFold (w.map params) 0

/-- Folding a concatenated parameter schedule splits at the prefix. -/
theorem gdnStateFold_append (s t : List (GdnStepParams K V)) (H : MatState K V) :
    gdnStateFold (s ++ t) H = gdnStateFold t (gdnStateFold s H) := by
  induction s generalizing H with
  | nil => rfl
  | cons p ps ih => exact ih (gdnStep p H)

/-- **The GDN run over a concatenated word splits at the prefix.** This is the
word-level form of `gdnStateFold_append`: processing `u ++ v` is processing `v`
from the state reached after `u`. -/
theorem gdnRun_append (params : GdnSchedule K V) (u v : List S5Tracker.AdjacentGenerator) :
    gdnRun params (u ++ v) = gdnStateFold (v.map params) (gdnRun params u) := by
  unfold gdnRun
  rw [List.map_append]
  exact gdnStateFold_append (u.map params) (v.map params) 0

/-! ## The state channel composes affinely (no nonlinear layer per word) -/

/-- **Composing two word-actions of a GDN state channel returns an affine map
with no additional nonlinear layer.** If folding a schedule `s` is the affine
map `A * · + C` and folding `t` is `B * · + D`, then folding `s ++ t` is
`(B * A) * · + (B * C + D)`. The linear parts multiply and the constants add;
no elementwise nonlinearity is introduced by longer words. This is the concrete
sense in which the GDN state channel has zero nonlinear temporal depth, in
contrast with the E97/E88 update, which adds one `tanh` layer per time step
(`CompositionalDepthDuality.e97_compositional_depth_grows_in_time`). -/
theorem gdnStateFold_affine_compose
    (A B : Matrix (Fin K) (Fin K) Real) (C D : MatState K V)
    (s t : List (GdnStepParams K V))
    (hs : ∀ H, gdnStateFold s H = A * H + C)
    (ht : ∀ H, gdnStateFold t H = B * H + D) :
    ∀ H, gdnStateFold (s ++ t) H = (B * A) * H + (B * C + D) := by
  intro H
  rw [gdnStateFold_append, ht, hs, Matrix.mul_add, ← Matrix.mul_assoc]
  abel

/-- Every word-action of a GDN state channel is affine in the initial state.
Direct specialization of `CompositionalDepthDuality.gdn_state_channel_depth_bounded_in_time`
to the parameter schedule induced by a word. -/
theorem gdn_word_action_affine (params : GdnSchedule K V)
    (v : List S5Tracker.AdjacentGenerator) :
    ∃ A : Matrix (Fin K) (Fin K) Real, ∃ C : MatState K V,
      ∀ init : MatState K V, gdnStateFold (v.map params) init = A * init + C :=
  gdn_state_channel_depth_bounded_in_time (v.map params)

/-! ## GDN realizers of the S5 tracker -/

/-- A GDN-family model realizes the S5 tracker: a per-token parameter schedule,
a decoder from the carried matrix state to `S5`, and the realization equation
`decode (gdnRun params w) = S5Tracker.run w` on every word.

`separates` is the injectivity-on-the-reachable-image condition: the decoder must
not glue together two reachable states. It is the strongest injectivity that can
be asked of a map from the uncountable matrix state space into the 120-element
`S5`, and it is what makes the factorization theorem below go through. -/
structure GdnRealizesS5 (K V : Nat) where
  /-- Per-token GDN gate/value/decay schedule. -/
  params : GdnSchedule K V
  /-- Decoder from the carried matrix state to `S5`. -/
  decode : MatState K V → S5Tracker.State
  /-- Realization: the decoded run reproduces the S5 tracker on every word. -/
  realizes : ∀ w : List S5Tracker.AdjacentGenerator, decode (gdnRun params w) = S5Tracker.run w
  /-- The decoder separates reachable states. -/
  separates : ∀ u v : List S5Tracker.AdjacentGenerator,
    decode (gdnRun params u) = decode (gdnRun params v) → gdnRun params u = gdnRun params v

/-- **The word-level GDN state map factors through the S5 element.** Two words
with the same S5 product — e.g. two spellings of the same permutation, or the
two words of the relation `s01 s12 ≠ s12 s01` — reach the same GDN state. This
is candidate (a)'s core: exact realization forces the per-token parameter family
to realize the transition-monoid action, since distinct words cannot be
distinguished by the state channel beyond their S5 product. -/
theorem gdn_realization_factors_through_s5 (R : GdnRealizesS5 K V)
    (u v : List S5Tracker.AdjacentGenerator) :
    S5Tracker.run u = S5Tracker.run v → gdnRun R.params u = gdnRun R.params v := by
  intro h
  apply R.separates
  rw [R.realizes u, R.realizes v, h]

/-- The canonical state assigned to an S5 element, read off from a chosen
representative word (`S5Inseparability.wordOf`, which exists because
`S5Tracker.run` is surjective). -/
noncomputable def s5StateOf (R : GdnRealizesS5 K V) (g : S5Tracker.State) :
    MatState K V :=
  gdnRun R.params (S5Inseparability.wordOf g)

/-- The word-level state is the canonical function of the S5 element alone. -/
theorem gdnRun_eq_s5StateOf (R : GdnRealizesS5 K V) (w : List S5Tracker.AdjacentGenerator) :
    gdnRun R.params w = s5StateOf R (S5Tracker.run w) := by
  unfold s5StateOf
  exact gdn_realization_factors_through_s5 R w
    (S5Inseparability.wordOf (S5Tracker.run w))
    (S5Inseparability.run_wordOf (S5Tracker.run w)).symm

/-- **The per-token fold implements right multiplication by the S5 element.**
Running the schedule for a suffix `v` from the state of `run u` returns the state
of `run u * run v`: the GDN per-token maps realize the right S5-action on the
reachable state image. -/
theorem gdn_word_action_is_right_multiplication (R : GdnRealizesS5 K V)
    (u v : List S5Tracker.AdjacentGenerator) :
    gdnStateFold (v.map R.params) (s5StateOf R (S5Tracker.run u)) =
      s5StateOf R (S5Tracker.run u * S5Tracker.run v) := by
  rw [← gdnRun_eq_s5StateOf R u, ← gdnRun_append R.params u v, gdnRun_eq_s5StateOf,
    S5Tracker.run_append]

/-! ## Headline routing bridge -/

/-- **`gdn_s5_realization_requires_per_token_action`.**

A GDN-family model that realizes the S5 tracker induces a function
`F : S5 → MatState` such that:

1. the word-level state map factors through the S5 element
   (`gdnRun params w = F (run w)`) — the per-token parameter family must realize
   the S5 transition-monoid action;
2. the per-token fold acts as right multiplication by the S5 element on the
   reachable image;
3. every such action is affine in the state (`A * · + C`), because the state
   channel carries no nonlinear temporal composition.

Reading: the recurrent state channel is affine (zero nonlinear temporal depth);
token-independent parameters are impossible
(`gdn_s5_constant_parameters_cannot_realize`); therefore the whole S5 action must
be routed through the per-token gate/value functions. See the module docstring
for the precise limits of this claim (it is a routing statement, not a
circuit-depth lower bound). -/
theorem gdn_s5_realization_requires_per_token_action (R : GdnRealizesS5 K V) :
    ∃ F : S5Tracker.State → MatState K V,
      (∀ w : List S5Tracker.AdjacentGenerator, gdnRun R.params w = F (S5Tracker.run w)) ∧
      (∀ u v : List S5Tracker.AdjacentGenerator,
        gdnStateFold (v.map R.params) (F (S5Tracker.run u)) =
          F (S5Tracker.run u * S5Tracker.run v)) ∧
      (∀ v : List S5Tracker.AdjacentGenerator,
        ∃ A : Matrix (Fin K) (Fin K) Real, ∃ C : MatState K V,
          ∀ init : MatState K V, gdnStateFold (v.map R.params) init = A * init + C) := by
  refine ⟨s5StateOf R, gdnRun_eq_s5StateOf R, ?_, ?_⟩
  · intro u v
    exact gdn_word_action_is_right_multiplication R u v
  · intro v
    exact gdn_word_action_affine R.params v

/-! ## Token-independence is impossible

The positive routing result is only meaningful if per-token dependence is
genuinely required. The two length-2 words `[s01, s12]` and `[s12, s01]` have
different S5 products — the adjacent transpositions do not commute — but a
schedule that ignores the token produces the same parameter list for both. -/

/-- The adjacent transpositions `(0 1)` and `(1 2)` do not commute: the S5
tracker produces different states for `[s01, s12]` and `[s12, s01]`. -/
theorem s5_run_s01s12_ne_s12s01 :
    S5Tracker.run [S5Tracker.AdjacentGenerator.s01, S5Tracker.AdjacentGenerator.s12] ≠
      S5Tracker.run [S5Tracker.AdjacentGenerator.s12, S5Tracker.AdjacentGenerator.s01] := by
  intro h
  have h0 := congrArg (fun p : S5Tracker.State => p (0 : Fin 5)) h
  simp [S5Tracker.run, S5Tracker.runFrom, S5Tracker.step, S5Tracker.transposition,
    Equiv.swap_apply_def, Equiv.swap_apply_left] at h0

/-- **A GDN schedule with token-independent parameters cannot realize the S5
tracker.** If every token maps to the same one-step parameters, the two words
`[s01, s12]` and `[s12, s01]` yield the same GDN state, but the S5 tracker
distinguishes them; hence no decoder can recover `S5Tracker.run`. Combined with
`gdn_s5_realization_requires_per_token_action` this shows the per-token
dependence is not decorative: without it the state channel alone provably
fails. -/
theorem gdn_s5_constant_parameters_cannot_realize
    (params : GdnSchedule K V) (hconst : ∃ p : GdnStepParams K V, ∀ g, params g = p) :
    ¬ ∃ decode : MatState K V → S5Tracker.State,
      ∀ w : List S5Tracker.AdjacentGenerator, decode (gdnRun params w) = S5Tracker.run w := by
  rintro ⟨decode, hdec⟩
  obtain ⟨p, hp⟩ := hconst
  have hmap1 :
      ([S5Tracker.AdjacentGenerator.s01, S5Tracker.AdjacentGenerator.s12] :
        List S5Tracker.AdjacentGenerator).map params = [p, p] := by
    simp [hp]
  have hmap2 :
      ([S5Tracker.AdjacentGenerator.s12, S5Tracker.AdjacentGenerator.s01] :
        List S5Tracker.AdjacentGenerator).map params = [p, p] := by
    simp [hp]
  have hstate :
      gdnRun params [S5Tracker.AdjacentGenerator.s01, S5Tracker.AdjacentGenerator.s12] =
        gdnRun params [S5Tracker.AdjacentGenerator.s12, S5Tracker.AdjacentGenerator.s01] := by
    unfold gdnRun
    rw [hmap1, hmap2]
  have hrun :
      S5Tracker.run [S5Tracker.AdjacentGenerator.s01, S5Tracker.AdjacentGenerator.s12] =
        S5Tracker.run [S5Tracker.AdjacentGenerator.s12, S5Tracker.AdjacentGenerator.s01] := by
    rw [← hdec [S5Tracker.AdjacentGenerator.s01, S5Tracker.AdjacentGenerator.s12],
      ← hdec [S5Tracker.AdjacentGenerator.s12, S5Tracker.AdjacentGenerator.s01], hstate]
  exact s5_run_s01s12_ne_s12s01 hrun

end S5GDNBridge
