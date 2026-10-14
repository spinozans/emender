# GAPS-D2 — the S5 routing bridge (task D2)

Module: `formal/lean/ElmanProofs/Expressivity/S5GDNBridge.lean`
(builds clean, in the `PaperCore` trust-gate closure).

This note records exactly what the bridge proves, what it does **not** prove,
and the residual gaps. Read it before citing the module in the paper.

## What is proved

Headline: `S5GDNBridge.gdn_s5_realization_requires_per_token_action`.

Given a per-token GDN parameter schedule (token → gate/value/decay params) and
a decoder from the carried matrix state to `S5` that (i) reproduces
`S5Tracker.run` on every word and (ii) separates reachable states
(`GdnRealizesS5`), the theorem produces `F : S5 → MatState` with:

1. `gdnRun params w = F (run w)` for all words — the word-level state map
   **factors through the S5 element** (`gdn_realization_factors_through_s5`,
   `gdnRun_eq_s5StateOf`);
2. the per-token fold acts as **right multiplication** by the S5 element on the
   reachable image (`gdn_word_action_is_right_multiplication`);
3. each such action is **affine** (`A * · + C`), inherited from
   `CompositionalDepthDuality.gdn_state_channel_depth_bounded_in_time`
   (`gdn_word_action_affine`).

Supporting structural facts:

* `gdnStateFold_affine_compose`: composing two word-actions of the state channel
  returns an affine map — no nonlinear layer is added per word. Concrete form of
  "the GDN state channel has zero nonlinear temporal depth", versus E97's
  one-`tanh`-per-step (`e97_compositional_depth_grows_in_time`).
* `gdn_s5_constant_parameters_cannot_realize`: a schedule whose parameters do
  not depend on the token cannot realize the tracker — `[s01, s12]` and
  `[s12, s01]` collapse to the same GDN state but are distinguished by
  `S5Tracker.run` (`s5_run_s01s12_ne_s12s01`). So per-token dependence is
  necessary, not decorative.

## The exact paper claim this licenses

> "The theory shows that a GDN-family model can match the S5 tracker only by
> carrying the S5 transition-monoid action in its **per-token gate/value
> functions**: the recurrent state channel is provably affine in the carried
> state (zero nonlinear temporal depth, in contrast to E97's one `tanh` layer per
> step), and a token-independent schedule provably cannot track S5. Hence none of
> the S5 state-tracking can be delegated to the recurrence — it must be produced
> per token."

## Claims this does NOT license (do not cite)

* **No circuit-depth lower bound.** Nothing here proves the per-token gate/value
  network needs depth ≥ k, or any particular size/nonlinearity. The result is a
  *routing* statement (where the S5 dependence must live), not a *cost* statement
  (how deep/wide the per-token network must be). The task's originally-hoped
  headline `gdn_s5_realization_requires_per_token_depth` is stronger than what is
  proved and must not be used.
* **Not "only because of depth".** See gaps G1–G3.

## Gaps

**G1 — input-only gates may be arbitrarily shallow.** Each token is one of four
discrete symbols. A per-token gate function over a one-hot token encoding can be
a plain lookup / affine map (`W · onehot(token) + b`), i.e. depth 1. A per-token
*parameter* schedule therefore needs no depth on its own. So this module gives no
depth lower bound for the per-token network, and in the state-independent regime
there is no formal obstruction to a shallow realizer. What would be needed to
claim depth: a theorem that the S5 action **cannot** be realized by the
*constrained* GDN state channel (affine, with per-token transition of the
rank-one form `splitGatedTransition 1 k b · decay`), forcing nonlinearity — and
then a circuit-depth bound on the function that supplies it.

**G2 — state-independent regime only.** The affinity results (and hence the
factorization and the affine-action law) are proved for schedules whose
parameters are a function of the token alone — exactly the regime
`gdn_state_channel_depth_bounded_in_time` covers. If the gates are allowed to
read the carried state, the recurrence is no longer affine in the state and the
argument does not apply. That state-dependent (dynamic-routing) regime is **not
modeled here**; it is the natural place a nonlinearity could enter the state
path, and the place to look for any genuine depth/cost result.

**G3 — the algebraic realizability question is open.** Whether the constrained
affine state channel (`splitGatedTransition 1 k b · decay`, i.e. decay minus a
rank-one term) can realize a non-solvable group representation — in particular
the S5 regular action — is not settled. If it can, then GDN can track S5 with a
linear state channel and possibly zero per-token depth (refuting the literal
depth-escape reading); if it cannot, the obstruction is *algebraic* (rank-one
transition structure versus a non-solvable representation), not about time-depth.
Resolving G3 is the load-bearing next step for any depth-escape claim.

## Relationship to D1

`CompositionalDepthDuality` (task D1, commit `32cfcc9c`) proves the *time* axis
over the concrete one-step maps. This module imports it and reuses its GDN
definitions `GdnStepParams`, `gdnStep`, `gdnStateFold`, `gdnStep_eq_affine`, and
`gdn_state_channel_depth_bounded_in_time`; it adds the S5/`run`-level bridge, the
affine-composition lemma, and the token-independence impossibility. No D1 file is
modified.
