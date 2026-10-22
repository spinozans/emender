# X20T — LIN(d) vs TANH(d) expressivity boundary

Operator-funded 2026-10-22 ("we want the expressivity to be clearly evaluated").
Formal receipt: `formal/lean/ElmanProofs/Expressivity/LinTanhExpressivity.lean`
(`lake build` green, no `sorry`; headline theorems depend only on
`propext`, `Classical.choice`, `Quot.sound`).

---

## Verdict

**COMPENSATION — and, in fact, LIN dominates.**  There is **no** TANH-favoring
separator of the asked form

> ∃ task family `F`, dimension `d₀`, such that some `TANH(d₀)` computes `F` while
> **no** `LIN(d)`, `d ≤ d₀`, with arbitrary encoders/decoders computes `F`.

It is false: at any fixed sequence horizon `T` over a finite alphabet, **`LIN(1)`**
— a *single* affine state coordinate — already realizes **every** function of the
length-`T` word, exactly, with an arbitrary readout. So for every `d₀` and every
target `F`, a `LIN(1)` readout (dimension `1 ≤ d₀`) computes the same `F`; the
premise "no LIN(d≤d₀) computes `F`" can never hold. The in-fold `tanh`
nonlinearity of the E97 arm is real, but it buys **no expressivity** over a linear
state channel once the encoder and decoder are arbitrary — it is a
*precision / parameter-efficiency* story, not a function-class gap at bounded
dimension.

(Consistent with the tracked constraints: S5 is not a separator — paper
`main.typ` lines 350–363; `S5GDNBridge.lean:49–80` disclaims lower bounds; the
trivial `T=1` second-difference witness is rejected because the decoder
compensates — the theorems below make that compensation exact and general.)

---

## The two objects

Fix a finite token alphabet and a horizon `T`; a word is `w : Fin T → α`.

* **LIN(d):** state `s ∈ ℝ^d`, `s_t = A(x_t) s_{t-1} + B(x_t)`, where `A`, `B`
  are *arbitrary* functions of the current token only (token-identifiable,
  state-independent) — the affine-fold structure of
  `CompositionalDepthDuality.gdnStep_eq_affine` /
  `gdn_state_channel_depth_bounded_in_time` (the fla-gdn arm). Readout
  `out = ψ(s_T)` with `ψ` arbitrary (the trained MLP/decoder).
* **TANH(d):** `s_t = tanh(A(x_t) s_{t-1} + B(x_t))`, same arbitrary encoders /
  decoders — the E97 arm (`StateActivation="tanh"`). Scalar step
  `LinTanhExpressivity.tanhStep A B s = Real.tanh (A·s + B)`.

Both are instances of the abstract `RecurrentModel σ α T` (a state type plus a
token-indexed one-step map); the compensation theorem quantifies over *any* such
model, so it applies to TANH at *any* dimension at once. In the Lean file the
`LIN(1)` realisation uses a state in `ℕ` (`linState`), a faithful stand-in for one
`ℝ`-coordinate carrying arbitrary precision (LIN(d) has no precision bound).

---

## What is proven (all in Lean, all sorry-free)

Read `RecurrentModel.stateAfter M w` = the fold of `M.step` over `w` from
`M.init`; `Realizes M ψ f := ∀ w, ψ (M.stateAfter w) = f w`.

1. **`readoutOf` / `readoutOf_apply`** — *the mechanism.* If the state map
   `S : Word → σ` is injective, then for **any** target `f` the readout
   `readoutOf S f` (which inverts `S` on its image) satisfies
   `readoutOf S f (S w) = f w`. Operationally: "arbitrary decoder" returns exactly
   what the state channel can injectively carry; it does not have to be able to
   *compute* the answer.

2. **`horner_mul`, `linState_fold_acc`, `linState_cons`, `linState_bound`** —
   the `LIN(1)` Horner fold `s ↦ b·s + x` over the digit alphabet `Fin b`
   (`b ≥ 2`):
   `linState b (x::xs) = x.val·b^{|xs|} + linState b xs` and
   `linState b l < b^{|l|}` (positional reading fits in `|l|` base-`b` places).

3. **`linState_inj`** — **`LIN(1)` is injective on equal-length words**: distinct
   base-`b` digit strings of the same length fold to distinct `LIN(1)` states.
   (Digit extraction via a bound-and-monotonicity argument, no division.)

4. **`lin_one_dim_realizes_any_function`** — **`LIN(1)` realizes every function of
   a length-`T` word.** Combine `linState_inj` (state map injective) with the
   mechanism (1). Dimension `1`, arbitrary readout, arbitrary target.

5. **`tanh_computations_are_lin_one_dim`** — **the compensation / no-separator
   theorem.** For **any** `RecurrentModel MT` over `Fin b` (in particular
   `TANH(d)` for any `d`) and **any** readout `ψT`, there is a readout `ψ` with
   `ψ ((linModel b T).stateAfter w) = ψT (MT.stateAfter w)` for all `w`.
   Hence every `TANH(d)`-computable function on length-`T` words is
   `LIN(1)`-computable. `tanh_scalar_model_compensated` is the concrete
   `tanhStep`-instantiated corollary.

6. **`lin_identity_token_preserves`** — the affine identity token `A=1, B=0` fixes
   the state exactly: exact latching against intervening/noise tokens needs no
   in-fold nonlinearity (see Corrections).

### Proof of the load-bearing step (summary)

`LIN(1)` with `s_t = b·s_{t-1} + x_t` computes the base-`b` positional integer of
the word: `s_T = x_1 b^{T-1} + x_2 b^{T-2} + ⋯ + x_T`. Positional encoding is
injective on length-`T` digit strings (digits `< b`), so the state separates all
`b^T` words; a decoder that inverts the encoding then emits any desired answer
`f(w)`. No `tanh`, no dimension `> 1`, no loss.

---

## What is NOT proven (honest labelling)

* **The dimension/precision gap as `T → ∞` at bounded state precision.** The
  `LIN(1)` universality uses a state whose magnitude grows like `b^T` (arbitrary
  precision). Under a *fixed* precision budget, `LIN(1)` cannot encode `b^T`
  words, and *both* `LIN(d)` and `TANH(d)` collapse to finite-state machines of
  bounded resolution. Whether, at a fixed *small* `d` and bounded precision, some
  task family is computable by `TANH(d)` but not `LIN(d)` is **open and not
  claimed here.** Full-state `tanh` saturates/renormalises per step, which could
  plausibly buy robustness or parameter-efficiency there; **no theorem in this
  development asserts or refutes such a gap.** This is the one honest residual and
  it is a *dimension-gap / efficiency* question, not a raw expressivity one.
* **Explicit `TANH(d)` realization of all fixed-horizon functions** (the symmetric
  "TANH is also universal at `d≈#states`" statement) is *not* separately
  formalized. It is unnecessary for the verdict: the compensation theorem already
  bounds `TANH` from above by `LIN(1)`. (It would follow from affine
  interpolation placing prefixes at affinely independent points in `(−1,1)^d` and
  `tanh`'s surjectivity — a standard, unformalized argument.)
* **Uniform-across-`T` architecture.** The Lean statements are per fixed horizon
  `T` (a parameter). If a task family requires one *shared* architecture across all
  `T`, the readout `ψ` may depend on `T` in the construction; the uniform version
  reduces to the precision caveat above and is not settled.
* The `ℕ` model of a `ℝ` state coordinate assumes "arbitrary precision real
  state", which is exactly what `LIN(d)`'s definition permits; it is *not* a claim
  about finite-precision floating point.

---

## Corrections to tracked theory

1. **"Affine state maps have no genuine attractors/bistability; tanh maps do"
   (task constraint 3) is not a task-relevant resource.** Per-token affine maps
   *can* contract to any chosen per-token fixed point
   (`s ↦ c·s + (1−c)·target`, `c<1`, or exactly `s ↦ s` for a null token), so a
   latch that must hold a value across intervening/noise tokens is affine-
   realizable — indeed *exactly*, via the identity transition
   (`lin_identity_token_preserves`). Genuine *bistability of a single map* exists
   only for `tanh`; but because the per-token map varies with the token, no
   word-level task can force a *single* map to be bistable. The
   attractor/bistability route therefore does **not** yield a separator.
2. **Constraint 1 (S5 is not a separator) is correct and can be strengthened.**
   Not only `S5` but *every* finite automaton up to `n` states is linear-realizable
   at dimension `≤ n` (one-hot / permutation basis; `S5` via the `5×5` permutation
   representation of `main.typ:350–363`). The present result strengthens this to
   the whole fixed-horizon function class: `LIN(1)` already captures *all* functions
   of a fixed-horizon word (unbounded precision).
3. **Constraint 2 (affine-fold theorem) is the reason no separator exists.** The
   combination "arbitrary `ψ` (constraint 2) + an injective state map" *is* the
   compensation. "Any LIN impossibility must survive `ψ`-arbitrariness" is thus
   not merely a constraint to respect — it is the proof that no impossibility
   exists at the function-class level. This is the (accurate) sharpening of the
   latching-sufficiency vs LIN-impossibility gap noted in constraint 3.

---

## X20b experiment spec

**Not written.** The task instruction is to write an X20b trainable task spec only
`if a separator survives`; none does, so no experiment is registered and **no
training was launched.** The only empirical question left open is the
precision/dimension-efficiency asymmetry above (e.g. whether `tanh` tracks a task
at a smaller `d` under bounded precision than `LIN`). That would be an
*efficiency* probe, not an expressivity separator, and is left to the operator.

## Reproduce

```
cd formal/lean && lake build ElmanProofs.Expressivity.LinTanhExpressivity
# headline check (standard axioms only, no sorry):
#   #print axioms LinTanhExpressivity.tanh_computations_are_lin_one_dim
```
