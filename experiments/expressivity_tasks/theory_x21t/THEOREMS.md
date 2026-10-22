# X21T — Streaming expressivity: unbounded horizon at bounded precision

Operator-funded 2026-10-22, immediately after accepting X20T ("unbounded time
horizon? how can we explore THAT?").

Formal receipt: `formal/lean/ElmanProofs/Expressivity/StreamingExpressivity.lean`
(`lake build` green, no `sorry`; every headline theorem depends only on `propext`,
`Classical.choice`, `Quot.sound`). **No training was launched and no experiment was
registered.**

X20T proved that at each *fixed* horizon `T`, `LIN(1)` with exact state and a
`T`-dependent terminal decoder realizes every function of the word, so no
TANH-favouring separator exists at fixed `T`. It left three escapes for unbounded
`T`: **(E1)** precision, **(E2)** one decoder shared by all lengths, **(E3)**
online readout at every position. This file formalizes those escapes as objects
and settles the bounded-precision branch. **A true small theorem beats an
ambitious unproven claim**: everything below is proved, and every open item is
labelled open.

---

## Verdict per question

**Q1 — characterization: CHARACTERIZATION (both directions proved).**
At bounded precision with budget `n` (`n = 2^B`, the state channel carries at most
`n` distinct values), the class of online stream functions computed by `S-LIN(d,B)`
— at *every* dimension `d` — is exactly the class computed by `n`-state
deterministic automata with output. Both halves are theorems:

* `exists_fin_model_of_abstraction` (`⊆`): an `n`-valued state channel is simulated
  *exactly*, with identical output at every position of every word, by an `n`-state
  model.
* `moore_realized_by_onehot_linear` (`⊇`): every `n`-state automaton is realized
  exactly — one shared readout, all lengths — by a **linear** state channel
  `s ↦ A(x)·s` of dimension `n` with `0/1` matrix entries, whose reachable states
  are the one-hot vectors (each coordinate `0` or `1`, i.e. **one bit**).

Consequently `bounded_precision_class_characterization` says: at budget `n` the
bounded-precision class *is* `n`-state finite automata, and the dimension needed is
at most `n`. **The class parameter is the budget, not `d`; and, surprisingly, the
binding resource is dimension, not precision** — a deliberate choice of code makes
every coordinate one bit.

Landing spots, stated precisely:

* bounded-precision `LIN` = bounded-precision `TANH` ⊆ `n`-state automata
  (`n = 2^B`), because both are governed by the collapse theorem;
* exact-arithmetic, unbounded-range `LIN(1)` is *strictly larger*: it counts
  (`countModel_out`: `s ↦ s + 1` on `a`, `s ↦ s` on `b`, exact count read online),
  while **no** model with an `n`-valued channel can count past `n`
  (`count_needs_unbounded_state`). This is the precision-collapse boundary in
  miniature, and it is what makes the classical (Schützenberger) rational-series
  picture different from the bounded-precision one.

**Q2 — separation or collapse: COLLAPSE at the class level; SEPARATION only as a
dimension hierarchy.**
As *classes of functions*, bounded-precision `LIN` and `TANH` collapse to the same
finite-state world at each budget (`⊆` above is activation-free; `⊇` shows the
linear side reaches all of it). No `s`-vs-activation separator can exist at fixed
budget — the X20T compensation lesson survives in the bounded-precision regime,
just relocated from the decoder to the state count. What *does* separate is
dimension, and this is proved in both directions at small `d`:

* `no_three_cycle_of_mono_or_antitone` + `lin_one_dim_no_three_cycle` +
  `tanh_one_dim_no_three_cycle`: at `d = 1` the step is `s ↦ a·s + b` (linear) or
  `s ↦ tanh(a·s + b)` (saturated); both are weakly monotone or weakly antitone, and
  **neither can cycle three states**. So at `d = 1` *both* classes are strictly
  weaker than a 3-state automaton.
* `rot3Pair_cube` + `rot3Model_three_cycle`: the two-dimensional **linear** channel
  `(x, y) ↦ (−y, x − y)` (matrix with characteristic polynomial `x² + x + 1`) cycles
  the three states `(1,0) → (0,1) → (−1,−1) → (1,0)` exactly, so `d = 2` suffices.

A genuine `LIN`-vs-`TANH` **activation** separation at equal `(d, B)` is **OPEN**
(Q2's strict form is unproven). Two natural witnesses were *constructed and
refuted* rather than asserted, per the X20T discipline:

* *latching* ("linear can hold a value, `tanh` cannot"): **refuted** —
  `tanh_one_dim_two_fixed_points` exhibits `a, b` with `s ↦ tanh(a·s + b)` fixing
  both `0` and `tanh 1` exactly, so a scalar `TANH(1)` model latches two values
  exactly, exactly as `LIN(1)` does.
* *counting* (`aᵏbᵏ`): **refuted for both** — `balanced_needs_unbounded_state`
  shows the discrimination is impossible at *any* dimension for *any* activation,
  because it needs unbounded state. It cannot be a `TANH`-only witness.
* *renormalization* has no formal handle here: at fixed resolution both sides are
  finite-state, so any witness must be a statement about *which* automata a given
  `(d, B)` reaches — i.e. a minimal-dimension question (open, see O3).

**Q3 — impossibility for both: BOTH-IMPOSSIBLE (proved), with a sharp survival
statement.**
`count_needs_unbounded_state` and `balanced_needs_unbounded_state` are
activation-free and dimension-free: the counting task and the `aᵏbᵏ` discrimination
are impossible for `S-LIN(d,B)` **and** `S-TANH(d,B)` at every `d` and every budget.
At X-series-like shapes (`d ≈ 32×32 = 1024`, `B ≈ bf16 = 16`, so `n = 2^16`), the
theory says exactly this:

* **What survives an arbitrary horizon shift**: every task computed by a fixed
  automaton with at most `d = 1024` states — `moore_realized_by_onehot_linear` is
  horizon-free (all lengths, one shared readout, one bit per coordinate), so such a
  task is stable at *every* length. Empirically "champions execute 4× beyond the
  training horizon" is *consistent* with this (a learned finite automaton is
  horizon-free), but that is an interpretation, not a measurement, and nothing here
  predicts it.
* **What dies**: anything requiring unbounded counting — precisely the counters
  that X20T-style exact arithmetic would have allowed.

---

## The objects

`StreamModel σ α β` (`formal/.../StreamingExpressivity.lean`): initial state,
token-indexed one-step map, and **one shared online readout** `out : σ → β` applied
after every token. `StreamModel.run` emits one output per input symbol at every
length; `StreamModel.stateAfter` is the shared prefix fold. `LIN(d)` and `TANH(d)`
are instances, and the class-level theorems quantify over the abstract structure,
so they cover either activation at any dimension at once.

`BoundedAbstraction M n`: the state channel is `n`-valued — a quotient
`q : σ → Fin n` that determines both the next quotient state and the emitted
output. `B` bits of precision is the case `n = 2^B`. **This is a semantic model of
bounded precision** (how many distinguishable values the channel carries, i.e.
"state confined to a bounded region at fixed resolution"). It does **not** model
rounding modes, and it does **not** model floating point with a bounded mantissa
but unbounded exponent (see O4 — under that different precision model the
hypothesis of every impossibility theorem below fails, and the class is larger).

Scalar instances: `countModel` (exact `LIN(1)` counter over `ℕ`),
`tanhStreamModel` (`TANH(1)` streaming), `rot3Model` (2-dimensional linear
3-cycle), `finLinModel` (the one-hot linearization of an arbitrary `n`-state
automaton).

---

## Literature anchors (verified vs secondary)

Every anchor the argument leans on, with its verification status in this session
(2026-10-22). "Verified" = I read the statement myself in the source or its
official abstract/summary page; "secondary" = reported by someone else, or from
memory, and *not* checked against the original.

| # | Anchor | Status | What it is used for |
|---|--------|--------|---------------------|
| 1 | Merrill, *Sequential Neural Networks as Automata*, arXiv:1906.01615 | **verified** (abstract + text: §3 Theorem 3.2 `L(SRN) = RL`; LSTMs strictly stronger, counter machines; "bounded precision" defined via asymptotic/saturated acceptance with `O(log n)` bits) | the saturated/tanh-RNN ≙ regular anchor for Q1; the counter axis for Q2 |
| 2 | Merrill et al., *A Formal Hierarchy of RNN Architectures*, arXiv:2004.08500 | **verified** (abstract: hierarchy from space complexity + "rational recurrence"; LSTM is not rational; builds on saturated-RNN theory) | the "rational recurrence" axis; activation differences at bounded precision |
| 3 | Weiss, Goldberg, Yahav, *On the Practical Computational Power of Finite Precision RNNs for Language Recognition*, arXiv:1805.04908 | **verified** (abstract: under finite precision + linear time, LSTM and ReLU-RNN are *strictly stronger* than squashing-activation RNN and GRU, via counting; and "RNNs are famously known to be Turing complete, this relies on infinite precision") | the float-like precision separation and the unbounded-precision attribution, O4 |
| 4 | Schützenberger 1961, *On the definition of a family of automata* | **verified at definition/attribution level** (weighted-automaton definition and attribution read on the Wikipedia *Weighted automaton* article; original not read) | exact-arithmetic linear recurrence = weighted automaton / linear representation, the classical name for the unbounded-precision `LIN(d)` side |
| 5 | Schützenberger's theorem: star-free = aperiodic = counter-free | **verified at summary level** (Wikipedia *Star-free language* article; original not read) | the sub-regular (counter-free vs counters) axis of Q2 |
| 6 | Delétang et al., *Neural Networks and the Chomsky Hierarchy*, arXiv:2207.02098 | **verified** (abstract: RNNs/Transformers fail to generalize on non-regular tasks; LSTMs solve regular and counter-language tasks) | empirical corroboration of the Q2/Q3 axes |
| 7 | Siegelmann & Sontag 1995 (JCSS), *On the computational power of neural nets*, and the super-Turing line | **secondary** — original not accessible (ScienceDirect 403; no open PDF located); the framing I rely on ("Turing completeness depends on unbounded precision") is the *verified* sentence in anchor 3 | the precision-collapse framing only; their super-Turing direction is **not** used or claimed |
| 8 | "the indicator series of `aⁿbⁿ` is rational but not regular" | **secondary** (not checked against a source) | not needed: the corresponding separation is *proved* here (`countModel_out` + `count_needs_unbounded_state`) |

Net effect: nothing in the Lean development depends on a secondary source. The
secondary entries supply vocabulary and context; the theorems stand on their own.

---

## What is proven (all in Lean, all sorry-free)

1. `StreamModel.runFrom_sim` — **the engine.** A state map commuting with the
   transitions and respecting the readout gives *identical online output streams*.
   Every realization statement is an instance.
2. `StreamModel.length_run`, `StreamModel.stateAfter_append` — the streaming
   bookkeeping (one output per token; the shared prefix fold).
3. `BoundedAbstraction.fold_congr` — states with the same `q`-value stay
   indistinguishable under *any* suffix.
4. `exists_fin_model_of_abstraction` — **collapse.** `n`-valued channel ⇒ simulated
   exactly by an `n`-state model (`Fin n`), same online output at every position.
5. `onehot_injective`, `onehot_entries`, `onehot_range_ncard` — the one-hot code:
   injective, exactly `n` states, **one bit per coordinate**.
6. `matVec_onehot` — the `0/1` transition matrix sends one-hot to one-hot; entries
   never leave `{0, 1}`. *This is the whole content of "a linear channel
   implements an automaton at finite precision".*
7. `finLinModel_fold`, `finLinModel_stateAfter`, `finLinModel_reachable_onehot` —
   the linear model's state *is* the one-hot vector of the automaton state, at
   every prefix.
8. `moore_realized_by_onehot_linear` — **every `n`-state automaton is an exact
   linear model of dimension `n`**, one shared readout, all word lengths.
9. `bounded_precision_class_characterization` — 4 and 8 combined: at budget `n`,
   bounded-precision streams *are* `n`-state automata, reached by `LIN(n)`.
10. `countModel_fold`, `countModel_out` — exact `LIN(1)` counts online
    (`s ↦ s + 1` / `s ↦ s`), unbounded precision.
11. `count_needs_unbounded_state` — a model reporting `n + 1` distinct counts
    cannot have an `n`-valued channel (pigeonhole on `Fin (n+1) → Fin n`).
12. `balanced_needs_unbounded_state` — same for the `aᵏbᵏ` discrimination
    (`SeparatesBalanced`): accept `aᵏbᵏ`, reject `aᵏ'bᵏ` for `k' ≠ k` is
    inconsistent with any finite state channel. **Activation-free, dimension-free.**
13. `no_three_cycle_of_mono_or_antitone` — a weakly monotone/antitone self-map of
    `ℝ` has no 3-cycle (proved by the two orientation cases; a 1-D map has only
    fixed points and 2-cycles).
14. `affine_mono_or_antitone`, `tanhStep_mono_or_antitone` — the two scalar steps
    are weakly monotone or antitone (`a > 0`, `a < 0`, or constant at `a = 0`;
    for `tanh` this uses strict monotonicity of `Real.tanh`).
15. `lin_one_dim_no_three_cycle`, `tanh_one_dim_no_three_cycle` — **at `d = 1` both
    activations are strictly below a 3-state automaton.**
16. `rot3Pair_cube`, `rot3Model_three_cycle` — **dimension 2 suffices** for a
    3-cycle, with a *linear* (`ℚ`-rational) channel and the explicit states.
17. `tanhStreamModel_state_bounded` — a scalar `TANH(1)` state stays in `(-1, 1)`
    after any nonempty word: the bounded region that fixed resolution discretizes.
18. `tanh_one_dim_two_fixed_points` — a correction: scalar `tanh` latches two values
    *exactly* (`a = 1/tanh 1`, `b = 0`, fixed points `0` and `tanh 1`).

Axiom check (`#print axioms` on 10 headline theorems): only `propext`,
`Classical.choice`, `Quot.sound` — no `sorryAx`, nothing exotic.

---

## Corrections to tracked theory and to the task's own candidate frames

1. **"Precision is the wall" is wrong in this model.** `moore_realized_by_onehot_linear`
   realizes the entire bounded-precision class with coordinates in `{0, 1}` — one
   bit each. Once the code is chosen deliberately, bounded precision costs nothing;
   what costs something is **dimension**: you need `d ≥` (number of automaton
   states), and counting needs unbounded `d`. So the honest scarce resources are
   state count/dimension, not mantissa bits. (X20T's E1 escape is thus real but
   almost free to close *given* enough dimension — the interesting version is
   "bounded `d` *and* bounded `B`".)
2. **E2 (shared decoder) dissolves at bounded precision.** `moore_realized_by_onehot_linear`
   exhibits *one* readout serving all lengths and all positions, so no
   horizon-dependent decoder is needed in this regime. E2 only bites for exact
   positional codes (X20T's `b`-adic fold has cross-length collisions like `[0]`
   vs `[0,0]` and needs a `T`-dependent readout).
3. **E3 (online readout) is fully in scope here.** Every theorem above is about the
   whole per-position output stream, not a terminal function; `length_run` records
   that one output is emitted per token.
4. **"`tanh` cannot latch / has no exact fixed point" is false.** See
   `tanh_one_dim_two_fixed_points`. The intuition only holds for a *single*
   contraction; `s ↦ tanh(a·s + b)` can have three fixed points.
5. **X20T's residual "dimension/precision gap as `T → ∞`" now has a definite shape**
   in the bounded-range precision model: at the *class* level it is a **collapse**
   (both finite-state), and the surviving gap is a **dimension** gap, provably
   present already at `d = 1` vs `d = 2`.
6. **The `aᵏbᵏ`-as-`TANH`-witness frame must be rejected.** It is impossible for
   *both* classes (`balanced_needs_unbounded_state`), so it can never separate them.
   Same for any latching witness. The X20T lesson (reject by construction any
   witness the decoder or a second coordinate can compensate) applies verbatim to
   every candidate in Q2's list.

---

## Open list (honest)

* **O1 — activation separation at equal `(d, B)`.** Open. Any witness must survive
  the shared readout, the two-dimensional escapes, and `tanh`'s multi-fixed-point
  latching (corrections 4 and 6). I prove no such separator and I explicitly do not
  claim one.
* **O2 — does `S-TANH(d,B)` reach *all* `n`-state automata?** Unknown. Collapse
  gives `TANH ⊆ finite-state`; the *converse* (exact realization of an arbitrary
  automaton by `tanh` steps at finite `d`) is a nonlinear-systems question and is
  not addressed. Equality of the `TANH` class with the `n`-state class is therefore
  **not** established; only the `LIN` side is proved to reach all of it.
* **O3 — minimal dimension.** Proved: `d = 1` cannot 3-cycle (both activations),
  `d = 2` can (`LIN`), `d = n` suffices for any `n`-state automaton. A general lower
  bound `d ≥ f(n)` growing with the automaton is not proved.
* **O4 — a different precision model.** Floating point with bounded mantissa and
  *unbounded exponent* (or "saturating vs linear accumulator") is a larger class
  where `BoundedAbstraction` hypotheses fail; anchor 3 (verified) proves a
  separation there — LSTM/ReLU-RNN strictly stronger than squashing RNN/GRU via
  counting. Nothing in this file models numeric error, so this file says nothing
  about that class. Formalizing it would need an explicit error/rounding model.
* **O5 — learnability, efficiency, average case.** Untouched. Nothing here explains
  why one arm ignites faster; no training was launched.
* **O6 — unbounded precision with a genuinely shared decoder over all lengths and
  online readout at the same time.** X20T left this open; the bounded-precision
  branch is what this file settles. The exact-arithmetic uniform-decoder question
  remains open exactly as in X20T.

No X21b task spec is written and no experiment is registered: the only empirical
question left is an *efficiency* probe (minimal `d` for a given automaton under a
fixed budget, `tanh` vs linear), which is an operator decision, not an expressivity
separator.

---

## Reproduce

```bash
cd formal/lean
lake build ElmanProofs.Expressivity.StreamingExpressivity   # green, no sorry
# headline axiom check (standard axioms only):
cat > /tmp/x21t_ax.lean <<'EOF'
import ElmanProofs.Expressivity.StreamingExpressivity
open StreamingExpressivity
#print axioms bounded_precision_class_characterization
#print axioms balanced_needs_unbounded_state
#print axioms tanh_one_dim_no_three_cycle
#print axioms rot3Model_three_cycle
EOF
lake env lean /tmp/x21t_ax.lean
```
