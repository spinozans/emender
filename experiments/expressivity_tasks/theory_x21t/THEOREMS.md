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
and establishes the bounded-precision results scoped below. **A true small theorem
beats an ambitious unproven claim**: formal results, interpretations, and open
items are distinguished explicitly.

---

## Verdict per question

**Q1 — surjective abstraction collapse plus finite-reachable one-hot realization.**
Both activation classes admit the finite-automaton upper bound under the stated abstraction hypothesis. Every n-state automaton has a dimension-n LINEAR realization. No TANH converse or equality at fixed dimension is established.

* `exists_fin_model_of_abstraction`: given `BoundedAbstraction M n` and a
  **surjective** quotient, an `n`-state model simulates the online stream exactly.
* `moore_realized_by_onehot_linear`: every `n`-state automaton is realized
  exactly — one shared readout, all lengths — by a **linear** state channel
  `s ↦ A(x)·s` of dimension `n` with `0/1` matrix entries, whose reachable states
  are one-hot (each coordinate `0` or `1`, i.e. **one bit**).

The formal headline for `bounded_precision_class_characterization` is therefore
**surjective abstraction collapse plus finite-reachable one-hot realization**,
not a stronger characterization. Its reverse conjunct proves stream equivalence
and reachable one-hot states but supplies no bounded abstraction on the ambient
state space. For a two-state swapping automaton with distinct outputs, the ambient
`finLinModel` has three distinguishable states: the two one-hot vectors and zero.
Zero stays zero and always produces the fallback output; swapping distinguishes
it from the corresponding one-hot state. Thus the ambient model need not admit
a two-valued abstraction. Likewise, `rot3Model.out = id` prevents any finite
abstraction on its entire rational-pair state space, despite its finite
initialized orbit. An invariant-domain/reachable-state bridge is not proved.

Landing spots, stated precisely:

* Both activation classes admit the finite-automaton upper bound under the stated abstraction hypothesis. Every n-state automaton has a dimension-n LINEAR realization. No TANH converse or equality at fixed dimension is established.
* exact-arithmetic, unbounded-range `LIN(1)` is *strictly larger*: it counts
  (`countModel_out`: `s ↦ s + 1` on `a`, `s ↦ s` on `b`, exact count read online),
  while **no** model with an `n`-valued channel can report `n + 1` distinct counts
  (`count_needs_unbounded_state`). This is a finite-state upper-bound separation,
  not an identification of arbitrarily decoded LINEAR streams with rational series.

**Q2 — finite-automaton upper bound; literal state-cycle realization separation.**
Both activation classes admit the finite-automaton upper bound under the stated abstraction hypothesis. Every n-state automaton has a dimension-n LINEAR realization. No TANH converse or equality at fixed dimension is established.

The dimension comparison below concerns **literal state-cycle realizations**
only, not automaton/stream-function expressivity:

* `no_three_cycle_of_mono_or_antitone` + `lin_one_dim_no_three_cycle` +
  `tanh_one_dim_no_three_cycle`: at `d = 1` the step is `s ↦ a·s + b` (linear) or
  `s ↦ tanh(a·s + b)` (saturated); both are weakly monotone or weakly antitone, and
  **neither can literally cycle three distinct real states**. This is not a
  restriction on observable three-cycle behavior with arbitrary readout.
* `rot3Pair_cube` + `rot3Model_three_cycle`: the two-dimensional **linear** channel
  `(x, y) ↦ (−y, x − y)` (matrix with characteristic polynomial `x² + x + 1`) cycles
  the three states `(1,0) → (0,1) → (−1,−1) → (1,0)` exactly, so `d = 2` suffices
  for a literal state cycle. Lean proves the cube identity and explicit orbit;
  the characteristic-polynomial calculation is explanatory algebra.

The reason for this restriction is the counterexample: scalar LINEAR transition
`s ↦ s + 1`, initial state zero, with `q(s) = floor(s) mod 3` and `out = q`.
This is a surjective three-valued abstraction and emits `1,2,0,…` at `d = 1`.
Its real state never cycles, but its observable behavior implements the three-cycle
automaton under arbitrary readout. A stronger hierarchy requires additional
hypotheses and proof.

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
  `(d, B)` reaches (open, see O3); the literal-cycle results do not settle this.

**Q3 — impossibility for both: BOTH-IMPOSSIBLE (proved), with a sharp survival
statement.**
`count_needs_unbounded_state` and `balanced_needs_unbounded_state` are
activation-free and dimension-free: the counting task and the `aᵏbᵏ` discrimination
are impossible for `S-LIN(d,B)` **and** `S-TANH(d,B)` at every `d` and every budget.
At X-series-like shapes (`d ≈ 32×32 = 1024` bf16 coordinates), distinguish:

* **Whole-channel finite-state upper bound:** `B` counts total channel bits, not
  bits per coordinate. The nominal bit-pattern bound is `2^16384` for
  `1024 × 16 = 16384` bits (subject to the stated abstraction hypothesis), not
  `2^16`.
* **One-hot LINEAR construction guarantee:** every automaton with **≤1,024
  states** has a horizon-free LINEAR realization using at most 1,024 coordinates,
  one shared readout and one bit per coordinate. This does not characterize every
  function available at that shape or supply the ambient abstraction bridge.
* **Empirical learning/generalization:** **4× transfer is consistent,
  interpretation only**. These results neither identify the learned automaton
  nor predict unseen-horizon correctness.
* **What dies under any finite abstraction:** anything requiring unbounded
  counting — precisely the counters that X20T-style exact arithmetic allows.

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
rounding modes or identify Weiss's numerical model (see O4). Unbounded numerical
range alone does **not** imply failure of `BoundedAbstraction`: the quotient
constrains observable distinctions, not the numerical range. The definition
constrains all ambient states; the collapse theorem additionally assumes a
surjective quotient.

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
| 1 | Merrill, *Sequential Neural Networks as Automata*, arXiv:1906.01615 | **verified** (abstract + text: §3 Theorem 3.2 `L(SRN) = RL`; LSTMs strictly stronger, counter machines; "bounded precision" defined via asymptotic/saturated acceptance with `O(log L)` bits at length `L`) | asymptotic/saturated regular-language and counter framing, distinct from our fixed total-bit model |
| 2 | Merrill et al., *A Formal Hierarchy of RNN Architectures*, arXiv:2004.08500 | **verified** (abstract: hierarchy from space complexity + "rational recurrence"; LSTM is not rational; builds on saturated-RNN theory) | the "rational recurrence" axis; activation differences at bounded precision |
| 3 | Weiss, Goldberg, Yahav, *On the Practical Computational Power of Finite Precision RNNs for Language Recognition*, arXiv:1805.04908 | **verified** (abstract: under finite precision + linear time, LSTM and ReLU-RNN are *strictly stronger* than squashing-activation RNN and GRU, via counting; and "RNNs are famously known to be Turing complete, this relies on infinite precision") | finite-precision states, input-bound counting and architecture-dependent range arguments; relationship to our abstraction requires explicit modeling, O4 |
| 4 | Schützenberger 1961, *On the definition of a family of automata* | **verified at definition/attribution level** (weighted-automaton definition and attribution read on the Wikipedia *Weighted automaton* article; original not read) | the weighted linear representation/channel and LINEAR-READOUT subcase only; arbitrary decoded LINEAR stream functions are not generally rational series |
| 5 | Schützenberger's theorem: star-free = aperiodic = counter-free | **verified at summary level** (Wikipedia *Star-free language* article; original not read) | the sub-regular (counter-free vs counters) axis of Q2 |
| 6 | Delétang et al., *Neural Networks and the Chomsky Hierarchy*, arXiv:2207.02098 | **abstract-level, parent-fetched** (verbatim abstract sentence below) | empirical outcomes **in their tested task subset**, not a universal claim about Q2/Q3 |
| 7 | Siegelmann & Sontag 1995 (JCSS), *On the computational power of neural nets*, and the super-Turing line | **secondary** — original not accessible (ScienceDirect 403; no open PDF located); the framing I rely on ("Turing completeness depends on unbounded precision") is the *verified* sentence in anchor 3 | the precision-collapse framing only; their super-Turing direction is **not** used or claimed |

Delétang et al., abstract (parent-fetched from `https://arxiv.org/html/2207.02098v3`):

> Our results show that, for our subset of tasks, RNNs and Transformers fail to generalize on non-regular tasks, LSTMs can solve regular and counter-language tasks, and only networks augmented with structured memory (such as a stack or memory tape) can successfully generalize on context-free and context-sensitive tasks.

The deleted rational-series claim was false: the indicator series of `aⁿbⁿ` has
Hankel submatrices indexed by `aⁱ,bʲ` that are arbitrarily large identity matrices,
so it is not a rational weighted series in the standard sense.

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
4. `exists_fin_model_of_abstraction` — **surjective abstraction collapse.** An
   `n`-valued channel with surjective quotient is simulated exactly by an `n`-state
   model (`Fin n`), same online output at every position.
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
9. `bounded_precision_class_characterization` — **surjective abstraction collapse
   plus finite-reachable one-hot realization** (4 and 8 combined). The reverse
   conjunct supplies no bounded abstraction on the ambient state space.
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
15. `lin_one_dim_no_three_cycle`, `tanh_one_dim_no_three_cycle` — **at `d = 1`
    neither activation has a literal three-state cycle**, not a stream-function
    expressivity separation (see Q2's scalar counterexample).
16. `rot3Pair_cube`, `rot3Model_three_cycle` — **dimension 2 suffices** for a
    literal state 3-cycle, with a *linear* (`ℚ`-rational) channel and explicit states.
17. `tanhStreamModel_state_bounded` — a scalar `TANH(1)` state stays in `(-1, 1)`
    after any nonempty word: the bounded region that fixed resolution discretizes.
18. `tanh_one_dim_two_fixed_points` — a correction: scalar `tanh` latches two values
    *exactly* (`a = 1/tanh 1`, `b = 0`, fixed points `0` and `tanh 1`).

X21TR axiom recheck (`#print axioms` on 13 headline declarations): at most `propext`,
`Classical.choice`, `Quot.sound` — no `sorryAx`, nothing exotic.

---

## Corrections to tracked theory and to the task's own candidate frames

1. **One-hot precision is a sufficient construction, not a dimension lower bound.**
   `moore_realized_by_onehot_linear` realizes any `n`-state automaton at `d = n`
   with coordinates in `{0, 1}`. Necessity is not proved; the two-dimensional
   three-cycle already contradicts blanket necessity of one coordinate per state.
   Counting requires unbounded distinguishable state, not necessarily unbounded
   dimension (the exact scalar counter suffices).
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
5. **X20T's residual dimension/precision gap is only partially resolved.** Both
   activations have a finite-automaton upper bound under the stated abstraction
   hypothesis. The `d = 1` vs `d = 2` separation concerns literal state-cycle
   realizations only; no automaton/stream-function dimension hierarchy is proved.
6. **The `aᵏbᵏ`-as-`TANH`-witness frame must be rejected.** It is impossible for
   *both* classes (`balanced_needs_unbounded_state`), so it can never separate them.
   Same for any latching witness. The X20T lesson (reject by construction any
   witness the decoder or a second coordinate can compensate) applies verbatim to
   every candidate in Q2's list.

---

### X21TR acceptance corrections

All six X21TR acceptance conditions applied verbatim per the X21TR review, with
condition 2's **downgrade** option; no theorem statements or proofs changed:

1. Replaced activation/fixed-dimension equalities with the exact three-sentence
   finite-automaton upper-bound / dimension-n LINEAR / no-TANH-converse text.
2. Downgraded to **surjective abstraction collapse plus finite-reachable one-hot
   realization**; documented the missing ambient bridge and the two-state
   swapping/zero-fallback counterexample (also the `rot3Model.out = id` issue).
3. Restricted separation to **literal state-cycle realizations**; included the
   scalar `s ↦ s + 1`, `q(s) = floor(s) mod 3`, `out = q` counterexample and removed
   the claimed dimension necessity.
4. Separated the `2^16384` whole-channel nominal bound, ≤1,024-state one-hot
   LINEAR guarantee, and empirical learning/generalization. **4× transfer is
   consistent, interpretation only**; these results neither identify the learned
   automaton nor predict unseen-horizon correctness.
5. Withdrew the Weiss bounded-mantissa/unbounded-exponent identification and
   blanket abstraction-failure claim; explicit modeling remains required.
6. Deleted anchor 8 with the Hankel identity-submatrix justification; restricted
   anchor 4 to weighted linear channels/linear readout, preserved Merrill 2019's
   distinct asymptotic `O(log L)` framing, scoped Delétang to their tested task
   subset with the parent-fetched verbatim abstract sentence, and kept
   Siegelmann–Sontag secondary-via-Weiss.

Validation after correction: targeted `lake build
ElmanProofs.Expressivity.StreamingExpressivity` passed (2,066 jobs, no target
warnings); full `lake build` passed (2,198 jobs, existing warnings in unrelated
modules). The 13-declaration axiom audit used only the standard axioms listed
above, with no `sorryAx`. Comparing Lean source with the pre-correction revision
after stripping block comments and whitespace gave identical SHA-256
`186dd0780adf6181f213b40003fd6022d1968f41219c47615ce037c17105f269`;
all declarations and proofs are unchanged.

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
* **O3 — minimal dimension.** Proved: `d = 1` cannot literally 3-cycle (both
  activations), `d = 2` can (`LIN`), `d = n` suffices for any `n`-state automaton.
  These are literal-cycle facts and a realization upper bound, not a general
  automaton/stream-function dimension hierarchy. A lower bound `d ≥ f(n)` growing
  with the automaton is not proved.
* **O4 — precision/range modeling.** Weiss's setting uses finite-precision states,
  input-bound counting, and architecture-dependent range arguments. The supplied
  passages support its counting separation, not an identification with bounded
  mantissa/unbounded exponent. Its relationship to this formal abstraction
  requires explicit modeling of precision, range, and rounding. Blanket failure
  of `BoundedAbstraction` from unbounded numerical range is **not** asserted;
  fixed mantissa with unbounded exponent also does not make unit-increment
  counting exact indefinitely.
* **O5 — learnability, efficiency, average case.** Untouched. Nothing here explains
  why one arm ignites faster; no training was launched.
* **O6 — unbounded precision with a genuinely shared decoder over all lengths and
  online readout at the same time.** X20T left this open; the bounded-precision
  branch has the limited results stated above. The exact-arithmetic
  uniform-decoder question remains open exactly as in X20T.

No X21b task spec is written and no experiment is registered: the only empirical
question left is an *efficiency* probe (minimal `d` for a given automaton under a
fixed budget, `tanh` vs linear), which is an operator decision, not an expressivity
separator.

---

## Reproduce

```bash
cd formal/lean
lake build ElmanProofs.Expressivity.StreamingExpressivity   # green, no sorry
lake build                                               # full build green
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
