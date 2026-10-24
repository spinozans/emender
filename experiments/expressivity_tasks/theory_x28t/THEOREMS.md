# X28T — Tape memory: the constructive counterpart of the both-impossible theorems

Pre-registered in `REGISTRATION.md` before any proof work. This document records
the delivered theorems, proof sketches, divergences, and the axiom audit.

**Status: DELIVERED — `lake build ElmanProofs.Expressivity.TapeMemory` passes
with zero errors and zero warnings; the axiom audit (§6) shows standard axioms
only.** Elaboration history: the initial worker run (30-minute ceiling) wrote
the module and pre-registration but timed out mid-build with errors; the parent
completed and corrected the proofs in-session. **No pre-registered statement
was changed** — every name and statement below is as registered; corrections
were proof-level only (see §5).

## 1. The contrast, stated precisely

The both-impossible theorems (X21T) bound the **total** state channel:

* `StreamingExpressivity.count_needs_unbounded_state`
* `StreamingExpressivity.balanced_needs_unbounded_state`
  (via `MemoryBoundary.context_free_discrimination_impossible`)

X28T's tape model keeps **per-step precision bounded in every component** — a
three-valued control, one-bit slots, and in fact every reachable slot holds the
single symbol `true` — while the **slot count grows without bound**. That axis
buys exactly what the impossibility theorems deny bounded-precision channels:
exact counting and the full `{aᵏbᵏ}` language decision on all inputs.

## 2. Objects

* `TapeMemory.TapePhase` — `pushing | popping | garbage` (finite control,
  3 values).
* `TapeMemory.tapeStep` — `a` appends one slot; `b` consumes one; malformed
  input falls into the `garbage` sink.
* `TapeMemory.tapeModel` — the `StreamModel` over `TapePhase × List Bool` with
  readout "never malformed, tape empty at the end".
* `TapeMemory.isBalancedAB` — constructive decider for the shape `aᵏbᵏ`.
* `TapeMemory.BalancedAB` — the language `{aᵏbᵏ : k ≥ 0}` as a word predicate.

## 3. What is proven (delivered; every pre-registered name honored)

* **(a) Unbounded counting** — the constructive twin of
  `count_needs_unbounded_state`:
  `tapeCount_slots` (after `aᵏ` the tape holds exactly `k` slots),
  `tapeCount_exceeds_bound` (for every `B` a word leaves more than `B` slots:
  `a^(B+1)`), `tapeCount_twin_countModel` (the slot count realizes exactly the
  count `countModel` reports online).
* **(b) The full language decision on all inputs**: `isBalancedAB_iff` (the
  constructive decider decides its own specification `BalancedAB`),
  `tapeModel_decides_balanced` (the tape model's readout equals `isBalancedAB`
  on every word), `tapeModel_accepts_iff` (acceptance is exactly membership in
  `{aᵏbᵏ : k ≥ 0}`), and the three explicit rejections:
  `tapeModel_rejects_leading_b` (a `b` before any `a`),
  `tapeModel_rejects_count_mismatch` (`aᵏ'bᵏ` with `k ≠ k'`),
  `tapeModel_rejects_trailing_a` (any `a` after the `b`-phase began).
* **Bounded per-step precision, stated explicitly**: `tapePhase_card` (the
  control is exactly 3 values), `tapeModel_cells_unary` (every reachable
  configuration is `(phase, replicate m true)`: the per-slot alphabet is the
  single symbol `true` — zero bits per slot — and all unbounded information
  lives in the slot count).
* **The pair**: `tapeModel_separates_balanced` (the tape model computes
  `SeparatesBalanced`), `tapeModel_no_finite_abstraction` (via
  `context_free_discrimination_impossible`, it admits NO bounded-precision
  total state channel), and `tape_memory_pair` — all four conjuncts side by
  side.
* Proof engines (declared in the original module write of 22:26:24 and its
  header docstring — not in `REGISTRATION.md`'s statement sections, see §5):
  `fold_pushTape`, `fold_garbage`, `fold_popping_sees_a`, `step_pop_nil`,
  `step_pop_cons`, `fold_popChain`, `fold_trichotomy`, plus
  `stateAfter_balanced` and `tapeModel_accepts_balanced`.

## 4. Proof sketches

* **Counting**: `fold_pushTape` (induction on `k` over any base tape) gives
  the exact slot count after `aᵏ`; `tapeCount_exceeds_bound` instantiates
  `k := B + 1` and `omega`. The twin theorem rewrites with
  `countModel_out` from the X21T framework.
* **The all-inputs decision**: `fold_trichotomy` — the central engine — from
  any live phase with `m` slots, after any word the machine is (i) in the
  `garbage` sink, (ii) holding a nonempty tape, or (iii) live and empty on a
  word of shape `aⁱb^(i+m)`. Case (i) kills the readout's `≠ garbage`
  conjunct, case (ii) kills `tape = []`, and case (iii) exhibits the exact
  shape, which `isBalancedAB` (takeWhile/dropWhile split + count match)
  accepts. The converse direction: a balanced word folds to exactly-empty
  (`stateAfter_balanced` via `fold_popChain`), hence accepted. Every
  malformed shape is rejected through one of the two dead-ends: landing in
  the `garbage` sink (`step_pop_nil` underflow, `fold_popping_sees_a`
  trailing-`a`, `fold_popChain` mismatch arithmetic) or finishing live
  with a nonempty tape (the `fold_trichotomy` case-(ii) route — e.g. `aab`
  ends in `(popping, [true])` and fails the `tape = []` conjunct).
* **Unary cells**: induction over the word with full case analysis on
  (token, phase, slot count); every reachable tape is a stack of `true`s.
  Constructed fully explicitly (per-step `rfl` rewrites plus `fold_garbage`),
  and in fact **axiom-free** (§6).
* **No bounded abstraction**: `SeparatesBalanced tapeModel` + the X21T
  both-impossible theorem `context_free_discrimination_impossible`
  (`balanced_needs_unbounded_state` via `MemoryBoundary`) — the tape model
  computes what no bounded-precision channel can, so its unbounded resource
  (the slot count) is the entire difference.

## 5. Divergences from the pre-registration

* **No statement weakenings.** Every theorem named in `REGISTRATION.md`'s
  pre-registered statement sections (a), (b), bounded-precision, and pairing
  is delivered with exactly the registered name and Lean proposition. One
  registration-precision caveat: `tape_memory_pair` is registered at
  `REGISTRATION.md:85-88` only in prose ("the four conjuncts side by side"),
  not as an exact Lean proposition; the delivered conjunction matches that
  stated intent component-for-component (unary per-step state / unbounded
  slot count / all-inputs decision / `NoBoundedAbstraction`), and no
  registered theorem was weakened.
* **Proof-engine provenance** (corrected on review): the fold-engine lemmas
  (`fold_pushTape`, `fold_garbage`, `fold_popping_sees_a`, `step_pop_nil`,
  `step_pop_cons`, `fold_popChain`, `fold_trichotomy`) and
  `stateAfter_balanced` / `tapeModel_accepts_balanced` were declared in the
  original module write (transcript timestamp 22:26:24, one minute after the
  registration write) and its header docstring, but are NOT listed in
  `REGISTRATION.md`'s pre-registered statement sections. They are proof
  machinery for the registered statements, not registered claims.
* **Proof-engine additions during parent completion** (not in the original
  module write; pure helpers, no claims): `replicate_true_cons`,
  `stateAfter_push`, `takeWhile_balanced`, `dropWhile_push`,
  `takeWhile_replicate_false`, `dropWhile_replicate_false`.
* **Elaboration history**: the initial worker session timed out at the
  30-minute ceiling with the module written but non-compiling (implicit-argument
  lemma misuse, a misordered `rcases` branch split in `fold_trichotomy`,
  `simpa`-of-a-value misuse, and an unfinishable final rewrite in
  `tapeModel_cells_unary`); the parent completed the proofs in-session with
  the build log retained. This is recorded as process provenance, not a
  scientific divergence.

## 6. Axiom audit

`#print axioms` on every delivered theorem (audit file
`formal/lean/AuditX28.lean`, output in the build receipts). **Only the three
standard axioms appear, and several theorems are axiom-free:**

* axiom-free: `fold_garbage`, `fold_popping_sees_a`, `step_pop_nil`,
  `step_pop_cons`, `tapeModel_cells_unary`
* `[propext]` only: `replicate_true_cons`, `fold_pushTape`,
  `tapeCount_slots`, `takeWhile_replicate_false`, `dropWhile_replicate_false`
* `[propext, Quot.sound]`: `stateAfter_push`, `fold_popChain`,
  `fold_trichotomy`, `tapeCount_exceeds_bound`, `tapeCount_twin_countModel`,
  `stateAfter_balanced`, `tapeModel_accepts_balanced`,
  `tapeModel_rejects_leading_b`, `tapeModel_rejects_count_mismatch`,
  `tapeModel_rejects_trailing_a`, `tapeModel_separates_balanced`
* `[propext, Classical.choice, Quot.sound]`: `tapePhase_card`,
  `takeWhile_balanced`, `dropWhile_push`, `isBalancedAB_iff`,
  `tapeModel_decides_balanced`, `tapeModel_accepts_iff`,
  `tapeModel_no_finite_abstraction`, `tape_memory_pair`

`Classical.choice` enters only through `simp`/`decide` infrastructure on the
  imported framework lemmas — the acceptance predicate
  `isBalancedAB` itself is a `Bool`-valued decision procedure with a
  constructive proof of its specification (`isBalancedAB_iff`); no classical
  shortcut is taken on the decision procedure.

## 7. Reproduce

```
cd formal/lean
lake build ElmanProofs.Expressivity.TapeMemory   # zero errors, zero warnings
lake env lean AuditX28.lean                       # the §6 axiom audit
```
Build receipt: `Build completed successfully (2070 jobs)`
