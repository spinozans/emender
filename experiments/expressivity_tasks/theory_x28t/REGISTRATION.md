# X28T — Tape-memory construction: the constructive counterpart of the both-impossible theorems

**Pre-registered before any proof work.** This leg adds no new hypotheses to
existing theorems and modifies no existing statement. It adds one new module,
`formal/lean/ElmanProofs/Expressivity/TapeMemory.lean`, extending the existing
theory-leg import chain (`StreamingExpressivity` → `MemoryBoundary` chain) with
new imports only. No training, no experiment registration.

## The precise contrast being formalized

The both-impossible theorems (X21T, `StreamingExpressivity.lean`) bound the
**total** state:

* `StreamingExpressivity.count_needs_unbounded_state` — no `StreamModel` whose
  channel carries at most `n` values can report `n + 1` distinct counts;
* `StreamingExpressivity.balanced_needs_unbounded_state` (via
  `MemoryBoundary.context_free_discrimination_impossible`) — no bounded-precision
  channel computes the `aᵏbᵏ` discrimination `SeparatesBalanced`, any dimension,
  any activation, any budget.

X28T exhibits the constructive object on the other side of that boundary: a
tape/stack memory model whose state is **finite bounded-precision control plus an
unbounded slot count, with bounded-precision slots**. Exactly the axis the
impossibility theorems do not bound — the number of slots — is the one that
grows. The point of the pair: what unbounded slot COUNT buys over bounded total
state is exactly what the impossibility theorems deny the native channels.

## The model (fixed before proving)

* `TapeMemory.TapePhase` — inductive control with three constructors
  (`pushing`, `popping`, `garbage`), `deriving DecidableEq, Fintype, Repr`.
* `TapeMemory.tapeStep : TapePhase × List Bool → Bool → TapePhase × List Bool` —
  `a` (`true`) appends one slot; `b` (`false`) consumes one slot; an `a` after
  the `b`-phase began, or a `b` with no slot to consume, falls into the `garbage`
  sink (malformed-input rejection). Slots are `Bool`: one bit of per-slot
  precision, stated explicitly in the model.
* `TapeMemory.tapeModel : StreamModel (TapePhase × List Bool) Bool Bool` —
  `init = (pushing, [])`, `step = tapeStep`,
  `out p = decide (p.1 ≠ garbage ∧ p.2 = [])` — "never malformed, tape empty at
  the end".
* `TapeMemory.isBalancedAB : List Bool → Bool` — the **constructive** decision
  procedure for the shape `aᵏbᵏ` (split at the first `b`, require the rest to be
  all `b`, require equal counts). No `Classical` shortcut on the acceptance
  predicate.
* `TapeMemory.BalancedAB (w : List Bool) : Prop := ∃ k, w = aᵏ ++ bᵏ`
  (including `k = 0`, the empty word, which the readout accepts: zero slots
  appended, zero consumed, empty at the end).

## Pre-registered statements (names fixed; statements honored as written)

(a) Unbounded counting — the constructive twin of `count_needs_unbounded_state`:

* `tapeCount_slots (k : ℕ) : (tapeModel.stateAfter (List.replicate k true)).2.length = k`
* `tapeCount_exceeds_bound (B : ℕ) : ∃ w : List Bool, B < (tapeModel.stateAfter w).2.length`
* `tapeCount_twin_countModel (k : ℕ) :
    countModel.out (countModel.stateAfter (List.replicate k true))
      = (tapeModel.stateAfter (List.replicate k true)).2.length`

(b) Full language decision on ALL inputs (construction: one slot per `a`,
consumed per `b`; accept iff slots empty at end; explicit malformed rejection):

* `isBalancedAB_iff (w : List Bool) : isBalancedAB w = true ↔ BalancedAB w`
* `tapeModel_decides_balanced (w : List Bool) :
    tapeModel.out (tapeModel.stateAfter w) = isBalancedAB w`
* `tapeModel_accepts_iff (w : List Bool) :
    tapeModel.out (tapeModel.stateAfter w) = true ↔ BalancedAB w`
* explicit malformed rejections:
  * `tapeModel_rejects_leading_b (w : List Bool) :
      tapeModel.out (tapeModel.stateAfter (false :: w)) = false` (b before a)
  * `tapeModel_rejects_count_mismatch (k k' : ℕ) (h : k ≠ k') :
      tapeModel.out (tapeModel.stateAfter (List.replicate k' true ++ List.replicate k false)) = false`
  * `tapeModel_rejects_trailing_a (k j : ℕ) (hj : 1 ≤ j) (v : List Bool) :
      tapeModel.out (tapeModel.stateAfter (List.replicate k true ++ List.replicate j false ++ (true :: v))) = false`

Bounded per-step precision, stated explicitly:

* `tapePhase_card : Fintype.card TapePhase = 3`
* `tapeModel_cells_unary (w : List Bool) :
    ∃ (p : TapePhase) (m : ℕ), tapeModel.stateAfter w = (p, List.replicate m true)`
  (every reachable slot holds the single symbol `true`: the per-slot alphabet is
  one value; all unbounded information is in the slot count)

The pairing statements:

* `tapeModel_separates_balanced : SeparatesBalanced tapeModel true false`
* `tapeModel_no_finite_abstraction : NoBoundedAbstraction tapeModel`
* `tape_memory_pair` — the four conjuncts side by side: (i) per-step state
  bounded-precision in every component, (ii) unbounded slot count, (iii) the
  full `{aᵏbᵏ}` decision on all inputs, (iv) `NoBoundedAbstraction tapeModel`.

Supporting fold lemmas (names may adapt, statements are means not claims):
`fold_pushTape`, `fold_garbage`, `fold_popping_sees_a`, `fold_popChain`,
`fold_trichotomy`, `stateAfter_balanced`, `tapeModel_accepts_balanced`.

## Gates (mandatory)

1. `lake build` passes clean; `lake build ElmanProofs.Expressivity.TapeMemory`
   passes clean (the theory-leg chain is built per-module, as in X22T).
2. Zero `sorry`, zero `native_decide`.
3. `#print axioms` on every new headline theorem shows at most `propext`,
   `Classical.choice`, `Quot.sound`.
4. No `Classical` shortcut where a constructive decision procedure exists for
   the acceptance predicate: `isBalancedAB` is a Bool-valued decision procedure
   and `tapeModel_decides_balanced` proves the model equals it; every proof in
   the file is constructive. Any `Classical` use (expected: none) would be
   flagged in `THEOREMS.md`.
5. Documentation in `experiments/expressivity_tasks/theory_x28t/THEOREMS.md`:
   exact names, statements, proof sketches, divergences, axioms output.
6. No git commit or push; parent reviews and audits first.

## Scoping commitments

* This is a construction inside the formal framework, not a claim about
  `HybridLadderLM` (no native tape or stack). It is the discrete idealization of
  a soft push/pop head; no approximation property of a soft tape is claimed.
* Unlike X22T's `stackModel` (block *discrimination* only — it also accepts
  `abab`, X22TR condition 6), `tapeModel` decides the language on **all**
  inputs; the added `garbage` control phase is what upgrades discrimination to
  recognition. This will be stated explicitly.
* Per-slot precision here is bounded and degenerate (unary reachable alphabet);
  no claim about tapes with informative slot contents.
* No `ω`-stream statement is claimed beyond the finite-word one.
