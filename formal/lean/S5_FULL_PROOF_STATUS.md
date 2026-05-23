# S5 Full Proof Status

This records the current trusted S5 proof boundary after adding the explicit
tracker modules.

## Fully Checked in Lean

- `ElmanProofs/Expressivity/S5Witness.lean`
  - fixed-precision online recognizers have finite state by construction;
  - `S3` has 6 states;
  - `S5` has 120 states;
  - `S5` is not solvable, via mathlib.

- `ElmanProofs/Expressivity/S5Tracker.lean`
  - defines the four adjacent transposition tokens for S5;
  - defines the explicit S5 online tracker DFA with state space `Equiv.Perm (Fin 5)`;
  - proves the tracker has 120 states;
  - proves tracker execution composes input words by S5 multiplication:
    `run (u ++ v) = run u * run v`;
  - proves appending one token composes by that token's adjacent transposition;
  - proves the Python tuple update used in
    `/home/erikg/elman/experiments/expressivity_tasks/tasks/s5_permutation.py`
    agrees with the Lean transition. The Python code swaps adjacent entries of
    the tuple-of-images representation; Lean proves that this is right
    composition by the corresponding adjacent transposition;
  - defines running prefix-product targets and proves there is exactly one
    target per input token.

- `ElmanProofs/Expressivity/S5NDMRealization.lean`
  - defines a finite associative lookup memory;
  - proves every `FixedPrecisionOnlineRecognizer` has an exact transition
    memory keyed by `(state, input-symbol)`;
  - proves the table-driven run is extensionally equal to the recognizer run on
    every word;
  - instantiates this for the S5 tracker;
  - proves the S5 transition table has `120 * 4 = 480` state/input keys.

## Mathematical Bridge to the Executable Task

The executable task currently represents a permutation as a Python tuple
`perm`, starts from `tuple(range(n))`, and applies a token by:

```python
p = list(perm)
p[i], p[j] = p[j], p[i]
return tuple(p)
```

For `n = 5`, Lean models the tuple as `Fin 5 -> Fin 5`. The checked theorem
`S5Tracker.pythonApplySwap_eq_step_tuple` proves that the Python swap update is
the same as the Lean S5 tracker update. The theorem
`S5Tracker.pythonRun_eq_tracker_tuple` lifts this to whole input words.

The remaining executable bridge not yet formalized is the exact lexicographic
integer ID assigned by Python's `itertools.permutations`. The Lean proof tracks
the permutation state itself, not Python's concrete class-index numbering.

## Still Unproven / External

- Barrington's theorem and the NC1-completeness of the S5 word problem are not
  formalized here. They remain cited mathematical background.
- No Lean lower bound is proved for linear, scan-compatible, solvable, or
  star-free model families. This should not be claimed from the trusted files.
- The finite lookup realization is a discrete exact transition-table theorem.
  It does not prove that a trained real-valued E88/NDM learns the table, and it
  does not provide a trainability or robustness theorem.
- Adjacent transpositions are used as the task generators, but the current Lean
  module does not separately prove the abstract group-generation theorem for
  `S5`; it proves the tracker over those generators and the resulting 120-state
  transition system.

## Exact Remaining Lean Lemmas for a Stronger Story

1. Formalize Python lexicographic class IDs:
   define lexicographic enumeration of `Fin 5` permutations and prove it agrees
   with `itertools.permutations(range(5))`.
2. Prove adjacent transpositions generate all of `S5`:
   every `Equiv.Perm (Fin 5)` is equal to `S5Tracker.run w` for some word `w`.
3. Add an acceptor/language layer:
   choose a target permutation or target subset and prove the online recognizer
   accepts exactly the corresponding S5 word problem.
4. Formalize a cited complexity theorem, if worthwhile:
   either import/prove the Barrington theorem interface or keep it explicitly
   outside Lean as a citation.
5. Only after a precise model class is defined, attempt a lower-bound theorem
   for linear/scan/solvable models. The current trusted surface intentionally
   has no such theorem.
