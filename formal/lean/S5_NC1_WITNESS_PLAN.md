# S5 / NC1 Witness Plan

This corrects the paper-facing complexity story.

## The Ceiling

A fixed-width, finite-precision recurrent recognizer that updates once per input
symbol has finitely many possible states. As a language recognizer it is a
finite automaton, so the fixed-resource ceiling is regular languages, and
regular languages are in NC1.

This means the claim is not "E88/NDM goes beyond NC1" at fixed precision. That
claim is false. The useful claim is inside the regular languages: which
transition monoids can a resource-bounded architecture learn and execute
reliably?

## The Witness

Parity and modular counters are not the right hard witness. They live in
solvable groups. Linear and scan-friendly models often have a path to those
tasks, which is why parity/mod-count results are useful engineering probes but
not the cleanest separation.

The sharper witness is permutation composition:

- `S3`: six-state solvable control.
- `S5`: 120-state non-solvable witness.

By Barrington's theorem, the S5 word problem is the canonical NC1-complete
regular-language witness. Empirically, the task should be prefix product
tracking plus length extrapolation: train at a fixed length, then evaluate at
longer lengths. A model that learned the algorithm should keep tracking; a
model that learned only a training-length surface should fall apart.

## What Lean Now Checks

`ElmanProofs/Expressivity/S5Witness.lean` is deliberately modest and trusted:

- Defines a fixed-precision online recognizer with a finite state type.
- Proves the recognizer state space is finite by construction.
- Proves `S3` has 6 states.
- Proves `S5` has 120 states.
- Imports mathlib's proof that `S5` is not solvable.

It does not yet formalize Barrington's theorem or the linear-scan lower bound.
Those should be cited in the paper first, then formalized only if doing so
improves confidence rather than producing another brittle scaffold.

## What The Experiment Now Tests

The Elman expressivity suite now exposes:

- `s3_permutation`
- `s5_permutation`

The canonical runner adds length extrapolation at `128, 256, 512, 1024` tokens
for both tasks. The intended first comparison is:

```bash
python experiments/expressivity_tasks/run_separation_suite.py \
  --tasks s3_permutation s5_permutation \
  --models E88_8M FLA_8M M2RNN_8M M2RNN_paper_8M \
  --seeds 42 123 456 \
  --gpus 0,5,6 \
  --use_triton_e88 \
  --output_dir experiments/expressivity_tasks/results/s5_witness_8m
```

## Paper Implication

The honest claim shape is:

> Fixed-precision NDM/E88 is not beyond NC1; it is a pure nonlinear recurrent
> route to the top of NC1-style finite-state computation under practical
> resources. The empirical question is whether the many-program nonlinear delta
> architecture learns non-solvable state tracking more reliably than linear
> scan models and M2RNN-shaped nonlinear matrix-state baselines.

That is a stronger, cleaner, and less fragile story than a parity-centered TC0
claim.
