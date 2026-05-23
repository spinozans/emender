# Trusted Proof Surface

This repository contains two different kinds of Lean work:

1. Trusted paper-facing proofs.
2. Historical research sketches.

Only the first category should be cited as formal evidence.

## Trusted Roots

The trusted roots are:

- `ElmanProofs.lean`
- `ElmanProofs/PaperCore.lean`

These roots are checked by:

```bash
scripts/check_trusted_no_placeholders.sh ElmanProofs.lean
scripts/check_paper_core.sh
```

The checks reject `sorry`, `admit`, explicit `axiom`, and `opaque`
declarations in the trusted import closure. `check_paper_core.sh` also rejects
`native_decide` in the paper core closure.

## Current Paper-Core Claims

The current trusted paper core supports:

- NDM/E88 and M2RNN are distinct one-step transition families.
- NDM/E88 has a delta-correcting matrix-state update; M2RNN has a fixed-right
  raw-write matrix-state update.
- A fixed-right raw-write M2RNN resource does not implement the mixed-key NDM
  delta correction in one step without an extra read-then-delta path.
- With that extra read-then-delta path, M2RNN can embed one NDM delta step.
  This is a resource separation, not an absolute computability separation.
- The 1.27B NDM/E88 production geometry is represented as a pure nonlinear
  recurrent many-program stack.
- The S5 witness scaffold is checked: fixed-precision online recognizers have
  finite state, S3 has 6 states, S5 has 120 states, and S5 is non-solvable.

## Historical Sketches

Many older files were written as theorem-discovery notebooks. They may contain
`sorry`s, placeholder definitions, or stale narrative claims. In particular,
the older TC0/parity-centered story is superseded by the fixed-state/NC1/S5
framing:

- Fixed-width finite-precision recurrence is finite-state as a recognizer.
- The fixed-resource ceiling is regular languages, hence inside NC1.
- Parity and modular counting are solvable-control probes, not the main
  non-solvable witness.
- S5 permutation composition is the paper-facing witness to target empirically.

Do not cite historical sketch modules as formal results unless their claims
have been moved into the trusted roots and pass the checks above.

## Workflow

When a historical sketch contains a useful idea:

1. Copy the needed statement into a new small module or an existing trusted
   module.
2. Weaken it until it is true and checkable.
3. Prove it without `sorry`, `admit`, explicit `axiom`, or `opaque`.
4. Add it to `PaperCore.lean` only if it is part of the paper-facing theorem
   surface.

The old files are a mine, not a debt pile.
