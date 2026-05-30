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

- Emender/E88 and M2RNN are distinct one-step transition families.
- Emender/E88 has a delta-correcting matrix-state update; M2RNN has a fixed-right
  raw-write matrix-state update.
- A fixed-right raw-write M2RNN resource does not implement the mixed-key
  nonlinear delta-memory correction in one step without an extra
  read-then-delta path.
- With that extra read-then-delta path, M2RNN can embed one Emender/E88 delta
  step.
  This is a resource separation, not an absolute computability separation.
- E97 split-gated delta algebra is checked: direct and expanded forms agree,
  all-one gates specialize to E88, and a concrete 1x1 split-write-gate witness
  leaves the E88 all-one-gate subfamily on the same state/input.
- E97's pointwise split erase/read gate also has a finite 2D transition
  witness: write direction `u = (1,1)` and erase/read direction `r = (1,0)`
  are nonparallel, E97 realizes `I - u r^T`, and no E88 coupled transition
  `mu I - p p^T` realizes that same transition factor. This is a one-step
  transition-factor result only.
- E97 and E88 have the same leading coarse per-head recurrent-state cost under
  `flopsPerToken`; applying precomputed split gates is represented separately
  as a linear `2*d` term and is bounded by one `d*d` state-scalar pass for
  `d >= 2`.
- GDN-2 is represented as matrix-state, split erase/write, no temporal
  state nonlinearity, and scan-compatible; E97 remains nonlinear-state and
  non-scan-compatible like Emender/E88.
- These resource theorems are cost/signature statements only. They do not
  prove empirical superiority, learning efficiency, hardware throughput, or
  FLOPs-per-bit convergence.
- The 1.27B Emender/E88 production geometry is represented as a pure nonlinear
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
