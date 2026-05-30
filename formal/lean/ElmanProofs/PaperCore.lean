/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Architectures.M2RNNComparison
import ElmanProofs.Architectures.MultiStepSeparation
import ElmanProofs.Architectures.OnlineMemory
import ElmanProofs.Architectures.RecurrentResourceFormalism
import ElmanProofs.Architectures.S5Inseparability
import ElmanProofs.Architectures.SplitGatedDelta
import ElmanProofs.Expressivity.E88ExceedsE1HCapacity
import ElmanProofs.Expressivity.NDMRealizesS5
import ElmanProofs.Expressivity.S5NDMRealization
import ElmanProofs.Expressivity.S5Tracker
import ElmanProofs.Expressivity.S5Witness

/-!
# Trusted Paper Core

This is the small paper-facing proof root.

Terminology:

* Emender is the public model family.
* A nonlinear delta memory is the recurrent mechanism historically called
  Nonlinear Delta Memory (NDM) in earlier notes and some formal identifiers.
* E88 is the current production Emender implementation lineage.

This root intentionally imports only the checked chain needed for the current
paper-space claims:

* M2RNN and Emender/E88 are distinct one-step transition families.
* M2RNN's raw outer write is separated from the delta-correcting write shared
  by ideal GDN and Emender/E88 pre-nonlinearity.
* Emender/E88 implements a mixed-key delta correction in one recurrent step that
  fixed-right/raw-write M2RNN resources with external row/column/cell forget
  cannot implement in one step; this witness embeds into every matrix state
  with key dimension at least two and value dimension at least one.
* Conversely, M2RNN-style raw writes exactly embed one Emender/E88 delta step
  once given the extra read-then-delta resource `v - H^T k`.
* The E97 split-gated delta algebra is checked: direct and expanded forms are
  equal, all-one gates specialize to E88, and the GDN-2-style linear core is
  the same split-gated read/write core applied to a pre-decayed state.
* Emender/E88 exposes the current 1.27B many-program production geometry.
* E88 has the checked matrix-state capacity separation over E1H.
* The S5 witness surface is checked: fixed-precision online recognizers have
  finite state, S3 has six states, S5 has 120 states, and S5 is non-solvable.
* The explicit S5 adjacent-transposition tracker is checked as a 120-state
  recognizer, its word execution composes permutations, and its transition
  matches the Python tuple-swap task semantics.
* Every finite fixed-precision recognizer has an exact finite transition-table
  realization; the S5 tracker instance uses 480 state/input keys.
* Nonlinear delta-memory parameters (orthonormal generator keys at `d = 12`,
  one-hot value family, `λ = 1`) realize the S5 transition memory: the delta
  core loads an orthonormal table whose readout, decoded and composed with any
  input state, reproduces `s5TransitionMemory.read` on every
  `(state, generator)` pair (`NDMRealizesS5.emender_realizes_s5_tracker`).

Use `scripts/check_paper_core.sh` to reject unfinished proof holes, explicit
assumptions, opaque declarations, and kernel-bypassing computation in this
import closure.
-/
