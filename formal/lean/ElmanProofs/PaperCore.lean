/-
Copyright (c) 2026 Elman-Proofs Contributors. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
-/
import ElmanProofs.Architectures.M2RNNComparison
import ElmanProofs.Architectures.OnlineMemory
import ElmanProofs.Architectures.RecurrentResourceFormalism
import ElmanProofs.Expressivity.E88ExceedsE1HCapacity
import ElmanProofs.Expressivity.S5NDMRealization
import ElmanProofs.Expressivity.S5Tracker
import ElmanProofs.Expressivity.S5Witness

/-!
# Trusted Paper Core

This is the small paper-facing proof root.

Terminology:

* NDM means Nonlinear Delta Memory, the model family.
* E88 is the current production implementation lineage of NDM.

This root intentionally imports only the checked chain needed for the current
paper-space claims:

* M2RNN and NDM/E88 are distinct one-step transition families.
* M2RNN's raw outer write is separated from the delta-correcting write shared
  by ideal GDN and NDM/E88 pre-nonlinearity.
* NDM/E88 implements a mixed-key delta correction in one recurrent step that
  fixed-right/raw-write M2RNN resources with external row/column/cell forget
  cannot implement in one step; this witness embeds into every matrix state
  with key dimension at least two and value dimension at least one.
* Conversely, M2RNN-style raw writes exactly embed one NDM/E88 delta step once
  given the extra read-then-delta resource `v - H^T k`.
* NDM/E88 exposes the current 1.27B many-program production geometry.
* E88 has the checked matrix-state capacity separation over E1H.
* The S5 witness surface is checked: fixed-precision online recognizers have
  finite state, S3 has six states, S5 has 120 states, and S5 is non-solvable.
* The explicit S5 adjacent-transposition tracker is checked as a 120-state
  recognizer, its word execution composes permutations, and its transition
  matches the Python tuple-swap task semantics.
* Every finite fixed-precision recognizer has an exact finite transition-table
  realization; the S5 tracker instance uses 480 state/input keys.

Use `scripts/check_paper_core.sh` to reject unfinished proof holes, explicit
assumptions, opaque declarations, and kernel-bypassing computation in this
import closure.
-/
