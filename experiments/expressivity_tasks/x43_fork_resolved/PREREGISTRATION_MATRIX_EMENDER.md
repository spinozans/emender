# Pre-registration — the matrix-state Emender (ops on a GDN2-class scan)

**Date:** 2026-11-13 · **Status:** registered before measurement · **Operator
authorisation:** "not about being cheaper than attention, but about being better and more
powerful… get an implementation that matches GDN2's performance."

## Why this exists

The tape-store Emender spends **78 % of its step in gradient-checkpoint replay**
(`ReplayBackward`, `reverse_kernel`, `Replay`; profile of the registered lattice arm,
`runs_150m/profile_lattice_emender_bounded.json`) because the state is a per-token
trajectory. Its store ops are 0.1 % of the step, and its slot budget (73) is a
search artifact, not a design target. Meanwhile **GDN2 beat the transformer on our own
150 M board (1.1407 vs 1.2807 nats/byte)** with a rank-1 matrix state and a chunked
parallel scan.

The five ops are rank-1 updates. So the Emender does not need a tape: it needs a matrix
state and a decision schedule — which is the algebra already proved chunk-scannable
(`MemoryAlgebra.AffineOperator`, `EmenderMemory.step_decomposition`,
`Event.combined_*_scan`, `FORMALIZATION_GAP_AUDIT` P1 closed).

## Architecture under test

- **State:** `S ∈ ℝ^{r×d_v}` per head, `r` = rank (memory capacity, NOT parameters).
- **Keys:** an orthonormal family; the active key table is a **prefix** `K[0:r]`.
  Capacity growth reveals further rows; no re-orthonormalisation, so CORRECT/POP are
  exact (`memoryTable_retrieves_orthonormal`,
  `linearDeltaWrite_overwrites_one_preserves_others`).
- **Writes:** APPEND `S ← S + v kᵀ`; CORRECT `S ← S + (v' − S k_j) k_jᵀ`;
  POP `S ← S − (S k_j) k_jᵀ`; DECAY `S ← λS`; MERGE = two rank-1 ops. All applied
  **given a fixed decision schedule** (`D ⊙ s + Σ e_i ⊗ δ_i`).
- **Decisions:** detached/architectural (as now: argmax match, thresholds, case codes),
  never differentiated through. The schedule is what the scan consumes.
- **Growth:** at **chunk boundaries only**, `r' = max(r + stride_floor, ⌈g·r⌉)` snapped
  to the precompiled shape set {64,128,256,512,1024,2048}. Growth is a zero-injection
  (linear embedding) and therefore scan-legal (`LiftedMemory`). The growth *trigger* is a
  deterministic chunk statistic, fixed before the run (write-residual mass and/or
  append-collision rate), reported per chunk.
- **Backward:** ordinary autograd through the chunked scan. **No checkpoint replay.**

## Claim set (registered)

- **C1 management.** At matched parameters, FLOPs and exposure, five-ops > append-only >
  no-management (plain GDN2) on held-out nats/byte. *Falsified if* the ops add nothing;
  then the 2.13 M 0.66-nat result is scoped as tape-specific and reported as such.
- **C2 capacity.** At **fixed parameters**, held-out nats/byte improves monotonically
  with rank. *Falsified if* the curve is flat.
- **C3 state class.** On a long-context / beyond-horizon probe where a fixed-rank state
  must alias, a grown state does not alias as much. *Falsified if* both alias equally.
- **C4 data efficiency.** A matched-exposure twin favours the managed store.
  *Falsified if* no sign.

## Arms

Wave A, fixed rank (parameter-matched; rank from the precompiled set):
1. `gdn2` — plain GDN2 backbone, no management (existing baseline, 1.1407).
2. `matrix_append` — orthonormal matrix state, APPEND only.
3. `matrix_ops` — all five ops.

Wave B, growth: the same three with chunk-boundary geometric growth; per-chunk rank
trajectory and the rank-vs-stop curve recorded.

Transformer at the same parameter count is carried as the third leg for context, not as
a claim.

## Stopping rule and reporting

2700 active seconds per arm (the standing rule); actual steps/bytes reported as
first-class deconfound columns; single seed per arm with the seed stated; matched-exposure
twins where a claim needs it. Every arm receipt binds this registration's hash, the
stream identity and the arm's rank trajectory.

## Gates before any claim

1. **Kernel reuse**: build on `/home/erikg/GatedDeltaNet-2/lit_gpt/gdn2_ops/chunk_gdn2.py`
   (Triton) — no new CUDA C++.
2. **Exactness**: with orthonormal keys, CORRECT and POP must be exact (bitwise on the
   affected basis direction) against a reference store; test that a POP leaves every
   other direction untouched to within the fp32 noise floor.
3. **Scan parity**: the chunked path must match the sequential per-token semantics within
   the established tolerance, and the strength/liveness channel must stay exact (death
   boundaries identical), as in `x44_chunk_scan`.
4. **Performance**: tokens/s must reach GDN2's class at matched parameters, and the
   receipt must show the replay kernels are absent from the profile.
5. **Determinism**: growth trigger and schedule reproducible from the recorded inputs.

## Honest scope

Rank `r` is capacity at fixed parameters — this tests Paper 1's state-class claim, not a
claim of being cheaper than attention: at large `r` the read is `T·r·d`, asymptotically
attention-shaped. The exponent-level win, if any, belongs to the exact-match selection
branch (`DESIGN_SUFFIX_ATTENTION.md`), not to this one. This branch exists to make the
ops claim **fast, exact and measurable**.
