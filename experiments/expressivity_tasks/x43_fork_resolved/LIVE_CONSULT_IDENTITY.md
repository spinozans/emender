# Live consultation as *the* mode — train/serve identity

**Decision (operator, 2026-11-10):** the consult path is dynamic — the same live
path serves training and deployment. The pre-extracted cache is retired
(`receipts/flagship_stream/REMOVAL_RECEIPT.md`) and will not return.

## Why this is the right architecture, not just an engineering preference

1. **Train/serve identity.** The system we measure is the system we deploy. A
   cached-feature arm is a *different* system from a live-consult arm: the cache
   fixes feature semantics at extraction time, while deployment queries the spine
   on demand. Training live removes the entire class of "it worked offline but
   not in production" mismatch — no door, no drift, no re-derivation.
2. **The bottleneck was the artifact, not the query.** The retired cache reached
   ~400 GB RSS for 822 M positions (~490 B/position). At training consumption
   (~131 k positions/s at T2048 × batch 64) the *dynamic* path with a maintained
   match-state ladder is projected at 300 k–1 M positions/s with O(1024) bounded
   memory. The dynamic mode is both the deployment mode and the cheaper mode.
3. **The spine is already a first-class artifact.** Frozen stream + frozen spine +
   deterministic queries ⇒ the "live" path is *replayable*, so dynamism costs no
   scientific rigor.

## Hard requirements on the live path (fail-closed, each receipted)

- **R1 — Rate.** Sustained throughput ≥ the arm's consumption rate
  (131 k positions/s at T2048 × B64; reported per arm as actual positions/s).
  Measured, not assumed; a shortfall is a result, and the arm reports reduced
  exposure rather than silently slowing.
- **R2 — Determinism.** Same (stream document, byte offset, spine) ⇒ byte-identical
  rows. No cache-warming dependence, no ordering dependence, identical across
  process restarts and across arms.
- **R3 — Parity with the verified chain.** The 17-context brute-force oracle plus
  full compact-row parity against the retired extraction's semantics. A mismatch is
  a stop, not a rounding note.
- **R4 — Replay receipts.** Every consulted arm logs per-step consult counts, rate,
  error rate, and its own stream/spine hashes, so any run can be re-executed
  exactly against the frozen stream + spine.
- **R5 — Isolation and restart.** A consult failure must not silently feed wrong
  features into training: the arm stops (fail-closed), retains its last complete
  checkpoint, and resumes from it. The consult process may restart independently.
- **R6 — No materialization.** The path reads bytes from the archive/pile by
  offset; it never stages a text copy (operator directive, `SXGC_EXTRACTION_PERFORMANCE_ASK.md`).

## Accepted cost (stated, not hidden)

The training run is no longer hermetic: it depends on a live process. That is the
deployment reality, adopted knowingly. R2 + R4 bound the scientific risk (replay),
R5 bounds the operational risk (fail-closed + checkpoint resume).

## Consequences for the registered boards

- **150 M lattice, arms 5–6** and **1.3 B flagship, arms 5–6** consult live. Dense
  arms (1–4 and the append-only control) are unaffected and fire on nomination.
- The registration's extraction-cost line becomes a **consult cost inside the arm's
  wall-clock**, measured and reported as a first-class column alongside actual
  bytes/steps — consistent with the standing matched-time rule.
- The deployment claim in the paper becomes an identity rather than an analogy:
  the trained consulted arm and the served system are the same code path.
