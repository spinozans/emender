# Cache removal receipt — the pivot to dynamic consultation (emender-xsa)

**Date:** 2026-11-09 (late) · **Decision:** the operator, in-session ·
**Action:** the extracted consultation cache at
`/home/erikg/flagship_consultation_features/` was REMOVED in full.

## Why

The extraction service reached ~400 GB RSS at 822M positions (~490 B/position
retention — see the COMPOSITION_AUDIT-era stop and the sxgc performance note).
The operator's ruling: the extraction must be streaming/bounded, and with the
maintained match-state ladder design (see the addendum in
`SXGC_EXTRACTION_PERFORMANCE_ASK.md`, delivered to ~/sxgc as
`NOTE_FROM_EMENDER_EXTRACTION_PERFORMANCE.md`), per-position consultation at
training-consumption speed (≥150k positions/s; ladder projection 300k–1M/s per
core, O(1024) bounded memory) makes a pre-extracted cache unnecessary.
**Consultation becomes live — "emender-xsa" — as the operator has directed.**

## State at removal (evidence preserved)

- Train phase: 822,524,650 / 1,466,137,768 positions (56.1%), 473,424 windows,
  ~4,600–4,956 positions/s sustained, 24 GB of sparse features on disk.
- Calibration and final_test phases: complete (journals preserved).
- Final-state evidence retained in this directory:
  `console_final_state.jsonl`, `service_journal_final_state.jsonl`,
  `calibration_journal_final_state.jsonl`, `final_test_journal_final_state.jsonl`,
  `train_journal_final_state.jsonl.gz` (11 MB), alongside the original
  registration/launch/validation receipts. Everything removed is re-derivable
  from the frozen stream registration and the spine container.

## Consequences (stated honestly)

1. **Dense arms (1–4) of the flagship and the 150M lattice: unaffected.** They
   never touch consultation features.
2. **Consulted arms (5–6) now gate on the live path:** the emender-xsa dynamic
   consultation must exist, pass the parity oracles (17-context brute-force +
   full row parity vs the retired verified chain), and sustain
   training-consumption rates before arms 5–6 fire. The sxgc ask is delivered;
   the native ladder path is the design.
3. **Matched-time doctrine:** live consultation moves query cost inside the
   arms' wall-clock budget. With the ladder this is negligible; it will be
   measured and reported as first-class columns (positions/s, consultation
   overhead per step) per the standing stopping-rule registration.
4. The leaky cache-build service is retired permanently; it will not be
   restarted.

## Registrations affected (annotations, not rewrites)

- PREREGISTRATION_1P3B.md and PREREGISTRATION_150M_LATTICE.md: their consulted
  arms' feature-delivery mechanism changes from "extracted cache" to "live
  emender-xsa consultation at parity with the retired verified chain." The
  stream, splits, windows, spine, stopping rules, and claims are unchanged.
