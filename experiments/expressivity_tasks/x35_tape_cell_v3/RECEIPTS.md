# X35 preregistration

Registered before X35 experiments (2026-10-27).

- (a) C=64, S=2944: complete pure-chunk forward >=3x X31 sequential.
- (b) Measure ladder/MQAR chunk pure fraction; prediction: a-runs >90%.
- (c) Complete mixed-chunk ladder training loop >=2x X31.
- (d) Zero bitwise trace, functional-state and gradient parity failures.

Do not substitute mutation-only timings for (a), inference for (c), or
1e-5 closeness for (d). Distinguish append-only mutation candidates from
certified entry-state-independent chunks. Density uses recorded X31 traces,
C=64, seed=0 initialized F12g projections, no learning. These are initialization
measurements, not trained-X31 checkpoint measurements. Historical JSONs contain
metrics, not model weights. Exclude incomplete chunks and report excluded tokens.
Separate all-a/all-b windows from whole-data statistics; padding is separate.

## Outcome

See REPORT.md and density.json. Targets (a), (c), and full (d) are NOT RUN:
the requested chunk-entry routing formulation has counterexamples before
performance engineering. Counterexample tests are successful falsification
experiments, not parity passes of an implemented v3 cell. No throughput claim.

## Addendum (part 2, preregistered BEFORE the trained-checkpoint measurement)

Registered 2026-10-27, before running density_trained.py. The open question
from part 1: the 50% event density was measured on an INITIALIZED X31 cell;
the certified stage-1 receipts say the trained instrument routes cleanly on
MQAR, so the 50% may be initialization noise. This addendum fixes the
protocol and the interpretation bands BEFORE any trained number exists.

### Protocol

- Instrument: the TRAINED tape_B0x4 arm. The frozen X28E package is READ-ONLY
  (verify_freeze re-verified at start, exactly the x33_pressure_probe/probe.py
  pattern): trainer.build_tape_model(configs/tape_B0x4.json),
  results_x28e_memory_head/tape_B0x4_seed0/best.pt loaded read-only,
  eval mode, no_grad, no training, no optimizer.
- Same part-1 instrumentation: the x35 classifier (MUTATION_CANDIDATE /
  ENTRY_PURE labels, event-token fraction, reasons breakdown), C=64 chunks,
  per-(example, head, chunk) stores; event tokens = case >= R-CONFLICT,
  POP fired, or any strength death. Full chunks only; excluded tail tokens
  reported.
- Ladder data, per bin: the exact positive word [SOS] a^k b^k [EOS] at the
  bin boundary k = 128 (bin ID / training top), 256 (bin 2x), 512 (bin 4x),
  1152 (bin 9x); 8 identical copies per bin (identical-token runs and head
  variability, exactly the part-1 protocol, not independent samples).
  Secondary band probe: per bin, 8 DISTINCT-k exact positive words evenly
  spaced in the registered band (clamped below at k=31 so every word has at
  least one full chunk), one word per forward (no padding).
- MQAR data: the first 8 words of the registered stage-1 validation pool
  (T=128, seed 28302; byte-identical to part 1's words, token hash
  cross-checked). The stage-1 embedding is stage-local and NOT persisted in
  best.pt (only the stage-3 vocab-5 embedding is). It is reconstructed by
  replaying the frozen deterministic construction sequence:
  torch.manual_seed(0) -> build_tape_model -> attach_multiclass_stage(65, 64)
  (the embedding draw is the first torch-RNG consumer after model
  construction; F12f stage-local embeddings are FROZEN random feature maps,
  never trained). The layer weights are the trained stage-3 weights. This is
  labeled a hybrid: trained layer + training-time stage-1 addressing
  geometry; it is NOT a receipt of stage-1-era behavior (the layer kept
  training through stages 2-3, and the stage-1 64-class head is not
  persisted, so no MQAR accuracy receipt is possible).
- Verdict datum: the LADDER bins (the curriculum's target task). MQAR is the
  associative-recall contrast stream only.

### Interpretation bands (registered BEFORE measurement)

- LOW: trained ladder event-token fraction < 10% -> speculative
  chunk-parallel execution (optimistic evaluation + journal rollback at
  first event) has long speculation runs; v3 viable.
- HIGH: trained ladder event-token fraction >= 40% (part 1 measured 50% on
  the initialized cell) -> the sequential correction core is intrinsic to
  the trained semantics; v3 pivots to read-GEMM batching + occupancy
  backing only.
- MIXED (10-40%): report both mechanisms with the measured numbers; no
  automatic verdict from the band alone.
- Phases are reported separately (all-a / all-b full windows) as in part 1.
- A LOW MQAR stream does NOT rescue speculative parallelism if the ladder
  datum is HIGH; it is reported as a contrast, not a substitute.

## Part 2 structural engineering preregistration (before measurement)

Registered 2026-10-27. Routing remains sequential; the trained-density HIGH
verdict is settled. X28E/X30/X31/X33 are read-only references.

1. Chunk-causal READ: C64, S2944, H4, K64/V128, width256, fp32. Target
   complete forward >=2x X31 using identical fresh-state end-to-end wall timing,
   allocation/synchronization included, B64, five warmups, twenty samples.
   Collect a matched X31 S2944 baseline: summary_part8.json itself has only
   S736/S11776 and cannot supply an S2944 denominator. No isolated read timing
   substitutes for this gate. All read versions are pre-token causal views.
2. Occupancy backing: live pools + preserved cold rows + stable logical map;
   public materialization, imported dead payloads, all first-order adjoints,
   owner mutation checks, snapshots, consuming continuation, exact undo.
   Fresh initialization allocation/work must scale with live occupancy rather
   than H*S. Re-run the original S11776 H16/H1 <=2 gate with unchanged B64/T16,
   five warmups/twenty samples, fresh-state allocation and synchronization.
   No subtraction of initialization or slowing H1 is permitted.
3. Timer wheel: replace heap insertion/cancellation with O(1) amortized death
   scheduling, exact absolute deadlines, long-horizon/decay=1 behavior,
   POP cancellation, post-POP versus end-decay phases, deterministic deaths,
   and journal undo/replay support. No stale-entry growth is allowed.
4. Parity precedes performance: all 109 X31 oracle tests applied to the v3
   implementation after any state representation change; outputs, native
   fields and first-order gradients at 1e-5, routing traces bit-exact. G10
   recorded-decision replay ownership remains unchanged. Component tests and
   unchanged-X31 passes do not establish v3 integration parity.

Only the X35 broker may allocate GPUs (physical 0-6, never 7); only the
project .venv/bin/python may run Python. No .pt artifacts. Report unimplemented,
failed and unmeasured targets explicitly, preserving original protocols.

### Part 2 outcome

FAIL to deliver the complete engineering scope. Causal-read and journal-aware
radix-wheel primitives have component tests; no complete v3 cell, compact
backing or replay composition is delivered. All registered performance gates
remain NOT RUN. Baseline-X31 and component results are separately identified
in REPORT.md and summary_part2.json. No component timing substitutes for any
gate. Earlier compile/expiry failures are retained as unsuccessful receipts.
