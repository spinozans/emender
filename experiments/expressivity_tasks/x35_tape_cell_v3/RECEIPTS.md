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

## Part 3 protocol and scope

The part-2 structural preregistration is unchanged. `gates_part3.py` imports
X31 part-8 `measure()` unchanged: B64, fp32, K64/V128, width256, fresh-state
complete cell including projections/allocation/synchronization, five warmups,
twenty samples, median wall milliseconds/token and CUDA-event samples. The
C64/S2944/H4 speed denominator is a newly matched X31 run with the same
weights and inputs; there is no S2944 number in historical summary_part8.json.
The S11776 H16/H1 gate keeps the original T16 protocol without subtracting
initialization or adding work to H1.

Memory is measured separately from latency after dropping timing samples and
clearing the CUDA cache: incremental allocated peak and retained allocation
for one complete fresh-state call, live-row count and total logical rows. The
B64/H16/S11776/T16 row explicitly reports sparse live occupancy. Dense base
state bytes are also collected by X31's original accounting function. CUDA
allocator reserved bytes are not mislabeled as occupied payload bytes.
Standalone native initialization is measured with the same five/twenty T16
procedure to receipt the remaining floor; it is not subtracted from any gate.

The v3 implementation integrates causal view construction and G10 replay
composition. Its backing is still dense, so the occupancy-proportional backing
and live-proportional memory requirements FAIL by structure. The timing and
memory receipts must say this even if a timing ratio passes. Timer-wheel cell
integration remains open, with the reference heap retained. Validation first
runs the unchanged 109-test X31 suite with only the public cell module redirected
to v3, plus X35 tests. Reference component tests still exercise the shared
reference primitives; passing the suite is not proof of a nonexistent storage
representation. `validation_part3.txt` and XML describe the final source;
compile-failure and intermediate logs are historical unsuccessful/intermediate
receipts, not acceptance substitutes.

### Part 3 measured outcome

All registered gates FAIL: complete C64/S2944/H4 speedup 1.879757x (<2),
S11776/T16 H16/H1 ratio 7.407990 (>2), occupancy backing not implemented,
and sparse-occupancy memory still dense. H16 native-init floor is
0.664066 ms/token. B64/H16/S11776 has 8,192 live of 12,058,624 logical rows,
10,095,140,864 base-state bytes, 10,658,721,792 incremental peak bytes, and
10,298,073,088 retained bytes. See gates_part3.json for all raw samples and
REPORT.md for exact remaining structures. No threshold, protocol or sample
selection was changed after observing these failures.

Final integrated validation: 141 passed (109 unchanged X31 suite + 32 X35);
separate no-redirection v3 integration validation: 4 passed. These establish
causal-view/recorded-replay composition with the existing dense representation,
not parity of unimplemented occupancy backing. Full engineering delivery is
INCOMPLETE. The timer wheel is not integrated.

## Part 4 continuation receipt

The >=2x C64/S2944/H4 and <=2 H16/H1 S11776/T16 thresholds are unchanged.
The final driver asserts AST equality of its timing loop with read-only X31
part-8 measure(): five warmups, twenty samples, synchronized wall/CUDA events,
medians per token, fresh complete cell calls and no initialization subtraction.
Only accounting understands deferred native tensors, outside timed samples.
All raw samples, memory measurements and component floors are retained.

- `validation_part4.txt` / `.xml`: complete unchanged 109-test X31 plus prior
  32-test X35 suites and added backing/emission tests.
- `parity_part4.json`: source hashes bound to a successful complete validation;
  gate execution rejects stale hashes, failures, errors, skips or short suites.
- `gates_part4.json` / `.txt`: final registered verdicts, physical-state bytes,
  incremental peak/retained bytes, initialization/projection/scan floors and
  the zero-live continuation counterexample.
- `summary_part4.json`: concise final status and explicit residual blockers.
- `gates_part4_before_emission.json` / `.txt`: earlier different-source run,
  retained rather than replaced to select a threshold-crossing sample.
- `validation_part4_import_failure.txt`, `validation_part4_usage_failure.txt`,
  `validation_part4_emission_interrupted.txt`: interrupted/failed development
  attempts; none is counted as final parity evidence. Intermediate inference
  and combined parity runs are labeled separately.

The last structural target is not delivered by an allocation-horizon prefix.
Live-row pools, separate exact cold-row storage, a journal-aware nonidentity
logical map, and live-proportional continuation/backward/undo ownership remain
open. Fresh sparse footprint is not substituted for that structural contract.
The timer wheel remains separate from the heap-based live scan. No benchmark
floor is subtracted or used to relabel a failed registered target as a pass.

## Part 5 — incomplete delivery and residual-cost adjudication

The original protocols and thresholds are unchanged. `gates_part5.py` asserts
AST identity of its full-cell timing loop with X31 part 8, requires a current
successful source-bound parity receipt, retains all five-warmup/twenty-sample
wall/CUDA timings, and adds the omitted S736 head pair. Its six verdicts include
full parity, forward, both head budgets, backing and memory. No initialization
subtraction or H1 slowdown is allowed. Gates are measured before diagnostics;
only the final complete run determines the part-5 result.

`structural_cost_part5.py` extracts the production `_tape_scan` allocation prefix
via AST, capturing its real scan arguments. Preparation timing includes that
prefix; core-only timing prepares a NEW state before each timed interval and
runs the captured real kernel. These independent diagnostics use B64/T16,
K=Q/V random inputs, writes on, POP/logits zero and physical S16. They do not
replace or decompose registered complete-cell samples. Core timing excludes
preparation explicitly, unlike every registered gate. Two regression tests
check production decisions/payloads and compatibility with registered memory
accounting without collecting timings during parity.

Final evidence:

- `validation_part5.txt/xml`, `parity_part5.json`: **160 passed**, no failures,
  errors or skips; unchanged 109 X31 and 49 existing X35, plus two diagnostic
  tests. Source hashes cover implementation, diagnostics, gate driver and
  validation runner. Parity ran before final performance collection.
- `gates_part5.json/txt`: forward **2.194987x PASS**; head ratio S736
  **2.348039 FAIL**, S11776 **2.349795 FAIL**; backing and memory **FAIL**;
  full parity **PASS**. Physical GPU 1 through the X35 broker.
- `structural_cost_part5.json`: exact fresh preparation H1/H16
  **0.031334/0.035144 ms/token**; sequential store core
  **0.056782/0.190631 ms/token**. All samples retained. The measured component
  ratio is not an irreducible hardware floor.
- `summary_part5.json`: `scope_complete=false`, `v3_closed=false`, explicit
  residual risks and failed/unfinished attempt chronology.
- `REPORT.md`: six-gate verdicts and parts 1–5 convergence table.

The fresh T16 path already allocates physical S16, independently of either
architectural budget. An O(architectural S) fresh initialization residual was
not found. The core still runs 64 versus 1024 sequential store-owning programs
at H1 versus H16. At zero live occupancy four T16 continuations grow physical
rows 16 -> 64 and base state 14,270,464 -> 55,410,688 bytes. This continues to
falsify occupancy-proportional backing; no fresh-start memory figure overrides
that structural result.

Unsuccessful/intermediate attempts are preserved: the initial new diagnostic
fixture used the wrong argument index (158 existing tests passed, one new test
failed); one diagnostic sample had the wrong accounting output type; another
duplicated `physical_slots` returned by accounting. The raw failures and the
successful pre-accounting/pre-duplicate-accounting validation/hash receipts
remain separately named. None counts as final-source acceptance or a favorable
benchmark selection. Final complete collection follows corrected diagnostic
accounting and renewed full parity.

**No live/cold pooling was implemented and no production representation was
changed in part 5. This fails the requested engineering work order.** Pooling,
nonidentity logical ownership, occupancy-proportional continuation/replay and
the heads gates remain open. The diagnostic result is not the permitted floor
adjudication after completed pooling. No universal impossibility is claimed.
Only X35 artifacts are changed; references and unrelated workspace edits are
preserved. No GPU7, .pt output, WG service/workers/delegation, Frontier/Slurm or
resilient training changes. `contact_supervisor` was searched first and is not
available in the callable tool catalog.
