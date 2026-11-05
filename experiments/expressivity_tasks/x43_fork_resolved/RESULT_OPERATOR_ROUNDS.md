# X43 A — operator rounds at generation: bounded diagnostic

Registered in 2c939baf; dense interface/compilation corrections recorded before
the valid dense sweep in ff317361 and 8f5b377e. Results in
`receipts/theory_rounds/results.json`; source and protocol SHA256s retained.

PA1 is refuted on this sample: increasing rounds worsens fork NLL for both
frozen checkpoints. This tests teacher-forced generation positions, not
free-running generation. Four 128-target document-bounded windows give 512
targets, including 169 forks. All selected windows are in calibration; no
final-test quality conclusion is claimed. No sample or threshold tuning.

| Arm | k | Fork NLL | Total NLL | Logical BRANCH reads | Seconds |
|---|---:|---:|---:|---:|---:|
| B1_v2 | 1 | 1.912021527 | 1.284314359 | 1024 | 0.832 |
| B1_v2 | 2 | 2.012622617 | 1.311142977 | 1360 | 7.873 |
| B1_v2 | 4 | 2.165865681 | 1.379526913 | 2038 | 12.950 |
| dense | 1 | 2.451763314 | 2.123275926 | 1024 | 0.143 |
| dense | 2 | 2.923046688 | 2.470931395 | 1362 | 8.480 |
| dense | 4 | 3.554979202 | 2.863157013 | 2038 | 7.483 |

PA2 has no robust support. At k=2, B1's mean layer-local rounds are
1.9821 (112 low-entropy layer/positions) vs 2.0000 (226 high); dense
is 2.0000 in both. At k=4 both arms hit the bound in every fork/layer,
so entropy does not distinguish halting there. The two B1 early halts at k=2
vanish at k=4 because previous positions evolve under different budgets.
Histograms and per-round CORRECT/strength-delta records are retained.
k=1 halting is the fixed bound, with no iterative op trace.

PA3's query-collapse watch does not fire: reads grow monotonically for both
arms. These are cached logical reads per layer round; live index RPCs=0.
The comparison includes one read at nonfork positions. No claim of index
latency or free emission through forced spans is established.

The k=1 gate passes byte-exact log-probability and per-target loss SHA256
fingerprints on both frozen checkpoints and all four windows. It delegates
to the original batched forward, preserving GEMM shapes; this establishes
API compatibility, not independent scalar-schedule equality. Extra rounds
use a residual stream and persistent window-local stores (B eager oracle,
dense native continuation). Repeated input observations are forbidden for B.
The dense native operator clock advances for every round. Slot budgets remain
fixed. Tests directly observe the same B store receiving clocks 0,0,1,1.

Failures retained: attempt1 unhashable list window metadata (exit1); attempt2
missing native telemetry scratch buffer (exit1), with an invalid nonzero-input
dense baseline excluded; attempt3 terminated during per-bound JIT compilation
(exit143). Final run exit0; conservative continuation bounds avoid the compile
storm. No production kernels, training interfaces, checkpoint files, or union
service changed. Physical GPU3 through the broker.

Residual risks: tiny calibration-only sample, FP32 rather than standing BF16,
k>1 scalar GEMM scheduling vs k=1 batched scheduling, eager B vs native dense
store implementations, layer-local halting, strength-only convergence metric,
no free-running generation, no live state-conditioned branch expansion. Dense
has zero consultation inputs; exact branch labels decide only where to spend
extra computation. This is a bound experiment on the frozen residual/store
policy, not proof of a general operator-search law or a serving speedup.
No resilient DiLoCo behavior changed; gap-matrix requirement IDs inapplicable.

Manual notes: contact_supervisor is absent from the active tool catalog. No
substitute orchestration service or subagents used. All three failed attempts
and the valid sweep are retained. Frozen checkpoint hashes are checked before
loading; no .pt files are staged. Unrelated pre-existing changes are preserved.

Validation: all 117 X43 tests pass on broker GPU3 in196.55 seconds, including
17 CUDA tests skipped by the separate CPU run (100 passed,17 skipped).
The initial test collection failure from missing importlib mode is retained;
`--import-mode=importlib` resolves existing relative test imports. Independent
receipt checks pass four k=1 fingerprints, six curve rows, halting conjunctions,
round bounds, and logical query accounting. Post-run frozen checkpoint hashes
still match the registered hashes exactly.
