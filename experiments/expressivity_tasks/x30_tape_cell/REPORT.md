# X30 practical tape cell v1

This package implements the timestamp store and a working trace-replay training
path. MQAR reached 1.000 mean per-class recall at update 500 on the full frozen
20,000-example validation pool. Full experimental acceptance also requires the
ladder phase diagram and the H=4/H=16 training runs; their status is reported
below rather than inferred from MQAR or from the native results.

## Implementation

`cell.py` stores head-parallel `[B*H, extent, K/V]` rows. Unallocated suffix
rows are implicit zeros; slot IDs are preserved. Each token performs two
batched matrix multiplies for the strength-weighted, pre-write softmax read.
The read scale is the executable oracle's `2/sqrt(K)`.

Birth timestamps replace age increments. An adjacency dictionary selects the
lowest-index slot born at `t-1`; linked birth groups maintain the newest live
slot for R-CAP. Sparse hierarchical bitmaps preserve tie breaking, including
imported age-zero ties. The free allocator uses at most three 64-bit bitmap
levels for supported budgets 1..262144. It needs no locks because each store
has one sequential owner; this is not a concurrent allocator API.

Strength references and touch timestamps replace full-store decay updates.
A cancellable integer timer wheel retires slots at their exact death boundary.
Retirement is amortized over allocations/touches; simultaneous expirations
can form a burst. POP and death events are sparse trace entries. Correction
and cap preserve the target's key, birth, and strength, as in the oracle.

`float_decay.py` addresses an easily missed correctness issue: analytic
`decay**age` can move fp32 death by one token. For example, at decay .8 and
threshold .20971521735191345, seven native multiplications leave strength
.20971523225307465 alive, while analytic power reaches the threshold early.
The scheduler jumps exact constant-decrement intervals in the IEEE mantissa.
A shared unit-strength decay curve represents the discrete exponential for
reads; arbitrary imported/popped references use exact scalar jumps on access.
This retains native fp32 rounding without per-token slot decay/death scans.
The shared curve grows with maximum requested reference age, independently of
slot count. Generic nonunit-reference reads have additional Python overhead.

The pressure register reports cap merges / append requests in the last 128
tokens, separately per example/head; windows containing no requests report 0.
It is returned as a detached `[B,T,H]` feature. The comparison decoder leaves
it available to downstream consumers and uses the native read-only feature
path for training comparability.

Forward saves fp32 chunk checkpoints (default 16 tokens) and sparse routing
traces. Custom autograd replays chunks in reverse. Replay consumes pop/write
indices, cases, deaths, and detached append errors; it never reruns threshold
or routing comparisons. Explicit trace overrides are supported. First-order
training works with `.backward()` and `torch.autograd.grad`; higher-order
derivatives are outside this implementation. Reverse replay uses functional
indexed tensor updates, which copy dense rows/stores. Checkpoint copies remain
O(S) per boundary; this is an important end-to-end cost limitation.

`naive_reference.py` and `routing_trace.py` are byte-identical copies of the
native eager oracle and its trace types. See `PROVENANCE.md` for hashes and the
F12/F12g/F12h binding. No score-argmax write correction was introduced.

## Validation

- CPU pytest: 51 passed. Tests cover all routing cases, allocation/reuse,
  cap pressure/window expiry, imported age ties and already-decayed rows,
  exact death rounding, streaming continuation, entry-state immutability,
  initial-state gradients, and replay after routing thresholds are changed.
- Outputs, functional state (keys/values/strength/live/age), and first-order
  gradients match at atol=rtol=1e-5. The unused native `usage` EMA telemetry is
  intentionally omitted; full telemetry-state identity is not claimed.
- A 2304-token default-decay test checks native outputs/state/query gradients.
  Exact scalar decay jumps are checked against sequential fp32 multiplication,
  including the complete approximately 1.38-million-token default fade horizon.
- Replicated H=4/H=16 heads have matching per-head outputs, discrete traces,
  pressure, and input gradients. This proves cell independence for the tested
  cases; it is distinct from completing full H=4/H=16 MQAR training runs.
- `runs/parity_cpu.json`: eight additional seeded stores, including B=736/H=16
  and B=11776/H=1. Exact traces; maximum floating absolute error 4.18e-7.
- `runs/parity_gpu.json`: the same eight cases on leased physical card 0;
  exact traces and maximum floating absolute error 2.33e-10.
- MQAR: `runs/stage1.json`, seed 0, B=736/H=1, batch 64, AdamW lr=1e-4,
  weight_decay=1e-4; full validation accuracy 1.000 at update 500, 322.7 s
  including evaluation. Frozen pool hashes are retained in the receipt.
  `runs/current_stage1.json` revalidates the frozen checkpoint with the final
  cell source: recall remains 1.000, with checkpoint and source SHA-256 hashes.

The T=128 MQAR dataset retains its exact generator/hash. Execution trims only
its unsupervised trailing pad suffix (31 executed tokens). A dedicated test
compares supervised loss and parameter gradients with full T=128 execution.

## CPU cost receipts

`runs/cost_cpu.json`, PyTorch 2.9.1+cu128, lambda01, one CPU thread. Times are
median warm measurements over five repetitions. Maintenance uses one full,
inert store with cap requests and no expirations during measurement.

| Slots | Eager maintenance us/token | Timestamp maintenance us/token | Ratio |
|---:|---:|---:|---:|
| 736 | 37.088 | 0.220 | 168.2x |
| 11776 | 78.797 | 0.186 | 423.0x |

The >=50x target is met **for this isolated workload**. It excludes reads,
content writes, checkpoint copies, and host/device synchronization. It is not
an end-to-end or arbitrary-pop-workload speedup. Native decay alone performs
S floating multiplies/token; its minimum listed maintenance work is >=7*S
scalar floating/integer/boolean operations. The isolated timestamp tick uses
<=20 estimated scalar/bitmap operations. Shared decay-curve preparation adds
at most one fp32 multiply per newly requested reference age. These are manual
operation estimates, not profiler-derived FLOP totals.

A complete forward comparison includes reads, routing and checkpoint copies
(B=1, H=1, K=V=16, T=16, full initial store):

| Slots | Eager us/token | Timestamp us/token | Ratio |
|---:|---:|---:|---:|
| 736 | 637.406 | 303.174 | 2.10x |
| 11776 | 1117.908 | 863.987 | 1.29x |

The empty-store head timer (B=1, T=32, K=V=16) reports 292.1 / 332.6 / 469.1
us/token at H=1/4/16. Host management still loops over independent heads;
head scaling is not exclusively through GEMM. These measurements establish
neither native-recipe GPU training throughput nor the full head-scaling target.

## Leased GPU cost receipts

`runs/cost_gpu.json` uses physical card 0. The timestamp maintenance number
times the CPU-side register/bitmap tick; the native number times CUDA tensor
operations with synchronization at repetition boundaries. This is an eager
baseline comparison, not a comparison with native fused Triton kernels.

| Slots | Eager maintenance us/token | Timestamp host tick us/token | Ratio |
|---:|---:|---:|---:|
| 736 | 150.186 | 0.231 | 648.8x |
| 11776 | 151.101 | 0.223 | 676.7x |

For the complete full-store forward, including reads and checkpoint copies,
the corresponding times are 1884.875 / 959.254 us/token at S=736 (1.96x),
and 1862.928 / 1342.568 us/token at S=11776 (1.39x). Dimensions and timer scopes
match the CPU full-store comparison. Empty-store H=1/4/16 times are
1215.6 / 1124.9 / 1268.7 us/token; launch/synchronization costs dominate this
small workload. Neither a 50x complete-cell speedup nor GEMM-only head scaling
was observed or claimed.

## Ladder pilot outcome

`runs/ladder.json` records 500 attended updates, effective batch 64, B=736/H=1,
and validation-only selection of update 500. The 2000-word validation balanced
accuracy is .786. Frozen 512-word pool results are:

| Pool | Balanced accuracy | Required |
|---|---:|---:|
| ID | .791016 | >=.99 |
| 2x | .597656 | >=.90 |
| 4x | .556641 | >=.90 |
| 9x | .332031 | <=.55 |

**The ladder phase diagram was not reproduced.** A low 9x result does not
certify the budget wall when 2x/4x already fail. Per-class recalls, confusion
matrices, class counts, and frozen pool hashes are retained in the receipt.
The process exits 2 for this failed experimental gate; it completed normally.
The pilot is right-censored at 500 rather than the native 30000-update cap,
so it does not establish an architectural failure or a result at full budget.

The training processes were launched before the final exact-fp32 decay
hardening and checkpoint-copy optimizations. Frozen checkpoints are evaluated
again by `evaluate.py` with source hashes, without further optimizer updates
or OOD-driven checkpoint selection. This distinction matters for interpreting
the pilot as experimental evidence for the final source.

## Deviations and remaining qualification

The executable native POP is multiplicative strength reduction followed by
threshold death, rather than unconditional kill. The actual native trainer's
stage embedding is frozen (F12f). Both are preserved and documented instead
of silently implementing the task shorthand differently. W_k and the write
enable are frozen/detached; W_q starts equal to W_k; POP bias starts at -4.

The measured MQAR run uses H=1 versus the certified config's H=4. Sampler
batches are sampled without replacement within each update rather than epoch
permutations. The ladder driver omits the intermediate stage-2 occupancy
curriculum and defaults to the native 30,000-update budget. The attended pilot
uses only 500 updates and therefore cannot establish that a longer run would
fail. No native GDN2/E97 training arm was rerun or disturbed.

Python host routing, exact nonunit-strength reads, sparse event processing,
and fp32 checkpoint snapshots are product risks. The forward allocator and
clock maintenance avoid slot scans, while the complete training path still
contains O(S) checkpoint/backward copies. A device-side implementation and a
broader event-heavy maintenance benchmark remain necessary before claiming
production training efficiency.

## Reproduction and isolation

Run from repository root with `PYTHONDONTWRITEBYTECODE=1`:

```bash
OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q -p no:cacheprovider experiments/expressivity_tasks/x30_tape_cell/tests
experiments/expressivity_tasks/x30_tape_cell/run_leased.sh train_stage1 --heads 4
experiments/expressivity_tasks/x30_tape_cell/run_leased.sh train_ladder --heads 4
experiments/expressivity_tasks/x30_tape_cell/run_leased.sh validate_gpu
experiments/expressivity_tasks/x30_tape_cell/run_leased.sh benchmark --output experiments/expressivity_tasks/x30_tape_cell/runs/cost_gpu.json
```

Each GPU command acquires one genuinely idle, unleased card from 0..6,
maintains a lease heartbeat, and releases it in the broker's trap. Card 7 is
excluded. GPU entry points reject CUDA unless the lease environment agrees.
The recorded training runs used physical card 0. No WG service/dispatcher,
workers, subagents, Slurm jobs, or external message tools were used. Checkpoints
are retained locally and ignored by git; receipts contain no model tensors.

The final-source frozen ladder re-evaluation in `runs/current_ladder.json`
returns exactly the same four bin scores as the pilot. It records source and
checkpoint hashes; no weights were updated or reselected during re-evaluation.

Final reference isolation check: all X28E package path/content hashes (excluding
`__pycache__`) still reduce to
`233117bca17483a4e7836a784801b74d07659e7880589e011ddfee036bf19bf4`.
`runs/source_manifest.json` records both endpoints and the final X30 code hashes.
The X28E tracked diff is empty; GPU 0 was released, and protected card 7's
long-running job remained active. `runs/operation_counts.json` gives explicit
manual fp32 multiply counts alongside the mixed integer/boolean counts.

Qualification summary: functional parity and H=1 MQAR passed; the full
telemetry-state contract, ladder phase diagram, H=4/H=16 training, and broad
workload maintenance-performance qualification remain incomplete. This package
is an experimentally documented v1 candidate, not a complete certification.
