# X31 experimental implementation — acceptance not achieved

X31 currently provides a working device-only sequential **inference baseline**,
a parallel append-mutation/rank-scan prototype, and a sparse rollback primitive.
It does not deliver the requested performance-first training implementation.
Training calls fail explicitly rather than silently using the X30 engine.
No training result, performance speedup or successful acceptance is claimed.

## Built and checked

`cell_fused.py` supplies F12g projections and a functional-state scan adapter.
`kernels.py` launches one program per batch/head store and loops over the whole
input chunk on device. Read precedes pop/write. POP uses query argmax; WRITE
uses the lowest-index age-1 slot, with youngest-live fallback only for CAP.
Allocation is lowest free index. Keys, values and strengths use fp32 HBM.
The kernel emits discrete decisions, deaths and a rolling 128-token pressure
register. The host contains no token loop or scalar token routing.

Part 2 ports X30's integer-mantissa interval jumps to Triton (`lazy_decay.py`).
The scan keeps reference strengths, birth and touch timestamps; access jumps
match sequential fp32 multiplication, including nonunit popped/imported
references. A cancellable device min-heap (`device_schedule.py`) supplies death
events and their exact terminal strengths. Inverse positions bound its storage
to O(S), including repeated touches. Death processing visits scheduled events,
with scalar metadata updates; it does not sweep for decay deaths. Already-low
imported live strengths retain the oracle's first-read-then-pop-death behavior.

Lowest-free allocation uses the three-level 64-bit GPU bitmap in
`device_free.py`. Birth timestamps replace recurrent age increments, but
birth-group target selection still reduces over the store: X30's sparse birth
index is **not implemented**. Import reconstructs the heap and bitmap each
chunk, so continuation still pays O(S) initialization and full state clones.
Dense read metadata spills/reloads and dense diagnostic death traces remain.
Read scoring still uses dimension-wise reductions rather than tensor cores;
head stores are independent programs, not a demonstrated GEMM batch.

Real GPU validation on lambda01, leased physical GPU 1 (RTX 6000 Ada),
PyTorch 2.9.1+cu128, Triton 3.5.1, Python 3.12.3:

- 31 tests passed in 3.38 seconds via `run_leased.sh -q -x`.
- 25 randomized imported-store cases: 5 seeds x slots {1,7,16,736,11776}; batch=2,
  heads=2, chunk=9, K=4, V=5, decay=.8. Outputs and functional state match
  X30 naive_reference at atol=rtol=1e-5; all discrete trace fields match exactly.
- Continuation splits each random case at token 4 and checks read/state and
  pressure equivalence. Imported age ties, dead rows and low strengths occur.
- Dedicated exact-death boundary, lowest-index age tie, zero-residual append,
  CAP pressure and 128-token pressure expiry checks pass.
- Free-rank scan, parallel append writes and sparse reverse rollback with
  repeated slot IDs/padding pass. Journal fixture is 60 bytes independent of
  backing S=7 vs 11776.

These tests establish **forward baseline parity on the checked cases only**.
They do not establish the full first-order gradient gate or production-dimension training parity.
An initial CPU Triton-interpreter attempt failed on unsupported tensor `**2`;
the implementation was corrected to explicit multiplication before GPU checks.
The first GPU subset passed 16 tests; the intermediate expanded GPU suite passed 20, and the final suite passed 31.

## Acceptance ledger

| Requested gate/deliverable | Result |
|---|---|
| Device-side sequential chunk scan | Implemented baseline; exact forward cases pass |
| Tensor-core-friendly batched scoring | Not implemented |
| Lowest-free sparse allocator and lazy scheduler | Device bitmap and exact lazy heap implemented; sparse birth index and persistent chunk metadata missing |
| Trace/state/first-order gradient parity | Partial: forward only; gradient gate unmet |
| Sparse captured delta checkpoints + fused backward | Not implemented; standalone rollback primitive only |
| MQAR >=99% by 5000 updates | Not run; training unsupported |
| Ladder >0.9 signal by update 2500, class recalls | Not run; no accuracy/recalls to report |
| S=736/11776 cost vs X30/X28E | Not measured; no accepted performance claims |
| >=10x X30 at S=11776 | Not established |
| H=16/H=1 <=2x | Not measured |
| Checkpoint overhead <=5% at S=11776 | Not measured; forward still copies state densely |
| Parallel decomposition | Written with fp32/routing qualifications |
| Append-phase measured speedup | Not measured; mutation prototype correctness only |

## Limits and next engineering work

The core mission remains incomplete. Finish the sparse birth index and
persistent metadata, then implement the tiled read path before treating this
as a performance candidate. Forward must capture old event fields at mutation time;
a boundary gather is wrong for repeated writes. Then implement reverse replay
adjoints with fixed recorded decisions (G10), test learned content, pop, query,
append and imported-state gradients, and remove per-chunk store copies.
Only after that complete parity gate may the registered timing and training
runs start. O(events) checkpoint metadata does not eliminate dense read and
read-gradient work. Usage EMA is intentionally not updated by this baseline;
it is excluded from the functional-state gate, matching X30's scope.

All changed source is confined to X31. X30 and X28E references are imported
read-only. No resilient DiLoCo behavior or Frontier job is involved; resilient
requirement IDs and Slurm Partition/QOS evidence are not applicable. No WG
service or agent delegation was used. The requested `contact_supervisor` tool
was not present in the available tool catalog; no supervisor contact occurred.

## Part 2 scope and validation

This is a **partial continuation**, not completion of the part 2 contract.
The tiled read, mutation-time captured journal, fused reverse replay and
custom autograd Function remain absent. No first-order gradient parity is
claimed. Consequently MQAR, ladder (including class recalls), and all cost
measurements remain NOT RUN. The numeric S=736 target is now explicitly >=1x
X30; the existing >=10x at S=11776, H16/H1 <=2x, and <=5% checkpoint targets
remain registered before any measurement.

New checks compare device jumps, exact horizons and terminal strengths
bit-for-bit with X30 across 128 references for each of seven decay rates
(.125, .5, .8, .99, .9995, 1-2^-24, 1), plus popped references and immediate
threshold neighbors. Cancellable-heap tests exercise 180 events on each of
two stores for three decay rates, including rescheduling, ties and no-death
decay=1. A bitmap fixture exercises high bits at all three levels up to slot
262143, full-store exhaustion, and reuse. These complement the original
25 randomized forward/continuation cases, not the missing gradient gate.

Development diagnostics initially found an int32 overflow in the interval
normalization constant and a mismatch between raw Python decay constants
and their fp32 ratios; both were corrected. Initial multiwarp integration
also encountered illegal memory access. Device mutation helpers now include
program barriers between dependent device metadata updates, and the
death drain is capacity-bounded with enabled device assertions. Intermediate
full-suite runs were interrupted while replacing implementations; they do
not count as validation passes. Large-store compilation time is not a timing
receipt. Final validation output is recorded separately in
`validation_part2.txt`.

Final leased GPU run on lambda01 physical card 1: **43 passed in 259.84 s**,
including compilation. All 25 seeded randomized forward and continuation
cases still pass at 1e-5, with exact routing/death traces. The 12 new checks
pass. This validates the checked forward implementation and scheduler/
allocator primitives; the complete forward-and-gradient acceptance gate is
still **UNMET**. No performance or learning success is established.
