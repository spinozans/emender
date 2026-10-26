# Part 6 execution update

Read scoring and local value weighting launch `(B*H, ceil(S/32))`
Triton programs for each token, using tf32x3 MMA. Empty tiles skip key/value
loads, MMA and partial payload writes. A second pass grids over stores and
eight-value output blocks to combine tile maxima, denominators and weighted
values with a stable softmax rescaling. A stream-ordered recurrent core then
handles routing and mutation. Heads share the launch grid; each store still
pads its query to 16 MMA rows.

Exact decay for the selected POP row is scalar in the recurrent core; reads
compute live-row strengths in their slot tiles. A final slot-tile pass
materializes strengths and ages once per chunk and records independent undo
groups. Small capture groups pack directly in registers, so concurrent slot
tiles do not share staging scratch.

This uses three launches per token plus final materialization once per chunk. Token groups, cooperative barriers and
integration of certified append phases into the full scan remain absent.
The core retains O(S) metadata vectors and register spills, but no longer
performs the O(S*V) weighted-value reduction. Token, start clock and journal
capacity do not specialize the core. Initialization builds free bitmap words
in parallel, visits live birth links rather than a scalar physical-slot loop,
and skips horizon calculation on inactive rows. Cold-state allocation is dense.

All-field recording packs changed bits at capture time. Score scratch is
now ephemeral and never mutates the persistent score buffer, so undo need
not record disposable scores. Persistent metadata byte rollback is retained.
Reservation uses initially live rows plus at most T new rows and heap depth;
counting imported live rows introduces a host synchronization. Private replay
copies and base state remain O(S). See REPORT.md for actual parity and gates.

The following decomposition is historical context; its older baseline status
is superseded by the part 5 and part 6 reports.

# Dependency decomposition and prototype

This is an analysis plus an unbenchmarked mutation prototype. Full
chunk-parallel training is not implemented. No measured speedup is claimed.
The executable correctness receipts are `tests/test_components.py`.

| Operation | Dependency | Parallel formulation | Qualification |
|---|---|---|---|
| Decay | Prefix product in real arithmetic | Scan decay factors, gather by birth/touch | Ordinary floating product scans change rounding; see below |
| Fresh append allocation | Running first-free | Exclusive/inclusive rank of the initial free bitmap | Valid only when no intervening frees or competing allocations occur |
| Age-1 address | Shift-by-one birth indicator | Previous-token append ID, plus imported age-1 tie handling | Target must survive pop/death; correction does not renew birth |
| Age-1 correction | Local update after addressing | Shift prior append IDs and scatter corrected values | Routing/error depends on prior contents; not universally independent |
| POP | Content-read argmax and strength recurrence | Sparse sequential event stub | Can affect all later read normalizations and free allocations |
| R-CAP | Youngest-live fallback and residual-dependent update | Sparse sequential event stub | Routing depends on previous events; not universally sparse |
| Read | Dense reduction over live slots | Batched score GEMM then weighted reduction | Work remains proportional to live context, even with sparse checkpoints |
| Pressure | Windowed request/cap counts | Prefix sums and shifted differences | Requests include CAP and exclude CONFLICT/NO_WRITE |
| Reverse rollback | Ordered writes on each slot | Reverse sparse event traversal | Repeated slot IDs must respect order |

## Exact decay is a constraint on the scan algebra

Native fp32 decay is `s[t+1] = round32(s[t] * d)`. A real-valued
prefix-product identity does not license replacing it with `s0 * d**age`,
or reassociating its floating products. The counterexample from X30 uses
`d=.8`, threshold `.20971521735191345`: seven native multiplies remain
alive; analytic power can free one token early. X31's test checks death at
token eight on a real leased GPU.

Function composition of rounded-multiply maps is associative, but an efficient
compact representation of that composition is not provided here. X30's
mantissa-interval jump scheduler is a useful route for exact timestamps;
this baseline instead executes the recurrence literally. That preserves death
semantics at the cost of an O(S) metadata sweep per token. The append prototype
accepts an exact unit-decay curve constructed by repeated fp32 multiplication.
It does not pretend analytic factors are a scheduler.

## Append phase prototype

`allocate_free_ranks` runs `tl.cumsum` on a free bitmap in one kernel per
store, producing the same increasing slot IDs as repeated first-free removal.
The next kernel grids over `(event, store)` and independently scatters keys,
gated values, age and exact final strength. It has no loop over tokens.
For a fixed, certified append-only run of length A, preprocessing costs O(S)
and mutation O(A*(K+V)), with two launches. Temporary IDs cost O(A).

The prototype is a **mutation subproblem**, not a complete tape scan: it omits
pre-write reads, decay of existing live rows, gate-error calculation and all
routing. The caller supplies already computed gates and guarantees distinct,
valid destinations and no deaths. Overflow is returned by rank allocation;
callers must stop before mutation when any store overflows.

An a-run alone is not a sufficient certificate. With sigmoid append gating,
the stored value need not equal the next raw value, so zero residual cannot
be assumed. POP and imported age ties also invalidate naive adjacency shifts.
Certification must use the binding F12g semantics, not token spelling.

Sequential correction stubs are the CONFLICT/CAP/POP branches of
`scan_kernel`. They are not yet integrated with the parallel mutation kernel.
To integrate, identify maximal certified runs on device, flush their deferred
writes before an event reads them, execute the event, and resume. In the worst
case every token is a dependent event. Sparsity is empirical, not guaranteed.

## Reverse replay

`DeltaJournal` is a tested sparse rollback primitive: old packed fields and
slot IDs are consumed in reverse, including repeated writes to the same slot.
Its 3-event/3-field fixture consumes 60 bytes at S=7 and S=11776. This is a
storage-size receipt, not a checkpoint-overhead timing receipt. It does not
implement gradients, capture deltas inside the forward kernel, or reconstruct
exact strengths from timestamps. G10 requires recorded routing/death decisions
to be inputs to a future backward kernel; rerunning forward routing is invalid.

## Measurement status

Sequential-vs-parallel speedup: **not measured**. The preregistered complete
first-order parity gate is unmet because fused backward is absent. Timing a
restricted mutation routine and presenting it as accepted tape throughput
would violate that gate. Full training parallelization remains future work.
