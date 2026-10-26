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

Literal fp32 decay with fusion disabled preserves the native death boundaries,
including imported/popped strengths. This is an exactness baseline, not the
requested lazy scheduler port. Dense age/live/strength sweeps, dense death
traces and full entry-state clones remain. The read uses dimension-wise
reductions rather than tensor cores; head stores are independent programs,
not a demonstrated GEMM-batched implementation.

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
| Lowest-free sparse allocator and lazy scheduler | Not implemented; dense reduction/recurrence baseline |
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

The core mission remains incomplete. Implement the exact timestamp scheduler,
GPU free/birth structures and tiled read path before treating this as a
performance candidate. Forward must capture old event fields at mutation time;
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
