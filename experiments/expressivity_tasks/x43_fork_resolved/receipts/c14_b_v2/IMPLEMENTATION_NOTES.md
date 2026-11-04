# Enabled-path bottleneck: implementation and gates

The dominant cost was scalar Python orchestration in NativeReplay's eager
backward, particularly per-slot DECAY, reads, stream operations, immutable slot
replacement and the resulting autograd graph. Production store kernels were
not the bottleneck being changed. kernel_boundary_audit.json verifies the B1
forward kernel's bytes and standing X31/X36/X40/X42 source against 8fa1786c.

The replacement batches rows/slots in ordinary differentiable PyTorch, compiles
one position's tensor orchestration, and captures the fixed launch schedule in
a CUDA graph. Ownership, expiry, valid lengths, keep masks, predictions, raw
bytes, all numerical inputs and upstream gradients are copied anew each call.
Every slot starts empty in every invocation. Returned gradients are copied
before reuse. Structural dependency bits preserve None versus zero gradients,
including the cancelling old value in old+(new-old), so AdamW does not start
applying decay to an originally unused parameter. The scalar oracle remains
independent; the unchanged native/replay tolerance check runs every backward.

Window-local integer ownership encoding is reused across heads/checkpoint
recomputation. Checkpoint backward skips only duplicate telemetry conversion;
the original forward retains its complete telemetry. Trace reductions use
NumPy integer sums/maxima; per-position occupancy and all counters remain.
Existing checkpoint cadence and exclusion rules are unchanged. No registration,
feature, op, parameter, corpus, frozen arm or document reset rule changes.

81 tests passed on leased card 2 in 32.07 seconds after the final telemetry
change. These include the 2,133,700 parameter inventory, exact force-disabled
dense parity, bit-identical raw cache answers, native/scalar lifecycle and
gradient comparison, BF16 16x128 finite backward, exact checkpoint comparison,
causality, shared budget, document boundaries and the full relevant C14 battery.
New tests cover batched/scalar gradients at capacities 1/2/32/64 and repeated
CUDA graph use with changed metadata, inputs, padding and upstream gradients,
including exact unused-gradient patterns. launch_gates.json binds sources,
tests, cache samples and the complete battery receipt.

The first full battery failed: a new test shadowed its batch-size variable;
also the compiler's default eight-variant limit was too small for the gate's
intentional budget/shape sweep. Corrected the test and scoped a 64-variant
allowance to this adapter. Failed output is retained in
launch_battery_failed_compiler_cache.txt; no training ran behind failed gates.

The baseline's last saved profile contains 11 steps, not the planned 200. The
initial PROFILE_NOTES.md discussed the then-current first 10-step snapshot;
step 11 completed before termination. The uncompiled batched prototype stopped
at 43. The graph, compiled and final optimized variants each completed 200
steps. These are isolated profiling runs, with repeated first-shard windows,
and no saved model; none enter PB verdicts. Final optimized warm median timings
under cProfile/synchronizing wrappers: forward 37.1 ms, backward 42.6 ms,
update 2.5 ms, total step 85.7 ms. The scalar baseline median was 15.73 seconds.
These instrumentation timings are not the registered training throughput.
profile_summary.json retains all counts, caveats and variant timings.

B1-v2 training must freshly initialize seed 3401 and run the unchanged 2700
active-second protocol using c14_b_v2 train. Its checkpoint is outside git at
/home/erikg/c14_dense_bytes/emender_c14_b1_v2_resume.pt. Its receipts stay here;
the first 237-step model/checkpoint/receipts remain in their original locations
and are the honest, throughput-confounded first attempt, never the PB verdict.
Further work: registered retraining, frozen-window paired PB1-PB4 evaluation,
then conditional ablations if PB1 moves. No numerical outcome is claimed yet.

Residual risks: remaining enabled-path cost and fixed-time exposure can still
differ from passive; one seed cannot establish generality; compiled reduction
rounding must continue to pass the fail-closed replay check. Ownership/occupancy
and nonzero read weights do not alone establish causal usefulness. Compilation
and graph capture costs in training are included in active time, not hidden.
No resilient DiLoCo behavior is changed/tested; gap-matrix IDs do not apply.
The runtime has no contact_supervisor tool (catalog discovery was the first
operation); no supervisor contact or decision is claimed. No WorksGood service,
workers, subagents, Frontier jobs or /mnt/nvme2n1 writes were used.
