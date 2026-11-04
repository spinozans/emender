# Conditional controls authorized by complete PB1

The complete B1-v2 PB1 delta is +0.07544000367724846, so the registered
append-only and unbudgeted controls are required. Fresh runs use separate
receipts c14_b_v2_append_only / c14_b_v2_unbudgeted and separate external
checkpoints. The old first-attempt control artifacts are retained untouched.
The original registered 2700-second, seed-3401, batch-16/context-128, BF16,
optimizer, dropout, source, document reset and frozen-window protocol is reused.

Both controls reuse the existing conditional-control forward kernels unchanged.
Their Python replay batching/capture, metadata reuse, telemetry reductions and
native training decode match the optimized B1 implementation. Append-only
preserves the registered array CORRECT/stale/DECAY exclusions, including the
stream-CORRECT guard for an array-owned target. Unbudgeted retains a finite
257-slot physical cap per head: a 128-position independent window can allocate
at most 256 slots, so this cap cannot bind. Aggregate peak and capacity-merge
counters must still verify the bound empirically for every training/eval run.

The first expanded battery failed two POP-counter checks for the 257-slot
configuration. The unchanged kernel uses S as an empty-read sentinel; with
BS=512 and S=257 that sentinel is also a padded lane. An empty read could thus
increment the POP trace without touching any live slot. Numerical outputs and
states were unchanged. A wrapper-only telemetry correction reconstructs exact
pre-read cardinality as previous post-decay live slots minus current stale
retirements plus current array allocations, and removes only these phantom
empty-store POP counts. No production/control kernel code was edited. The
failed output remains control_battery_failed_phantom_pop.txt.

After this correction all 90 relevant C14 tests passed on leased GPU 2 in
105.98 seconds. New gates compare native/scalar operation counts and gradients
at capacities 1/32/257 in both control laws, and verify registered-shape finite
backward, bit-exact checkpoint forward/gradients, disabled-dense exact parity,
and nonbinding 257-slot capacity. All existing core/feature/document/cache gates
also passed. control_gates.json binds every runtime/control source, tests,
complete battery log, triggering PB1 receipt and frozen B1 checkpoint.

The runtime still exposes no contact_supervisor tool; the new counter discovery
is receipted here without claiming a delivered progress update. This remains a
wrapper/telemetry fix inside the operator boundary, not a production-kernel
re-certification change. Actual throughput/exposure and control deltas remain
unmeasured until the independent registered runs and frozen evals complete.
