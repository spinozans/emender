# Live causal read integration

`cell_fused.tape_scan` runs one sequential core program per (batch, head)
for the complete input chunk. No chunk-entry independence is assumed. The
core uses the X31 birth index, exact lazy decay, free bitmap, cancellable
heap and typed journal mutation primitives without changing reference files.
POP needs a query argmax before routing; it computes that argmax over the
current keys inside the core only when the POP threshold is exceeded.
There is no sequential value-mixture read.

Immediately before each token's POP/write/decay, the core emits logical-slot
payload-version IDs and rounded strengths. Initial payloads have stable slot
versions. An append or correction creates one immutable new version after
its journaled mutation; death removes visibility, and reuse creates another
version. Multiple later tokens share unchanged payload versions. The read
therefore cannot see its own write, future corrections, or future reuse.

After the core, `causal_read.chunk_read` scores every query in two launches,
preserving X31's 32-slot arithmetic. Version storage is
O((reachable slots + chunk length)*(K+V)); view metadata is
O(chunk length*reachable slots). This avoids per-token payload snapshots.
Known empty-origin continuations use the same cumulative allocation bound
as X31. Arbitrary imported stores require the full logical slot range.

The production custom autograd Function owns the exact journal and private
recorded trace. Backward rolls back to each token's boundary before applying
that token's read adjoints to the restored keys, values and strengths. Thus
shared-version gradients are accumulated at their proper mutation boundary,
and an allocation's adjoint is routed to its original input before rollback
continues. Backward recomputes read arithmetic from the restored state; it
does not recompute routing or liveness decisions. The forward's version/view
buffers need not be retained for backward. The standalone read primitive
continues to reject direct grad-bearing calls; callers use the cell's replay.

A local typed-journal class is required because undo constructs a continuation
handle and snapshot rebases replay addresses. Returning X31's FusedState from
v3 undo would cross implementation ownership boundaries. Low-level immutable
X31 primitives remain shared, while v3 journal handles and replay construction
are local. Tests must exercise the real v3 public module as well as the
unchanged oracle tests redirected to that module.

## Unimplemented storage change

Native fields and persistent metadata still have dense logical-slot backing.
Live/cold row pools, a journaled logical-to-physical map and materialize-on-
access public tensors are NOT implemented. A reachable-slot bound on reads
is not occupancy-proportional backing. The implementation must not claim
O(H*live) initialization or live-proportional memory. The full death trace is
also dense in (batch, chunk, head, logical slot), and training replay clones
dense owners. These structures remain even if causal reads improve latency.

The timer wheel stays a tested component. The integrated core retains X31's
heap and its sorted-slot tie order. Wheel ownership and journal-capacity,
continuation and replay changes are not silently substituted into the cell.
