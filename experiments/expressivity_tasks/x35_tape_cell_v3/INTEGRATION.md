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

## Part 4: elastic prefix and deferred public tensors (partial scope)

`backing.py` adds an empty-origin allocation horizon for inference and training.
The native tensor prefix and all scan metadata use a physical capacity bounded
by cumulative chunk lengths, capped at architectural S. At most T appends can
occur in a T-token chunk; starting from a complete previous prefix, no smaller
physical capacity can exhaust before the last possible append unless it is S.
Logical IDs remain unchanged. R-CAP, free ordering and pressure therefore retain
the architectural meaning. Sparse training runs the same recorded G10 replay
with prefix-sized owners, then differentiably pads public tensors on access.

The public native facade subclasses TapeState. Accessing a tensor promotes the
entire handle to its exact dense public representation, including dead payloads,
terminal strengths, ages and untouched zero rows. It expands the heap/birth/decay
owners and rebuilds free words at the same boundary. Once materialized, the
existing dense continuation preserves public buffer addresses. Snapshots also
promote before copying/rebasing. Explicit payload/all-field recording promotes
before entering the existing dense journal path. Replay watches the versions
of promoted public native and metadata tensors, so promotion cannot bypass
in-place mutation detection. Weak ownership avoids retaining timed samples
until cyclic collection; orphan public native facades remain materializable.
The public RoutingTrace subclass similarly defers only its zero death-trace
tail; its slicing and concatenation API still materializes the exact shape.

Within a chunk, causal read strengths can advance by one native fp32 multiply
at each token boundary rather than repeating a timestamp jump at every emission.
POP multiplication precedes allocation reset, which precedes decay. Dead lanes
are masked and reuse resets strength to one. Reference/touch/death metadata and
the recorded replay remain authoritative and unchanged. The diagnostic
`IncrementalViews=False` specialization retains timestamp-jump emission for
bitwise comparison across imports, continuation, POP, death and reuse.

**This is not the requested occupancy-proportional representation.** The prefix
retains dead touched rows, and even no-write chunks increase its conservative
capacity. There is no live-row pool, separate cold-row store, or nonidentity
logical-to-physical mapping. Growth reconstructs metadata from the complete
previous native prefix. Snapshot/public materialization and explicit recordings
still allocate dense S-sized owners. An O(live) persistent/backward/undo memory
claim is therefore false. The zero-live continuation probe in gates_part4.json
provides an executable counterexample, independent of any fresh-start timing.
The timer wheel is still a standalone component; the live core keeps the heap.

The carried causal-strength vector is enabled only for reachable prefixes of
at most 256 slots. Wide imported states retain timestamp-jump emission to bound
register lifetime across the token loop. The attempted unrestricted variant's
validation was interrupted during compilation, not counted as a pass; its raw
log is retained as validation_part4_emission_interrupted.txt.
