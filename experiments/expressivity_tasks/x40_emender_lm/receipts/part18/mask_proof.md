# Mask-only candidate: operand/lifetime argument

This is a source-level value-neutrality argument, conditional on correct
execution of the existing kernel. It does not excuse the shared scratch races
or certify a candidate for production. `candidate.py` changes only the cache
load predicate, independently of the 11 part-17 diagnostic barriers.

Let `c = Start+t`. For `t>0`, the unchanged selection is
`base = reference if touch >= c-1 else cached`, followed by the unchanged
multiply and `strength = reference if touch >= c else next_strength`.
The candidate suppresses a cache read exactly when `touch >= c-1` (or the
slot is not live). On that branch, `base` is `reference`; the cache operand
is not selected. At `touch==c-1`, the reference receives exactly the original
rounded multiply. At `touch>=c`, the final selection is the reference.
For `touch<c-1`, the cache load and every arithmetic expression are unchanged.
The proof includes unit decay and preserves signed zero/NaN bit patterns in
every selected operand; it makes no floating-point algebraic substitution.

The initialization invariant for the remaining loads is: a live slot with
`touch<c-1` at `t>0` was live at the preceding read and had its strength stored
to the cache then. POP updates touch to its mutation clock; allocation and
reallocation likewise set touch to that clock, so a slot created or changed
after the preceding read instead satisfies `touch>=c-1`. Death sets touch
to the following clock but also clears live. There is no other live-enabling
path. A previously live, unchanged slot retains both its last cache store and
its touch. Allocation reach is monotone within a chunk; every live slot in
the supported native continuation is within the scanned bound. Unknown native
imports scan the full slot grid. Thus the induction is not defeated by a tile
being skipped while empty or beyond the prior allocation reach.

Each continuation allocates a fresh cache. This does not require preserving
the cache across chunks: at local `t==0` the kernel bypasses it altogether
and evaluates `advance_unit(reference, Start-touch, ...)`. This first read
stores the cache for all live slots. This is also the initial step for imported
TapeState/FusedState values, including non-unit strength references and nonzero
ages. The mask introduces no initialization or changed prior-strength value.
Zero-filling would remove undefined reads as well, but is unnecessary for this
operand argument and is not used.

`lifecycle.py` tests imported and empty origins, native chunk continuation,
POP/death/reallocation, all four production policies and decay exactly one.
It compares unchanged source with finite and NaN cache poisoning to the masked
source with NaN cache poisoning. It compares bytes of public values, native
state, traces, pressure, evidence/errors, history and defined native metadata.
It requires observed POPs, frees, allocations and reused slots, rather than
assuming that random inputs exercise those branches. Chunk lengths include
one-token chunks and boundaries surrounding slot-tile width 32.
This suite does not substitute for the all-shape X31/gradient gates.
