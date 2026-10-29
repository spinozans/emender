Tool metadata searched for contact_supervisor before work: no matching tool exists.
No alternate messaging, WG, dispatcher, workers or subagents used.

Source finding: the production batched read is X35 causal_read._tiles/_reduce,
which uses elementwise FP32 multiply/reduce, not FP32 GEMM. The reverse uses
scalar-dimension FP32 loops, not GEMM. A BF16 GEMM rate-class explanation cannot
be assumed; a tensor-core read candidate changes both datatype and algorithm.

UPDATE: first complete T512 diagnostic (B64 accumulated as B32) observes
~4.89 s fused reverse and ~0.24 s combined initial/recompute batched read
within ~11.97 s instrumented wall. This is not a sustained rate. Additional
instrumentation is separating forward sequential core from dense work.
Reverse score/softmax and event adjoints remain fused, so no invented split
or direct conversion of component microbenchmarks into full-step savings.

UPDATE: side experiment replaces only the chunk read with BF16 tensor-core
QK and weighted-value products; all store kernels and production sources are
untouched. Existing FP32 replay is explicitly a surrogate gradient. The CPU
mathematical spec is not a bitwise MMA oracle. This limits the decision evidence
and blocks production adoption even if the paired pilot slice looks benign.

UPDATE: T2048 forward-core attribution is 15.47 s (initial plus checkpoint
recompute), fused reverse 36.91 s, batched read 1.85 s, reverse setup/teardown
0.32 s within 72.38 s instrumented wall. The sequential core is ~21%, so
its smallness is not confirmed under this fixed B32 checkpoint envelope.
Pure and control use identical accumulation/checkpoint/input storage, with
pure-specific journal packing retained. Leases may select different approved
physical cards; per-arm IDs and hardware inventory are retained.

UPDATE: eight recorded-view BF16 forward cases completed. Candidate/native
speedup is about 1.02–1.29 for shallow/T512 cases and ~0.80 for deep T2048.
All eight CPU prefix comparisons fail bitwise equality despite ~1e-7 numerical
differences. Inputs are bytewise unchanged. This is a preserved certification
failure, not a relaxed pass. The isolated cached query-adjoint matches its
uncached comparator bytewise in all eight cases, but is not a full replay.
Two deep-lane caches at physical B32 across 24 blocks cost 1.5 GiB (T512)
and 12.9375 GiB (T2048); the actual budgets are 736/1472.

UPDATE: paired pilot runs completed 5,013,504 tokens each with matching token
and target hashes. Max absolute loss divergence is 2.8014183e-5; mean
6.3001720e-6; final difference 7.6293945e-6. This applies to the documented
BF16-forward/FP32-surrogate-replay recipe only; it does not repair its missing
bitwise read/adjoint certification. Fixed-envelope sustained arms are running
sequentially; their raw times will replace no historical receipt.

Preserved non-numerical failure: the initial finalizer assumed that part9's
production JSON included a sources field. It does not; the corrected finalizer
requires part8's source-bound production gate plus part9's fresh pressure gate.
The initial KeyError log is retained as finalize_initial_failed.log. No numeric
assertion or parity threshold was weakened.

UPDATE: unchanged production T512 fixed B64/B32 matched-envelope arms both completed 3 warmups and 20 updates: pure 2761.32 tok/s, transformer 6110.27 tok/s, step-time ratio 2.2128. This is a new fixed-envelope measurement, not a claimed improvement over part8 per-arm envelopes. T2048 sustained timing is in progress.

Measured profile source is archived as profile_source.py; the earlier profile
without core isolation is profile_initial_source.py. The final profiler CLI
removes an unused --steps option (it always runs exactly one diagnostic update).
Receipt validation binds the measured versions to those byte-identical archives.

UPDATE: added an opt-in CTA-clock reverse diagnostic to probe the fused region
rather than treating it as opaque. It will require fresh pressure-inclusive
parity before profiling B64/B32 at each context. Device cycle fractions and
cycle-weighted device-time estimates will be labeled estimates, not additive
wall phases; added barriers/registers can perturb execution. Production code
is unchanged and no optimization is implied. The attended sequential driver
will execute these after the sustained arms and read/store component checks.

UPDATE: unchanged T2048 pure arm completed all 3 warmups and 20 measured updates at 1808.83 tok/s. Matched transformer timing is now running sequentially; no partial timing is being reported as its sustained rate.

UPDATE: all four fixed-envelope sustained arms completed. T512 pure/control
ratio is 2.2128; T2048 is 3.2073 (1808.83 vs 5801.37 tok/s). Fresh unchanged
B128 pilot is 55,429.54 tok/s; 100k fails. B32 read-only BF16 also slows deep
T2048 lanes (~0.84x / ~0.89x). Identical-input store/routing byte checks pass.

UPDATE: the T512 clock-instrumented reverse passes fresh pressure-inclusive
parity. CTA cycles: undo/input 8.68%, strength/store adjoints 16.37%, repeated
score/softmax 4.07%, read adjoints 70.88%. Cycle-weighted score/softmax device
estimate is 197.45 ms, not the entire ~4.9 s fused reverse. This is an
instrumented diagnostic, not an exact wall split or a measured caching speedup.
T2048 clock parity/profile is running sequentially.

UPDATE: T2048 instrumented reverse also passes fresh pressure-inclusive parity.
CTA cycles: undo/input 4.87%, strength/store adjoints 18.80%, repeated score/
softmax 11.79%, read adjoints 64.53%; cycle-weighted score/softmax device estimate
4,363.98 ms. Both instrumented full-shape diagnostic losses exactly match their
uninstrumented counterparts. This supports targeting read-adjoint execution,
not calling the whole reverse a duplicated read forward. All 18 attended,
sequential GPU commands completed successfully; BF16 bitwise read certification
still fails and neither 1x nor 100k is accepted.

Final whitespace check found Matplotlib SVG line-end spaces and extra EOF blank
lines in two experimental sources. SVG whitespace was normalized. Exact measured
IEEE/clock source bytes are retained in measured_sources/*.py.gz; current sources
remove EOF blanks only. Validation checks decompressed measured hashes and the
whitespace-only relation to current sources. The generator/plot helper now emit
clean whitespace. No arithmetic or measured result changed.
