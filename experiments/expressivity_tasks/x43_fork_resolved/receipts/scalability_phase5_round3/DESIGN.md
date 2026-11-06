# Round3 profile and pre-implementation design

## Controlled profile: slot dimension is material

axis_profile_fixed.json measures the round2 selected native kernel, not an
independent reference. Three synchronized samples in each of two batches;
nonzero inputs, same fixture seed, physical4 broker lease. The first attempt is
retained but INVALID for isolating axes: install_rowkeys silently reduced BS to
min(S,2*T). The fixed observer overrides BS after that wrapper and records the
full allocation. Repeat hashes check stability only, not independent parity.

| T | S257 / BS512 ms | S513 / BS1024 ms | S1025 / BS2048 ms |
|---:|---:|---:|---:|
|128|10.066594|18.050146|35.383240|
|256|19.650748|35.844771|70.445921|
|512|33.941958|66.919216|132.024449|

At fixed T, doubling the slot tile nearly doubles cost. At fixed slot tile,
time is approximately linear in T. Width is fixed64/128; no claim to have
profiled actual1.3B widths or hardware counters. Time-only paging cannot remove
this slot cost: `for t in range(T)` is a runtime loop, while `ss=arange(BS)`
and slot-width layout conversions size the body. Therefore pivot to slot
splitting, as authorized by candidate2. Occupancy selection alone is insufficient.

## Candidate: split the step around globally parallel slot work

Retain the frozen per-token control arithmetic but split launches into:
pre-control (correction, expiry, array append); per128-slot score CTAs;
normalization (reference BS reduction topology); ordered256-slot READ pages;
post-control (pop, stream append/correct, decay, traces). Each next token waits
for post-control on the same CUDA stream. Initialize once per document/window.
No state crosses a document boundary. Padded tokens output positive zero and
leave state unchanged, with zero trace.

Carry: existing global FinalK[B,S,64], FinalV[B,S,128], FinalS[B,S] and
FinalMeta[B,7,S] are persistent store, not O(1) carry. Dedicated score[B,BS],
weighted-strength[B,BS] and step scalars[B,8] avoid scratch aliasing state.
READ carry[B,2,128] FP32 is O(1) in context/slot count. Its two lanes correspond
to even/odd slot rows. Each page extends those two accumulators in ascending
row order; first pair initializes without an extra zero addition, last page
uses unchanged fixed_sum across the two parity lanes. Page size256, multiple
of2, masks beyond BS. Score CTAs are128 slots; dimension64 reduction unchanged.

Composition-parity obligation (hard gate): for every valid token and every
page boundary, carry[p,c] must have identical FP32 bits to the original
row-parity accumulator after exactly those rows, including initialization,
rounding, signed zero, NaN payloads and multiplication association. No partial
page sums may be added together: floating addition is not associative. At the
step boundaries all seven metadata fields, strength, keys, values,18 raw trace
fields and reads must equal the monolithic reference. This is an induction over
READ rows nested inside an induction over token transitions. Kernel boundaries
alone do not prove compiler arithmetic parity; adversarial primitive, all ten
native pairs, state/replay/gradient and masked-reset gates are mandatory.

Control reductions still use a BS-wide scalar tile to preserve original
floating softmax reduction topology. Thus this candidate bounds value/key
width work and READ code, but does NOT yet prove all register state and compile
complexity bounded independently of BS. No such stronger claim is permitted.
If BS-wide control remains limiting, its exact reduction tree must be split
with tree-frontier carry, rather than regrouping page sums.

The ten-shape sweep and T2048 reference compilation retain their original
shapes/arithmetic. Reference compilation failure is an incomplete gate even if
the candidate compiles. Five paired timing samples; project native-only cost
at22 layers/two heads/two passes/2B targets. GO requires the standing real1.3B
standard, not a local gain. No production substitution or training authorized
by a primitive pass. DiLoCo requirement IDs inapplicable.

Supervisor transport discovery searched tool names/descriptions for
contact_supervisor/supervisor/intercom and found no callable tool. No contact
or supervisor decision is claimed.
