# Round 2 decision before implementation

Read 3cd3fb63 phase5 result and EXECUTION_NOTES. The inherited claim that
T2048 is green is contradicted by those receipts: it was unrun. Re-run all
three long fixtures for this candidate rather than inheriting that claim.
No contact_supervisor/intercom tool is exposed (searched name and description).
No supervisor contact is claimed. No WG, delegation, or training is used.

Static profile of retained T1024 compiler-input PTX is in static_profile.json.
Dominant expanded instructions are whole-context value loads/updates and layout
conversions: line73 21507 instructions, line60 14342, line177 7195. This is
compiler-input attribution, not hardware counters or measured physical liveness.
The source makes the structural problem explicit: BS/2 values per column lane
remain simultaneously live, including in previous-value selection and READ.

Choose candidate 1: bounded value intermediates. Serialize value updates in
128-row pages, load the selected previous value directly, and stream READ with
two row-parity accumulators per column, retaining the final two-warp reduction.
The expected reference per-thread accumulation is ascending even/odd row order;
byte gates must establish it for this compiler. Do not claim full bounded kernel
state: scalar metadata still scales with BS/256. No promotion from spill counts.
Compare matched-input reference/candidate times, all ten shapes, repeats, full
X31/C14 battery, short gradient/shape gates and T512/1024/2048 masked resets.
Any byte failure or slower long-shape ms/head rejects the candidate. Full trained
model/paired-target/full-width gates remain required for production promotion.
No resilient DiLoCo behavior change, gap-matrix requirement IDs inapplicable.
