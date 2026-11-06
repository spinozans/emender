# Phase5 work checkpoint: column-chunk candidate rejected

**The requested long-context fix is incomplete. No production kernel is changed
or promoted. The first internal rewrite reduces compiler spills but is rejected
for cost. The 1.3B/24h class remains unqualified.**

Phase2/3/4 receipts and the X31 part8 ledger define the requirements. Implemented
a separate explicit-opt-in kernel: 16-column global value staging, 128-row key
staging, original explicit reduction layouts, and private output-buffer scratch.
The original kernel and historical registrations are unchanged.

The first two attempts fail compilation. The next fails T64 reference and repeat
byte fingerprints; producer/consumer layouts require CTA barriers. Those
failures are retained. With barriers, **all ten nonzero native read/final-state/
raw-trace and repeat fingerprint pairs match phase2 exactly**. These are native
parity gates, not trained-model or long-context gradient certification.

| B | T | cap | median ms/head | registers | compiler spills |
|---:|---:|---:|---:|---:|---:|
|1|32|65|1.756022|248|0|
|1|64|129|4.810967|255|72|
|1|96|193|32.084826|255|528|
|1|128|257|46.239726|255|530|
|16|128|257|44.521244|255|530|
|1|129|259|218.653853|40|1472|
|1|257|515|1178.415656|40|3270|
|1|512|1025|2118.172970|40|3224|
|16|512|1025|2196.162302|40|3224|
|1|1024|2049|6937.956855|40|6154|

Five synchronized wall samples per row, broker physical card4. Values are
nonzero constant vectors; phase4 used zero values and ten samples. Thus these
are not matched-input speedup measurements. Candidate cost itself rejects it.
Compiler spill counts are not hardware spill transactions. Retained T1024
compiler-input PTX has 181835 lines; its virtual-register declarations are not
physical register use. Nsight hardware-counter permission remains unavailable
from prior evidence; no new hardware-counter profile is claimed.

Why the initial change fails: preserving the reference two-row/four-column warp
layout leaves duplicate column lanes/warps when the tile shrinks to sixteen
columns. BS2048 still assigns 1024 row positions per unique column lane. Both
previous-value and READ sums still span all BS rows. The staging reduces some
overlap while adding repeated state traffic; it does not bound the value
reduction's live range. This is source/layout analysis backed by compiler spill
counts and timings, not a measured instruction-level hardware lifetime profile.
See `receipts/scalability_phase5/EXECUTION_NOTES.md` for the decision and next
implementation constraints.

For the registered scenario, measured candidate head cost times 22 layers,
two heads and two native passes at T1024 gives 0.596230667 seconds/target,
1.677203 targets/s and **13801.636 native-only days for 2B targets**. This is a
hypothetical raw-byte budget-count scenario excluding all other work. It does
not replace the phase4 production-path diagnosis of 1429.604 days. Required
throughput remains 23148.148 targets/s; even with other work free the head must
cost **at most 0.502691 ms**. A tenfold improvement over phase4 still takes
142.96 days and cannot qualify the 24h class.

Next work must change the distributed value layout while explicitly preserving
the reference floating-point reduction tree with bounded intermediates. Merely
summing contiguous row chunks changes parenthesization. Direct selected-row
previous-value loads and row-local updates are additional candidates, subject to
signed-zero/NaN and exact update-expression gates. After a qualifying internal
path, deterministic cross-CTA sequence execution or another measured occupancy
solution is still needed; the single-sequence grid and full training/dense/replay
cost remain. No numeric engineering-cost or eventual-speedup estimate is invented.

The new long fixture supports T512/T1024/T2048, cap2T+1, configurable high-live
peak, seven masked tail positions, fresh document identities, complete native
state/replay/gradient fingerprints and document-reset identity. Fresh T512 and
T1024 runs at peak257 both pass exact fingerprints and fresh-document reset
identity, including masked tails. T2048 and configurable maximum-live fixtures
remain unrun. These tests do not cover every operation-rich case.
No paired >=2M real-target slice or 150M/1.3B-width training is run for this
rejected speed candidate. All those gates remain mandatory before promotion.

Fresh broker card5 regression: **109 X31 tests pass in13.62s; 108 C14/operator
tests pass in55.57s**, with no skips. The current glob includes three more tests
than phase2. **8/8 nonzero shape/state/read/trace pairs and 4/4 native/state/
replay/gradient pairs pass**, including repeats. All four terminal exits are0.
Sequential long T512/T1024 fixture terminal exits are also0 on broker card4.
This scoped battery is green; it is not the complete promotion battery.

The new source registration binds the separate candidate, launchers, probes and
current regression sources in this same commit. Historical source bindings match
and require no rebind because no bound historical source changed. Exact resume
is in `receipts/scalability_phase5/EXECUTION_NOTES.md`; completed receipts refuse
overwrite. No training or training checkpoint exists to resume from this work.
The next candidate must address the explicit reduction layout before another
expensive training gate. The implementation and operator objective remain open. No resilient DiLoCo changes; gap-matrix IDs are
inapplicable. No .pt artifact, WG, delegation, protected-card operation, sxgc or
/mnt/nvme2n1 change. AGENTS.md and CLAUDE.md remain unchanged.
