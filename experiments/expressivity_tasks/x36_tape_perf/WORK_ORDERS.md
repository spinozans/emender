# Remaining X36 work after part 6

The overall target set remains **FAILED**. [REPORT.md](REPORT.md) contains the
parts 1–6 convergence table, the final measured gates and limitations.
[summary.json](receipts/summary.json) is the machine ledger;
[part6_failure_arithmetic.json](receipts/part6_failure_arithmetic.json) retains
decomposition and required savings without asserting an irreducible floor.
[PART5_DIAGNOSTIC.md](PART5_DIAGNOSTIC.md) is historical diagnostic evidence.

Part 6 implements exact-union sparse adjoints and hot private key/value replay
payloads for short nonempty entries. It separates measured entry cardinality
from logical reach in journal sizing and removes the reproduced raw-output
owner cycle. The original 143 tests and 36 new regressions pass. Fifty
production-path imported/continued capacity points survive without OOM,
including S-32 occupancy stress. This is an implementation delivery with
performance failures, not closure of X36.

1. **Close the expanded continued capacity gate without slowing a baseline.**
   Live1024 and live2800 matched-entry doublings pass. Live17 and live256 still
   fail the first S2944→S5888 doubling; the second doubling now passes. Full
   logical-S operands, native forward traversal and the oracle's 4→16-warp
   layout remain. A narrow eight-warp variant failed exact bytes and was
   reverted before performance. AoS high-occupancy baseline inflation is
   rejected explicitly. Preserve stable logical IDs, immutable cold backing,
   all journal address targets/restoration, required cold native gradients and
   exceptional signed-zero arithmetic. Measure both imported and continued
   paths, every capacity and small through near-budget entry sets. Keep literal
   S-32 growing-occupancy ratios labeled and reported; they are not matched
   occupied-work controls. Production float row pitch must remain the exact
   maximum entry/allocation-union cardinality, never capacity or padded BS.
2. **Remove the new continued-state sizing synchronizations.** Forward obtains
   entry cardinality and exceptional cold-tail flag in one host transfer;
   backward obtains the union maximum in another. Both are inside measured
   ranges. Continued zero steady sync/DtoH FAILS. Fresh remains device-only.
   Preserve exact sizing and fail-closed overflow checks; do not move metadata
   or allocations outside registered ranges to hide costs. Long nonempty
   chunks >512 still use dense replay; extending this pool is unfinished.
3. **B64/T4096 <=2x.** Re-measurements remain FAIL; see the current ratio and
   wall shortfall in the ledger. Sparse short-entry replay does not remove the
   long dense fallback, repeated occupied streaming or logical reductions.
   Retain the matched whole-LM wall gate and pinned ingestion. Logical payload
   counts and compiler spills are evidence, not DRAM counters or a hardware
   floor. A floor claim needs measured decomposition and arithmetic proof;
   none is established by part 6.
4. **Inclusive launches <=10.** Fresh still has 59/61 inclusive activities;
   the continued profile adds union construction and metadata work and also
   FAILS. Narrow seven/eight custom counts do not satisfy the inclusive gate.
   Fuse remaining initialization, fills, native copies, reductions and mapping
   transfers within registered phases, then reprofile the actual autograd
   thread. The prior outer backward range missed that thread; its attribution
   is superseded by the direct Replay.backward wrapper profile.
5. **Preserve the proof and memory discipline.** Require all 179 tests,
   continued-state byte comparisons, expanded/strided and native adjoints,
   retained/consuming state, signed zeros and complete journal restoration.
   Keep both checkpoint-free 512-step/2,097,152-token curve comparisons. Retain
   per-store union/event/worst-case sizing and peak allocated/reserved receipts.
   Candidate performance nodes must be cleaned through public output lifetime,
   without private-reference disposal. Historical oracle-only cleanup after
   independent byte snapshots is documented, along with its raw OOM. The
   original raw-owner negative control grows 40,844,800 bytes/chunk; the
   candidate standalone lifetime series is flat. Do not infer all ownership
   paths are closed from that one regression.

Reproduce final sweeps/gates with `receipts/part6_run_final.sh`, after the
live17 sweep and parity command listed in REPORT.md. Rebuild reports using
`receipts/part6_analysis.py`, `part6_failure_arithmetic.py`, and
`part6_write_report.py` with `/home/erikg/emender/.venv/bin/python` only.
All GPU work uses run_leased.sh and physical whitelist 0/1/5/6. No card7,
certified 2/3/4, .pt checkpoints, WG, subagents or parallel debugging.
