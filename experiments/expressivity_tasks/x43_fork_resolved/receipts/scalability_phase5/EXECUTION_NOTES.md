# Phase5 attended execution notes (incomplete investigation)

Only `/home/erikg/emender/.venv/bin/python` was invoked. No WG or subagents.
Supervisor/intercom tool discovery by name/description returned no callable
transport. No supervisor contact or permission is claimed.

First compile failed because Gluon does not broadcast scalar store values like
ordinary Triton; second failed because Gluon has no gather. Both tracebacks and
exit1 receipts are retained. The third candidate passed T32 but failed both
reference parity and repeat parity at T64. CTA barriers for changed global
scratch producer/consumer layouts corrected this observed nondeterminism in the
subsequent ten-shape probe. Failure receipts are not qualifying measurements.
`quick.json` precedes the barriers; `sweep_barrier.json` is the final variant.

The candidate deliberately retains reference reduction layouts. This preserves
addition order in tested fingerprints, but is also a reason column chunking
alone cannot establish bounded registers: BlockedLayout([1,1],[1,32],[2,4],[1,0])
places only two warps along rows and four along columns. A 16-column tile leaves
column lanes/warps duplicated instead of reallocating them to rows. BS2048 still
assigns 1024 row positions per unique column lane. Merely shrinking the column
extent cannot eliminate that per-thread row reduction state. Row tiling of keys
reduces its materialized dot-product extent; value READ and previous-value sums
still span all BS rows. Repeated full-state global loads/stores add substantial
cost, even with reduced compiler spills. This is a static layout/source analysis,
not a measurement of dynamic hardware register liveness or spill transactions.

Next implementation must change the value reduction's distributed layout and
preserve the reference floating-point reduction tree explicitly, with bounded
intermediates. Contiguous row-chunk partial sums cannot be assumed byte exact:
the parenthesization must match reference TTGIR/PTX. Direct selected-row previous
loads and row-local updates are separate candidates; signed zeros/NaNs and the
existing exact `values + (v - values)` arithmetic must be considered before
claiming equivalence. Keep candidates separate; do not edit source-bound phase2
or production kernels to promote this rejected experiment.

The phase5 protocol file was written after exploratory runs began. The new
registration explicitly says so. No retrospective preregistration claim.
Historical phase2 registration remains unchanged and guard matched.
No resilient DiLoCo behavior change; gap-matrix IDs are inapplicable.
AGENTS.md/CLAUDE.md remain unchanged. Unrelated workspace changes are preserved.

## Exact resume

This is a failed-candidate checkpoint, not a completed optimization phase.
No training/checkpoint was started. Start the next candidate from the unchanged
`c14_phase2_gluon_rowkeys_kernel.py`; retain this candidate as negative evidence.
Run both historical and phase5 source guard checks with the approved interpreter
before another experiment. Never overwrite completed output paths. The probe
wrapper accepts a fresh label: `bash .../run_c14_phase5_probe_leased.sh NEW_LABEL`.
Validation wrapper deliberately refuses its existing terminal receipts; copy
its explicit recipe to a fresh receipt directory for a genuinely new candidate.
Long fixtures are explicit `-m ...c14_phase5_long_parity --tokens 512|1024|2048
--peak PEAK --output NEW_PATH` commands inside an approved broker lease.
Do not claim the configurable maximum-live or T2048 cases were tested merely
because the CLI supports them. Source/battery binding and trained fingerprint,
2M-target resume, and full-width/memory bills still apply after a speed candidate
passes. No old source registration is silently re-bound to a different kernel.

Final scoped regression is 109 X31 and 108 C14/operator tests (current collection,
three more than the phase2 count); all pass without skips. Eight shape and four
short-context gradient/replay fingerprint pairs pass. Long T512/T1024 at peak257
also pass state/replay/gradient and masked/fresh-document reset equality; both
terminal exits0. T2048, maximum-live, full-model and 2M-target gates remain open.
All own probe/validation/long-wrapper processes are absent at completion; fresh
cards4/5 show2MiB and0% utilization. No training remains from this work.
One registration-guard command used the invalid spelling `precommit` and exited2;
the corrected `pre-commit` command passes without warnings. This is a CLI error,
not a source-binding failure. Historical check passes; final new bindings are
verified against working/index source bytes before commit and HEAD after commit.
