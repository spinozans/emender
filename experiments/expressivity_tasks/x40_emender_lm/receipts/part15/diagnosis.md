# X40 part 15: reproducible failure, no certified fix

The work order is incomplete. Production sources remain byte-identical to
4ff85f4f. No production edit was retained, no new certification is claimed, and
no budget-scaled training/evaluation was launched.

With CUDA_LAUNCH_BLOCKING=1 and the completed depth checkpoint, production
execution fails in the 9x pool at batch start 80 (B8, T2286, one head per scan,
S2944). The first pass completed batches 0 through 72. Repeating only batch 80
fails at scan call 10: zero-based LM block 2, head 1, CUDA block 1. See
localize_console.txt and single_console.txt. The exact reduction still resolves
to Triton's shared sum combiner rather than an identified source expression.

The proposed shape-product explanation is unsupported by this reproduction.
The freed-event tensor has 53,839,872 one-byte elements, keys have 1,507,328
fp32 elements, and values have 3,014,656 fp32 elements. The disabled all-field
journal is absent from this evaluation. Direct integer sums count at most 32
live flags or 128 history flags; helper boolean sums have at most 4096 lanes.
These bounds do not establish an int32 arithmetic overflow. Store-owner n
and freed-address arithmetic are already int64.

The long-context receipt really completed T8192. context_probe constructs B1
inputs and the LM launches each head separately with budgets 64/256/736/1472.
Its largest freed-event tensor has 12,058,624 elements. Neither that product nor
the failing depth product crosses 2^31. An explanation of their different
outcomes needs more evidence; length alone does not explain it.

Diagnostic evidence makes a passing instrumented run especially unreliable:

- Captured scan-10 input, replayed alone through the unchanged kernel, passes.
- Replacing integer sum combiners with labeled diagnostic combiners passes the
  full failing batch. This is diagnostic execution, not certification.
- Promoting only the live-tile count still fails, now at call 18.
- Promoting only the history-request count passes this batch, but promoting
  only history-cap count still fails at call 14. These process-local changes
  were never written to production source.
- Changing only equal-size assertion function-name strings in generated PTX,
  then reassembling with lineinfo and fmad=false, also passes the full batch.
  The helper verifies all other PTX text is unchanged. This is consistent
  with code-layout/sanitizer sensitivity; it does not establish its cause.

No such diagnostic pass can stand in for localization or the required full
X31 values/state/traces/pressure, 109-oracle, graph/eager and 2M-token gates.
Those gates were not run because no justified production fix was established.
The 9x result remains missing and budget-depth causality remains unsettled.

All GPU work used an exclusive broker lease on physical card 1, released at
process exit. No services were started, stopped or preempted; no other card
was used. The broker is used directly, without WorksGood orchestration.
Only /home/erikg/emender/.venv/bin/python was used. Temporary tensor capture is
/tmp/x40_part15_scan10.pt and is not a committed artifact.

Reproduction scripts are evidence scaffolding in this receipt directory.
From the repository root, acquire a broker lease with GPU_LEASE_VISIBLE=1
and set CUDA_LAUNCH_BLOCKING=1 and PYTHONPATH=/home/erikg/emender before invoking
localize.py or single.py with the approved interpreter. Run capture.py before
isolated.py. tagged.py and label_run.py import neighboring diagnostic helpers.
Their source substitutions and cubin replacements are confined to the process;
none is a deployment or a certified kernel implementation.

Coordination limitation: no contact_supervisor tool was exposed (searched
ALL_TOOLS names and descriptions before work). No alternative messaging channel
was used. This receipt does not claim supervisor approval for an unproven
compiler/sanitizer workaround or permission to weaken the certification gates.
