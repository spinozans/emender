# Part 9 diagnostic decision — implementation and 1x gate remain open

This is an incomplete diagnostic receipt, not acceptance of the requested
three levers. No production arithmetic, kernel, checkpoint helper, architecture,
or training definition was changed. No new speedup is claimed.

## Duplicate-forward premise

`part6_checkpoint.recorded_block` already enables recording during both the
initial checkpoint pass and its backward recomputation. The initial graph is
discarded; the recomputed graph's `Replay.forward` records the all-field journal.
`Replay.backward` consumes that journal and its retained final buffers directly.
It calls `reverse_kernel`, not `tape_scan` or `scan_kernel`.

The instrumented B1 single-block audit uses the actual scale widths and FFN
hidden dimensions. At T512, both 22M and 1.3B execute four initial native scans,
four checkpoint-recompute native scans, and four reverse invocations. There are
zero native forward invocations inside those reverse invocations. These are
call counts, not throughput or full-model capacity measurements. T2048 results
are retained in `call_audit.json` only if their fresh production gate passes.

There IS repeated read mathematics: `reverse_program` restores each pre-token
state with `undo_to`, recomputes scores/probabilities, and applies read and store
adjoints in reverse token order. That is not a second complete forward scan.
The final loss adjoint is unavailable during the checkpoint's forward scan.
Removing read recomputation requires retaining more forward intermediates or
changing the reverse algorithm/kernel. This audit does not prove such an
optimization impossible, nor establish its memory cost or speedup. It does
reject attributing an achieved saving to removal of a nonexistent extra native
forward launch. The requested ~1.2x remains an unverified expectation.

The part5/6 decompositions are pilot profiles, not 1.3B checkpointed profiles.
Their `recomputed_native_forward_device_ms=0` and ~70% replay share cannot be
used to infer a 1.3B full-forward duplication or a guaranteed wall-time saving.

## BF16 read decision evidence is missing

The historical part3 BF16 receipt fails reads, keys, values, strengths and
routing. It does not isolate read-only BF16 arithmetic from store accumulation.
FP32 store update preservation is a necessary constraint for a future candidate;
it does not establish semantic equivalence of BF16 reads.

No side-branch BF16 read implementation, isolated read throughput measurement,
or 5M-token candidate/reference curve has been produced by this diagnostic.
There is no evidence-backed recommendation to adopt it. The ~1.0x number is
only the operator's projection, not a result or a measured extrapolation.
If selected after evidence exists, certification must explicitly define BF16
read casts, accumulators, reduction order, softmax and adjoints while retaining
FP32 store accumulation, update the CPU reference, then demand bitwise parity
against that new definition. Current FP32 certification cannot certify it.

## Event suppression and ratio interpretation

No deeper suppression change is accepted. Write density ~0.5 alone does not
identify dead read adjoints: a no-write token can read live values and propagate
query, value and strength gradients. Existing case-based store adjoints and
whole-scan value-adjoint elision remain unchanged. A trained-state, per-event
backward profile and a proof of zero contribution are still required.

`historical_ratios.json` recomputes part8's individual pure/transformer step-time
ratios from raw sustained receipts, preserving envelope metadata. At 1.3B the
ratios are 1.5854/1.3174 at T512 B64/B128 and 3.4187/3.4888 at T2048 B64/B128.
Thus "1.6x at 1.3B" describes one cell, not both context lengths. These are
historical measurements; there are no part9 before/after sustained rates.
The pilot 100k gate remains failed at the prior qualified 54,836.10 tok/s.

## Execution limitations and preserved failure

Tool metadata discovery found no `contact_supervisor` or Intercom tool, so
the requested decision/progress channel could not be used. No alternate
supervisor messaging channel, WG service or delegated worker was used.

The first diagnostic reached and passed T512 production parity and both call
audits, then failed in the unchanged X31 T2048 oracle's address-rebase launch
with `Triton Error [CUDA]: invalid argument`. Its raw log and JSON are retained
as `call_audit_unadapted_oracle_failed.*`. The rerun uses the existing part5
`TransposedLaunch` oracle adapter for T2048, preserving floating arithmetic;
this is the same documented grid-axis repair, not a relaxed parity criterion.

Only the approved `.venv/bin/python` and `run_leased.sh` on physical card 1
were used for GPU execution. Protected PID 634691 and cards 0/2/3/4/7 were
not signaled or modified. No `.pt` artifacts were created. No Frontier,
Slurm or resilient DiLoCo work occurred; their requirement IDs do not apply.
AGENTS.md and CLAUDE.md are unchanged.

Residual work: actual recomputation/caching candidate, unchanged production
parity plus a bitwise 2M-token matched curve, full before/after timing and
phase profiles at both scales, BF16 side branch plus its 5M-token evidence,
trained-density event suppression and exactness gates, fresh matched-envelope
T512/T2048 ratios after each accepted lever, and a fresh pilot gate.
