# X40 part 19: bounded diagnosis and isolated Birth operand candidate

FAIL / incomplete. Production unchanged. No certified journal fix, race repair,
full recertification, 9x result or depth verdict is claimed.

## Journal failures: two different limits on the reference path

`audit.py` calls the actual X31 and X36 capacity functions for 72 combinations,
independently checks their formulas, and records all values in `audit.json`.
It inspects all 12 pure part-8 surface receipts, rather than extrapolating from
one successful model. These are static allocation bounds, not new GPU runs.

At B64/T2048, policy 3 (S1472), X31 reserves 20,208,128 events per store.
The existing rebase launch requests grid `(64,78938)`. CUDA grid.y is limited
to 65,535: this explains the invalid launch independently of address arithmetic.
Its 1,293,320,192 total entries pass the existing int32 guard. Both the store
index in X31 rebase and its journal count are already int64. Simply promoting
an allocation-size expression does not fix this launch-axis failure.

At B128/T2048, the same per-store capacity requires 2,586,640,384 entries,
which exceeds 2**31 and is rejected before allocation. The three journal
arrays alone require 51,732,807,680 bytes, before state, replay clones, inputs
or allocator overhead. Removing the guard would not establish a runnable
certification on the part-8 card's 50,876,841,984 bytes. Keep this guard;
removing it without auditing every journal caller's index width is unsafe.
The part-18 adapted B128 failure is allocation, not a backward launch failure.

Production uses a different capacity implementation: X40 forward imports
X36's `AllFieldJournal`, which overrides X31's `capacity_for`, including when
the inherited allocation classmethod calls `cls.capacity_for`. X40 backward
also has its own rebase. For S1472/T2048 its capacity is 1,130,496/store,
grid.y 4416, and B128 has 144,703,488 entries (6.74% of 2**31). Thus this
specific ceiling is on the independent certification reference path, not the
production envelope. This does not certify every future imported-state bound.

Both 1.3B T2048 production receipts selected checkpoint_B32, unpacked and
without offload. Each memory policy is called separately with one head by
`Cell.forward`; multiplying journal stores by four would be incorrect.
The surface passes `tokens[:,:-1]`, so the real scan length is 2047. The audit
includes both 2047 and the requested direct-certification length 2048. The
22M and 150M T2048 successes used unsplit checkpoint envelopes. Packing is
post-forward trimming and cannot rescue a capacity guard that fires during
allocation; no T2048 pure winner used packing. X40's mutation-count reservation,
not a hidden large-int promotion, accounts for the different allocation path.

The six inspected part-8 source hashes match current files except backward.py,
which changed in part 12 (90817ba6). Its current source still uses the X40
rebase, and forward/allocation dispatch matches the recorded source. Historical
surface success is retained evidence, not a fresh part-19 certification.
No splitting, pressure exclusion or oracle substitution was introduced here.
Direct B128/T2048 reference certification remains blocked, and direct B64/T2048
still needs the already validated launch-axis adapter or a fully gated repair.

## The 33,070 printed uninitialized reads are one retained site

Lossless parsing of the terminated part-18 log finds exactly 33,070 printed
accesses, all `materialize_state+0x350 in kernels.py:378` in X31. The raw log
hash agrees with part 18. This is a complete inventory of retained printed
accesses, not of all possible errors in a run that terminated with exit 143.
There is no evidence in this log for a handful of separate X40 empty_like sites.

The source operand is Birth, allocated uninitialized on fresh state. A mixed
32-slot tile enters materialization when any lane is live, then loads Birth
for all valid lanes. Dead never-allocated lanes have no Birth initialization.
Allocation initializes Birth before making a slot live; death removes live;
reallocation sets Birth again. Materialization only writes age on `valid & live`.
The process-local candidate masks the Birth load with that identical predicate.
For live lanes the operand and arithmetic are unchanged. For dead lanes the
computed age cannot reach the masked journal/store, including Record=True.
An entirely empty tile bypasses the branch. This is an operand argument,
not a reason to waive sanitizer errors in the unchanged reference.

The bounded original control at B64/S64 reproduces initcheck failure, exit 86,
15,616 sanitizer-reported errors, while its defined-output assertions pass.
The masked candidate passes 15 mixed/empty/full fixtures at B8/S2944 and
B64/S64/S256/S736 plus B128/S1472, with zero initcheck errors. A second 15-fixture
run enables all-field recording and checks journal counts as well as outputs;
it also passes with zero initcheck errors. These are isolated kernel tests,
not the full lifecycle or certification gates. No source candidate is promoted.

The identical Birth load pattern also exists in X40's own materialization.
An additional 15 recorded fixtures apply the same process-local operand mask
there and pass initcheck with zero errors. This extends isolated coverage to
45 passing fixtures across X31/X40; it does not attribute any of the 33,070
reference-side accesses to X40 or establish a whole-program pass.

The first new masked harness run failed: PyTorch's masked scalar assignment
to partially uninitialized Birth itself reads uninitialized lanes. It was
deliberately terminated (143), and its compressed raw output is retained.
The retry uses a tiny Triton masked store to initialize live Birth lanes without
reading dead lanes. This correction changes fixture construction only.
An audit enhancement initially asserted that all historical source hashes
matched current files; it failed because backward.py changed in part 12.
The corrected audit retains the mismatch explicitly and does not assert a
historical-to-current certification.
The standing part-18 lifecycle byte-comparison correction is already committed
in 868ab9a2: `numpy().tobytes()` is present. No second source correction is needed.

## Residual gates

All race inventory IDs remain unresolved; no group-count subtraction is made.
The isolated Birth fix has not undergone imported-state/continuation differential
lifecycle proof, full X31 parity, graph/eager, oracle-suite or curve matching.
The complete certified-shape sweep and 9x shape sanitizer acceptance remain
incomplete. No payoff/depth experiment is justified yet.

Only the approved venv was used on lambda01. GPU tests were sequential and
broker leased physical card 2 with allowlist 2/3/4. No searches, forbidden card,
WG service, agents, Slurm, resilient training or committed .pt files.
`contact_supervisor` was searched first and by name/description; no callable
tool exists in this session. No alternate supervisor channel was used.
