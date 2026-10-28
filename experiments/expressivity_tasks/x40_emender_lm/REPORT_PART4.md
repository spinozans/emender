# X40 part 4 — 100k program blocked by oracle pressure synchronization

**FAIL / incomplete.** No new training-throughput or quality gate passes.
The target was preregistered in RECEIPTS.md before measurement: >=100,000
sustained training tokens/s at B>=64, T512/T2048, one RTX 6000 Ada,
bitwise production parity including pressure, and >=2M matched training tokens.
No scale timing was run after the failed scale-policy gate. No performance
floor is claimed. The historical ~30k tokens/s remains historical evidence.

The first lever's assumed policy-translation cause is contradicted by a
standalone reproducer: **unchanged X31 produces different pressure on repeated
identical inputs**, without importing X40 model/scan or the X42 adapter.
Outputs, native keys/values/strength/live/age and the compared routing fields
remain bitwise identical. This prevents reliable bitwise certification against
that unchanged reference when capacity pressure is exercised.

## Evidence and narrow diagnostic

`receipts/part4/scale_gate/parity_T512.json` reproduces the actual production
scale certification failure: 142 pressure mismatches, max 0.0166666731.
X31 disagrees with the independent case-window calculation by 0.0166666657;
X40 disagrees by only 5.96e-8 (division rounding). Values, traced decisions
and gradients passed the existing assertions before the pressure failure.
The gate remains failed; no assertion/tolerance/reference was weakened.

`pressure_probe.py` separately runs B1/T512/S64, the first X42 retention
policy, seed4001, a fixed externally supplied beta and three repetitions each
with/without autograd recording. It uses X31 alone. The JSON retains exact
mismatch counts as well as errors >1e-6; the latter separate stale-window
errors from Triton/PyTorch division-rounding differences.

The temporary diagnostic fork inserts one `tl.debug_barrier()` immediately
after the scalar history-ring `journal_store` and before the 128-lane reload.
The original kernel has no barrier at that transition. The fork is generated
in a temporary directory, uses unchanged X31 dependencies, and is discarded.
The minimal patch and source hashes are retained. No upstream file changes.

The diagnostic barrier eliminates semantic-window errors and repeated-run
pressure differences in all six tested calls. Other compared outputs/state/
routing remain bitwise equal to the original. This supports a cross-thread
history-store/reload synchronization defect, rather than a policy translation
mismatch. It is a narrow diagnostic, **not** a full production oracle repair:
no backward-gradient comparison or complete X31 suite against the fork is
claimed. Full candidate certification must follow any authorized reference
repair; a corrected diagnostic oracle is never substituted into the real gate.

Fresh original X40 B64/T512 certification passes all four production policies:
bitwise values/state/trace/pressure/gradients, with existing evidence-wire
comparison tolerances. Local tests: 5 passed. Exact commands and outcomes are
in `receipts/part4/commands.json`. GPU execution used run_leased.sh, physical
card1, the mandated Python, sequentially. The wrapper now excludes card0 in
both the broker allowlist and held-card validation, protecting the depth run.

## Source audit and decomposition relevant to the remaining program

The requested X35 causal-read source and report, X36 part6/7 reports and raw
receipts, and X40 part3 reports/summary were read before work. X35 chunk_read
scores all chunk queries over immutable payload versions in two launches,
preserving X31's 32-slot reduction tree; this is not an unrestricted GEMM
replacement. Its 1.94–2.20x forward results are standalone historical receipts,
not an X40 gain. X40 kernels.py still performs tile reads inside its token
loop and does not import that chunk-wide primitive.

Contrary to the missing-port hypothesis, X40 backward.py already includes
both X36 part6's entry/allocation-union pool (around line479) and part7's
journal-driven event-consumer suppression (around line172). The pool requires
nonempty entry and T<=512; suppression requires a pool and no native-value
consumer. X40 model.scan always passes a fresh origin, so copying those same
features again would not activate them for this composition. Fresh-prefix and
long-chunk replay use their existing fallback. This audit does not establish
that a new fresh-origin pooled algorithm is exact or faster.

No new post-port replay profile exists because the first gate failed. Historical
part3 exact-decay profiles attribute 214.287ms to scan and 658.357ms to reverse,
separately from the 1.065205s synchronized wall median. At B64/T512 the 100k
budget is 327.68ms/update, requiring 3.251x over that median; reverse busy time
alone is about twice that budget. These are engineering targets and historical
component evidence, not an additive wall identity or irreducible floor.

## Unfinished work and residual risks

Reference pressure synchronization must be repaired and fully certified before
scale timing. The X42 B64/B128 T512/T2048 150M/1.3B matrix, chunk-batched X40
reads, new fresh-origin backward work/profile, occupancy B128/B256, fusion,
and the >=2M matched curve were not run. Neither interim target (60k/85k) nor
the 100k target is established. No memory feasibility or amortization claim is
made for the requested larger batches. The one-barrier experiment covers one
policy/shape and does not establish the absence of other synchronization bugs.

The contact_supervisor tool was searched first and is unavailable. The needed
reference-scope decision and discovery are recorded in coordination.txt; no
substitute service, WG machinery or agent was started. No .pt files, Frontier,
Slurm or resilient-training changes; DiLoCo requirement IDs are inapplicable.
AGENTS.md and CLAUDE.md remain unchanged, as do unrelated workspace edits.
