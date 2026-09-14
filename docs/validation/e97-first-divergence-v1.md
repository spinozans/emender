# E97 first-divergence trace: current composite reference

## Question and frozen method

Find the earliest differing module boundary, rather than try another numerical
policy and inspect only its final probabilities. Target the current sole failed
position: `onpolicy-task-004-sample-0`,turn1,generated position23.

The two completed `fp32-linear-candidate-v1` workers are the reference. Their
measurement files are byte-identical (SHA
`64ba00df08cf48bc87e87ca1afc9c5257bfb88bb25addb63b03cedd88812e34e`), and their
summaries match SHA
`490092c0e3aa4f9154aee5437350aec0a9b548889205c4ab9abeccd56fb4c7c8`.
Reference recipe:
`916b1e87207a161d17ecf61881e5be3952a4fcd23ac3e6e6a6fe81d282045d15`.

No numerical/architectural changes, no gradients, sampling or updates. Use the
same weights, composite policy, actor prefill/tokenwise calls and teacher forward
configuration. Execute the entire recorded turn to preserve original shapes and
check **every generated score**, but capture only logical input rows0 through
`len(prefix)+23-1`: the complete causal history for the failing prediction.
Later generated inputs and alignment padding are excluded from the trace, not
removed from the original execution. This avoids blaming a later-token mismatch
for an earlier failure.

1. **Coarse:** embedding output and all18 block inputs/outputs, over every causal
   token. Actor and teacher each run twice.
2. Require exact FP32 score binding to the current reference, exact teacher CE
   mean/padded length, and exact repeated tensor bytes. Fail closed if the
   isolated turn or observer changes the measured failure.
3. **Fine:** only the first differing layer, selected by the predeclared coarse
   ordering. Observe outer normalization, mixer/projection inputs and outputs,
   post-mixer normalization, and MLP inputs/projections/outputs. Two executions
   per path again. At most8 evaluations in one worker. If the coarse difference
   lies outside a layer, retain that boundary without an automatic new profile.

"First" means earliest site in layer/topological order, then earliest logical
causal-token coordinate within that site. Independent branches do not have a
unique chronological ordering across the two layouts. This localizes an interval;
it does not automatically prove which GPU instruction caused the discrepancy.
In particular, recurrence/pointwise operations between projected operands and
output-projection input may require an operand replay after localization.

## Observation and storage

`Rows` in `scripts/e97_first_divergence.py` accumulates each site's rows across
prefill segments, projection subchunks and tokenwise calls. Hooks returnNone.
When fused outer normalization bypasses module hooks, a scoped wrapper calls the
original function and returns the **same result objects unchanged**, restoring
the original function on exit. No shadow result replaces a model output.

Private tensor files contain contiguous CPU copies with original values/dtypes.
Receipts include original call shapes, strides, offsets and addresses. These are
**not** full storage/alias-preserving replay capsules and do not restore GPU
addresses. Compare tensor bytes (including signed zero), not only norms. Retain
all site comparisons, first coordinates, maxima and hashes. No prompt/token or
activation arrays are published in documentation.

One bank<=2GiB trace payload, at most three live banks during comparisons plus
bounded scratch (8GiB trace-payload ceiling). Real-model peak allocated HBM<=16GiB.
This is not a whole-process RSS limit. Source/parameter/buffer fingerprints and
zero recurrence autotune entries are checked; all parameters remain BF16 and no
gradients are permitted.

## CPU preflight and execution

**19 passed,3 CUDA skips.** Tests cover logical-row alignment across differing
outer and projection chunks, tail exclusion, no output replacement, hook removal,
first-site/token ordering, signed-zero detection, bounds, incomplete coverage and
fused-normalization wrapper restoration.

The initial CPU-only norm fixture assumed an optional CUDA symbol existed and
failed with `AttributeError`. It is retained in `first-divergence-cpu-v1/failure.json`.
The fixture now installs its fake symbol explicitly; production instrumentation
is still gated on the actual model's fused-normalization configuration.

Plan SHA:
`5971714805b7b5c9d395af68a615a3e42087fa75d7053f0441dc4be5dfc04ac8`.
CPU authority preflight retained in `first-divergence-v1-preflight`.
Run root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/first-divergence-v1`.
Driver `scripts/run_e97_first_divergence.sh`; executor
`scripts/diagnose_e97_first_divergence.py`.

Immutable export; composed before/after source checks and lease release on
success/failure; one checked GPU lease; local CUDA selection; NUMA binding;
isolated cache; both expandable allocator variables. Worker1,200s, outer1,800s,
teardown30s. No retries, no overlapping GPU work, no delegated workers.

Results pending. This is a single-record training-forward diagnostic, **not the
actual64K packed multi-document trainer qualification**. The real SFT trainer
packs independent records contiguously, resets state at record starts, masks
cross-record targets, and pads only the unused pack tail. The existing numerical
gate remains failed; no admission, promotion, training or budget expansion.
