# E97 packed64K forward and reset-isolation qualification

## Frozen scope

Follow-up to the [uniform-workspace forward pass](e97-uniform-workspace-v2.md).
Weights remain exact correction-y; policy stays `fp32-linear-v2`. This stage
executes only the forward portion of the real packed SFT objective, under
`no_grad`, with its token-aligned valid/reset masks and prediction-aligned loss
mask. It does **not** call `packed_objective` itself, because that function also
calls backward. Returning already-computed final states adds an observation, not
a change to the training layout or recurrence.

Use **pack0 from the actual closed correction-training authority**, not synthetic
repeated padding or a concatenation of shortened turns:

- Context65,536; token/target storage65,537.
-24 complete records,60,893 real tokens,7,894 assistant targets.
-4,644 unused tail token slots (4,643 invalid forward input steps).
- No internal padding and no reset between turns inside a record.
- Source masks are preserved; targets crossing record boundaries are excluded.

Authority SHA `bf2645fc1c41fbb013bd11c9b8fe539aba6ac88b618e5f03dd5e759a7f7d2a92`;
pack manifest SHA `41c47a919b0cd82dc6a0486329fbf34592d60496f26fd9efe045c1a1d9f887ae`.
All source payload hashes are verified. The freezer reconstructs the original
layout and requires exact equality to `MaskedSFTPackedDataset.pack_at_with_boundaries`.
The loader source SHA is
`3d49215c5aad63186aecbc0fdf699c26776e8167bde2f3db4aa4e55f1e84889c`.
Unrelated working-tree loader hardening is not incorporated: freeze and GPU
execution use an immutable committed export and reject another loader identity.

Three sentinel records are selected by geometry/mask availability alone: nearest
to original offsets0,32,768 and65,536, with length≤8,192 and a contiguous32-token
assistant span. On each, observe the first eight **unsupervised diagnostic** prompt
predictions plus32 consecutive assistant predictions. Diagnostic prompt probes
never become supervised targets. No model outputs influence record selection.

## Comparisons and interventions

Six fixed packed profiles, each twice in the same worker:

1. Original authority order.
2. Whole-record cyclic rotation placing the first sentinel near the pack middle.
3. Whole-record cyclic rotation placing it last, beyond position55,000.
4. Profile3 with every preceding token changed by `(token+1)%50281`, while keeping
   all record boundaries, masks, lengths and the protected sentinel unchanged.
5. Profile4 with **only the protected sentinel's reset removed**. Its cross-record
   loss target remains masked. This is a deliberate invalid-boundary negative control.
6. Profile1 with only invalid tail token values changed.

Profiles4–6 are diagnostic counterfactuals, not new training records. Whole-record
rotations preserve the original target count and membership. Anchor offsets must
cover three different residues modulo512 and16 without adjusting any chunk size.

For each sentinel/probe span, replay actual actor probabilities from a fresh cache
and full causal within-record prefix, twice. Also evaluate each complete sentinel
alone with alignment16, twice. The standalone actor/teacher comparison must pass
before packed runs. No historical behavior probabilities are replaced.

Predeclared gates:

- Current actor versus standalone and each of the three real packed orders:
  maximum absolute log-probability gap≤.05,p99≤.02. Packed versus standalone
  trainer scores must satisfy the same bounds. These cover120 selected positions,
  **not exhaustive actor replay of all7,894 assistant targets**.
- Exact repeated selected score vectors, hidden/logit hashes, CE and final-state
  hashes. No tolerance fallback for repeatability.
- Full native CE versus independently summed actual-head log probabilities over
  **all**7,894 supervised targets: mean difference≤.0001; finite logits and states.
- Changed predecessors with resets: selected sentinel hidden/logit hashes and
  scores **bitwise identical** at fixed position/shape.
- Negative control: removing the reset must change the sentinel's initial-boundary
  logits, demonstrating that the isolation test can detect a missing reset.
- Changed invalid tail values: all selected probes, native CE and all18 layer
  final-state hashes exactly unchanged. This tests tail-value invariance, not all
  possible padding lengths or a new long-context stability oracle.
- Persistent BF16 weights, unchanged parameter fingerprint, no gradients/optimizer
  updates, actual FP32 heads, uniform workspace on18 mixers,163 FP32 Linears,
  no recurrence timing autotuning, no mutation of input/control tensors.

The passive head hook returns `None` and hashes actual hidden/logit rows; it never
replaces logits. Final-state observations require18×60×64×64 FP32 finite elements.
The12 packed forwards retain their original65,536-step shapes. Instrumented forward
timings are reported separately; they are not production training throughput.

## Operations and limitations

CPU tests: **98 passed,14 CUDA skips** before freezing. New tests cover whole-record
packing, cross-record target masks, prefix/reset/tail intervention bounds,
next-token probe alignment, passive head/CE reconstruction, coverage/finite checks
and signed-zero exactness. Existing14 CUDA checks already passed under this policy;
this experiment adds the real4B packed workload rather than rerunning that suite.

One model worker,12 actor spans,6 standalone forwards and at most12 packed forwards;
no retries or automatic expansion. Worker≤3,600s, outer≤4,200s, teardown30s. Allocated
HBM≤40GiB on one leased48GiB GPU; this is a new long-layout capacity bound, not a
relaxation of any numerical threshold. Original16GiB short-assay evidence stands.
Immutable export/inventory, before/after source audits composed with owned lease
cleanup, local CUDA/NUMA routing, isolated cache and both expandable allocator vars.
Private token/mask/score/logit artifacts stay outside Git and are never overwritten.

Driver `scripts/run_e97_packed_forward.sh`; freezer/worker
`scripts/qualify_e97_packed_forward.py`; helpers `scripts/e97_packed_forward_probe.py`.
Artifact root `/mnt/nvme2n1/erikg/e97_systematic_posttraining/packed-forward-v1`.

This is one historical pack plus fixed transformations, not general pack coverage,
full4B backward, eight-rank, cache/restart, on-policy data admission or behavioral
qualification. The880+32 training budgets stay closed. No resilient/distributed
training claim is made.

## CPU export preflight

The initial export (`2a402f26`) stopped at `placement coverage` before GPU work:
choosing the nearest middle offset without the residue constraint selected34,016,
sharing residue0 modulo16 with the original start. The guard correctly refused it.
`packed-forward-v1-preflight/freeze-failure.json` is retained. Selection now
minimizes distance **among** whole-record rotations satisfying the predeclared
residue constraints; no model output was consulted. A regression uses the actual
pack's record lengths. Repaired suite: **99 CPU passed,14 CUDA skips**.

Sentinels are source records0,12,23. The anchor's frozen offsets are
**0,31,043,58,236**, residues0/3/12 modulo16 and0/323/380 modulo512.
Independent CPU `layout-audit.json` rechecked all24 whole-record slices and masks
in each real order, all cross-record exclusions, tail-only padding, bounded
interventions and96 assistant/24 diagnostic prompt probe positions.

Frozen plan SHA:
`af9339afe3819cd495ec1a052b77c5d46761bb060ac0d910ac0f73b2240d1fbf`.
Private input capsule SHA:
`7340c9d573acce2b1f9e79c77cc25a3130f4759c80726cab72f13ca00e9da7b9`.
Successful CPU export preflight: `packed-forward-v1-preflight-r2`.
These identities must reproduce exactly in the leased runner before any GPU work.
## GPU attempt: numerical prerequisite failed; packed forwards not executed

Source **`57cbebc2`**, process `proc_2077`,243s,exit1 at the declared
`standalone gate failed before packed runs`. This was a numerical acceptance
failure, not CUDA/OOM or a broken observer. The original plan/input hashes
reproduced exactly in the runner.

All12 actor spans and6 standalone forwards completed. Each repeated artifact
pair is byte-identical, including the standalone selected hidden/logit and18
final-state hashes. The120-position standalone actor/trainer comparison failed:

| Metric | Observed | Frozen limit |
|---|---:|---:|
|Maximum absolute gap|**.056125640869140625**|≤.05|
|p99 absolute gap|**.030076026916503906**|≤.02|

All three standalone native CE/head consistency checks passed, worst mean delta
1.2365210827592235e-9. CPU recomputation separates the failure by probe kind:

| Source record | First8 prompt predictions: max |32 assistant predictions: max |
|---|---:|---:|
|0|.056125640869140625|.000005244466592557728|
|12|.030076026916503906|.00003610987914726138|
|23|.030076026916503906|.0000070315145421773195|

The maximum occurs at **record0, logical prediction row5** (predicting token6).
The other largest deviations are also among the first eight prompt predictions.
These prompt probes are unsupervised diagnostics using very short fresh-cache
prefixes; they are not generated assistant responses. The tightly agreeing
96-assistant-token subset does **not** reclassify the frozen120-position gate as
passed or justify dropping its failed probes.

**Zero of the12 planned64K forwards ran.** Consequently there is no result yet
for packed placement, reset isolation, tail invariance or64K memory/performance.
This failure does not demonstrate that packing is broken: it precedes packing.
The previously passed57-turn generated-token gate remains valid within its own
scope, but does not establish universal actor/trainer agreement.

Independent CPU `result-audit.json` verifies the18 baseline artifact hashes,
byte-identical repeats,120 score/position identities and finite scores, original
record target counts, the reported max/p99 and subset diagnostics. All17,694
source files and the inventory verified before/after. Cleanup shows eight idle
GPUs, no compute processes and no lease files without reaping.

The early-stop path did **not** retain a post-run in-memory parameter fingerprint
or peak allocated HBM; neither is claimed retrospectively. Initial parameter and
production-geometry checks passed before baseline execution. No optimizer/update
path was invoked; no checkpoint was written. Terminal safety receipts on early
failures should be hardened before another GPU experiment.

| Artifact | SHA256 |
|---|---|
|Source inventory|`4772c8ae35db16931bea568705ecb9898c3b4608e1e6088c21fc2d84200c28dd`|
|Baseline failure|`d7090def9005bd97079b94e092b86e9ed688448ec7aa0faf64972fa3251f56f8`|
|Independent audit|`7463dbef82ca33376b92c66ec94d7f66b9e03bc5113bed8591758bd05396d9ca`|
|Worker failure|`303a3122984e20748dfbf2f30ca7c5502e84178832fc48582611e0aed6dd71e9`|

No further GPU run or automatic retry was launched. Next localization should bind
the early record0/row5 mismatch directly to these repeated native scores before
attributing it to the previously observed projection-layout issue. Any separate
continuation of the unrun packed diagnostics needs a new explicit diagnostic plan;
this failure and the original thresholds must remain intact. Full4B backward,
eight-rank/cache/restart, training eligibility and behavioral gains remain unqualified.
