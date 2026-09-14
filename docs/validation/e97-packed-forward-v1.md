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
training claim is made. Results and frozen input/plan identities pending CPU export
preflight; no GPU work starts before those identities are fixed.
