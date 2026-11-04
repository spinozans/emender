# X43 C14-B — memory operations over the array

Status: registered before B implementation and training. B1 is the deep
integration arm. This registration does not authorize modifying the frozen
C14 dense/v1/v2 arms, their receipts, or the running transformer extraction
service. The companion `receipts/c14_b/registration.json` binds the files below
by SHA256 and Git object identity. The registration commit must be an ancestor
of the implementation and training commits; no training before that commit.

## Design frozen before training

Start from the native byte `c11.Boosted('emender', 2000000)` construction:
2,133,700 trainable parameters, two layers, width 256. Retain the standing
shared 260-to-256 trainable content projection (`device_input`) and existing
content/value maps. No new learned adapter, slot table, gate, key embedding,
or layer is admitted. The unused standing parameters remain counted.

Consultation answers enter the STORE, never the input embedding channel. The
query policy is C14's deterministic maximal-Wikipedia-supported causal suffix
per position, clipped at the independent window reset. The committed
`registered_emender/` cache IS the answer oracle, keyed by source document,
window and position. Zero live array queries during training. Reuse the shared
v2 decoder and require bit-identical float32 answers including all metadata;
never reconstruct answers from target bytes. Reconstruct the queried suffix
from the already observed raw bytes and cached context length, checking its
bounds. No final target or future byte can enter a key or an answer.

APPEND inserts the answer using the store's native APPEND key at the query
position, exactly as a stream slot at that position (amendment #2 below).
The full raw queried suffix is slot metadata, not an addressing input. Content is all 260 scalars through the shared
trainable projection and the standing content/value maps. Reads use ordinary
content-addressed addressing, with the standing learned query coordinates.
Array and stream slots share the SAME addressing normalization and physical
capacity; a parallel unbounded answer bank or an embedding hook is not B1.

CORRECT observes the actual byte only after the corresponding prediction is
scored. When the array prediction was wrong, update the originating live slot
by the standing residual/delta learning rule; an observed-byte update may
influence subsequent predictions only. Never pass shifted targets to a
pre-prediction forward path. DECAY/POP retire stale array answers when their
queried suffix ceases to be a suffix of the running context, and apply the
standing strength/age/free-threshold law. MERGE uses the existing capacity
routing/delta rule, preserving frozen keys and allocating no extra slot.
Record stale retirement separately from natural decay and capacity merge.

The ACTIVE physical budget is the tier-1 construction's production budget:
per layer/head 32 and 64 slots (two heads), horizons 128 and 512, decay rates
`exp(log(.001)/horizon)`, pop threshold .1, and existing free threshold .001.
These are the actual C14 `Candidate(Policy(N=2, budget_min=32,
budget_ratio=2., h_min=128.))` settings, NOT the six-layer X40 family's
64/256/736/1472 capacities. Preserve the existing stream operations. The
operator's stale-suffix retirement and delayed array correction are additional
array event causes; report them explicitly, not as unchanged native behavior.

## Matched training and immutable comparison

Seed 3401, batch 16, context 128, natural UTF-8 bytes/vocabulary 256, BF16
CUDA autocast, AdamW lr 3e-4, decay .01, clip 1, 2700 ACTIVE seconds. Preserve
C14 source order/sampler, independent window/document resets, tail padding loss
mask and document-disjoint calibration/final-test halves. No cross-document
model, array-slot, stream-slot, or consultation state. Independent keep=.5 per
position on all 260 consultation scalars together, with no rescaling; dropped
answers do not create array slots. No target filtering, hard routing, or fork
loss mask. All real targets count. All-input-zero evaluation disables insertion
of new array answers; delayed residual corrections must not create new array
slots. Physical GPUs 2/4 only through `scripts/gpu_lease.sh`; do not touch the
P3 stacking run or the union extraction service. `/mnt/nvme2n1` stays read-only.
Only `/home/erikg/emender/.venv/bin/python`; no checkpoint committed to Git.

Reuse the exact registered Emender training stream/cache: 1,785,536 windows,
225,880,702 targets, journal SHA256
`3ae1aadac6688019d7c7195ce80e7163a7483ca456a8b67adda006f4c79895cb`.
The cache bounds the maximum registered source exposure. Stop at 2700 active
seconds or exhaustion, recording the actual reason; do not cycle or extend it.
Report actual steps, targets, active seconds, checkpoint stalls, end-to-end
cost, lease wait, and initialization/compile costs under the v2 timer definition.
An exhaustion before the requested budget is a protocol limitation, not a
2700-second success. Preserve partial checkpoint, optimizer/RNG state, stream
cursor, event telemetry and accumulated timer outside Git for exact resume.

Evaluation uses the frozen C14 windows: 3,431,242 windows / 434,204,084 targets;
windows SHA256
`2ab51ce59ad94010d78f6817f29fa06fc5ee64feea766376ff0e35347f196d4d`,
journal SHA256
`00e5c17a75180a367162cef6be5046a721b3d9d58ec273ef2e4a90f68522ea3b`.
Primary final test is 214,097,022 targets. Calibration is reported separately.
Use the shared stratum definitions, float32 per-target CE and float64 sums,
BF16 autocast, deterministic settings/nonfinite guards of the paired evaluator.
Frozen controls are reused without retraining or selecting a new checkpoint.

## Predictions (both directions retained)

PB1, headline: report B1 total final-test NLL minus primary v2_on
**1.509260010** (display 1.509260). Negative: store door better. Exact zero:
door indifferent at measured precision. Positive: input channel better. No
post-hoc equivalence band or claim of statistical significance. Every sign is
informative; report the dense comparison 2.276133549 as a separate anchor.

PB2: report identical-denominator supported-target, all-zero-target and fork
columns against v2_on **0.811639725 / 3.517122903 / 2.010901269** (display
.812/3.517/2.011). Retain nonempty-zero-target, support/empty, reset/interior
and exact sums/denominators; strata overlap and never filter total CE.

PB3, budget pressure: report sampled occupancy over training and peak/final
occupancy per layer/head, stream/array ownership, APPEND/POP/DECAY/CORRECT/MERGE
counts, stale-answer retirements, and refused/merged insertion requests.
Occupancy must remain within the active shared capacity; stale array answers
must actually leave. Zero observed pressure or zero corrections/merges is
reported as absent evidence, never manufactured by relabeling events. Bounded
occupancy alone does not prove useful management or causal contribution.

PB4, unplug/decay: freeze B1 weights. At the SAME independent evaluation window
reset state, suppress all new array answers, apply standing decay-only aging
until array slots are empty (record ticks, thresholds, residual occupancy),
then evaluate with blank consultation context and ordinary stream operations.
Use no held-out warmup from another document. Compare to the frozen dense
control on identical windows/targets, including reset 1–16. Report absolute
NLL delta and whether <= .02 nats/byte (the C14 graceful-degradation margin).
Because resets start empty, also retain a synthetic populated-store decay
receipt to prove that decay really empties array slots; reset-only blank
performance alone is not evidence for eviction. Do not age stream slots or
carry a synthetic context into held-out windows.

## Conditional ablations and launch gates

Only if PB1 moves in either direction (nonzero measured delta), run the matched
append-only control (CORRECT and DECAY/stale retirement off for array slots;
standing capacity/merge behavior explicit) and unbudgeted control (remove array
insertion eviction pressure; retain geometry, other operations and protocol).
Record any finite machine cap and verify it never binds. Keep all three signs,
exposure and telemetry; these controls isolate which operation carries a delta.
No ablation replaces B1 or changes its selected checkpoint.

Before any training: registration committed alone; source/cache hash checks;
2,133,700 trainable count and unchanged shape inventory; force-disabled B path
exact forward parity against dense with identical weights; cache oracle bit
parity; causal delayed-correction test; shared-capacity pressure, stale suffix,
merge, decay-to-empty, document reset, padding and backward-gradient tests;
full relevant test battery green. Test native forward AND replay/checkpoint
backward, including consultation projection gradients. A passing disabled
branch alone cannot certify the enabled path. No results or throughput are
claimed by this registration.

Read authorities: PREREGISTRATION_C14_BYTE.md, RESULT_C14_PAIRED_EVAL.md,
RESULT_C12.md, RESULT_C13.md, their hash-bound receipts, c11/c6 head construction,
x42 velocity_candidate and x40 store forward/backward plus the eager five-op
reference. Resilient DiLoCo is not changed/tested; gap-matrix IDs inapplicable.

## Supervisor key ruling amendment #2 — 2026-11-04

Provenance: supervisor decision delivered in the attended Intercom task channel,
“X43 C14-B1 — SUPERVISOR KEY RULING AMENDMENT #2”. This supersedes the
2026-11-04 tail-64/zero-padding ruling in commit 060445bc and the qualifications
in ca2e775a / receipts/c14_b/key_ruling_resume.json. Those commits remain
historical evidence, not active addressing requirements. No PB1–PB4 denominator,
training budget, frozen arm, oracle or evaluation window changes.

The prior padded-literal construction was flawed: 0x00 is a real byte, not a
padding sentinel; literal suffix b'b' and context b'ab' yielded different keys.
With the reported median suffix of 12 bytes, this was a majority-case filing
inconsistency, not a rare long-suffix collision. The prior implementation and
receipt own this flaw; their tests did not establish shared native addressing.

Array slots use the store's NATIVE APPEND key at the query position, identical
to a stream write at that position. Existing velocity_candidate.Cell.forward
computes z = W_in(x), features = z + features[h](ids), then
k = geometry[h]['key'](features).detach().unsqueeze(2). W_in is trainable;
features[h] is a frozen byte embedding; the key geometry is frozen. Detaching
keys prevents key gradients but does not freeze contextual features over
optimizer steps. velocity_kernels.scan_kernel writes the supplied K payload
into Keys on allocation. Preserve this complete rule, including contextual
W_in(x), dtype, detach and ordinary native read normalization. Do not replace
stream keys or add a suffix fold, byte padding, tail truncation or learned key
adapter. Reuse the actual stream key tensor for an array APPEND at that query
position; no recomputation with projected consultation content as key input.

Content is all 260 answer scalars through the shared trainable device_input
projection and standing content/value maps. Each array slot additionally owns
the full raw queried suffix bytes as stored metadata: no embedding, padding or
parameters. Numeric addressing and reads use the ordinary shared similarity
machinery. Metadata supplies identity for CORRECT (same suffix, wrong/stale
answer) and MERGE (same-suffix duplicate consolidation). Preserve delayed causal
correction and the tested window-local suffix ownership DECAY/POP component.
Array and stream slots share physical capacity and native addressing.

Empty suffix creates no APPEND. Suffixes longer than 64 bytes use the native
rule unchanged; there is no tail-truncation question. Similarity and operations
handle addressing collisions. Native contextual similarity supplies staleness
weighting with explicit ownership cleanup; no universal monotonic-similarity
theorem is asserted for arbitrary or repeated contexts.

PB3 must include array-slot read opportunities, slots found/read, and hit-rate
with explicit denominators, alongside ownership and lifecycle telemetry. A run
whose array slots are never read must expose zero hits. Bounded occupancy or
successful insertion is not a findability receipt. All enabled native forward,
replay/checkpoint backward, finite-gradient and frozen-control parity gates
remain mandatory before any training.
