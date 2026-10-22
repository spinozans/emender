# X21E eval-only horizon ladder — preregistration

Registered before any X21E evaluation. Authority: taskX21E-horizonladder.md and
supervisor measurement-definition ruling in the attended session (2026-10-22).
No optimizer, backward, training, checkpoint reselection, or scheduler submission.
Resilient DiLoCo R01–R16/NDP/V21S/ISP requirements: not applicable to this
single-device, independent-instance evaluation; no resilient conformance claimed.

## Predictions (fixed before results)

(a) Cold T16 E97 and GDN2 retain their first four learned columns but cliff near
that learned prefix, not at the new sequence end. Cold T64 E97 seed2 retains two
columns, cold GDN2 roughly five; all have chance-like far tails, rather than
unbounded execution. Thus aggregate accuracy decays approximately inversely with
horizon even when the surviving prefix is flat. No architecture winner predicted
from unmatched recipes. Compare cold arms at the same horizon/AdamW recipe;
T64 E97 seed2 versus GDN2 seed0 remains a seed-confounded descriptive contrast.
(b) Conditional on genuinely new beyond-prefix transfer, suffix/structural
features should dominate; in these specific champions predict start-anchored
surviving columns, which would refute that generic transfer prediction. Report
both the conditional hypothesis and observed shift maps, never call a retained
short prefix unbounded transfer. No designated structural events exist in this
iid-driver task: structure/event-relative alignment is not identifiable. Do not
invent events. Start/suffix ties are ambiguous, not arbitrarily classified.
(c) Precision fork: counting/fold-like state predicts failure horizon increasing
with state bit budget B; compressed features predict flat transfer range until a
minimum precision threshold, then a cliff. Predict the latter here: fp32 and bf16
retain similar short prefixes; fp8/int8 lose some marginal columns; int4 may
collapse even the prefix. Measure both forks rather than infer a theorem from a
finite ladder. Native bf16-autocast accumulators are fp32; bf16 STATE is the first
quantization rung, not the native baseline.
(d) Staged curriculum and wean retain more learned prefix columns than cold T64
E97 on every rung (staged >= cold in count of >=480/512 columns), but neither is
predicted to acquire an extending tail. This is a testable staging prediction,
not a matched-recipe causal architecture contrast.

## Fixed measurement rules

Six receipted champions: X10 AdamW T16 E97 best step468000, X11 AdamW T16 GDN2
best step482500, X12 T64 GDN2 final1M, X13 T64 E97 seed2 final1M, X16E curriculum
extension fixed-final T64 step500000, X17 wean fixed-final1M. Verify existing
checkpoint receipt SHA256 before deserialization, snapshot outside old runs;
missing/mismatching checkpoints are unusable and recorded, never regenerated.
Existing X18 frozen controls bind the four cold champions. Staged controls use
their original frozen validation pools and fixed-final per-position counts.

Fresh 512 examples at each T_train * {1,2,4,8,16,32}, shared across champions of
the same training horizon; fixed deterministic seeds 202610220000 + actual T.
Generate each actual T only once; scalar-check every target. Full-input hashes
and zero within-pool duplicates; task distribution unchanged (uniform headers,
iid drivers). Rungs overlap in actual T by design; this is not a training-data
novelty replay claim. Evaluate all columns, joint/X/Y/OOR counts and CE.
Standard no-scaffold evaluation; X17 preserves mandatory target-independent
neutral slots and gathers driver logits, never reads target carry values.

Native ladder: original bf16-autocast forward, fp32 parameters and state
accumulators. Sequential GPU0 on lambda01, deterministic settings, TF32 off,
CUBLAS :4096:8. Batch32 where memory permits; OOM/nonfinite/exception is censored
with receipt, not zero and not a scientific failure. No adaptive reseeding.

Sweep: state after EVERY token, BEFORE readout and the next transition, cast and
dequantize to bf16, fp8 E4M3FN (saturate finite +/-448), symmetric signed int8
[-127,127] or int4 [-7,7], round-to-nearest-even. Integers use one dynamic fp32
maxabs/qmax scale per example/head (32 metadata bits per head); all-zero scale=1.
Only recurrent matrix state is quantized; weights, projections and GDN convolution
buffers remain native. Wrapper fp32 unquantized control separately compared to
native; numerical/backend deviations disclosed, never silently called quantization.
Quantization is simulated, not a packed-memory/speed benchmark. Effective state
bits = elements*B + 32/head for integers (no scale metadata for floats).

Failure horizon: first rung losing any column that scored >=480/512 at native
fresh1x below480/512; evaluate all rungs regardless. No such native1x column =>
already-failed1x; first loss at1x => already-failed1x; no loss through32x => right
censored >32x. Report absolute T and multiplier, not extrapolated infinity.

Alignment: for each surviving >=480/512 column at each rung, map to nearest
native1x surviving column under (i) same 1-indexed start position and (ii) same
distance from last driver. Exact equal offsets imply start-anchored or
suffix-relative; both exact is ambiguous; neither exact is unclassified. Include
all offsets and both candidates. No structural-event classifier without events.
Curves: per-position counts and losses primary; aggregate/prefix/number-surviving
secondary. Every model/precision/rung AND minibatch gets hash-bound receipts;
exceptions record infrastructure censoring. Preserve original checkpoint hashes
and compare model tensors before/after; training updates always zero.
