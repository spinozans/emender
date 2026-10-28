# X40 EmenderLM — preregistration

Registered before training, 2026-10-28. Status: implementation/preflight only.
No completed training or scientific outcome is asserted by this registration.

Six residual blocks, each norm -> four-head emender -> norm -> MLP, width
256, FFN1024, identical X34 embedding/head/vocabulary; zero attention.
One shared learned W_in per block precedes frozen head geometry (3701–3704).
Frozen token features follow X39. Keys detach; query READ gradients train
W_in. Learned W_q is the query transform in otherwise frozen key geometry.
The X39 match wire is extended locally with detached post-POP residual
magnitude, empty store zero: W_write(h)+error_coef*error+match_coef*match.
Match is pre-POP certified READ maximum, scale 2/sqrt(64), empty zero.
Error coefficient starts zero; match coefficient -1; W_write starts bias 2.
The threshold remains strict sigmoid > .5; padding never writes.
POP/append coefficients are frozen policy inputs; append amplitude remains
content-only. ISA/thresholds unchanged; all scans use X36 ReplayBackward.
No upstream experiment is modified.

Head budgets 64/256/736/1472 (sum2528 <=2944); pop thresholds .01/.1/.5/.9.
Retention horizons 128/512/2048/8192 for untouched strength1 at free=.001:
decay=exp(log(.001)/horizon), receipt actual float32 T2 arithmetic.
This explicitly corrects X37's near-one rates, whose T2 horizons are much
longer than their informal time constants. Refresh/POP can alter lifetimes.
Budget-capped occupancy alone is not evidence of semantic compression.

Protocol: seed3401, existing TinyStories tokenizer/data, B64 T512, AdamW
3e-4 betas .9/.95 decay .1, clip1, fp32; 3052 updates =100,007,936 tokens.
X39 gate cost .01 times mean actual wired sigmoid supplies the enable-wire
surrogate; detached threshold still has no routing gradient.
Curves each step, quality every256 and step611/final. Existing B64 control
6.8564175640 is the reference; no new trained control. Widen the X34 base
replacement MLP to match total parameters within .1%, receipt trainable too.
Frozen token tables count toward total; this is a capacity receipt only.

Predictions: final quality within5% of B64 control; adverse if >5% worse.
Long context T512/1024/2048/4096/8192: two fixed windows, reset per window,
full-position NLL, 128-token NLL bins, all 24 head occupancy profiles,
mean/peak/final live, routing fractions, R-CAP and pressure. Expect graceful
loss growth (8192/512 PPL <2). Compare existing attention/hybrid receipts
only at their available lengths; missing lengths remain missing.
Fit log peak/log T separately per block/head across the five lengths;
zero occupancy -> undefined slope, no pseudocount. Predict mean slope<=.5;
report every head, capacity and pressure alongside slopes. Hierarchy should
produce unequal head peaks; count equal-head stories on first16 validation
stories. Semantic compression prediction may fail in a pure six-block stack.

Depth probe: original stage3 budget30000 updates, AdamW1e-4 decay1e-4,
B64, permutation epochs dropping short remainder, seed0, validation every500,
latest-best selection. This original budget supersedes the informal word
brief; the X31 2500-update pilot is not the X28E stage3 protocol.
After pretraining use frozen X28E stage3 ladder data, final-only
binary supervision, matched updates and S2944/head. Preserve fixed ID/2x/4x/9x
512-word pools and binary metrics. Predict pure model balanced accuracy>.6
at4x and9x where the fine-tuned transformer <=.55. Honest null: no separation,
chance OOD accuracy, or collapse despite in-range learning. Comparability
requires an identified fine-tuned transformer receipt and exact update
protocol; no invented update count or claim from ordinary LM depth prompts.

Execution: exact project .venv/bin/python, broker cards0/1/5/6 only, exclude
2/3/4 and7. Fresh exact-policy oracle parity before every run; every step
checks all24 ReplayBackward nodes. Source/config/data hashes bound to run.
No .pt committed. DiLoCo/Frontier requirement IDs: N/A (single local GPU).
Contact-supervisor tool discovery returned no tool; no substitute service used.
Code-corpus arm deferred until items1–4 complete. Failures stay failures.

## Part 2 preregistration — 2026-10-28, before v1 execution

PURE STACK v0 is a completed ignition failure, not evidence for an operating
memory architecture. Preserve every v0 receipt in place. CPU term audit is
`receipts/part2_diagnosis.json`: +2 bias was present; initial gates can open.
Thus “never ignited” describes sustained utilization, not proven zero writes
from initialization. No saved weights permit terminal causal reconstruction.

Pre-register one diagnostic rescue, **ignition-v1 (always-enable baseline)**:
freeze W_write weights at zero and bias at +2, freeze match_coef/error_coef
at zero, set gate-cost lambda to zero. Preserve key/query/content paths,
POP, append, budgets, horizons, ISA and thresholds. All nonpadding tokens
must have sigmoid(2) > .5 in every block throughout optimization. This
intentionally removes learned selective enable; it tests whether guaranteed
write opportunities rescue the pure stack, not semantic gate compression.
No lambda sweep or quality-driven variant selection. Same seed/exposure and
quality/context/depth protocols. Prediction: nonzero live stores in every
block and held-out PPL improves over 36.8497193; original <=5% family band
remains a separate, stronger test. Failure to improve is a failure.

Before parent launches 100M training: CPU novel-content construction must
open and retain slots with all four actual policies; GPU short six-block
forward/backward must show every lane enabled, nonzero stores, 24 replay
nodes and finite gradients; fresh oracle/parity preflight must pass. The
runner performs that preflight before training. Use a new receipt directory;
never overwrite v0 or train into an existing curve. No long run in this task.

## Part 3 performance registration (2026-10-28, before measurement)

Ordered gates: baseline X31/X40 parity, unchanged-v1 decomposition, optimization
parity, timing, graph parity/timing, scale probes, bf16 parity, matched training.
B64/T512 target: <=2.5 s/update including AdamW and clipping (aspiration ~2s).
T2048 target: >=2x T512 tokens/s at the same scale shape and microbatch.
Scale: X42 pilot_pure/final_pure widths/layers/heads and microbatch=1,
T512 and T2048, 200 measured updates each after 3 warmups; same-shape X42
transformer controls; pure target >=0.5x control tokens/s. Report peak allocated
and reserved CUDA memory, actual parameters and lane count. No inference from
analytical memory to measured headroom. Training slice: 153 B64/T512 updates
(5,013,504 tokens), seed3401, batches/optimizer identical to ignition-v1;
per-step loss tolerance 1e-5, failures remain failures. bf16 native store must
pass the unchanged bitwise production parity gate before timing/enabling.
No claim that X36 standalone scan timings predict heterogeneous four-policy
lanes: measure the actual composition before changing it. Profile GPU busy
activities separately from synchronized unprofiled update wall times; these
quantities are not additive. Retain full launch ledger, sync/DtoH counts,
journal capacity/bytes and fill/copy activities. No .pt artifacts.

Coordination: tool metadata discovery found no contact_supervisor/intercom
callable. No alternative service or worker machinery is used.

## Part 4 — 100,000 tokens/s program (2026-10-28; before measurement)

Acceptance requires >=100,000 tokens/s sustained TRAINING (forward, backward,
clipping and optimizer) on one leased RTX 6000 Ada at T512 or T2048, B>=64.
Production certification must pass bitwise values, state, traces, pressure and
input gradients versus unchanged X31. A >=2M-token curve must match ignition-v1
at matched steps (loss tolerance 1e-5). Interim targets are >=60k after reads
and >=85k after backward work. Kernel-only timings cannot satisfy these gates.

Order: resolve X42-policy pressure parity; measure 150M/1.3B pure shapes at
T512/T2048 B64/B128; measure and certify chunk-batched causal reads; measure
and certify sparse pooled replay and journal event suppression; occupancy at
B128/B256 and both contexts; then remaining launch fusion. Decomposition
precedes each lever, and each lever is measured before the next. OOM and
exactness failures are retained as failures; no floor declaration before
reads and backward work are exhausted. No .pt artifacts. Only broker GPUs
1/5/6; GPU0 depth probe and certified arms 2/3/4 are protected, GPU7 forbidden.
No WG, workers or delegation. contact_supervisor was searched first in callable
tool metadata and is unavailable in this session; no substitute service used.
