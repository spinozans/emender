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
