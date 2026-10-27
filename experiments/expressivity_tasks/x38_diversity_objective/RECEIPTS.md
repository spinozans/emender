# X38 arm (e): preregistration before training

2026-10-27. No X38 training has occurred at registration. X37 b/c both
returned 16/16 identical peak occupancies. Occupancy is append dominated;
it is a secondary architectural receipt, not a read-specialization test.

Use exactly X37 c geometry seeds and learned W_in, X34 six-block skeleton
with lane at block 3, X36 replay, unchanged ops/thresholds/allocation.
Objective: CE + lambda times the SUM over six unordered head pairs of
mean cosine similarity of read vectors, averaged over batch and positions
where BOTH vectors have norm > 1e-8. Each pair with no valid positions
contributes zero. Zero reads are excluded, with coverage receipted. Signed
cosine, not absolute or squared cosine; minimization rewards anticorrelation.
Evaluation primary is the unweighted mean over six pairs and the same16
held-out stories (each story weighted equally); coverage retained per pair.
No slot-balancing term in this registration. No new trainable parameters.

Primary lambda .01. Sweep {0,.001,.01,.1} for 611 B64/context512 steps
(20,021,248 tokens). Choose lowest read correlation among candidates whose
validation perplexity <=1.02 times lambda0; ties choose lower lambda.
If none pass, lambda0 remains selected; receipt an adverse/null outcome.
Continue the selected checkpoint to step3052 (100,007,936 tokens), without
resetting AdamW. Never describe the sweep as independent 100M runs.
Seeds/data order3401; AdamW3e-4, betas .9/.95, decay .1, clip1.
X34 a historical endpoint100,003,840 differs by4096 tokens and B8 updates;
quality comparisons to a are near-matched and confounded by batching.

Predictions at init,64,512,611,1536,3052 (all available milestones):
1. Mean pairwise read cosine falls monotonically and final <.3. Report every
   violation and nonzero-read coverage; init need not actually be ~1.
2. W_in pairwise distances and distances from init grow monotonically;
   final pair distances exceed initial pair distances (initial W_in equal).
3. Slot attention-mass JS grows. This requires exact read-mass telemetry;
   missing instrumentation is a missing receipt, never a zero divergence.
   Eval-only single-token shadow replay accumulates un-decayed soft-strength
   read mass by physical slot, normalized per head; six pairwise JS in bits.
   Full-read parity is asserted; empty distributions produce null, not zero.
4. Peak-live remains16/16 identical on the same first16 held-out stories;
   this is expected, not failure.
5. Final perplexity within2% of a; >5% worse is an adverse result.
6. Full X34 depth/context/story probes on selected final checkpoint.
Honest null: decorrelated reads with quality outside the band buy diversity
at too high a price at this scale. Do not infer causality from occupancy.

Oracle and mechanism gates green against exact source hashes before training;
ReplayBackward every step. Parameter receipt vs widened base <.1%.
Broker whitelist0,1,5,6 only, never7 or certified2/3/4. Exact .venv Python.
x31/x34/x36/x37 source read-only. No checkpoint files committed.
Resilient DiLoCo requirements not applicable: no resilient behavior changed.
