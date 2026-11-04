# B1-v2 baseline profile and decision boundary

The first 10 completed real-cache steps (16x128, BF16, seed 3401, leased GPU 4)
show 158.957 seconds under cProfile; this is instrumentation time, not an
unprofiled throughput claim. NativeReplay backward consumes 156.310 seconds,
including 115.232 seconds constructing the scalar eager replay and 41.015
seconds differentiating its graph. These nested totals must not be added.
Scalar SharedStore decay alone consumes 41.507 seconds; read 29.236; stream
21.869; dataclass replacement 10.542 (also nested). Native forward and metadata
are minor by comparison. The baseline planned 200 steps was deliberately
terminated after the decisive 10-step sample; baseline_profile.json records the
last completed step, and its status is running because termination did not
complete the requested 200-step loop. No training checkpoint was produced.
A 200-step post-change profile is still required.

The dominant cost is the B-only Python replay wrapper issuing scalar slot
operations, synchronizations and constructing a very large autograd graph,
not the standing production store kernels. Wrapper batching is within the
operator boundary. The independent scalar oracle, B1 Triton forward and all
production X40/X42 kernels are retained. A batched PyTorch replay candidate has
passed 8 CPU output/gradient comparisons against the scalar oracle, covering
capacities 1/2/32/64, enabled/disabled stream writes, incorrect next-byte
predictions, repeated suffixes and zero/padded rows. GPU parity, registered
shape finite backward, checkpoint parity, full battery and the 200-step profile
remain gates, not claims.

First-attempt training receipts/checkpoint and frozen arms are untouched.
training.json's 14 MB is bounded sampled telemetry plus repeated serialization;
checkpoint stalls were already excluded (about 7 seconds over the first run).
Those writes cannot explain a roughly 11-second step. Throttling journals alone
would leave the real bottleneck intact.

Tool discovery first searched for contact_supervisor, then all supervisor or
Intercom tools; none are available in this runtime. No supervisor contact or
reply is claimed. No WorksGood orchestration or subagents were used. Existing
append-only first-attempt evaluation on GPU 2 was observed and left untouched.
No resilient DiLoCo behavior is changed/tested; gap-matrix IDs are inapplicable.
