# Revised round8: graph benefit measured; split fusion compiler-rejected / NO-GO

Round6's 6145-launch observation was eager forward store execution. Existing
backward graph capture does not make that forward profile a graph profile.
New measurements hold the warmed R5 store, inputs and shape fixed and alternate
seven eager and captured replays, with synchronization around each sample.
Capture includes initialization; all six output/state/trace buffers match eager
bytewise for each tested batch. Profiling occurs separately from wall timing.

|B/T1024|eager ms|graph replay ms|speedup|eager / graph host calls|device kernels, both|
|---|---:|---:|---:|---|---:|
|1|188.983311|88.979067|2.124x|6145 kernel / 1 graph|6145|
|16|189.292280|92.992223|2.036x|6145 kernel / 1 graph|6145|

Receipts: graph_b1_final.json and graph_b16.json. Initial graph.json is a prior
B1 repeat (189.283 / 88.992 ms), before adding batch selection to the harness.
These are **preprojected forward store invocation times**, not training-step,
backward, projection or optimizer times. No end-to-end step improvement is
claimed. Graph replay removes about 100 ms B1 / 96 ms B16 from this eager
baseline; the remaining ~89/93 ms includes actual device work and graph
scheduling. It is not all removable launch overhead. This materially shrinks
the fusion target relative to eager-only assumptions. Backward's already
captured PyTorch replay remains unchanged (c14_b_control_graph.py).

## Portable candidate outcome

The inherited two-kernel probe failed at B1/T32, before execution
(split_terminal.json exit1). The revised three-kernel candidate separates
pre-control and post-control from score + normalize + ordered READ. It also
fails ConvertTritonGPUToLLVM at B1/T32 with SmallVector<long> idx<size()
assertion / PassManager::run failed (three_terminal.json exit1). The failure
persists without fused control: attributing it solely to mega-kernel complexity
was insufficient. The exact compiler/layout interaction remains unresolved.

Three candidate kernels per position plus initialization is an intended source
schedule only. No successful candidate launch count, timing, byte fingerprint,
shared-memory staging, or speedup exists. Compositional fixtures, ten native
pairs, T512/T1024/T2048 masked-reset gates, 109 X31 and 108 operator tests are
NOT RUN for this failed candidate. The baseline graph/eager comparison is not
candidate certification. Do not promote or change production/registration.

## Scenario projection

projection.json retains the historical 2B bytes *88 native passes formula.
B1 stress: eager 375.943 days; replay 177.006 days.
B16 whole-batch timing divided by16: eager 23.535 days; replay 11.562 days.
These forward-native-only extrapolations use a nominal second pass and exclude
actual backward, dense work, optimizer, data and checkpoint costs. B16 is the
registration-facing comparison, but remains far above one day even before
excluded work. No new certified registration-facing training number exists.

## Portability and operator constraints

All new implementation uses torch and Triton experimental Gluon. No new CUDA
C++ kernel, driver operation, or hardware-counter attempt. Existing X31 CUDA
production sources untouched. Actual execution is NVIDIA RTX6000 Ada only.
Torch graph and Gluon source usage alone do not prove ROCm/AMD, Intel, or other
vendor support; those executions are untested. No cross-vendor bit exactness or
performance claim. The working 180.08 MB/head round6 traffic estimate remains
analytic and occupancy conditional; no new physical traffic measurement.

Runs were serial, attended, exclusive broker physical4. No training, WG,
subagents, card1 work, sxgc or /mnt/nvme2n1 access; no .pt artifact committed.
Final physical4 check: 0% utilization, 2 MiB, wrappers completed and leases
released. Unrelated workspace changes preserved. AGENTS.md/CLAUDE.md unchanged.
No resilient training behavior was changed/run; DiLoCo validation IDs do not
apply. Tool discovery found no contact_supervisor capability, so no supervisor
contact or approval is asserted.

## Exact resume

This stop records a reproducible compiler blocker, not a claimed runtime ceiling
or successful completion of the revised round8 implementation. Start from
c14_phase5r8_three_cycle.py and three_console.txt; minimize score/normalize/READ
layout transitions independently, then split further or use torch-native
operations if needed. Preserve fixed normalization tree, parity accumulation,
signed zeros, masks and per-call state initialization. Never add CUDA C++.
Use new receipt labels (existing wrappers refuse overwrite), approved venv
Python, exclusive permitted-card leases. Obtain compiling nonzero B1/T32 and
byte parity first, then all standing exactness gates before timing or scaling.
Measure a complete training step eagerly and with existing backward capture
before claiming model-level improvement. Validate an actual ROCm backend before
claiming portability. GO remains blocked by compilation, unrun exactness gates,
unmeasured full-step behavior, model-level and paired optimizer/checkpoint gates.
