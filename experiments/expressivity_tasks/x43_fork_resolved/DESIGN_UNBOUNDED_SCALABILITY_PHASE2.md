# Phase 2: register-pressure baseline and first experiment

Before kernel mutation, `c14_phase2_profile.py` repeats the phase-1 controlled
native curve with compiler metadata, emitted PTX and CPU/CUDA operator timing.
All work is sequential on leased card3; only the project .venv interpreter.
Phase-1 registration check passes before profiling. Supervisor/intercom tools
were searched by name and description and are unavailable.

The BS256 controlled 128-live case still spills (2276 compiler spill slots,
40 registers/thread); BS512/257-live has 4276. These synthetic specializations
are distinct from the trained-batch 2246/4078 metadata in phase 1. The grid has
only 16 CTAs on 142 SMs: its global warp occupancy upper bound is 1.878%,
regardless of the 100% per-SM resource ceiling. This is an analytic bound,
not a hardware achieved-occupancy measurement. Live occupancy is a separate
quantity. The 257-live case necessarily uses BS512; a BS256 launch cannot
represent 257 simultaneously reachable physical slots.

First experiment: retain the original reduction extents, 8-warps layout and
arithmetic, but place persistent keys/values in the existing final-state global
buffers between tokens. Explicit reloads/stores shorten loop-carried register
ranges; this may reduce compiler spills but adds deliberate global traffic.
No slot identity, arithmetic, policy, capacity, or replay change is intended.
Use a private copied kernel and opt-in wrapper, never mutate shared production
sources. Exact fingerprints decide acceptance; reject a byte difference even
if tolerance tests pass. This experiment does not address the small grid by
itself. Splitting the token recurrence across blocks needs a separate design
and full battery if this experiment is insufficient.

No promotion or training-scale use before standard bitwise parity, all four
live-occupancy state/replay/gradient fingerprints, 109 oracles, graph/eager,
2M-token bitwise curve and fresh training-runner gates pass. No full-battery
claim may be inferred from an isolated candidate pass. No resilient DiLoCo
behavior changes: resilient requirement IDs are inapplicable.

First staging candidate: rejected after the full trained-batch fingerprint
failed (84 differing hash leaves, including loss and gradients). Synthetic
zero-valued curve fingerprints passed, illustrating their limited coverage.
The 8.151 diagnostic steps/s is not an accepted speedup. TTGIR explains the
risk: staging changes inferred 2D reduction layouts from the original
[1,1]/[1,32]/[2,4] value and [1,1]/[1,32]/[4,2] key layouts to vectorized
[1,4] layouts. The next experiment uses Gluon explicit reduction layouts
copied from the retained reference TTGIR, with global staging still opt-in.
Other shape specializations require independent exact fingerprints; these
layout constants alone do not certify them. No failed fingerprint is waived.

Explicit-layout candidate passes the complete trained-batch fingerprint,
including loss, physical state, raw traces, replay and all model gradients.
Fresh first-batch diagnostic compute is 6.90254 steps/s (reference 4.45781);
it is not a high-live full-model measurement. Compiled BS256 has 255 registers,
860 spills. The controlled zero-value BS256 case has 858 spills and roughly
6.2ms native invocation; BS512/257-live remains 3616 spills and 51.4ms.
The nonzero four-occupancy fingerprints and standing battery are pending.

Next isolated experiment (`c14_phase2_gluon_rowkeys_kernel.py`): keys only ever
change by exact row assignment on allocation. Initialize FinalK once, write
only allocated key rows in each array/stream allocation branch, load the key
matrix immediately before the unchanged score reduction, and retain no key
matrix across the remaining token phases. This removes redundant full-matrix
key stores and narrows key register liveness. Value storage and explicit
reference reduction layouts remain unchanged. This is still kernel-internal;
all exact and standing gates apply anew. No production source is edited.

Baseline phase composition on the real 16x128 batch: instrumented native calls
77.10ms forward plus 73.55ms checkpoint recomputation, replay/autograd41.71ms,
telemetry16.07ms. These are synchronized diagnostic component timings and
must not be added to or substituted for uninstrumented training throughput.
At 128 controlled live slots the fused kernel accounts for99.85% of CUDA
operator time; individual READ/APPEND/DECAY timings cannot be separated by
an ordinary operator profiler. Nsight Compute is not installed; measured
hardware achieved occupancy and additive internal-phase timing are not claimed.
