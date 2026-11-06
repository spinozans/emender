# X40/X43 unbounded-store scalability phase3

**Real training sustains 8.13792 steps/s after warmup. The 1.3B/24-hour budget class is not qualified; the straight sequential kernel path has a named long-context/low-grid wall. No search was launched.**

## Sustained training receipt

Leased physical card3, RTX6000Ada, Phase2 exact row-key kernel installed process-locally. Registered C14 shape B16/T128, two layers/two heads, cell256/key64/value128, raw vocabulary256, ops on, cap257 nonbinding. Actual model: 2,133,700 trainable / 2,218,964 total parameters. This is a C14 training-path receipt, not a150M measurement.

4,833 real sequential AdamW updates / 9,781,410 masked document-local targets in 600.038887 active seconds; 648.355840 wall seconds including setup. Receipt/checkpoint writes exclude 21.814840s; active plus those writes is 621.853727s. Setup and final accounting account for the remaining wall time. Full-run rate is 8.05448 active steps/s; setup-inclusive wall rate is 7.45424 steps/s.

First20 updates use8.610018 active seconds and39930 targets. The remaining 4,813 updates / 9,741,480 targets take 591.428869s: **8.13792 steps/s, 16471.093 targets/s, 60.712 us/target**. Loading/hash verification, decode, assembly, transfers, native telemetry/aggregation, backward/replay agreement, clipping and optimizer completion are inside active time. Unlike Phase2 microbenchmarks, these updates are retained and never restored outside timing.

Natural end-of-step live occupancy averages18.11010; layer/head means are0/0:18.92892 / 0/1:28.78242 / 1/0:6.03180 / 1/1:18.69726. Transient per-head peaks are 0/0:117 / 0/1:115 / 1/0:81 / 1/1:115. This does not certify a model held at110 live slots on every head. Capacity merges and refused insertions are zero throughout. Full per-step operation counts and occupancy sequences are retained losslessly in `sustained/steps.jsonl.gz`; aggregate events and samples are in `sustained/result.json.gz`. Both have raw SHA256 and decompression checks in `compressed_artifacts.json`. See `occupancy_trace.svg` and the full occupancy histogram in `projection.json`.

Initial state is the immutable trained unbounded model, seed3401, fresh AdamW and cursor0; this is further training on a possibly historically seen prefix, not a fresh quality verdict. Every batch uses verified raw document bytes and cache identity/length checks; padding is masked and store state resets per window. The original checkpoint remains SHA256056b876cec5f5aae0b9bcea24544e9e3efccc2791fe9bb91654ca192cfb3eb06. The retained diagnostic atomic checkpoint, optimizer/RNG fields and terminal cursor are hash/load checked; no model is promoted.

## Fixed-length occupancy versus cost

Fresh B16/T257/cap257/BS512 head invocations,20 synchronized samples after5 warmups; all four reference/repeat pairs are exact. Controlled zero-value schedules ramp from empty to the stated peak; peak and time-integrated occupancy are not interchangeable. These are native forward milliseconds, not model training steps/s.

| Peak live | Median ms | Sample min–max ms | Compiler spills |
|---:|---:|---:|---:|
|48|35.704511|35.297081–37.948994|3204|
|110|38.579246|38.256865–40.255377|3204|
|128|39.228821|39.036086–41.881870|3204|
|257|45.481490|45.285095–47.618229|3204|

Descriptive affine fit: **ms = 33.384365 + 0.046881411 * peak_live**, R²=0.999349. Marginal constant:46.881411 us/live-slot/head invocation, or11.401121 ns/(input byte*live slot) after dividing by16*257. The large33.38ms fixed-extent intercept remains: this is an approximately affine four-point curve, not evidence of proportional live-only scaling or a universal sane constant. All rows scan the same physical BS512. See `occupancy_cost.svg` and raw samples/fingerprints/compiler resources in `fixed_curve/result.json`.

## Search and terminal projections

At the C14 shape only, steady-state extrapolation to2700 active seconds gives21972.4 updates / 44,471,952 targets. Reusing the observed write/setup overhead gives approximately2824.7 wall seconds, not2700 wall seconds. This is extrapolated from600 seconds, not a completed2700-second arm and not a150M forecast. The draft clock always budgets2700 active seconds plus synchronized whole-step overshoot; full proxy setup, checkpoints, evaluation and exposure remain unmeasured.

The fixed-T curve alone cannot cross the physical-extent boundary. An additional single-stream forward-only probe holds peak257 and measures:

| B | T | cap | BS | Native median ms | Compiler spills | Qualification |
|---:|---:|---:|---:|---:|---:|---|
|1|257|257|512|43.218856|3204|repeat exact only; no gradient/reference qualification|
|1|512|1025|1024|146.664125|7696|repeat exact only; no gradient/reference qualification|
|1|1024|2049|2048|unmeasured|unmeasured|optional compile stopped; failed exit1 retained|

The BS2048 ptxas exploration was deliberately stopped after over300 CPU seconds and over5GiB compiler RSS; PTXASError(-15) and wrapper exit1 are retained. This is an agent-imposed exploration limit, not proof that compilation is impossible. No T1024 runtime is claimed. `extent_abort.json`, the compiler snapshot and lossless input PTX identify the exact next experiment.

For the150M draft (12 layers, N2, B1/T512), direct measured native-only multiplication is0.146664125 s/head *12*2*2 passes /512 =0.013749762 s/byte,72.729 bytes/s and196367 bytes per2700 native seconds. Forward plus checkpoint recomputation are counted; replay, dense work, decode, telemetry and optimizer are additional unmeasured work. This is a scenario at the controlled peak257 schedule, not a full-model throughput ceiling. N1 halves and N8 quadruples this head-work scenario. Natural proxy occupancy is unknown.

For the historical terminal budget class (`x42_pile_scale/configs/final_pure.json`): width2048,22 layers,B1/T1024,accumulation256,2B target count and86400 wall seconds. The source is GPT-2-tokenized; adopting2B raw-byte targets is a budget-count scenario, not token/byte equivalence. Current velocity nominations leave terminal duration/corpus registration pending. Required raw-byte scenario throughput =2e9/86400 = **23148.148 targets/s**.

Conditional T1024 model: double T and BS relative to the measured B1/T512/BS1024 head, assuming cost proportional to T*BS. Head invocation =146.664125 ms *4 =586.656499 ms. With22 layers,N2,and2 native passes: 0.586656499*22*2*2/1024 = **0.050415793 s/byte**, or19.835055 byte/s.2e9 targets take **1167.032 native-only days** under those assumptions. This excludes other work and is not a certified bound; BS2048 runtime and the scale-natural schedule remain unmeasured. It provides no basis to call the24-hour class tractable.

Backbone width does not widen this native kernel: the committed cell projects backbone width to256, with fixed key64/value128. Layers and head count multiply serial head invocations. Wider dense/FFN/projection cost and utilization are unmeasured; larger model width cannot be used to hide the measured single-stream store latency. Physical allocation depends on reset length, not learned occupancy: max2T live rows, cap2T+1, BS=nextpow2(min(cap,2T)); terminal T1024 has cap2049/BS2048, held-out T2048 cap4097/BS4096. This is bounded within each document/window, not a store accumulating across the entire corpus.

Per native invocation, allocated state is B*S*(4*(64+128+1)+7*4) =800*B*S bytes:1,639,200 bytes at B1/S2049, excluding replay/graphs/activations. FP32 parameter+gradient+two Adam moments alone would be20.8GB at1.3B trainable parameters. Native value staging logically loads/stores1024*min(S,BS) bytes/input/head/pass; withBS2048,L22,N2,two passes this is184,549,376 logical bytes/input byte. This is not measured HBM traffic or a bandwidth roofline; cache residency matters. Full training peak memory is not qualified.

For sensitivity only, parameter-proportional total-cost scenarios are also retained in `projection.json`:150M gives234.296 targets/s and632599 targets/2700s;1.3B gives27.034 targets/s and856.256 days/2B targets. They assume unchanged utilization and proportional total cost to trainable parameters, and are not forecasts. Their disagreement with head-specific scenarios is another reason not to promote a small-model rate to scale.

**Next target:** BS1024/2048 spilling/local-memory pressure and the one-CTA-per-sequence grid; then full proxy/terminal-width raw-byte consultation training with long-context oracle/gradient and memory receipts. The mean-live curve is no longer the only question: the physical scan extent and tiny single-stream grid remain expensive.

## Registration, validation and exact resume

`UNBOUNDED_VELOCITY_FAMILY_DRAFT.md` and `family_registration_draft.json` register ops-on, nonbinding2T+1 capacity, all applicable velocity dimension ranges,150M/2700-active-second protocol and the exact unchanged score utility. Budget coordinates are removed. Raw-byte document boundaries, score-denominator changes, pending initialization/corpus/long-context gates and the lack of executable launch readiness are explicit. No search or terminal training was launched.

Fresh document tests:4 passed. Phase2 source registration matches before and after; no kernel, production runner or frozen-arm source changed. The previous109-oracle/105-operator/fingerprint/training-runner gate is inherited source-bound evidence, not falsely described as rerun here. Fresh controlled curve pairs are exact; every real backward retains its standing native/replay comparison. Final checkpoint hash/load/counters/RNG-field check passed; no resumed-update parity is claimed. Full report arithmetic and all occupancy/event invariants are recomputed by `c14_phase3_report.py`.

Detached internal broker wrapper had PPID1, matching owner/starttime, heartbeat renewal and advancing steps. Training and curve exit0; optional extent exit1 is retained. All owned processes are now absent, card3 lease is released and fresh nvidia-smi shows2MiB/0%. No .pt files committed; cards1/2/7, union service, sxgc and /mnt/nvme2n1 untouched. No resilient DiLoCo change; gap-matrix IDs inapplicable.

Core run is complete. Exact diagnostic checkpoint: `/home/erikg/c14_dense_bytes/emender_phase3_sustained.pt`, SHA256d70b924097522f5de75ec0e56d1ca32975b368abbe564bf02a33634883ca8720; step4833,next_window77328,9781410 targets. No training remains active. Future continuation must restore its optimizer/all RNG/cursor fields through a newly registered runner; current runner has no resume CLI. Fresh reproduction and the exact remaining offline compiler command are in `receipts/scalability_phase3/EXECUTION_NOTES.md`. Do not overwrite completed receipts/checkpoints or launch from the family draft.

Residual risks: natural mean occupancy18.11 differs from a sustained110-live stress fixture; all scale throughput/memory and long-context gradient correctness remain unqualified; optional BS2048 compilation was interrupted; family normalization/corpus/initialization changes need final executable registration. **The operator’s1.3B tractability goal remains open.**
