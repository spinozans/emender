# Phase5 round3: slot splitting compiles T2048, rejected for promotion

**NO-GO.** The split candidate is slower than round2 at matched T1024:
279.310420 -> 368.719367 ms/head. Native-only
1.3B scenario projection is **733.491217 days**. T2048 candidate
compiles and executes, but the mandatory unchanged reference still times out
before candidate comparison; its full state/replay/gradient byte gate is not green.
No production kernel changed and no training was launched.

## Profile, design, implementation

Profile and design were committed/pushed in84aaeba0 before implementation;
implementation checkpoint9c98ff35 was pushed before the complete battery.
The controlled fixed-axis profile shows nearly proportional cost in slot count
at fixed T. Time-only pages leave the BS-sized body unchanged. Per operator
candidate2, pivoted to slot splitting. Width remains64/128; actual1.3B hardware
width/throughput is unmeasured. Initial profile silently retained reachable-tile
clamping and is invalid for axis isolation: its source/receipts remain, while
axis_profile_fixed.json overrides BS after the wrapper and is authoritative.
Both profile timing arms run the same round2 selected kernel; their equality
is a repeatability check, not independent parity.

The opt-in split kernel separates pre-control,128-slot score CTAs, normalization,
ordered256-slot READ launches and post-control. READ carries two FP32x128 parity
accumulators, never page sums. Dedicated scratch separates normalized products,
scores and scalar carries from persistent store. The design states the exact
composition induction and signed-zero/NaN/rounding obligations explicitly.
BS-wide scalar control and normalization remain; only slot-width score/read
work is bounded. This is not a proof of O(1) total kernel state or complexity.
The initial scalar-store Gluon compile error is retained with source and exit1;
explicit distributed zero tensors fix it. No failure is relabeled a pass.

## Ten-shape native sweep

Five synchronized wall samples per arm, same nonzero inputs, physical4 broker
lease, wrapper/allocation included. All ten complete reads/final physical states/
raw traces match the unchanged reference byte-exactly, including repeats.
Registers/spills below describe **post-control only**, not total pipeline
resources or measured hardware spill counters.

| B | T | monolithic reference ms/head | split ms/head | post registers | post spills |
|---:|---:|---:|---:|---:|---:|
|1|32|0.622831|5.989128|255|6|
|1|64|1.391312|11.512997|255|0|
|1|96|3.472275|16.506024|255|0|
|1|128|4.776027|22.656862|255|0|
|16|128|5.799508|23.339634|255|0|
|1|129|21.576461|24.774956|254|0|
|1|257|79.742464|59.731305|255|0|
|1|512|148.545813|120.782323|254|0|
|16|512|190.126322|123.253637|254|0|
|1|1024|719.189629|446.030798|255|26|

The separate matched_r2.json directly compares round2 selected with split on
identical T1024 inputs: five samples each, full fingerprints and repeats exact.
Split median 368.719367 versus 279.310420 ms/head,
1.320106x slower. This separate sweep's
446.030798ms median is also retained: cross-run variability is visible and no
precision/thermal/performance stability certification is claimed. Even the lower
matched median is strongly NO-GO; no occupancy selector can promote a losing
variant from these measurements. No selector was added.

## Gates and compile behavior

- 109 X31 tests pass (17.00s);108 C14/operator tests pass (81.94s).
- Eight short shape/state pairs and four replay/gradient pairs pass with repeats.
- T512/T1024 peak257, seven masked-tail tokens, distinct document reset fixtures
 pass complete state/read/trace/replay/gradient fingerprints.
- All24 adversarial READ/selected-previous-value pairs pass byte comparison:
 BS128..4096, random/cancellation/signed-zero/NaN. READ page boundaries preserve
 the parity accumulator rather than regrouping floating additions.
- T2048 full mandatory gate: unchanged reference stage prints native_forward begin,
 then exits124 under the180s diagnostic bound. No candidate or gradient result
 exists for this gate. Round2 retained stack traces locate its unchanged LLVM
 optimizer bottleneck; this round's stage log alone is not a fresh LLVM stack.
- Separate T2048 candidate-only compile probe: first compile+wrapper+execution+hash
 completes in14.140219s; subsequent median
 1095.749087ms/head (two samples), stable repeated complete fingerprints,
 post-control255 registers/82 spills. This proves finite candidate execution at
 S4097/BS4096, **not independent monolithic byte parity**, replay, or gradient.
 It does not qualify the failed mandatory gate or measure compiler-only time.

All terminal exits, hashes, timing samples and source SHA256 bindings are retained
in receipts/scalability_phase5_round3; validation_summary.json is the index.

## Projection and twin state

With22 layers, two heads, two native passes, T1024 and2B targets, the matched
split measurement gives0.031686821s/target,
31.558862targets/s and733.491217native-only days.
The24h lower bound requires23148.148148targets/s and <=0.502691ms/head even if
all other work were free. Dense/replay/optimizer/source/checkpoint costs are
excluded. The existing round2 candidate remains the better unqualified path
(557.274949days in its historical receipt); no production selection changes.

Seed3402: both trainings completed35419 updates/71677042 real targets with
exit0. Budgeted full evaluation exit0, final-test NLL1.582652269831; unbudgeted
full evaluation still active, no terminal/table, so **no second-seed sign**.
Two read-only snapshots prove broker owner PPID1/starttime matches, heartbeat
renewal and advancing evaluation windows. Card2 was not leased or disturbed.
Seed3401 matched delta remains -0.019429161nats/byte. Single-seed evidence stands;
two-seed hardening/flipping is pending, with no significance claim. Snapshot
hashes and checkpoint identity receipts are in twins_state.json.

Residual risks: failed T2048 independent gate; BS-wide scalar control; severe
READ serialization/launch overhead; unqualified operation-rich maximum-live long
cases, full-model fingerprints, >=2M paired real-target optimizer/RNG/cursor/
checkpoint-resume identity,150M/1.3B measured throughput/memory. Rejected candidate
receipts are progress, not a real1.3B GO or a future-performance promise.

Manual notes: serial attended work, broker cards4/5 only; no WG service,
delegation, protected-card workload, corpus change, sxgc or /mnt/nvme2n1 access,
or .pt commit. AGENTS.md/CLAUDE.md unchanged. No resilient DiLoCo behavior change;
gap-matrix IDs inapplicable. Supervisor/intercom discovery found no callable
transport, so no supervisor contact is claimed. Unrelated existing worktree
changes were left untouched; broad git diff --check reports pre-existing blank
lines in unrelated growth reports. Only scoped staged changes are committed.
