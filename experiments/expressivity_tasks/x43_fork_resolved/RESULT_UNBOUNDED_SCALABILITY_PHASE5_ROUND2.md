# Phase5 round2: bounded value live ranges

The first streaming candidate improves long-context speed, but the registered
1.3B/24h scenario remains **NO-GO**. No production kernel is promoted. The
unchanged historical reference and rejected round1 kernel remain available.

The static compiler-input profile and candidate decision were committed and
pushed before implementation (c3318bb1). Whole-context value loads, updates and
layout conversions dominate expanded code. Streaming replaces context-sized
value temporaries with 128-row update pages, a directly selected previous-value
load, and two ascending row-parity READ accumulators. Metadata remains
context-sized, so this is bounded value state, not an O(1) complete kernel.
These are static/compiler observations, not hardware register-liveness counters.

Five synchronized wall samples per arm on the same nonzero inputs, broker
physical card4. Full native read/state/raw trace and repeat fingerprints match
for all ten rows. Timings include the native wrapper and output allocation.

| B | T | reference ms/head | streaming ms/head | registers | compiler spills |
|---:|---:|---:|---:|---:|---:|
|1|32|0.633221|0.866712|255|16|
|1|64|1.408122|1.650462|255|18|
|1|96|3.450835|4.424777|255|22|
|1|128|4.782257|6.298429|255|20|
|16|128|5.704218|7.110240|255|20|
|1|129|21.407941|11.424216|255|122|
|1|257|79.459854|43.133461|255|492|
|1|512|147.976652|82.343598|255|504|
|16|512|189.858091|86.988135|255|504|
|1|1024|718.757878|638.255253|40|1524|

T1024 is 1.126129x faster than the matched reference, with1524 compiler spills
rather than round1's6154 or historical baseline13802. Spill reductions alone
are not acceptance: the short rows regress and this raw candidate is **NO-GO
as an unconditional replacement**, while its long class is an internal speed
improvement. A small/long shape dispatch is the next concrete specialization.

At22 layers, two heads, two native passes and T1024, the measured candidate
projects to0.054850061 seconds/target,18.231520 targets/s and1269.677333 native-
only days for2B raw-byte scenario targets. The24h target still requires
23148.148148 targets/s and<=0.502691ms/head even with all other work free.
This excludes dense/replay/optimizer/data cost and is not a training benchmark.

Fresh broker card5 validation:109 X31 tests pass in13.93s;108 C14/operator tests
pass in54.73s. Eight short shape and four native/state/replay/gradient pairs
pass, including repeats. T512 and T1024 peak257 masked-tail/fresh-document
state/replay/gradient fixtures pass. T2048 is pending at this work checkpoint.
The inherited task's T2048-green claim was not supported by round1 receipts;
this round runs it explicitly.

Residual risks: the selected-value shortcut and READ order still need the
adversarial primitive gate; full trained-model and>=2M paired-target exact
optimizer/RNG/checkpoint-resume gates, maximum-live operation-rich long cases,
and150M/1.3B-width throughput/memory remain unqualified. One CTA per sequence
still leaves severe global underutilization. No eventual speed or engineering
cost is estimated. No training is started for this unqualified scenario.

No resilient DiLoCo change; gap-matrix IDs inapplicable. No WG, delegation,
protected-card workload, sxgc, /mnt/nvme2n1 or .pt commit. AGENTS.md and
CLAUDE.md unchanged. Supervisor/intercom tool discovery returned no transport;
no supervisor contact is claimed. Exact final status follows in round2 receipts.
