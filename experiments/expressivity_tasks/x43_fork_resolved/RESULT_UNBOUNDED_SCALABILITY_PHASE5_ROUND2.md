# Phase5 round2: structural live-range reduction, no promotion

**T1024 improves from719.876410 to280.137051ms/head (2.569729x), with68
compiler spills. The1.3B/24h class remains NO-GO. The mandatory T2048 byte gate
is incomplete because the unchanged reference stalls in LLVM O3 compilation.
No production kernel is changed or promoted.**

## Profile and decisions

Static compiler-input instruction attribution and the first decision were
committed/pushed before implementation (c3318bb1). Whole-context value loads,
updates and layout conversions dominate expanded code. This is a static
compiler/source profile, not measured physical register liveness or hardware
spill counters; Nsight permission limitations are not represented as resolved.

Candidate1, streaming v1, replaces context-sized value temporaries with128-row
update pages, directly selected previous-value loads and two ascending row-
parity READ accumulators. All ten native/repeat byte pairs pass. T1024 drops
from718.757878 to638.255253ms (1.126129x), compiler spills1524. Short rows
regress, so v1 is NO-GO as an unconditional replacement. Its retained projection
is1269.677333 native-only days. Sources and receipts were checkpointed/pushed
in c64e3197 with the then-pending T2048 status stated explicitly.

The separately receipted follow-up also removes both context-sized key-store
address tensors: an allocating write addresses only scalar dst and64 columns.
Reference is retained for BS<=256; the streaming specialization is selected for
BS>=512. This removes register intermediates rather than introducing more state
passes. An initial Gluon scalar-mask compile failure is retained with its source
snapshot; explicitly broadcasting the mask to64 columns fixes it. Failed sweep
receipts do not qualify. Only selected/sweep_fixed.json is the final sweep.

## Matched measurements

Five synchronized wall samples per arm on identical nonzero inputs, broker
physical card4; native wrapper and output allocation included. All ten complete
read/final-state/raw-trace fingerprints and repeats match. Small shapes execute
the same reference specialization; their observed timing differences are
sampling variation, not a different implementation or a certified speedup.

| B | T | reference ms/head | selected ms/head | registers | compiler spills |
|---:|---:|---:|---:|---:|---:|
|1|32|0.796321|0.783971|255|32|
|1|64|1.529242|1.527942|255|298|
|1|96|3.538495|3.546715|255|858|
|1|128|4.899547|4.895257|255|862|
|16|128|6.321279|5.978018|255|862|
|1|129|21.555531|10.262885|255|26|
|1|257|80.357275|37.402683|255|42|
|1|512|150.408075|73.987806|255|44|
|16|512|189.995121|80.166195|255|44|
|1|1024|719.876410|280.137051|255|68|

Round1's rejected T1024 cost was6937.956855ms with6154 spills, historical
baseline718.648853ms with13802 spills. Matched round2 timing, rather than spill
reduction alone, supports an internal long-class speed improvement. Final
candidate is **NO-GO for promotion** because T2048 and full training gates are
unqualified. No eventual performance or engineering-cost estimate is invented.

## Projection and gates

At22 layers, two heads, two native passes and T1024, the selected cost projects
to0.024074278seconds/target,41.538110targets/s and**557.274949 native-only days**
for2B raw-byte scenario targets. The24h requirement is23148.148148targets/s and
<=0.502691ms/head even if all other work were free. This excludes dense,
replay, optimizer, source, and checkpoint cost; it is not sustained training.

V1:109 X31 tests pass in13.93s;108 C14/operator pass in54.73s; eight shape
pairs and four native/state/replay/gradient pairs pass with repeats. T512 and
T1024 peak257 masked-tail/fresh-document complete fingerprints pass. All24
primitive pairs (BS128..4096, random/cancellation/signed-zero/NaN inputs) pass
READ and selected-previous-value byte equality.

Final selected path: fresh109 X31 and108 C14/operator tests pass; eight shape
and four replay/gradient pairs pass; T512/T1024 masked-tail/fresh-document
fixtures pass. Terminal receipts retain exact counts, timings and exits.
The final primitive check and T2048 terminal status are recorded in
receipts/scalability_phase5_round2/validation_summary.json.

T2048 is not green. Round1 receipts explicitly say unrun, contrary to the
inherited task description. First round2 attempt was stopped with143 after
>15min of CPU work without a result, under an agent-selected diagnostic budget
(not an operator/runtime ceiling). A separate180s faulthandler retry exits124:
stack traces locate the unchanged reference in Triton make_llir's
llvm.optimize_module(OPTIMIZE_O3), before candidate execution. The final
selected fixture adds stage logging and an explicit180s bound. Missing results
are failed/incomplete gates, not passes and not proven native byte mismatches.
The original source and compiler options are retained rather than silently
weakening reference arithmetic or reducing the required context shape.

Residual risks: scalar metadata still scales with BS/256, so only dominant
value/key-store intermediates are bounded. Severe one-CTA-per-sequence global
underutilization remains. Maximum-live operation-rich long cases, trained-model
fingerprints,>=2M real masked targets with optimizer/RNG/cursor/checkpoint-resume
identity, and150M/1.3B-width training throughput/memory remain unqualified.
No training was started for this rejected/unqualified scenario.

No resilient DiLoCo behavior change; gap-matrix IDs inapplicable. No WG,
delegation, protected-card workload, sxgc, /mnt/nvme2n1 or .pt commit. AGENTS.md
and CLAUDE.md unchanged. Supervisor/intercom discovery exposed no transport;
no supervisor contact is claimed. Source bindings, failures, resume commands
and final process/lease checks are retained in the round2 receipts.
