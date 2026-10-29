# X40 part 11 — prerequisite counterexamples; optimization stopped

**Task FAILED / incomplete. No production optimization landed.** The requested
zero-support prerequisite fails on native B32 head shapes at T512 and T2048.
The operator's “any counterexample is a receipt and a stop” rule was applied.
[Raw gate](receipts/part11/support_gate.json),
[console](receipts/part11/support_gate.log), and
[decision](receipts/part11/decision.json) retain failures explicitly.

The deterministic probe uses all four production policies, key/value widths
64/128, and physical batch 32, shared by the 1.3B stack. It tests three fixtures
per policy/context: no writes; a single allocation at T-2 followed by pop=1;
and a single allocation at tick zero followed by natural decay. These are
head-level native-state probes, not trained-model telemetry, a full 1.3B
update, reverse parity, or a performance experiment.

All eight popped fixtures leave dead written slots with **nonzero key/value
payloads**. A representative B32/T512/head-0/slot-0 witness has live=false,
strength=0, key[0]=0.25, value[0]=0.25. Thus popped slots are not zeroed.
Natural death also leaves nonzero payloads and **nonzero terminal strength**:
T512/head-0/slot-0 strength is 0.0009999986505135894 (FP32 bits 981668451).
This is threshold death, not floating-point underflow to zero. Never-written
rows remain zero in the sampled fixtures. Absence of a counterexample in those
fixtures is not a proof for arbitrary imported states or all reverse ticks.

The native code matches the witnesses: pop/death update live/strength and
allocator metadata without clearing payloads; decay records its terminal
threshold strength. Changing that storage behavior would change visible state
and journal semantics. No such change was made. No sparse skipping was enabled,
no parity tolerance was weakened, and no failed candidate needed reverting.

A crucial distinction: these witnesses **do not refute X36's entry/allocation
union pooling**. Both witnesses are written rows and belong to that union even
when dead. X36's union cannot be replaced with the current live set on this
evidence. Its documented cold-gradient backing and logical-S reduction operands
also remain necessary. The stronger popped-zero/strength-zero prerequisites in
this task fail; a revised proof based on retained union membership could support
future work, with full bitwise values/traces/gradients/pressure certification.

Source inspection shows X40 already imports X36's build/prepare/unpack machinery
for T<=512. With X40_FRESH_POOL=1 (part 10's run environment), fresh T512 replay
already uses that union. Full-S reduction reconstruction remains for exact
addition order. T2048 retains streamed dense/prefix replay, without this pool;
its strength/probability scratch indexing must be redesigned and certified
before extending pooled addressing. No T2048 port was attempted after the stop.

The requested new synchronized, child-subtracted fused-reverse wall split is
**not completed**. Part 10's CTA-cycle fractions identify read adjoints as the
likely dominant component (70.88% / 64.53%), but are not exclusive wall seconds
and are not promoted to such here. Its original fused wall remains 4.926609 /
36.914377 seconds. There is no new launch-occupancy profile, warp/fusion result,
matched-envelope ratio, sustained arm, or pilot measurement. Historical
measurements retain their original envelope and date; none are after-optimization
results. The 100k pilot gate remains open/failed, with part 10's fresh result
55,429.54 tok/s.

Validation used only the approved Python via run_leased.sh on physical GPU 1.
The final probe completed all 24 fixtures and returned exit 1 for the failed
prerequisite. Two initial harness failures (missing required wire/evidence
arguments) are preserved separately; the probe was fixed, not production.
PID 634691 was alive before/after inspection and was not disturbed. Cards
0/2/3/4/7, the BF16 side branch, and unrelated working-tree changes were left
alone. No .pt files, WG, subagents, Frontier, Slurm, or resilient training work;
resilient DiLoCo requirement IDs do not apply. AGENTS.md/CLAUDE.md unchanged.

Residual work: exclusive wall decomposition; a union-based T2048 proof and
bitwise reverse gate; sequential launch occupancy/tuning; new diagnostic and
sustained comparisons; and the 22M 100k gate. This commit records the stop,
not an accepted speedup or completion of the requested implementation.
