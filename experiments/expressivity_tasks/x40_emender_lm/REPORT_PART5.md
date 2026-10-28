# X40 part 5 — scale surface + 100k levers: FAIL

**The 100,000 tok/s acceptance gate is not met.** Best qualified sustained
training is **54,812.6 tok/s**, B128/T512 on one RTX
6000 Ada. The complete scale surface was attempted, but **18/24 arms OOM**;
only **1/12 matched ratios** exists. This does **not** confirm throughput
across the scale curve. Both matched training curves pass. No irreducible
hardware or algorithmic floor is established, and no gate is waived.

Canonical machine evidence: [summary](receipts/part5/summary.json),
[scope/decisions](receipts/part5/decision.md),
[commands](receipts/part5/commands.json),
[raw surface](receipts/part5/surface/).

## Scale-probe parity and the second launch blocker

The operator's option (b) scopes pressure out of **only the scale probe**.
Values, native payload/strength, live/age, routing traces and gradients still
compare exactly; evidence wires retain rtol=2e-5, atol=2e-6. Production
certification still includes exact pressure by default. The decision cites
part4's nondeterministic unchanged-X31 pressure receipt and the X30/X31
instrumentation-exclusion precedent. The directly located X31 REPORT.md:854
exclusion specifically concerns usage EMA; it is not misquoted as a blanket
historical pressure exclusion. Explicit operator authorization and part4's
experiment authorize this narrow new probe scope.

T512 then passes. T2048 exposes a separate unchanged-X31 rebasing launch whose
grid.y exceeds CUDA's 65535 limit. The probe-local adapter transposes its
store/log-block grid axes while preserving every integer address expression.
Small two-store mappings are bitwise equal to the original; 65537 log blocks
execute successfully. Both context gates then pass. The original failure,
launch-adapter control and passing gates are retained. The final surface also
requires an independent B1/T2048 production-policy gate, including pressure.
All initially failed long arms are rerun after that gate; their earlier
diagnostic JSONs are archived under surface_pre_long_gate/. Upstream oracle
sources and production comparison thresholds remain unchanged.

## Full-stack scale surface

Names mean pure **trainable** count: X40's original “22M” pilot has 21.719M
trainable / 72.163M total parameters. New targets are 21.999M / 150.005M /
1,300.013M trainable. Width/layers are 256/6, 768/12, 2048/24. Every pure block
has four memory heads; this is not X42's historical one-lane adapter.
Controls match actual **total** pure capacity, including frozen features,
within 0.1%, following X40's registered convention. Both counts are receipted;
this is not a matched-trainable comparison.

Every successful cell contains three warmups and **20 synchronized complete
forward + backward + gradient clip + AdamW updates**. Both arms use the same
vocabulary, nonpadding synthetic repeated token batches, dense bf16 autocast,
fp32 native stores, checkpointing and fused optimizer. These are the baseline
backend before the opt-in levers below. Actual OOMs remain failures and have
no invented throughput or ratio. The larger configurations' memory failures
are implementation failures; bounded projection/checkpoint strategies could
change feasibility and no hardware floor follows from these receipts.

| Pure trainable scale | T | B | Pure tok/s | Transformer tok/s | Pure/control |
|---|---:|---:|---:|---:|---:|
| 22M | 512 | 64 | 7,095 | 117,278 | 0.0605 |
| 22M | 512 | 128 | OOM | 111,871 | — |
| 22M | 2048 | 64 | OOM | OOM | — |
| 22M | 2048 | 128 | OOM | OOM | — |
| 150M | 512 | 64 | OOM | 41,734 | — |
| 150M | 512 | 128 | OOM | 38,054 | — |
| 150M | 2048 | 64 | OOM | OOM | — |
| 150M | 2048 | 128 | OOM | OOM | — |
| 1.3B | 512 | 64 | OOM | 6,904 | — |
| 1.3B | 512 | 128 | OOM | OOM | — |
| 1.3B | 2048 | 64 | OOM | OOM | — |
| 1.3B | 2048 | 128 | OOM | OOM | — |

## Pilot levers, measured in order

The original fp32 ignition-v1 model/data/optimizer protocol is used here.
Each successful throughput point has three warmups and 20 measured updates;
rate is total measured tokens / total synchronized update time. These points
are separate from the bf16 scale surface above.

| Receipt | B | Sustained tok/s | Mean update seconds |
|---|---:|---:|---:|
| pilot_baseline | 64 | 6,793.1 | 4.823695 |
| pilot_reads | 64 | 6,524.3 | 5.022490 |
| pilot_backward | 64 | 36,126.6 | 0.907033 |
| pilot_fused | 64 | 36,063.9 | 0.908609 |
| pilot_B128_bounded | 128 | 54,812.6 | 1.195637 |

1. **Causal reads.** X35's two-launch immutable-version read is composed with
   X40's policy wire and recorded reverse. Exact pre-read strengths and payload
   versions preserve the original reduction tree. Full production B64/T512
   parity, including pressure, passes. The separately profiled forward range
   improves **242.371 -> 174.467 ms**; replay stays **663.341 -> 655.238 ms**.
   The >=60k after-reads gate fails. Fresh wall is slower than historical part3;
   its historical 30k result is not substituted for the new samples.
2. **Replay.** Existing X36 nonempty-entry pooling and event suppression were
   already present. This workload has no pool initially, and all 24 scans have
   32768 value-consuming events, so no-event suppression cannot remove them.
   The separate CPU profile identifies 48 address-order-dependent mapping
   module loads (0.793 s of a 1.091 s profiled step). Byte-pointer views and
   runtime integer sizes preserve the identical mapping without specializing
   on allocator permutations. Exact union pooling is also activated for fresh
   T<=512 origins. Production parity passes; replay busy time falls to
   **569.352 ms**, with per-head pool pitches 64/256. Exact sizing introduces
   **24 extra DtoH transfers** and host waits (26 transfers total including
   step telemetry). The >=85k after-backward gate fails. The CPU-only
   one-sample diagnostic is not marketed as a sustained throughput result.
3. **Occupancy.** Original B128 and B256 paths actually OOM. Early Python
   logits release also OOMs at B128 and is reverted. X36's existing bounded
   vocabulary projection enables the successful B128 result above; every
   update asserts execution of all **24 native replays**. Projection roundoff
   follows the existing X36 component contract; native production gradients
   retain zero-tolerance certification. B256 with an additional reentrant
   checkpoint prototype fails after four updates with an actual
   **invalid heap inverse position** device assertion. The checkpoint code
   is reverted; its patch and failure are retained. That cause remains open
   and is not relabeled as pressure telemetry.
4. **Fusion.** Seven fresh native/history fills become one launch. The
   full-update profile removes exactly **144 kernels**, 2526 -> 2382, across
   24 cells. Wall remains about 36k tok/s. This is partial fusion; closure of
   the requested approximately 59 inclusive launches is **not achieved**.
   Eight-warp global replay and eight-warps only at S>=736 both fail exact
   gradients (maximum errors 9.54e-7 / 5.96e-7). Their scoped overrides restore
   the original launch; neither rejected layout is timed.

## Quality, arithmetic and acceptance

All four candidate flags remain **opt-in**: X40_BATCHED_READS,
X40_FRESH_POOL, X40_STABLE_MAPPING, X40_FUSED_INIT. Production defaults retain
the existing path. Fresh production gates pass for reads, backward and
fusion. All **109 X31 + 70 X36 = 179** imported native fixtures pass with
these flags enabled; all **9** local tests pass. The matched v1 curves are:

- Dense original projection: **5,013,504 tokens**, maximum loss
  difference **0**, tolerance 1e-5.
- Bounded projection: **2,097,152 tokens**, maximum loss difference
  **2.86102295e-06**, tolerance 1e-5.

At the best B128 point, the 100k budget is **0.65536 s**;
measured mean is **1.195637 s**. Closure still requires
**0.540277 s** per update / **1.824x**
speedup. Its separate profile attributes 210.030 ms to forward and 596.760 ms
to replay; these busy times are not asserted to be an additive wall identity.
Remaining work includes exact replay parallelism, full launch fusion and
memory-feasible larger/longer training. The failed warp and checkpoint
experiments establish neither a floor nor permission to relax exactness.

Residual risks are the incomplete scale ratios, absent 100k/interim passes,
replay sizing synchronizations, unclosed inclusive fusion, opt-in scope of the
new paths, and the unexplained reverted-checkpoint heap assertion. Opt-in
nonrecording batched reads now **fail closed**; fast-mode evaluation is disabled
until the heap issue is resolved. Recorded training has a fresh post-guard
production gate and matched bounded curve. Fast-mode T2048 scaling is not
established by the baseline scale receipts.

All GPU work was sequential through run_leased.sh with physical-card allowlist 1/5/6; physical
0's depth PID 634691 remained active, 2/3/4 were protected and 7 was excluded.
CUDA OOM messages name logical GPU 0 after visibility remapping. Python was
/home/erikg/emender/.venv/bin/python throughout. contact_supervisor was searched
first and is unavailable; coordination is recorded in decision.md. No WG,
subagents or .pt artifacts were used. No resilient DiLoCo or Frontier/Slurm
behavior changes; their requirement IDs and queue evidence are inapplicable.
AGENTS.md and CLAUDE.md are unchanged. Unrelated workspace edits are preserved.
