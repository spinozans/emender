# Phase5 round4: streaming within pages, compositional T2048 green, real 1.3B NO-GO

**Selected PAGE 1024 costs 256.591488ms/head versus matched round 2 279.278910ms/head (8.12% lower time). The native-only 1.3B scenario still projects to 510.435903days: NO-GO.** No production kernel changed and no training started.

## Synthesis and sweep

Source inspection found R3 already contained R2's 128-row update streaming,
selected previous-value loads and two-row READ register discipline. R4 reuses
those exact JIT functions and changes READ launch composition. Each page streams
pairs into two FP32 x 128 carries; PAGE does not allocate a PAGE x 128 value tensor.
Score stays 128 rows; control/normalization still have BS-wide scalar metadata.
The new opt-in wrapper exposes page sizes 256/512/1024. At T1024/BS2048, READ
launches/token drop 8 ->4 ->2; total launches/token drop 12 ->8 ->6.

Five synchronized wrapper/allocation-inclusive wall samples on identical nonzero
inputs, broker physical4. All three complete native/state/raw-trace byte and
repeat fingerprints match round2. The independent ten-shape anchor is below.

|READ page|matched R2 ms/head|R4 ms/head|
|---:|---:|---:|
|256|279.422650|351.188002|
|512|279.403880|277.494837|
|1024|279.278910|256.591488|

All 72 independent adversarial primitive pairs pass (24 per page size; BS128..4096,
random/cancellation/signed-zero/NaN). PAGE 1024 READ uses 28 registers / 0 compiler
spills at T1024; post-control 255 / 26. These are per-kernel compiler resources,
not total pipeline resources or measured hardware spill transactions. No Nsight
permission or register-lifetime profile improvement is claimed. Select 1024 by
lowest measured T1024 median. Source snapshots retain the tuning default 512;
explicit page arguments controlled tuning. All execution terminals exit0.

The chosen path saves 26.94% versus same-session PAGE 256, and 8.12% versus R2.
This is a modest native long-shape improvement, far below the real scenario bar.
Small shapes still regress strongly; no unconditional production selector is added.

## Ten independent monolithic pairs

Broker physical5, same nonzero inputs and five samples/arm, complete read/final
physical-state/raw-trace fingerprints and repeats all byte-exact. Timing differs
across runs/cards: selected T1024 here is 254.791934ms. No sustained thermal/
performance-stability certification is claimed. Resources describe READ and
post-control only, not the score/normalization/control pipeline in aggregate.

|B|T|monolithic ms/head|R4 ms/head|READ regs/spills|post regs/spills|
|---:|---:|---:|---:|---:|---:|
|1|32|0.909182|8.801352|32 / 0|255 / 6|
|1|64|1.633042|15.633902|32 / 0|255 / 0|
|1|96|3.669996|22.521573|32 / 0|255 / 0|
|1|128|5.315238|27.692590|32 / 0|255 / 0|
|16|128|6.718789|28.282621|32 / 0|255 / 0|
|1|129|21.931782|25.638167|32 / 0|254 / 0|
|1|257|79.645404|56.613521|32 / 0|255 / 0|
|1|512|148.025742|109.720307|32 / 0|254 / 0|
|16|512|190.277412|99.138622|32 / 0|254 / 0|
|1|1024|714.824872|254.791934|28 / 0|255 / 26|

## T2048 validation strategy and standing gates

Use the operator-authorized T1024 monolithic anchor plus carry-composition
proof, implemented before the long validation. See COMPOSITION_PROOF.md and
DESIGN.md. The unchanged monolithic T2048 compiler remains unqualified; no new
unchanged-reference compile success is claimed. R2/R3 timeout evidence stands.
R4 and R3 share all control/score/normalization JIT functions. The only changed
READ boundaries store/load exact FP32 parity accumulators, never regrouped page
sums; induction on pairs/pages and then token state establishes composition.
The independent full-matrix primitives test the reduction-order premise through
BS4096. T2048 uses PAGE 256 as the anchored decomposition reference, explicitly
labelled in the receipt; it is not presented as an independent monolithic run.

- 109 X31 tests pass in 17.69s; 108 C14/operator tests pass in 41.65s; no skips.
- Eight short state/shape pairs and four short native/replay/gradient pairs pass with repeats.
- T512/T1024 original-monolithic long gates pass full native/replay/gradient bytes.
- T2048 S4097/BS4096 compositional gate passes full native/state/read/trace/replay/gradient bytes.
- All three long fixtures have peak 257, seven masked tail tokens and distinct-document fresh-state reset equality.

These certify the changed page composition in this battery. They do not qualify
every operation-rich maximum-live long case or all derivatives/full trained models.

## Projection, twin state and limits

At 22 layers x2 heads x2 native passes, T1024 and 2B raw-byte scenario targets,
the matched cost gives 0.022050831s/target, 45.349765targets/s and
**510.435903 native-only days**. The 24h lower bound is 23148.148148
targets/s and <=0.502691ms/head with all other work free. Dense/replay/optimizer/
source/checkpoint costs are excluded. This remains NO-GO by the standing real 1.3B
projection standard despite the measured native speed gain. No training/checkpoint
exists for this round. Global underutilization and BS-wide scalar work persist;
>=2M paired real-target optimizer/RNG/cursor/resume and measured 150M/1.3B throughput/
memory remain mandatory before promotion. No eventual speedup is promised.

Seed3402 completed during this round: both trainings: 35419 updates / 71677042 targets,
both full eval terminals exit 0, 66 rows per arm. Independently checked every row's
original denominator (also matching all 66 seed3401 rows) and NLL sum/target identity, both checkpoint bindings and
matched source journal. Budgeted final-test NLL 1.582652269831; unbudgeted
1.564147680122; PT1 delta **-0.018504589709 nats/byte**, holding the negative sign
and exceeding the descriptive .005 near-zero threshold. Seed3401 delta is
-0.019429160810. The negative direction now holds across TWO seeds; no statistical
significance claim. Capacity/implementation/trajectory remain confounded. The
existing delivery independently committed/pushed 067ee73c; round4 binds full
source hashes and tables in twins_verdict.json. Card2 was never disturbed.
Initial finalizer observed the just-released lease as FileNotFoundError and
stopped before producing a verdict; the finalizer then harvested completed
terminals instead of claiming an active evaluator.

Manual notes: design f37f356c and implementation b281035b committed/pushed before
final battery; all final receipts indexed in validation_summary.json. Historical
phase2/R2 guards match and all bound implementation bytes match HEAD. Administrative
CLI/key-lookup failures are retained in ADMIN_NOTES.md; no failed parity gate is
hidden. Serial attended work, broker cards4/5 only, approved venv interpreter;
no WG service/delegation, protected-card workload, corpus change, sxgc,
/mnt/nvme2n1 or .pt commit. No DiLoCo change; gap-matrix IDs inapplicable.
AGENTS.md/CLAUDE.md unchanged. Supervisor/intercom discovery exposed no callable
transport; no supervisor contact claimed. Unrelated workspace changes preserved.
