# X40/X43 protocol-shape full-step gap

**At B16/T128, physical S=257 versus S=48 measures 1.474241x full-step cost;
the ~1.6x small-model relationship is not reproduced at the registered context.**
T256 measures 1.660072x. Both arms fit; no protocol-shape OOM occurred.
The inherited S=2049 stress allocation gives different, larger ratios.

|Context|Unbounded S|Observed U/B peak|U full step (ms)|B full step (ms)|U/B|U/B peak allocated (GiB)|U/B allocator margin (GiB)|
|---|---:|---|---:|---:|---:|---|---|
|128|257|128/48|1286.057|872.352|1.474241x|24.255/24.252|21.748/22.113|
|256|257|256/48|2492.735|1501.583|1.660072x|24.273/24.270|21.488/22.135|
|128|2049|128/48|1794.606|872.352|2.057204x|24.276/24.252|18.549/22.113|
|256|2049|256/48|3642.795|1501.583|2.425971x|24.294/24.270|15.131/22.135|

For S=257, the conditional BF16 full-step scenarios are **1.405064x at T128**
and **1.406169x at T256**. Traffic-only ratios remain **1.571911x / 2.760742x**
when both payloads halve. At B16/T128, modeled BF16 payload per head invocation
is 53.363 MB unbounded and 33.948 MB budgeted; at T256 it is 207.389 MB
and 75.121 MB. These are logical payload quantities, not resident model memory.
The optimistic all-allocation-halves envelope is about **12.13 GiB per arm**;
store-only BF16 total peak cannot be inferred without resident storage attribution.
At S=257 the floating K/V/strength state for one B16 head changes from
3,174,464 to 1,587,232 bytes; S=48 changes from 592,896 to 296,448 bytes.

The S=2049/T128 common-cost fit is rejected (negative C); no full-step BF16
forecast is emitted for that control. Nonnegative fits on the primary pair
remain assumptions, not measured precision gains.

Results and conditional BF16 scenarios are in `receipts/protocol_gap/summary.json`.
This measures a disposable 1,301,052,902-parameter byte model at B16/T128 and
B16/T256 with R5 native forward graph replay, captured replay backward,
checkpoint recomputation, gradient clipping and AdamW update. No training run
or retained weight/optimizer/checkpoint update was made.

## Scope and occupancy

The previous B16/T1024 grid is a long-context stress case. T128 is the registered
training context. The same controlled R5 fixture starts empty per document
window, repeats the first-token value, disables pop/array insertion, appends
stream writes up to 257 positions and uses decay 1. Both arms use identical
raw document-bounded inputs and zero consultation features. Existing dense
BF16 autocast and FP32 store arithmetic remain unchanged.

The primary unbounded configuration has physical S=257, while the budgeted
control has S=48. A separate oversized S=2049 control retains the prior stress
allocation. Existing reach-bounded launch uses BS=256/512 for unbounded
T128/T256 and BS=64 for the budgeted arm. Replay backward uses physical S,
so the 2049-slot control is not interchangeable with the 257-slot sibling. Physical
capacity is distinct from actual occupancy. With one stream write per position
and fresh state, peak occupancy is 128 at T128, 256 at T256 and 48 for the
budgeted sibling. **A literal peak-257 R5 fixture at T128 is unreachable.**
We did not prefill/carry state or enable different routing to manufacture it.
These are the unchanged fixture's true reachable protocol shapes, not a
claim that 257 live slots were measured at T128.

Each arm runs five disposable full updates, restoring the same initialized
weights and clearing gradients outside timing. The median uses the last three
samples, excluding capture/compile and early warmup. AdamW moments persist
between samples; all model updates are discarded when the process exits.
Native graph read/trace bytes match eager R5 before timing; the existing native
versus replay tolerance check executes in backward. Data assembly and model
restoration are excluded. Forward, backward and update times remain separate
in each raw JSON receipt. Memory peaks include warmup, capture and optimizer
allocation. Allocator margin is device total minus peak reserved bytes, not a
measurement of all process/driver memory available to a larger job.

## Failed preliminary attempt

`stress48_t128` and `stress2049_t128` are excluded. The reused input helper
loads cached consultation features at T128, unlike its zero-feature T1024
branch. Although fixture array values were zeroed, keep metadata remained live
and allocated one array slot. The unbounded arm completed update 0 at peak 129
and failed the new peak-128 assertion. This was an assertion failure, not OOM.
The budgeted arm's timing is also excluded from the matched comparison.
The `_fixed` pairs zero both features and context lengths before metadata
construction, so array insertion is disabled identically at both contexts.
Exact per-attempt sources, logs, lease ownership and terminal codes are retained.
Older source `scope` text says zero consultation even for the preliminary pair;
this audit corrects that description without rewriting raw evidence.

The first T256 budgeted Python measurement completed, but the launcher was
edited during execution to improve fail-closed lease acquisition. Its shell
then failed parsing the shifted remaining file contents (exit 2) before writing
its terminal receipt. That attempt is also excluded and rerun as `_fixed`.
One earlier T256 lease admission failed before GPU execution and was retried
after a read-only lease/physical-GPU check. `attempt_audit.json` retains both.

## Capacity matters to the ratio

The inherited S=2049 stress control at T128 measured 1794.606 ms against the
same 872.352 ms budgeted sibling: **2.057204x**, with peak 128/48. The S=257
sibling measured **1286.057 ms / 872.352 ms = 1.474241x**. Neither difference
is cross-head sharing. Both use BS=256 in native forward, but PyTorch replay
backward allocates/operates across `cfg.slots`. Thus unused physical capacity
changes full-step cost even when peak occupancy and native launch shape match.
Registration must name physical capacity as well as observed occupancy. The
S=257 pair is the primary 257-capacity comparison; the S=2049 pair preserves
the exact prior stress allocation as a separately labeled control.

## Precision scenario

The unchanged R6 payload model counts pre-read live slots, query/features and
parity carries. Halving both siblings' floating payload leaves the traffic-only
ratio unchanged. `summary.json` includes FP32 and BF16 logical bytes and a
conditional two-point fit `time = C + A*payload`; its BF16 scenario is
`(C + A*M_U/2)/(C + A*M_B/2)`. Two points do not validate the model, even when fitted coefficients are nonnegative. Padding,
launch costs, backward and dense/optimizer work differ from logical payload;
this is not an implementation or certified full-step forecast.

For memory, a store-only conversion gives `P - F/2`, where F is simultaneously
resident convertible floating store storage. F was not measured separately
from graph pools, backward and optimizer storage. The receipt gives exact
per-head capacity arithmetic and the optimistic `P/2` envelope if *all*
allocated bytes halve. Integer metadata, FP32 accumulations and unchanged dense
or optimizer tensors prevent interpreting that envelope as expected BF16
store-only device use. The operator's precision implementation decision remains
held; no BF16 store code was introduced.

## Validation and residual risks

GPU measurements ran serially under exclusive leases on physical card 4, with
`/home/erikg/emender/.venv/bin/python`. Card 1, existing checkpoints, `~/sxgc`
and `/mnt/nvme2n1` were untouched. Input windows use local materialized raw bytes,
source-document hash verification, document-index receipts and masked tails.
No WorksGood service, dispatcher, worker, completion machinery or subagent was
used. No resilient DiLoCo behavior was changed or run; gap-matrix IDs do not apply.

Residual risks: random initialized scale model and controlled routing rather
than trained 1.3B occupancy; uniform 48-slot control rather than historical
per-head 32/64 budgets; only three steady timing samples per arm; NVIDIA-only
execution; uncalibrated BF16 storage residency and traffic-time assumptions;
literal peak 257 is not exercised at T128; the registration must bind physical
slot capacity because replay cost depends on it. The 2.13M reference (~13/8.14 steps/s)
is operator-supplied context, not a matched hardware/fixture remeasurement.
The tool catalog exposes no `contact_supervisor`; no supervisor contact is
asserted.
