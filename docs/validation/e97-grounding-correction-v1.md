# Bounded grounding correction v1

## Frozen experiment

Operator authorized correction after native execution and supplied-read tests
failed. This is a **new 32-update experiment**, not an extension of the closed
880-update program. No automatic expansion, promotion, first-party registry
admission, or independent final-holdout claim.

- Start from exact u880 train/y checkpoint
  `6b529dd37a1237b13728e8901fe1469b18aa2a0a2c5e4282fc01064cdd4233b1`.
- Fresh BF16-SR Schedule-Free state, LR **1e-5**, 32 updates, unchanged qualified
  `b8ee034f` numerical trainer: eight-rank full-world DDP/K4, no outer merge,
  65,536 context, group3/MLP4096/checkpointed FP32 CE128, seed927413, bucket262144,
  BF16 persistent state and no CPU Adam/FP32 master weights.
- **1,024 complete verified authored trajectories**: 256 each lookup, sum,
  edit/verification and missing-file recovery. Actual qualified native executor
  supplies every observation. Arithmetic uses an executed Python calculation
  before finish; editing uses real create/read-back checks. All finish answers
  are independently checked against fixture truth. Private analysis is null;
  supervision emphasizes actions, arguments and answers rather than long prose.
- Half of each family uses the minimal system message, half the exact source
  system message. New paths and values use a separate frozen training seed;
  diagnostic answer values are excluded. No existing diagnostic trajectories
  or generated failed rollouts are trained verbatim.
- Copy unchanged replay from the admitted 50M-agent mixture: at least 100,000
  native, 200,000 conversation and 50,000 tool-retention assistant targets,
  whole-record overshoot counted. All corrective records enter once in the
  assembled authority. The 32-update runtime traversal may repeat records;
  exact exposure and sample IDs freeze in the schedule before training.
- Experimental internal authorization does not mutate original raw eligibility,
  claim legal clearance, or admit a first-party source to the production registry.

## Measurements and stopping rule

After 32 updates, evaluate pre/post y/x on the unchanged eight execution cases
and eight fresh-value cases frozen before training. Retain private full episodes
and authenticated isolated filesystem outcomes. No supplied reads in this gate.
Primary behavioral gate: correction-y completes at least **4/8 fresh cases**.
This is same-family fresh-value transfer, not independent task-family transfer.

Run matched likelihood/retention panels as well. Both new x/y must retain tool
accuracy >=.98, conversation NLL <= pre-y+.15, and native-development NLL <=
pre-y+.10. Report all failed criteria. Even a pass does not automatically launch
more training or promote a checkpoint; a failure does not trigger an automatic
retry, changed recipe, or relaxed threshold.

The sandbox remains nonroot, no-network, no-host-mount/no-GPU, read-only-root,
resource-bounded and owner-cleaned. Model jobs use checked GPU leases, explicit
local-rank devices, NUMA placement, isolated caches and both expandable-segment
allocator variables. ADR-003 safety intent: R07/R12 committed state identity,
R14/NDP13 bounded teardown, R16 evidence discipline; no elastic/native-data-plane/
async, Frontier or numerical-continuation equivalence qualification.

Recipe: `configs/pi/e97-grounding-correction-v1.json`.
Data builder: `scripts/build_e97_grounding_correction.py`.
Execution/measurements and exact scheduled target totals are pending.
