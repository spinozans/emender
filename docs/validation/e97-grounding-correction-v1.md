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
## Data verification completed

`proc_9ab4` completed in 211 seconds from immutable builder commit `5070bf52`.
All 1,024 authored complete trajectories passed real-executor observation,
arithmetic/edit/recovery outcome checks and native codec/runtime parity.
The source and evaluation cases were frozen before training; the separate
fresh-value evaluation seed is excluded from training answers.

Assembled authority: 1,583 records, 4,561,878 input tokens and 522,278 assistant
targets: correction 170,337; native replay 100,464; conversation 201,392;
retention 50,085. No records were clipped. These are inventory counts, not the
32-update repeated runtime exposure; the exact runtime schedule is generated
and verified before model loading.

Authority manifest SHA:
`bf2645fc1c41fbb013bd11c9b8fe539aba6ac88b618e5f03dd5e759a7f7d2a92`.
Data root: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/grounding-correction-v1-data`.
The builder preserved its actual calls in `correction-verification.json`.

Controller: `scripts/prepare_e97_grounding_correction.py` reuses the qualified
numerical launcher and complete-state collector; the launcher substitutes only
new data, committed u880 parent, fresh-stage policy and the 32-update budget.
A synchronous terminal checkpoint is required. No earlier checkpoints are
removed. `scripts/run_e97_grounding_correction.sh` performs packing/schedule,
training, autonomous execution, likelihood/retention and final evidence gates
in one bounded sequence. It halts on infrastructure/child failure, without retry.
Training deadline 5,400 seconds, each evaluation 3,600 seconds, overall 10,800
seconds, each with 30-second kill grace. A measured behavioral failure remains
a negative result, not a signal to restart or expand.

Controller/episode CPU validation: **26 passed**, including rejection of
insufficient fresh successes or retention damage, and no automatic promotion
or expansion even after a positive correction gate.

Training root: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/grounding-correction-v1-train`.
Training and post-training measurements remain pending.
