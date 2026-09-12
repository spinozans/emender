# Bounded grounding correction v1

**Completed:** genuine behavioral improvement, but the predeclared gate was
missed. Both corrected x/y achieved 7/16 autonomous completions (4/8 old,
3/8 fresh) versus 0/16 before correction. First requested calls improved from
0/16 to 16/16. Fresh-value gate required 4/8; no expansion or promotion.

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
## Training completed

`proc_d872` reached the verified `LARGE_SFT_SEGMENT_COMPLETE 32` boundary.
All 32 updates passed exact runtime sample/count/cumulative-clock checks.
Actual scheduled exposure: **15,488,679 input tokens / 1,759,435 assistant
targets**: correction 580,858, native replay 319,232, conversation 688,629 and
retention 170,716. These include the predeclared runtime traversal repeats;
they are not additional unique corrective trajectories.

Complete finite BF16 checkpoint:
`checkpoints/checkpoint_agent_sft_u000032_loss_0.6173.pt`, SHA
`48dffaf72900419a6481bae05bfd149710627f4bf684a152d7d17804cf55b3e6`.
Recipe SHA `74499140ac7b2d5c52bcf09e1d864c533785bb5d4afa616a20681d83fe2ff74e`.
Rank-0 peak allocated HBM: 36,128,739,328 bytes. The checkpoint filename loss
is a training average, not a behavioral or retention result.

## Autonomous execution and retention completed

The same bounded process (`proc_d872`) completed the full training/evaluation
sequence in 3,815 seconds (~63.6 minutes). Both execution and retention
measurements completed; source inventories passed before and after. This process
success is not a pass of the behavioral acceptance threshold.

| Representation | Exact first requested call | Old completions | Fresh completions | Total completions |
|---|---:|---:|---:|---:|
| Pre-correction y | 0/16 | 0/8 | 0/8 | 0/16 |
| Pre-correction x | 0/16 | 0/8 | 0/8 | 0/16 |
| Correction y | 16/16 | 4/8 | 3/8 | 7/16 |
| Correction x | 16/16 | 4/8 | 3/8 | 7/16 |

Every episode was autonomous: zero supplied reads and zero assisted credit.
All 64 agent and 64 reader containers have cleanup receipts. Both corrected
representations completed all four sum tasks and three of four lookup tasks.
Sum solutions read the actual input, executed a Python calculation, and returned
the exact result. These are genuine grounded execution wins, not inference from
loss or a syntactically valid first frame. Fresh-value cases remain same-family
probes, not independent task-family or repository generalization.

Remaining failures in both representations:

- One fresh lookup read the right file but corrupted the observed 20-character
  value in its final answer. Exact information fidelity is still unreliable.
- All four edits failed. A representative trace attempted `create` on the
  existing **input** path rather than the requested output path, then read the
  unchanged input and declared `done`; the required output file was absent.
- All four recovery cases repeatedly read the missing path and eventually
  produced an invalid frame. The correct first missing-file attempt did not
  turn into effective recovery.
- Corrected models issued 12 finishes each, but five were not successful
  outcomes. The filesystem/answer oracle, not a finish claim, determines success.

| Matched retention metric | Pre y | Correction y | Correction x |
|---|---:|---:|---:|
| Tool token accuracy | 100% | 100% | 100% |
| Conversation NLL | 1.5088 | 1.5812 | 1.5778 |
| Native development NLL | .8717 | .9239 | .9216 |

All six retention limits passed, **with measured likelihood degradation**, not
unchanged retention. The primary fresh-value behavioral criterion was **3/8**
against the frozen **4/8** minimum. Therefore `positive_correction_evidence:false`
in the gate is preserved; the threshold is not relaxed despite the real gains.
`automatic_expansion:false` and `checkpoint_promotion:false` remain in force.
No numerical fresh-continuation equivalence is established.

The result demonstrates that this model/training setup can substantially improve
exact first-path use and short task completion. It does not establish the cause
of every earlier failure or reliable general copying. A subsequent separately
frozen experiment should address value fidelity, destination/source distinction,
and recovery from actual failed actions rather than merely repeat the same data
or scale the old SFT budget.

### Final evidence identities

- Training summary: `d0923a3904bde51a1f8a89d7c41f16c1f3e085f19d0f36940235f6564f71ca4a`.
- Execution panel: `7f03244fca9437ebdfbc7044287c9ddeff61d1953c8a80f062032e46f097ad67`.
- Execution summary: `5c8e7dace01b528fd643194109e36b1d55cffcf8e59290952fc623b1dc0f577b`.
- Execution detail audit: `e001ff305e54dd65d22f9cd553362f47d7d4d453bde64082b6b90648e3909c7a`.
- Learning/retention summary: `14322f8696c7bd95ae98567a7a9e7556d7fc5b38d237988c195c61032bd72a26`.

Artifacts are under the training root's `evaluation/`; private generated
reasoning remains only in private episode evidence.
