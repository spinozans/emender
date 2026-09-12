# Live actor probability-capture audit v1

The [complete-generation diagnostic](e97-actor-logprob-repeatability-v1.md)
reproduced forced replay exactly, but did not explain its disagreement with
historical actor probabilities. Original actor files bind disk checkpoint and
args, not effective in-memory parameters/buffers and runtime math settings.
This bounded audit captures that missing evidence at the live collection path.

## Frozen scope

Repeat the original sample-0 lookup and edit episodes from
`native-onpolicy-canary-v2`: **eight training episodes**, four per worker. Reuse
original tasks, seeds, prompts, checkpoint y, native sandbox/oracle and verified
teacher-continuation behavior. Preserve original per-family collection order.
Two fixed actors replace the original eight-actor launch; this is an explicit
diagnostic difference, not a claim to recreate every historical condition.
No new independent evaluation or capability-improvement claim is made.

Source panel SHA: `795ef4e090d6606ddc099c0f630b6e876c39833dc9ae8494263429ac1fea7be8`.
Source summary SHA: `8278969f8ff65afd6f49251abb8a05c758ef6586c748824d3cdbc8345e187e6d`.
Checkpoint SHA: `48dffaf72900419a6481bae05bfd149710627f4bf684a152d7d17804cf55b3e6`.

Record parameter and registered-buffer hashes before collection, after live
collection and after replay. Capture model configuration, weight mode, MLP
settings, dtype, thread count, Python hash seed, GPU, Torch/CUDA and math flags.
Use a local fingerprint implementation rather than importing numerical-probe
modules into the actor path. `PYTHONHASHSEED=0` matches the original collector.

Run the unchanged native episode/generation code and preserve newly recorded
probabilities, rewards and teacher repairs. Only after all four live episodes on
a worker, force-replay every nonempty sampled turn from both fresh and historical
episodes using the same loaded model. The audit calls cannot affect later
sampling within that worker's collection sequence. Do not replace actor
probabilities with replay values. Report fresh self-consistency and historical
differences separately; keep the .0001 actor-replay tolerance unchanged.

## Safety and bounds

- Zero optimizer updates, BF16 persistent parameters, bounded FP32 probability
  arithmetic, candidates still not training eligible, no automatic expansion,
  promotion or first-party admission.
- Existing nonroot/no-network/no-GPU/no-host-workspace native tool containers;
  paused external snapshots and ownership-checked cleanup; teacher assistance
  never earns autonomous reward.
- Existing eight-turn/4,096-per-turn/8,192-per-episode and time limits. Teacher
  continuation remains in the same container and leaves model prefixes masked.
- Two checked leased GPUs, explicit local-rank device and NUMA placement,
  isolated Triton caches and both expandable allocator settings.
- 4,200-second worker deadline, 4,500-second outer bound, 30-second kill grace;
  no retries, communicator reuse, external services or resilient training claim.

Scripts: `scripts/audit_e97_live_actor_capture.py` and
`scripts/run_e97_live_actor_capture_audit.sh`.

CPU validation: **73 passed**, including parameter/buffer fingerprint regression
checks and the existing numerical candidate/native runtime tests.
Artifacts: `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-actor-capture-audit-v1`.
## Completed: live capture and historical replay agree exactly

`proc_00d3` completed in **422 seconds** from `d93a7361`. All eight repeated
live episodes reproduced the historical generation records exactly, including
prompt IDs, generated IDs, probabilities and stop labels. Both fresh and
historical captures then matched same-process forced replay with **maximum
absolute difference 0**. These were repeated failed lookup/edit episodes, not
new capability improvements; repairs remained separately attributed.

Both actors matched the standalone assay's unchanged parameter hash
`f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd`; no registered
buffers were present. Captured settings included eval mode, train-y weights,
MLP chunks 0 for all 18 layers, float32 default dtype, one effective Torch thread,
Python hash seed 0, TF32 disabled, BF16 reduced-precision reduction disabled,
highest matmul precision and the expected CUBLAS workspace configuration.
Source inventories passed before/after. Zero optimizer updates occurred.

- Recipe SHA: `34290c6c836be0120a287706490f249199e2dc1d7067f13cd28adc2b9354e1fe`.
- Summary SHA: `f5b5ed20209235314b5f6d65ac6211df7cba44e6759ab726398899f7cc994b92`.

The historical records are reproducible under the live collector path; the
standalone assay mismatch is not explained by differing parameter values.
This is not yet a demonstrated cause or a pass of all 57 original assay turns.
The earlier failed gate and `rl_optimizer_ready:false` remain unchanged.

### Next bounded initialization comparison

The live collector explicitly sets `PYTHONHASHSEED=0`; standalone assay wrappers
did not set it. The current parent shell reports it unset, but that is not a
retrospective record of every historical process. Test **explicit 0 and 123**
using the exact same immutable `dc444435` standalone replay export, all four
selected inputs, all four generation modes, two repetitions and two workers.
Run the two conditions sequentially in fresh roots/caches; compare each against
recorded actor probabilities and against the other condition. Assert identical
replay recipe bytes, and report cross-condition parameter identity. This tests
a hypothesis; do not claim Python hashing is the cause before measurement.

Controller: `scripts/run_e97_actor_hashseed_audit.sh`. No numerical model code,
weights, data, rewards or thresholds change. No child retries. Each condition
has a 1,900-second outer bound around the original 1,800-second worker bound;
the two-condition program has a 4,200-second bound, all with 30-second kill grace.
Numerical source inventories are checked before/after each condition. Artifacts:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-actor-hashseed-audit-v1`.
Results pending.
