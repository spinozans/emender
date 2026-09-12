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
Results pending. The original failed probability gate and `rl_optimizer_ready:false`
remain unchanged regardless of process exit status.
