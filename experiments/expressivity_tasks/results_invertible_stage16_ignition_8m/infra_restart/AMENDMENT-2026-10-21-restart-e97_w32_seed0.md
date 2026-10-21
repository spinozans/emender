# e97_w32_seed0: approved checkpoint restart amendment, 2026-10-21
The original registration and the exact-replay admission remain immutable.
The supervisor explicitly approved this separate re-admission after the exact gate failed.
Path: RESTART-FROM-CHECKPOINT, not replay-verified resume.
Checkpoint experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/e97_w32_seed0/T16/step_245000.pt, step 245000, SHA256 2812c3ca67a66a840d38185a2050f3a918470dcd54ebf3f94b03d20d3be54fa9.
Model, AdamW moments, NumPy/torch/CUDA RNG restored and exactly compared before training.
Historical full updates 245001..245475 and partial death
row 245476 are SUPERSEDED by re-execution; training.jsonl marks this boundary
in EVERY row. No historical post-checkpoint validation is copied into this continuation.
Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.
Exact replay divergence: ValueError('replay mismatch step=245104 grad_norm: 0.03506257385015488 != 0.03506257012486458')
Receipt: experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/infra_resume/e97_w32_seed0/interrupted.json, SHA256 1dc9aedab9ca61b9a3a2b51bbb369e1db399e7bd0fe5c07004253de289bd7f61.
Independent X15 divergence: experiments/expressivity_tasks/results_invertible_stage15_probes_8m/muon_resume/LAUNCH-INTERRUPTION-2026-10-21.md, SHA256 0ad3afa52bfd49504a1c6062c95e1aa84194db790be6114efdf41ee4a6aa0a10:
Muon's update433044 muon_update_rms0.11409317095890362 vs0.11409317090657782.
CUDA floating-point nondeterminism is a plausible justification for checkpoint restart,
not an independently isolated cause or proof that the trajectories coincide.
No tolerance was weakened; failed exact-replay launches are retained as failed evidence.
Continue original schedules and ORIGINAL ceiling 500000, no LR/carry reset.
X16 curriculum retains original teaching criterion and live-moment/RNG cosine extension
for500K T64 updates only after that criterion; no new scientific arm or readout.
Card1 only with atomic no-wait gpu_lease; no automatic retry.
The registered200GiB fail-closed write guard, atomic checkpoints, original evaluation,
retention and stopping rules remain in force; original checkpoints never pruned.
Tiny emergency terminal/interruption metadata is the below-floor write exception.
Concurrent external disk exhaustion can still defeat writes; no absolute ENOSPC guarantee.
