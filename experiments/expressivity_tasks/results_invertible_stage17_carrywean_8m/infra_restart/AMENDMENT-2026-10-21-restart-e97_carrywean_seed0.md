# e97_carrywean_seed0: approved checkpoint restart amendment, 2026-10-21
The original registration and the exact-replay admission remain immutable.
The supervisor explicitly approved this separate re-admission after the exact gate failed.
Path: RESTART-FROM-CHECKPOINT, not replay-verified resume.
Checkpoint experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m/e97_carrywean_seed0/step_160000.pt, step 160000, SHA256 d55ce04d3b0a816ed8148bd1358e9ebc4e4718b85b615e32604601c59be83592.
Model, AdamW moments, NumPy/torch/CUDA RNG restored and exactly compared before training.
Historical full updates 160001..160667 and partial death
row 160668 are SUPERSEDED by re-execution; training.jsonl marks this boundary
in EVERY row. No historical post-checkpoint validation is copied into this continuation.
Restart from exact checkpoint state; historical post-checkpoint updates superseded. NO at-death-restoration or exact replay claim.
Exact replay divergence: ValueError('replay mismatch step=160022 loss: 2.1394128799438477 != 2.1394338607788086')
Receipt: experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m/infra_resume/e97_carrywean_seed0/interrupted.json, SHA256 7de99e2661e0a9cfecd2094dc115d1e587ec8aca856ceb64a5c2ad3195699c9f.
Independent X15 divergence: experiments/expressivity_tasks/results_invertible_stage15_probes_8m/muon_resume/LAUNCH-INTERRUPTION-2026-10-21.md, SHA256 0ad3afa52bfd49504a1c6062c95e1aa84194db790be6114efdf41ee4a6aa0a10:
Muon's update433044 muon_update_rms0.11409317095890362 vs0.11409317090657782.
CUDA floating-point nondeterminism is a plausible justification for checkpoint restart,
not an independently isolated cause or proof that the trajectories coincide.
No tolerance was weakened; failed exact-replay launches are retained as failed evidence.
Continue original schedules and ORIGINAL ceiling 1000000, no LR/carry reset.
X16 curriculum retains original teaching criterion and live-moment/RNG cosine extension
for500K T64 updates only after that criterion; no new scientific arm or readout.
Card4 only with atomic no-wait gpu_lease; no automatic retry.
The registered200GiB fail-closed write guard, atomic checkpoints, original evaluation,
retention and stopping rules remain in force; original checkpoints never pruned.
Tiny emergency terminal/interruption metadata is the below-floor write exception.
Concurrent external disk exhaustion can still defeat writes; no absolute ENOSPC guarantee.
