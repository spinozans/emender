# e97_curriculum_seed0: 2026-10-21 attended infrastructure amendment
Original registration and interrupted run artifacts remain immutable.
The approximately 21:00 UTC ENOSPC window is an INFRA-INTERRUPTION, not architecture failure.
Resume checkpoint: experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/e97_curriculum_seed0/T16/step_252000.pt at 252000, SHA256 bcf96443ac0090ef66c317ba3b92db037f4be918325e81739e040681405e9886.
Last FULL training row: 252678; partial death row: 252679.
Restore model, AdamW moments, NumPy, torch and CUDA RNG exactly; replay all full rows,
then check only readable diagnostic fields of the partial row. Abort on ANY diagnostic mismatch.
Exact replayed diagnostics are evidence, not proof of bitwise full-state trajectory identity.
Continue ORIGINAL global LR/carry schedules to 500000; no warmup or wean reset.
X16 curriculum retains its original criterion, live moments/RNG and 500K T64 extension,
with the original attained-LR extension cosine; no T64 launch without the criterion.
Original evaluation, stopping, predictions, memory/wall limits and checkpoint selection apply.
New checkpoints alone are pruned with deletion hashes; original retained evidence is never deleted.
Card 6 only, atomic no-wait gpu_lease; no automatic restart or lease retry.
Before every normal runner write, require root-free >=200 GiB. On guard failure retain the
last hash-receipted checkpoint, abort nonzero, and attempt tiny emergency interruption/terminal
metadata (the sole below-floor write exception). Atomic checkpoint staging remains required.
This pre-write guard cannot guarantee writes against simultaneous external disk exhaustion.
All zero-copy/full-state/DiLoCo/Frontier/scale/overlap conformance claims remain unclaimed.
