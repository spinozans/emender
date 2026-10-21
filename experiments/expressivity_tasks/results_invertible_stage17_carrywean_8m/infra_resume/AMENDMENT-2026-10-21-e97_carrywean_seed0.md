# e97_carrywean_seed0: 2026-10-21 attended infrastructure amendment
Original registration and interrupted run artifacts remain immutable.
The approximately 21:00 UTC ENOSPC window is an INFRA-INTERRUPTION, not architecture failure.
Resume checkpoint: experiments/expressivity_tasks/results_invertible_stage17_carrywean_8m/e97_carrywean_seed0/step_160000.pt at 160000, SHA256 d55ce04d3b0a816ed8148bd1358e9ebc4e4718b85b615e32604601c59be83592.
Last FULL training row: 160667; partial death row: 160668.
Restore model, AdamW moments, NumPy, torch and CUDA RNG exactly; replay all full rows,
then check only readable diagnostic fields of the partial row. Abort on ANY diagnostic mismatch.
Exact replayed diagnostics are evidence, not proof of bitwise full-state trajectory identity.
Continue ORIGINAL global LR/carry schedules to 1000000; no warmup or wean reset.
X16 curriculum retains its original criterion, live moments/RNG and 500K T64 extension,
with the original attained-LR extension cosine; no T64 launch without the criterion.
Original evaluation, stopping, predictions, memory/wall limits and checkpoint selection apply.
New checkpoints alone are pruned with deletion hashes; original retained evidence is never deleted.
Card 4 only, atomic no-wait gpu_lease; no automatic restart or lease retry.
Before every normal runner write, require root-free >=200 GiB. On guard failure retain the
last hash-receipted checkpoint, abort nonzero, and attempt tiny emergency interruption/terminal
metadata (the sole below-floor write exception). Atomic checkpoint staging remains required.
This pre-write guard cannot guarantee writes against simultaneous external disk exhaustion.
All zero-copy/full-state/DiLoCo/Frontier/scale/overlap conformance claims remain unclaimed.
