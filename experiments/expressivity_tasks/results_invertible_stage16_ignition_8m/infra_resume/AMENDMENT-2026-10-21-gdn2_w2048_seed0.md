# gdn2_w2048_seed0: 2026-10-21 attended infrastructure amendment
Original registration and interrupted run artifacts remain immutable.
The approximately 21:00 UTC ENOSPC window is an INFRA-INTERRUPTION, not architecture failure.
Resume checkpoint: experiments/expressivity_tasks/results_invertible_stage16_ignition_8m/gdn2_w2048_seed0/T16/step_130000.pt at 130000, SHA256 741c4b394b5d11024e544d0c6620fa9f42423130ff31f668accbc9522c8bf616.
Last FULL training row: 130148; partial death row: 130149.
Restore model, AdamW moments, NumPy, torch and CUDA RNG exactly; replay all full rows,
then check only readable diagnostic fields of the partial row. Abort on ANY diagnostic mismatch.
Exact replayed diagnostics are evidence, not proof of bitwise full-state trajectory identity.
Continue ORIGINAL global LR/carry schedules to 500000; no warmup or wean reset.
X16 curriculum retains its original criterion, live moments/RNG and 500K T64 extension,
with the original attained-LR extension cosine; no T64 launch without the criterion.
Original evaluation, stopping, predictions, memory/wall limits and checkpoint selection apply.
New checkpoints alone are pruned with deletion hashes; original retained evidence is never deleted.
Card 3 only, atomic no-wait gpu_lease; no automatic restart or lease retry.
Before every normal runner write, require root-free >=200 GiB. On guard failure retain the
last hash-receipted checkpoint, abort nonzero, and attempt tiny emergency interruption/terminal
metadata (the sole below-floor write exception). Atomic checkpoint staging remains required.
This pre-write guard cannot guarantee writes against simultaneous external disk exhaustion.
All zero-copy/full-state/DiLoCo/Frontier/scale/overlap conformance claims remain unclaimed.
