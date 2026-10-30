X40 part 12 — checkpoint backport, managed reproduction in progress

The discarded depth run is recorded in discarded.json; its approximately 26k
updates and 42.6 hours do not constitute a saved model or a completed receipt.
No claim is made that those weights survived termination.

The runner now uses X42's fsync + atomic rename checkpoint helpers for model,
AdamW, Python/NumPy/Torch/CUDA RNG state. LM checkpoints occur every 128
updates, plus the final update, beneath training/runs/ (gitignored).
The initial FP32 eager attempt, seed 3401, B64/T512, learning rate 3e-4,
betas (.9,.95), weight decay .1 and clipping 1.0 are unchanged. Every observed
training loss must equal the original ignition_v1/curve.jsonl value exactly;
any mismatch writes v1_divergence.json and stops before depth training.

Depth checkpoints occur every 32 updates and include the epoch permutation,
next minibatch offset, NumPy generator state, validation history and best
validation weights. On resume, speculative trace rows are truncated. The
midpoint acceptance is scheduled for updates 14977/14978 after checkpoint
14976: the parent sends SIGKILL to its own worker group, resumes, and compares
loss, model, optimizer, Torch/CUDA/Python/NumPy RNG and sampler hashes.
This live test is PENDING until kill_roundtrip.json says passed.

The CPU depth regression exercises the actual fine_tune loop, interrupting
inside an epoch and comparing final model, optimizer, cursor (including best
weights), validation rows and Torch RNG against an uninterrupted reference.
A second regression checks fail-closed checkpoint registration mismatch.
Both pass. This is not a substitute for the pending GPU SIGKILL acceptance.
The unchanged oracle suite passed 109 tests and production B64 parity passed.
The targeted CPU regression set passed 9 tests.

Parent command: /home/erikg/emender/.venv/bin/python -m
experiments.expressivity_tasks.x40_emender_lm.part12_parent --output
experiments/expressivity_tasks/x40_emender_lm/receipts/part12_checkpoint
Parent PID and child PID are recorded in parent.json and watcher.json.
Physical GPU 0 only. Search parent PID 1599083 was observed alive; no signals
or changes were sent to that process or the other GPUs.

The initial eager attempt was stopped after 24 exactly matching steps because
a 20-second sample measured 5.00006 seconds/update. Its stopped receipt is
retained. The replacement run is ../part12_checkpoint_graph, using certified
per-chunk graphs (historical slice: 153 steps, zero loss difference). Its
parent PID is 1003693 and initial child PID 1003698. The live watcher is the
authority for subsequent child PIDs. Parent launch requires card 0.

Residual risks: full 3052-step trajectory equality, final checkpoint identity,
30000-update depth bins and live SIGKILL equivalence remain unaccepted until
corresponding receipts exist. A source edit during the run invalidates the
fingerprint gate; do not change runner/model sources beneath the active run.
Depth uses the existing promoted stable-mapping, fresh-pool, fused-init and
batched-read flags, scoped to depth. Depth parity runs under those flags before
any update. Their values are bound in checkpoint identity. Each 32-update
checkpoint emits hash and seconds/update telemetry.
The supplementary shared-helper CPU SIGKILL process test passed bit-identical
continuation; it does not replace the scheduled GPU depth acceptance.
The parent and watcher are local processes, not a reboot-persistent service.
The final depth row is published beside depth_control.json only after the
worker completes and the live roundtrip passes. No synthetic placeholder bins
are published. No resilient DiLoCo behavior is modified; its requirement IDs
are not applicable to this local X40 single-GPU task.

Final active launch: ../part12_checkpoint_managed (parent PID 1021053, initial
child PID 1021063). The graph pace experiment was stopped after verifying
approximately 1.00 seconds/update so that depth optimization flags and their
checkpoint identity could be finalized before the long run. All attempts are
retained; neither stopped attempt is claimed as a completed retrain.
