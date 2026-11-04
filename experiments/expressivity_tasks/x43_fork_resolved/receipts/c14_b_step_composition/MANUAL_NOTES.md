Fresh trained B1-v2 step composition, card 3, seed 3401, real hash-checked document-local 16x128 batch. The checkpoint was loaded read-only; its weights and optimizer/checkpoint files were never updated. Profiling AdamW updates use disposable state and restore weights outside timing. No retraining occurred.

The original slicing reference and optimized implementation both match the original CURRENT baseline fingerprint byte-for-byte. This includes every native read, final physical K/V/strength/bookkeeping, full operation trace, replay output/gradient, model gradient, telemetry and scalar loss. The run-to-run comparison also passes in each process. fingerprints preserve dtype, shape and raw bytes, including signed zero and one-ULP changes. Native-versus-replay numerical parity still uses the existing tolerance: strict reproducibility is a separate, stronger cross-run/cross-change check, not a claim that two different numerical algorithms are identical.

The removable driver was backward slice-scatter work inside the already captured graph. Original replay took x[:, i] separately for seven full-window inputs at every time position. Each differentiable slice created full-window scatter/accumulation backward work. Unbinding the time dimension once preserves all transition arithmetic and makes autograd stack the disjoint token gradients once per input. No production store kernel, numerical store operation, parameter, optimizer law, checkpoint policy or trained artifact changed. All 92 C14 tests passed before and after, without skips, on the actual leased card 3. Four test lease guards now permit operator-authorized cards 3/5/6 as well as their existing cards.

Metric | Original slicing | Unbind
--- | ---: | ---:
Forward median, ms | 20.696 | 21.882
Backward median, ms | 40.092 | 30.378
Update median, ms | 1.829 | 1.431
Compute steps/s from mean step duration | 14.732 | 17.288
Replay forward reconstruction GPU ms, sum of head medians | 9.387 | 9.187
Replay differentiation GPU ms, sum of head medians | 14.204 | 6.764

Compute rate improved 17.4% in these 50-step repeated-batch samples. These are compute measurements, not a new sustained training rate. Do not substitute them for historical 13.118 steps/s. Every raw timing and host outlier is retained. The reference ablation restores original slice indexing in memory, changes no file, and also byte-matches the untouched initial implementation's fingerprint. Split replay graphs only diagnose the existing transition and gradients; every split graph byte-matches the standing combined capture.

Forward/backward breakdown follows. Synchronizing wrapper measurements are per-step means; they perturb execution and must not be added to uninstrumented medians. Projection categories contain forward work only, with checkpoint recomputation listed in the backward phase; their derivative GEMMs remain in residual backward work.

Component, ms | Original forward | Original backward | Unbind forward | Unbind backward
--- | ---: | ---: | ---: | ---:
260-scalar content projection | 0.251 | 0.245 | 0.207 | 0.271
Stream + array value projection | 0.686 | 0.642 | 0.546 | 0.665
Key/query projection | 1.400 | 1.248 | 1.128 | 1.277
Addressing projections | 1.471 | 1.366 | 1.173 | 1.448
Native fused store including wrapper/encoding | 8.062 | 4.231 | 7.256 | 4.378
Captured store replay + differentiation | 0.000 | 24.443 | 0.000 | 17.371
Telemetry conversion | 9.066 | 0.000 | 8.819 | 0.000

GPU profiling independently sees eight native launches per original checkpointed step, about 6.4ms total: four initial forwards and four checkpoint recomputations. The original combined replay is about 24ms GPU, versus about 16ms after unbind. Split captures attribute the reduction to differentiation, with forward reconstruction nearly unchanged. Thousands of native elementwise slice-backward kernels disappear from the graph. The 260-scalar projection is about 0.2–0.3ms per pass and is not the dominant driver; checkpointing duplicates this small projection but does not reproject content inside the store replay.

Occupancy test: retain all trained K/Q/V/array tensors, slot capacities and window metadata; suppress only external-insert keep flags. All four heads have peak_array=1 when live, zero when suppressed. Peak stream counts are 32/64/5/16 with external answers and 32/64/6/15 without. These are separate per-head/time maxima, not simultaneous totals. Native kernel medians, live -> suppressed, ms:
Head 0, S=32: native 0.582 -> 0.519; combined replay 4.251 -> 4.239.
Head 1, S=64: native 1.351 -> 1.177; combined replay 4.458 -> 4.441.
Head 2, S=32: native 0.534 -> 0.472; combined replay 4.245 -> 4.231.
Head 3, S=64: native 1.225 -> 1.042; combined replay 4.453 -> 4.432.

This intervention does not support a 2.5x occupancy explanation on this batch. Both paths reduce over the full configured S/BS slot axis even when slots are masked dead. The native live/suppressed difference includes insertion branches and changed stream trajectories, so it cannot isolate read-only occupancy cost. It is not legitimate to label that difference as read time. It is also not legitimate to label the entire residual intrinsic architectural cost. Per-token addressing, reads, CORRECT/APPEND/POP/DECAY and dependency bookkeeping remain fused in native or compiled transitions. Splitting their native times would require instrumentation within the production kernel; that boundary was not crossed.

Fresh passive-channel measurements load the actual trained consultation-v2 checkpoint read-only and use the same frozen batch/answer mask. In the reference process its forward/backward medians were 13.140/25.100ms; in the optimized process 11.664/23.142ms. The historical premise that enabled forward alone exceeds the entire passive step does not reproduce: enabled forward is about 21ms here. Passive timing has a large host outlier (optimized backward mean 53.71ms, median 23.14ms). No stable exact passive ratio or 2.5x ceiling is inferred from these small samples. Dense bypass measurements are separately labeled in the final receipts; they are not the passive channel.

Residual classification: verified removable work was slice-scatter reconstruction, yielding roughly 7.6ms less replay GPU time and 17.4% higher measured compute steps/s. Current implementation costs include ~9.2ms replay forward reconstruction, ~6.8ms replay differentiation, checkpoint native recomputation, full-budget addressing/read/operation tensors, metadata encoding and telemetry. Removing replay reconstruction via a production reverse kernel would have an optimistic upper bound of about 9ms GPU/step from this profile; correctness, synchronization and memory costs can reduce that yield. It would cross the operator's production-kernel boundary, so STOP: no reverse-kernel or store-kernel surgery was attempted. Further telemetry/checkpoint-policy changes are unverified possibilities, not claimed wins or intrinsic limits.

X31 re-certification planning estimate, not a measured promise: at minimum rerun the 109-test X31 parity/gradient/state/journal battery plus registered cost, append, storage and structural/checkpoint gates, then the complete 92-test C14 battery and strict fingerprints. Historical X31 REPORT.md records 109 tests in 15.52s but leaves its large-store H16/H1 gate failing; do not turn that historical failure into a green certificate. Budget roughly 0.5–2 GPU-hours plus 1–3 engineering days for re-certification and failure diagnosis after a proposed kernel implementation; actual wall time is unmeasured and implementation effort is additional. No X31 certification is required or claimed for the view-only unbind change; kernel_boundary_audit.json hash-checks unchanged X31/X36/X40/X42 sources and the entire B1 native module.

Failures retained as failures: initial fingerprint test collection used a relative import without a package; corrected to the repository's absolute-import convention, then 2 tests passed. The first diagnostic split capture used retain_graph=True, which AOT donated buffers reject. failure.log and partial receipts are retained under c14_b_step_composition_final. Removing retain_graph for the single capture-time differentiation fixed the diagnostic, and split/combined byte comparisons subsequently passed.

Resume/reproduce, from the project root with /home/erikg/emender/.venv/bin/python only, a real lease restricted to cards 3/5/6, OMP_NUM_THREADS=1, CUBLAS_WORKSPACE_CONFIG=:4096:8 and TORCHINDUCTOR_COMPILE_THREADS=1: run c14_b_step_composition --candidate --steps 50 --compare experiments/expressivity_tasks/x43_fork_resolved/receipts/c14_b_step_composition/fingerprint.json --output <new-directory>. Add --reference-slicing for the original-slicing ablation. Outputs refuse overwrite. The --candidate flag bypasses the OLD trained implementation hash gate for this explicitly compared profiling candidate; it does not authorize retraining. New validation_manifest.json binds the candidate code, tests and logs without editing the trained launch receipt. No model/optimizer profiling state needs resuming; no .pt files are added.

No contact_supervisor tool exists in this runtime (initial catalog discovery found none). No supervisor contact is claimed. Work stayed single-threaded in the attended session, without WorksGood service/workers, subagents or Frontier jobs. Live card-2 eval processes, lease and checkpoint writes were untouched; card 1 pure search, union service, ~/sxgc and /mnt/nvme2n1 were not operated on. Offline union-service unit tests are part of the existing battery. No resilient DiLoCo behavior was changed or tested; gap-matrix requirement IDs are not applicable. AGENTS.md and CLAUDE.md were not changed.
