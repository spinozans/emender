# Unbounded-store scalability phase 1 design (before implementation)

Decision: first measure occupancy-aware launch variants (8 reference, 16 and
32 warps) without changing the kernel, reduction extent, capacity, policy or
checkpointing. Promote only a byte-exact winner after scoped certification.
Then investigate shape reduction with explicit full-state initialization and
reduction-tree preservation; finally tile/stage/split persistent K/V storage.

Evidence read: RESULT_C14_UNBUDGETED_PROFILE.md,
RESULT_C14_DATA_EFFICIENCY.md, c14_b_step_composition_unbind/composition.json,
X31 RECEIPTS.md and X40 part18/diagnosis.md. The native control kernel holds
BS x 64 keys and BS x 128 values across its token loop. Cap 257 means BS512,
4078 spills and 84.86% measured GPU time in native kernels. Native and replay
are already batched over slots; no per-slot Python loop remains to batch.
The 9.045x statistic is descriptive improvement per exposure, not matched-loss
sample complexity. The historical 110 peak is not the first profiled batch.

| Candidate | Change and expected yield | Certification |
|---|---|---|
| Launch variants first | Increase threads sharing persistent state; potential spill reduction; 0–2x hypothesis, no promised gain | Exact fingerprints outputs, physical state, traces, gradients, telemetry at standard shapes and controlled occupancy; C14 suite; graph/eager remains checked by existing backward |
| Reach-limited launch shape | At most two allocations/token from empty store; 128 tokens implies reach <=256 even at cap257. Simply lowering BS is unsafe: final physical slot256 would be unwritten, reductions change, and empty-read POP sentinel currently occupies padded lane257 | Treat as kernel-internal change if adding initialization/tree logic: full 109 oracles, parity/state/journal, graph/eager, bitwise 2M-token slice, registration rebind in same commit |
| Register-pressure tiling/staging/splitting | Reduce persistent 512x192 floats per program; potential >2x if spills dominate, unmeasured. Splitting adds launches and memory traffic | Full standing battery and registration rebind |
| Occupancy-aware shape selection | Select compiled variant by safe static envelope, never shrink configured cap or use live count as allocation bound | Scoped for warp-only selection; full for arithmetic/tree changes |
| Batch scans | Already batched. Reorganize fused scans only if profiling supports it | Full battery for internals |

Expected steps/s at ~110 and cap257: reference historical training 2.644903,
first-batch compute ~2.6; neither is a controlled 110-live measurement. Warp
variants have no defensible point estimate; hypothesis 2.6–5.2 on the same
batch, below the >=8 goal. A hypothetical 2x reduction in all native time gives
~1.74x overall from Amdahl at 84.86%. >=8 requires roughly 3.08x overall and
~4.9x native acceleration if the profile fraction holds. No measured cap257
fully-live rate exists. Measure controlled 48/110/257 peaks independently of
capacity and retain raw timing, compiler metadata, and parity failures.

No production source changes before this design commit. No training, protected
GPU/service, checkpoint or corpus changes. Python restricted to project .venv.
Supervisor/intercom transport was searched by name and description and is
absent; no contact or supervisor approval is claimed. No resilient DiLoCo
behavior changes; requirement IDs are inapplicable.
