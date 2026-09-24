# BF16 update-resolution investigation

Status: synthetic CPU mechanism and real checkpoint endpoint audits complete;
no production optimizer change. Date: 2026-09-09.

## Observation and hypothesis

`ndm/schedulefree_offload.py` stages parameters, gradients, `z` and second
moments in parameter dtype and performs in-place arithmetic. With BF16
parameters, small writes can round back to their old values. The source audit
establishes this mechanism exists; it does not establish how many historical
E97 updates were lost or prove why the analysis-enabled task failed.

The data/boundary audits did not measure effective weight movement. The new
read-only checkpoint audit therefore compares the parent train/y starting point
with saved/x, CPU-reconstructed train/y and z endpoints in the old 2e-6 q8,
old 2e-6 u240 and new 5e-5 q8 checkpoints. It reports per-parameter and aggregate
unchanged fractions and delta norms. Endpoint equality alone cannot distinguish
rounding, small gradients and cancellation. CPU basis reconstruction is not a
GPU per-step writeback trace.

## Concrete synthetic evidence

`bf16-update-diagnostics-v1/synthetic-writeback.json` under
`/mnt/nvme2n1/erikg/e97_systematic_posttraining` records 100 constant
normalized-gradient subtractions on 65,536 coordinates. Example:

- Initial BF16 value: 0.010009765625; LR 2e-6.
- Ideal arithmetic endpoint: 0.009809765625.
- Ordinary BF16 nearest-rounding endpoint: 0.010009765625 (unchanged).
- Stochastic-rounding mean endpoint: 0.009809757582843304.

At an initial value of 1.0, even LR5e-5 remained unchanged under ordinary
rounding in this synthetic experiment; stochastic rounding tracked the ideal
0.995 endpoint in the mean. A higher LR is not a general numerical remedy and
can also create biased oversized rounded steps. These are CPU synthetic
constant-gradient results, **not Adam, Schedule-Free, GPU or convergence passes**.

## Real checkpoint endpoint results

All 4,045,972,080 parameter coordinates were compared with the parent train/y
starting point, using checked tied-storage deduplication and optimizer-slot shape
mapping. Results are endpoint comparisons, not per-step update measurements:

| Checkpoint | Train/y unchanged | Saved/x unchanged | z unchanged | Train/y RMS delta |
|---|---:|---:|---:|---:|
| 2e-6, q8 (806,195 targets) | 98.7571% | 98.7572% | 97.2581% | 4.993e-7 |
| 2e-6, u240 (~50M targets) | 98.2478% | 92.9874% | 91.0771% | 7.169e-6 |
| 5e-5, q8 (806,195 targets) | 49.1959% | 49.2147% | 28.3668% | 4.583e-5 |

The two q8 runs consumed identical target/input counts and started from the same
parent. Their comparison is particularly informative: the 25x LR increase
produced approximately 92x larger net train/y RMS movement. This is consistent
with low-precision update suppression but does not separately quantify rounding,
gradient magnitudes, changed gradient trajectories or Schedule-Free basis effects.

In the terminal 2e-6 checkpoint, 99.6949% of embedding train/y coordinates still
matched the parent. All 37 normalization-weight tensors were unchanged in x/y/z
in every audited checkpoint. Gradient activity of those tensors was not measured,
so zero endpoint change is not independently attributed to rounding for each one.
Saved/x and reconstructed train/y differ: **do not summarize this as “98% of all
updates were discarded” or “98% of gradients were zero.”**

Receipts in `bf16-update-diagnostics-v1`:

- `lr2e6-q8.json`: `5461ff65afa36413b00da078b137e2685c9e9a683049a640a0c2887f7b9b31d2`
- `lr2e6-u240.json`: `d2104f66760eb4de67fb2dd19b4610762aa0e48192910f289d0fc12c5f9eddaf`
- `lr5e5-q8.json`: `1c89ee216dc8ab9e83da2c63a7fb3c9b7479f2717e971ff384eb812f4265b7d5`

The read-only CPU audit process completed in 466 seconds. These observations
justify further precision investigation, not a model-capability or convergence
claim. The immutable 5e-5 experiment resumes to its already declared u64 gate;
stochastic rounding remains separate and unqualified for production.

## Proposed remedy, not yet admitted

Prototype `ndm/bf16_rounding_probe.py` performs stochastic BF16 writeback of
finite FP32 temporaries using an explicit caller-owned RNG. It requires no
persistent FP32 master weights and is not imported by the production optimizer.
Eight rounding tests and two endpoint-audit tests passed, including positive
and negative values, small-update mean preservation, RNG state replay and
nonfinite/overflow rejection.

Before a production optimizer variant is considered:

1. Measure endpoint motion and sampled actual proposed-versus-applied updates;
   distinguish gradient quantization, parameter/z writes and basis conversions.
2. Compute update arithmetic in bounded GPU FP32 scratch, then round once at
   each intentional persistent BF16 write. Do not merely apply stochastic
   rounding after an update was already rounded away.
3. Specify Schedule-Free x/z/y and train/eval conversion semantics. Save or
   evaluation must not accidentally consume rounding RNG or perturb live y.
4. Bind per-rank/per-parameter rounding RNG to resumable state and test bucket
   partitioning, DDP synchronization, interrupted reload and exact RNG replay.
5. Qualify CUDA behavior, memory/throughput and controlled optimization against
   a high-precision reference. Keep the existing immutable run unchanged; use
   a separately versioned recipe if the implementation is accepted.

The 5e-5 q8 checkpoint completed numerical/mmap reload qualification:
SHA-256 `3f1cd3f103a9d60d856e07190a4682be3aabc1431cc5a74c03d264cec4fb4516`,
806,195 consumed assistant targets. After the CPU audits completed, clean exact
resume to the planned u64 milestone was launched as `proc_2c72`; no behavioral
result or promotion is inferred from q8 loss or parameter movement.

Architecture scope: ADR-003 safety intent R07/R12 (checkpoint semantics),
R14/NDP13 (bounded work), R16 (evidence) and NDP15 atomicity only. No elastic,
native or asynchronous conformance claim. No training snapshot was modified.
