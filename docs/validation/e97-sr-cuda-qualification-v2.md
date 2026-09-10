# E97 BF16 SR candidate: local CUDA qualification v2

Date: 2026-09-10 UTC. Status: **bounded local gate passed**.

## Scope and authority

This is an independent optimizer prerequisite for the sustained learning program.
No E97 checkpoint or training dataset is loaded. It uses a small BF16 linear model
and synthetic gradients. A separate small FP32 reference is a test oracle, not
persistent master weights in the candidate optimizer. No production CPU Adam
arithmetic or FP32 masters are introduced.

The compute-pool design authority is ADR-003 in
`docs/RESILIENT_DILOCO_COMPUTE_POOL.md`, with the production crosswalk in
`docs/RESILIENT_DILOCO_GAP_MATRIX.md`. This fixture exercises local safety intent
for **R07/R12/NDP15** (atomic committed state and fresh-process continuation),
**R14/NDP13** (bounded process lifetime), and **R16** (exact-source evidence).
It is not a complete production conformance or fault campaign. The synthetic
numerical checks do not claim elastic R15 changing-membership semantics.

Explicitly unclaimed: elastic membership R02--R06/R08--R11; NDP02 no-all-rank
semantics; native NDP17 scale gates; V21S01--V21S17 and ISP01--ISP07. Fixed-world
NCCL collectives are deliberate. No Frontier/ROCm, communicator shrink, failed
communicator reuse, automatic restart/retry, or scheduler requeue is exercised.
The planned resume is a separate successful-path job, not recovery from a failed
child.

## Frozen execution

Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/precision-sr-cuda-qualification-v2`.
Inventory SHA-256:
`a9fd3e8b46905d2af58264b4da33f3be5e07f31f8f33d56317d31beefebda31d`.
Qualifier SHA-256:
`94c7b7d92882eccfceee0fc43969ed5306122782cd6adb380ef001013524b64f`.

Managed process **`proc_e7f3`**, `e97-sr-cuda-qualification-v2`:

```text
CPU regressions -> exclusive eight-GPU lease -> produce -> fresh resume -> aggregate
```

The original precision candidate snapshot is verified and copied, never modified.
Overlays are the corrected qualifier with explicit LOCAL_RANK selection and route
receipts, the fail-closed aggregator, and routing/aggregation tests. Optimizer and
`train.py` bytes match the earlier CPU-qualified candidate. The older snapshot
with incorrect CUDA routing is not launched.

Before CUDA, the source-frozen CPU regression suite must pass. The two torchrun
phases use `--max-restarts=0`, distinct standalone rendezvous, NUMA placement,
per-rank Triton caches, both expandable-segment allocator variables, and the
CUBLAS deterministic workspace setting. The runner has a 3,600-second total bound,
1,200 seconds per CUDA phase, and 900-second collective bounds. Failure stops the
chain and retains partial output; no automatic retry occurs.

## Required measurements

- Pure DDP, pure avg-DiLoCo, and hybrid DDP islands with genuinely different local
  gradients; synchronized replicas agree on rounded parameters.
- Actual `train.diloco_merge` rebuilds merged y from averaged x/z, rather than
  restoring pre-merge y; local moments and clocks remain local.
- Independent produce and resume jobs reach identical per-rank state digests
  with changed optimizer bucket sizes.
- Explicit rank/local-rank/current-device/visible-device evidence on every rank.
- Tiny-update SR movement against a separate FP32 reference and the legacy BF16
  optimizer; CPU/CUDA counter-store and bucket-boundary parity.
- Bounded scratch at fixed bucket size as synthetic parameter count increases;
  persistent candidate parameters and optimizer tensors remain BF16.
- Atomic no-replace receipts and checkpoint hashes; aggregation requires all
  sixteen produce/resume rank reports plus arithmetic/memory evidence.

A pass will not establish full-model E97 fused backward correctness, long-context
behavior, practical full-model throughput, production trainer/loader schema
integration, an optimal LR, or SFT convergence. Those remain distinct gates.

## Completed results

`proc_e7f3` exited zero in **177 seconds**, including the final source hash check.
CPU regressions: **50 passed, one GPU-only test skipped** before leasing CUDA.
Both eight-rank CUDA jobs then passed their explicit CUDA checks. All sixteen
rank reports reconciled; DDP, avg-DiLoCo, and hybrid cold-resume state digests
matched uninterrupted execution despite changed optimizer bucket sizes.

Summary SHA-256:
`c86721330b719981a93601a9f93b7721da58272da78dd181524a87ea7a443094`.

For 64 synthetic LR2e-6 updates over 65,536 coordinates:

| Arithmetic | Final parameter mean |
|---|---:|
| FP32 reference | 0.9999274015426636 |
| BF16 SR candidate | 0.9999271631240845 |
| Legacy BF16 | 1.0 |

SR changed 1,208 endpoint coordinates; legacy BF16 changed zero. Of 4,194,304
nonzero live-y proposals, 4,194,271 would have stalled under nearest-even BF16
storage. Unlike the earlier checkpoint endpoint comparison, these are instrumented
per-update synthetic proposal counts. They do not measure historical E97 stalls.

With a fixed 262,144-coordinate bucket, allocated scratch peak remained exactly
**17,563,648 bytes** for both 1,048,576 and 16,777,216 coordinates. Persistent
optimizer state increased from 4,194,304 to 67,108,864 bytes, as expected for two
BF16 state tensors. Single-step times were 0.00897 and 0.10886 seconds; these are
small synthetic observations, not full-model throughput benchmarks.

All scope exclusions above remain in force. The next numerical gate is an
independent fused-gradient reference and full-model masked-loss/backward check,
not an SFT restart or checkpoint promotion.

## Historical LR evidence

Both the mature Frontier 4B checkpoint's `args.json` and the local continuous 4B
run's `args.json` record **LR = 0.00047431158698290157**, Schedule-Free, clipping
1.0, and zero configured warmup steps. Their batch sizes differ (4 versus 32 per
rank in the recorded arguments); they are not identical post-training regimes.

The exact 4B rate is about **237 times 2e-6** and **9.49 times 5e-5**. This supports
examining substantially higher post-training rates, but it does not establish
that pretraining LR transfers to masked SFT or the new precision candidate.
The operator's recollection of approximately 1e-3 is the same broad scale; these
records establish the 4B configured rate, not the exact earlier CMA-ES provenance
or the effective schedule at every training step.

Evidence paths:

- `/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_frontier_100b_hf/checkpoints/step_024448_tokens_99723771904/args.json`
- `/mnt/nvme1n1/erikg/diloco_8gpu/e97_4b_continuous_primary/runs/emender_E97_4.0B_20260823_051037/args.json`
- `configs/frontier/e97_4b_hybrid_ddp_8n.json` repeats that configured rate and
  explicitly describes its 2,048-token context and fixed-world island topology.
