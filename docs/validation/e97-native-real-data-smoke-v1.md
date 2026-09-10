# Eight-GPU native real-data SFT smoke v1

2026-09-10: **passed**, managed process `proc_f27d`, 1,016 seconds.
This is actual supervised training, not an authored-fixture numerical assay.
It is not a sustained run, selected learning rate, or model promotion.

## Frozen inputs and recipe

- Execution commit: `b8ee034fcfcb3708f6a8a0c3f43837e80aafeb65`.
- Evidence root:
  `/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-real-data-smoke-v1`.
- Recipe SHA: `d85ed1e00477aae89a838dabbe91300c6cac490de2003e5a210e63b0053b1a39`.
- [Mixture and loader report](e97-native-training-mixture-v1.md).
- Parent train/y SHA:
  `aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39`.
- Eight-rank full-world DDP; no outer merge; context 65,536; complete-record
  reset/valid/loss masks. Sampler key 974117, epoch permutation.
- BF16 SR Schedule-Free, seed 927413, CPU-offloaded BF16 state, device arithmetic,
  262,144-element optimizer buckets. No persistent FP32 master parameters.
- LR `5e-5`, warmup 0, weight decay .01, gradient clip 1; this is a smoke rate,
  **not an experimentally selected SFT learning rate**.
- Activation checkpoint group 3, MLP chunk 4,096, checkpointed FP32 CE chunks
  128, reduced-precision BF16 GEMM reductions disabled.
- Eight updates; K/save interval 4; one child with `--max-restarts=0` under the
  canonical eight-GPU lease, explicit local-rank device/NUMA binding, isolated
  Triton caches, both expandable-segment allocator variables, and a 2,400-second
  TERM deadline plus 30-second KILL grace.

## Observations

All eight updates completed with finite reported losses and gradient norms.
Runtime sample IDs and global input/target counts matched every row of the
CPU-materialized schedule exactly.

| Exposure | Tokens |
|---|---:|
| Inputs | 3,328,172 |
| Assistant targets | 1,490,478 |
| Native agent targets | 415,097 |
| Conversation targets | 922,093 |
| Retention targets | 153,288 |

Rank-0 peak allocated HBM was 36,128,739,328 bytes (33.65 GiB). This is rank-0
telemetry, not a measured maximum across all ranks; all eight ranks completed.

Per-update losses:
`1.455287, 1.656723, 1.449338, 1.475024, 1.382576, 1.364423, 1.291691, 1.407923`.
The eight-update mean is approximately **1.4354**. Batches differ, so these
numbers do not establish matched loss improvement, retention, transfer, or
observation-dependent tool capability.

Both checkpoints were atomically published, SHA-verified and loaded on CPU.
The collector checked complete optimizer slots and exact update clocks,
4,045,972,080 optimizer coordinates, full live-y backup coverage, BF16 floating
state, and finiteness in bounded chunks. The terminal `latest.pt` resolves to
update 8. Checkpoints remain read-only and unpromoted.

| Update | File under `checkpoints/` | SHA-256 |
|---|---|---|
| 4 | `checkpoint_agent_sft_u000004_loss_1.5091.pt` | `d4ef86a065e49bbf003304193e1b4c9beb182483c38e9e81a2abea9c71f84034` |
| 8 | `checkpoint_agent_sft_u000008_loss_1.4354.pt` | `b19c9b9b99b3041c6bfde8f613427e08d749d480ae2d5f5a3ea55e8356c2beaa` |

Retained evidence includes `recipe.json`, `run.sh` (exact launch and collector),
`inventory.sha256`, `input.sha256`, `source.sha256`, clean `worktree/`,
`hardware.csv`, `console.log`, `training.jsonl`, and `summary.json`. Bound source
and input identities were checked before and after execution.

## What this closes—and does not

Real-data admission/loading, all-target 64K packed execution on eight GPUs,
finite updates, actual source-exposure accounting, and complete checkpoint
publication/readability now pass together. No more standalone basic-loader or
bitwise-execution tests are prerequisites for beginning real SFT.

Numerical fresh continuation has **not yet been measured** for this run. Its
comparison tolerances were frozen before training in `recipe.json`; exact
checkpoint integrity/restoration remains required. The earlier authored-fixture
bitwise failure remains a failure under its original criterion. Next learning
work needs matched fitting/development measurements, retention and execution
evaluation, and an exposure-matched learning-rate comparison. No improved-model
claim or automatic expansion follows from this smoke result.

Architecture scope: ADR-003 safety intent **R07/R12** committed atomic state,
**R14/NDP13** bounded failure, **R16** frozen-source evidence, and **NDP15**
checkpoint atomicity. No elastic/native-data-plane/async/Frontier/ROCm or
communicator-shrink qualification is claimed.
