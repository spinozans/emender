# E97 4B native-trajectory numerical qualification v2

2026-09-10: **passed**, four full-model forward/backward cases in 310 seconds.
This was a no-update numerical test. It trained no model, executed no source
commands and admitted no dataset.

## Result

Two complete training-side native trajectories were selected deterministically
from predeclared length bins. Their lengths were **14,841** and **65,423** tokens.
Three original supervised Analysis openings were tested in each: early, middle
and late. The long trajectory's last tested opening was at token **64,608**.

The reference used an unchunked MLP; the candidate used the trainer's
**4,096-token MLP chunks**. Both used BF16 parameters, grouped activation
checkpointing (group 3), checkpointed 128-token CE chunks, temporary FP32 CE
logits and disabled reduced-precision BF16 GEMM reductions.

| Result | Short trajectory | Long trajectory |
|---|---:|---:|
| Reference/candidate mean opening loss | 10.63631820678711 | 11.645539283752441 |
| Absolute mean-loss difference | **0** | **0** |
| Reference peak allocated HBM | 19.676 GiB | 40.493 GiB |
| Candidate peak allocated HBM | **16.613 GiB** | **26.082 GiB** |

Across every parameter and both comparisons, worst relative gradient L2 error
was **0.00481890338 (0.482%)**, below the frozen **5%** limit. The frozen
mean-loss tolerance was 0.05; CE-to-direct-FP32-NLL tolerance was 1e-4. Causal
prefix embedding gradients were nonzero and future embedding gradients exactly
zero. The parameters were unchanged before/after:
`9250d077297f1755c1393d173acb6c264f9c05093c301d785c99676590289ad2`.

This supports the **specific tested configuration**, not every chunk size or
context. It does not show that disabling reduced-precision GEMM reductions alone
fixes the historical 256-token MLP failure; that failure remains retained.
Sparse opening supervision, two single trajectories and one GPU do not qualify
all-target packed training, DDP memory or optimizer/restart behavior.

## Evidence

Root:
`/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-sft-numerics-v2`

- Process: `proc_6a20`, success, 310 seconds; one GPU leased through the canonical
  helper, explicit local-rank selection, NUMA binding, isolated Triton cache and
  both expandable-segment allocator settings.
- CPU preflight: **29 passed** from the frozen clean export.
- Recipe: `5bf3342ea55fcaac935f74852e14196f6a89b5262ace30e910f92065d6d7cf2a`
- Source inventory: `ffa23de47df8bff5feb2c25f19fbad8c535af00d4622108525d4963da1bdd56f`
- Summary: `0246cf1d2d2cbf4701cb7655fbac7feb7c4205c5bef8b11bca39bf6299945f9e`
- Parent checkpoint: `aae654aa1db004ba9b9805fce54e005332361bbf536168a6618a1c22cce7af39`
- Native authority: `255b02f3b3bbdc85eb15a89d22e7ff2c4bf69f151c2dfb5941a22468711993f5`
- PyTorch 2.9.1+cu128; CUDA 12.8. Individual case measurements, hardware inventory
  and raw logs are retained. Bound model/data inputs and source files were
  rechecked after execution.

The recipe's LR=0 is a **no-update assay placeholder**, not an SFT learning-rate
choice. No training optimizer was constructed. Parent train-y restoration may
instantiate a temporary Schedule-Free loader object; it performs no optimizer
step.

## Retained preflight failure

`native-sft-numerics-v1` stopped before GPU work because its proposed
8,193–12,288-token bin contained no complete eligible trajectories. A metadata
census found a 14,841-token minimum. The replacement predeclared the populated
12,289–16,384 bin and retained the 60,000–65,536 long bin. The original invocation
also had an incomplete model-args digest; the freezer now validates full digests
and checks model inputs before recipe publication. No numerical measurement or
tolerance was changed in response to a failed model result.

## Remaining gates

Actual effective optimizer updates, all-target packed/DDP execution, atomic
checkpoint publication and fresh-process continuation are next. Source admission,
mixing/sampling and the higher-LR-inclusive learning experiment remain separate.
The native authority is still `training_eligible:false`.

Architecture scope: [ADR-003](../RESILIENT_DILOCO_COMPUTE_POOL.md), **R16** exact-source
evidence discipline. This no-update single-GPU assay does not qualify **R07/R12**
restart, **R14/NDP13** distributed failure containment, or **NDP15** checkpoint
publication. Elastic/native NDP02/NDP17, V21S and ISP requirements are unclaimed.
