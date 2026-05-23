# Frontier Distributed Training — Research Notes

Synthesized from two deep-research passes on scaling E88 (nonlinear sequential
matrix-state RNN) to OLCF Frontier: 64-node × 24-hour runs on AMD MI250X,
~20K node-hour budget.

## Headline: ParaRNN (Apple, Oct 2025)

**arXiv:2510.21450** — parallelizes arbitrary nonlinear RNNs across sequence
via Newton's method on a block-bidiagonal Jacobian. Reports 665× speedups.
Validated on 7B GRU/LSTM variants with transformer-comparable perplexity.

Code: https://github.com/apple/ml-pararnn

- Accepts user-defined `recurrence_step(h_{t-1}, x_t)` — works in principle
  for E88.
- Published examples are vector state. E88 has matrix state (per head, n×n).
  Newton on matrix state means solving a block system with block size n²×n² —
  for n=32 that's 1024×1024 per block per layer. Solvable but expensive.
- Earlier DEER (arXiv:2407.19115) had cubic state-size complexity (showstopper
  for matrix state). ParaRNN claims improvement; untested for our case.
- Convergence of Newton on `tanh(decay·S + outer(v−Sk, k))` is empirically
  untested — needs a prototype.

**If ParaRNN works on E88**, sequence-parallelism becomes available and the
"nonlinear RNNs can't use SP" axiom evaporates. Change the Frontier plan.

## Critical batch size

- **McCandlish 2018** (arXiv:1812.06162) gradient-noise-scale framework.
- **Marsden 2025 NeurIPS** (arXiv:2505.23971): GNS *underestimates* true
  B_crit by orders of magnitude. Must measure empirically.
- **arXiv:2507.07101**: batch sizes down to 1 are *more* robust per-FLOP for
  recurrent models — supports the user's "diminishing returns" observation.
- **Recommendation**: target 1M–2M tokens global batch. Fork-and-compare
  measurement at 480M costs <1% of budget.

## Parallelism strategy (assuming no ParaRNN)

| Dim | Value | Rationale |
|-----|-------|-----------|
| TP | 8 intra-node | 200 GB/s Infinity Fabric, no cross-node |
| PP | 4, interleaved 1F1B v=2 | E88 depth 10–30; deeper PP wasteful |
| DP | 16 cross-node | ZeRO-1 sharding |
| Microbatch | 1–4 | Memory at 128K ctx |
| Global batch | ~1M–2M tokens | Measured B_crit |

**Memory math (7B)**: 14 GB bf16 params + 28 GB optimizer states (ZeRO-1
shards these). Activations at 128K × depth=30 bf16 ≈ 33 GB/replica with
gradient checkpointing. Projection-chunk recompute essential.

**Pipeline bubble**: `(p−1)/(m·v + p−1)`. At p=4 v=2 m=16: ~4.5%. At p=8
v=1: ~30%. E88's sequential per-layer recurrence means each stage's forward
time ≈ L_stage × T, so bubble-in-seconds grows with context — pipeline
depth >4 is wasteful at long context.

## Frontier-specific

- **Framework**: ROCm/Megatron-LM (AMD official,
  `github.com/ROCm/Megatron-LM`, primus:v25.10 container). NOT Microsoft's
  Megatron-DeepSpeed.
- **Networking**: `aws-ofi-rccl` plugin MUST be built + LD_PRELOAD'd.
  Default RCCL uses TCP/IP on Slingshot, catastrophic. Script:
  `eminorhan/frontier-accelerate/blob/master/aws_ofi_rccl.sh`.
- **DeepSpeed JIT ops fail on ROCm** — pre-build at image time.
- **Storage**: Orion Lustre (679 PB, 10 TB/s). Per-node NVMe (2×1.92 TB at
  `/mnt/bb/<uid>`) via `-C nvme` flag for hot data.
- **Checkpointing**: PyTorch DCP with `SHARDED_STATE_DICT`. Tiered — NVMe
  every 15 min, Orion every 2 hr.
- **MI250X** = 4 cards/node × 2 GCDs = 8 effective GPUs/node, 64 GB HBM each.
- **Known gotcha**: `torch.compile` + bf16 + ROCm NaN issues on Megatron
  layers (2025).

## Port strategy: Triton, not HIP

**Rewrite E88 kernels in Triton**. Triton on ROCm works (aotriton mature for
FA-style kernels, 15–25% of AITER). Tradeoffs:

- Same source runs on CUDA + ROCm
- ~1-week port vs 3–6 weeks HIPifying
- Expected ~1.5× perf penalty vs hand-tuned HIP
- ~30% throughput loss (≈6K node-hours) worth it for engineering velocity

Starting point: the step-kernel Triton prototype already exists
(`elman/models/e88_step_kernel.py`). Extend to full sequence forward +
backward.

**Alternative**: hipify-clang for the hot kernels only. `__nv_bfloat16` →
`__hip_bfloat16` auto; warp-shuffle needs `warpSize==64` fix for AMD
(wavefront is 64 on CDNA2, not 32 like NVIDIA warps). cuBLAS → rocBLAS.

## Scaling laws — the paper hook

No published Chinchilla-style scaling law for nonlinear RNNs since the
~2019 LSTM era (Kaplan 2020 included LSTMs, pre-Chinchilla). Mamba/RWKV/
GLA papers are architecture papers, not scaling law studies.

**Hypothesis**: E88's nonlinearity may require different token/param ratios
than transformers (20 tokens/param Chinchilla). Nonlinearity → slower
saturation → potentially higher token/param optimum.

**Published scaling-law comparisons** report token-matched curves. The
standard for non-transformer work is to report (a) matched-tokens AND (b)
matched-FLOPs side-by-side, with (c) matched-wall-clock as a system-cost
footnote. Gu's 2025 blog post argues this explicitly.

## 13-run budget allocation

| # | Size | Purpose |
|---|------|---------|
| 2 | 4×8 node-hr each | Port validation @ 480M, RCCL wiring, FSDP resume |
| 1 | 32×12 node-hr | B_crit measurement, 5-way fork |
| 1 | 32×12 node-hr | PP/TP shape sweep @ 1B |
| 1 | 64×24 | **ParaRNN prototype @ 1B** (high-risk, high-reward) |
| 4 | 64×24 each | Scaling ladder: E88 at 480M, 1B, 3B, 7B Chinchilla-optimal |
| 1 | 64×24 | Over-Chinchilla @ 3B (60 tok/param) |
| 1 | 64×24 | Long-context @ 7B, 32K → 128K progressive |
| 2 | 64×24 each | Slack / reruns / failure recovery |

Total ≈ 12.5 runs, ~0.5 run of slack.

## Pre-Frontier work checklist

1. **Prototype ParaRNN on E1H locally** — 1-2 day spike, see if Newton's
   method converges on our recurrence before burning Frontier budget.
2. **Port E88 CUDA → Triton** (forward first, then backward). Builds on
   existing step-kernel prototype.
3. **Add FSDP + pipeline parallel to `train.py`** — validate on local 2–4 GPUs.
4. **Integrate E88 into ROCm/Megatron-LM** model registry (or keep custom PyTorch
   and just use FSDP+PP).
5. **Pre-tokenize commapile** to mmap'd uint16 arrays on Orion.

## Paper pitch if it all works

> *Scaling Laws for Nonlinear Sequential Recurrent Networks*
>
> 1. Fitted scaling exponent α for E88 from 480M → 7B, contrasting with
>    transformer and linear-RNN scaling.
> 2. First published critical batch size measurement for a nonlinear RNN.
> 3. [If ParaRNN works] First parallelized training of a matrix-state
>    nonlinear RNN at 7B params.
> 4. 128K context capability at 7B, contrasting with MinGRU/MinLSTM's
>    failure to use long context.
> 5. Generation quality measurement — diversity/repetition vs perplexity
>    dissociation observed at 480M extending to scale.

## Key uncertainties (need empirical verification)

- **ParaRNN on matrix state** — Newton step is O(n⁶) per block; convergence
  untested for tanh recurrences. Prototype required.
- **FLA-GDN context parallelism on Megatron** — chunkwise SSD supports it in
  theory; no merged Megatron implementation as of 2025. Custom ring-reduce
  needed if we include FLA-GDN in comparisons.
- **ROCm FA-2 at 128K** — published numbers mostly ≤16K ctx. Verify before
  committing to long-context runs.
- **Triton-on-ROCm perf gap for E88's specific pattern** — extrapolated from
  FA-style kernels; benchmark before deciding port strategy.

## Sources

- [ParaRNN: Parallel Training of Nonlinear RNNs (arXiv 2510.21450)](https://arxiv.org/abs/2510.21450)
- [apple/ml-pararnn](https://github.com/apple/ml-pararnn)
- [McCandlish, Empirical Model of Large-Batch Training (1812.06162)](https://arxiv.org/abs/1812.06162)
- [Marsden, Critical Batch Size Revisited (2505.23971)](https://arxiv.org/abs/2505.23971)
- [Small Batch Size Training (2507.07101)](https://arxiv.org/html/2507.07101)
- [Yin et al., Optimizing Distributed Training on Frontier for LLMs (2312.12705)](https://arxiv.org/html/2312.12705v2)
- [LUMI 100B training on MI250X](https://lumi-supercomputer.eu/scaling-the-pre-training-of-large-language-models-of-100b-parameters-to-thousands-of-amd-mi250x-gpus-on-lumi/)
- [Databricks MPT-7B on AMD MI250](https://www.databricks.com/blog/training-llms-scale-amd-mi250-gpus)
- [Frontier User Guide](https://docs.olcf.ornl.gov/systems/frontier_user_guide.html)
- [ROCm Megatron-LM](https://github.com/ROCm/Megatron-LM)
- [Gated Delta Networks (ICLR 2025)](https://arxiv.org/pdf/2412.06464)
- [DEER: Parallel Nonlinear RNNs via Newton (2407.19115)](https://arxiv.org/abs/2407.19115)
- [Zero Bubble Pipeline Parallelism (2401.10241)](https://arxiv.org/html/2401.10241v1)
- [Seq1F1B (2406.03488)](https://arxiv.org/html/2406.03488v1)
