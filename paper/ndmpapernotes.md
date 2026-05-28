# Historical Paper Design Notes

These notes use the older NDM naming for the nonlinear delta-memory mechanism.
The public paper and repository name is Emender.

Working title:

> **Nonlinear Delta Memory: Scaling Pure Recurrent Language Models by
> Multi-Programming**

## One-Sentence Thesis

Pure nonlinear recurrent language models can be trained at billion-parameter
scale when the recurrent computation is organized as a multi-programmed GPU
workload, and the useful memory resource is a nonlinear delta-correcting matrix
update rather than matrix state or temporal nonlinearity alone.

## Abstract Shape

Modern sequence models are usually made efficient by parallelizing the sequence
operation itself: attention, convolution, linear scan, or hybrid mixtures. We
study a different route. Nonlinear Delta Memory (NDM) keeps recurrence serial
along time, but exposes massive parallelism across many small memory programs:
heads, state tiles, and batch elements. The resulting architecture is a
pure recurrent language model with matrix-valued state and a nonlinear
delta-correcting update.

The paper has three linked claims. First, architecture search over recurrent
geometry finds many-head shapes that train smoothly at 1.27B parameters. Second,
an optimized Triton implementation makes the nonlinear recurrence competitive
with hand-tuned recurrent baselines in wallclock training. Third, controlled
state-tracking tasks identify the mechanism: NDM's read-then-delta write
outperforms both linear recurrent baselines and M2RNN-style raw-write matrix
updates on S5 permutation composition.

The formal contribution is deliberately bounded: a checked update-family
resource separation between NDM and M2RNN, a finite-state ceiling for
fixed-precision recurrence, and a formalized S5 tracker. The empirical result is
that a pure nonlinear recurrent stack can be trained in the same loss regime as
strong linear recurrent models while exposing a different state-tracking
profile.

## Audience

The first audience is an ML systems/modeling reader who knows Mamba, DeltaNet,
Gated DeltaNet, hybrid recurrent-attention models, and the current assumption
that pure serial RNNs cannot compete at scale.

The paper should not read like a repo history. It should read as an answer to a
field-level question:

> Did pure nonlinear recurrence fail because it was the wrong computation, or
> because it was parallelized along the wrong axis?

## Main Contributions

1. **Architecture:** NDM, a pure recurrent matrix-memory architecture with a
   nonlinear delta-correcting update.
2. **Training geometry:** CMA-ES and follow-up search identify many-head,
   multi-programmed recurrent shapes that train at 1.27B parameters.
3. **Systems:** a fused Triton recurrence kernel with L2 key/query
   normalization, sparse checkpointing, output gating, and portable CUDA/ROCm
   semantics.
4. **Language modeling:** 1.27B pure recurrent NDM enters the same wallclock
   loss regime as strong linear recurrent baselines under matched training.
5. **Expressivity:** S3/S5 state-tracking tasks show that NDM separates from
   FLA-GDN, Mamba2-style linear recurrence, and M2RNN-style raw-write matrix
   updates.
6. **Formal core:** a Lean-checked resource separation between NDM and M2RNN
   update families, plus a checked S5 tracker and finite transition-memory
   realization.

## Model Families In The Paper

| Name | Role | Key property |
| --- | --- | --- |
| NDM/E88 | main model | nonlinear delta-correcting matrix memory |
| FLA-GDN | linear recurrent baseline | delta-style memory with linear temporal state |
| Mamba2 | selective SSM baseline | strong linear-time recurrent baseline |
| M2RNN-paper | related nonlinear matrix RNN | raw-write matrix-state update in published geometry |
| M2RNN-CMA | searched M2RNN control | M2RNN-style update reshaped toward many independent programs |
| NDM hybrids | control | tests whether mixing with GDN/attention helps or hurts |

The M2RNN comparison should be written carefully. The paper should concede that
M2RNN validates the broad nonlinear matrix-state direction. The NDM claim is
narrower and more useful: the delta correction and many-program geometry are
the resources that make pure recurrence train and track state.

## The Core Update

One NDM head has matrix state `S`. Per token:

```text
r_t       = S_{t-1}^T k_t
delta_t   = v_t - r_t
S_t       = tanh(d_t S_{t-1} + k_t delta_t^T)
y_t       = S_t^T q_t
```

Interpretation:

- `k_t` addresses memory.
- `r_t` is what the memory currently returns at that address.
- `delta_t` is the correction required to make the memory return `v_t`.
- `k_t delta_t^T` writes the correction.
- `tanh` bounds the state after the write.
- `q_t` reads from the updated memory.

Contrast:

```text
M2RNN: H_t = f_t H_{t-1} + (1 - f_t) tanh(H_{t-1} W + k_t v_t^T)
GDN:   delta-style update, but linear temporal state
```

This contrast makes the mechanism testable. If "nonlinear matrix state" were
enough, M2RNN should also win the state-tracking tasks. If "delta correction"
is the useful resource, NDM should separate from raw-write M2RNN.

## CMA-ES And Geometry Search

CMA-ES should be presented as part of the scientific method, not as incidental
hyperparameter hacking. The search object is recurrent geometry:

- number of heads;
- state dimension per head;
- depth and model width;
- expansion ratio;
- key/query normalization;
- decay law;
- output/write gates;
- M2RNN q/k head sharing versus many independent address programs.

The important empirical pattern is that the same update family can look weak or
strong depending on geometry. Paper-shaped M2RNN is brittle under the local
schedule-free language-modeling setup. The tied/CMA-ES-shaped M2RNN trains much
more smoothly. E88/NDM is the current optimized point in the nonlinear
delta-memory family.

This supports the multi-programming thesis: recurrence becomes trainable when
the model exposes many independent address/memory programs to the hardware and
to the optimizer.

## Evidence Stack

### Figure 1: Mechanism And S5

This is the first figure to build.

Panel A: update schematic comparing NDM, M2RNN, and GDN.

Panel B: train-length S3/S5 bars.

Panel C: S5 length-extrapolation curves.

Current matched 8M S3/S5 run, three seeds, train length 128:

| Task | Model | Mean acc | Min | Max | Baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| S3 | E88/NDM | 1.0000 | 0.9999 | 1.0000 | 0.1667 |
| S3 | FLA-GDN | 0.7185 | 0.6122 | 0.8516 | 0.1667 |
| S3 | M2RNN | 0.3124 | 0.2742 | 0.3529 | 0.1667 |
| S3 | M2RNN-paper | 0.3773 | 0.3669 | 0.3855 | 0.1667 |
| S5 | E88/NDM | 0.7918 | 0.6232 | 0.8880 | 0.0083 |
| S5 | FLA-GDN | 0.3552 | 0.3148 | 0.3850 | 0.0083 |
| S5 | M2RNN | 0.2157 | 0.1856 | 0.2309 | 0.0083 |
| S5 | M2RNN-paper | 0.1698 | 0.1555 | 0.1844 | 0.0083 |

Length extrapolation on S5:

| Model | T=128 | T=256 | T=512 | T=1024 |
| --- | ---: | ---: | ---: | ---: |
| E88/NDM | 0.7900 | 0.4158 | 0.2150 | 0.1104 |
| FLA-GDN | 0.3544 | 0.1843 | 0.0974 | 0.0521 |
| M2RNN | 0.2142 | 0.1120 | 0.0593 | 0.0339 |
| M2RNN-paper | 0.1696 | 0.0884 | 0.0488 | 0.0283 |

Interpretation:

- NDM separates at the training length, not only under extrapolation.
- NDM remains ahead under extrapolation, but all models degrade.
- M2RNN underperformance supports the update-rule claim: nonlinear matrix
  state alone is not enough.

### Figure 2: CMA-ES Geometry

Show how the searched geometry differs from naive or paper-shaped settings:

- many independent heads;
- stable q/k address programs;
- state size per head;
- depth/width tradeoff;
- relation to GPU occupancy and recurrent program count.

The point is not that CMA-ES is magic. The point is that recurrent models have
non-obvious geometry, and search found shapes that make serial recurrence
trainable.

### Figure 3: 1.27B Language-Model Racers

Plot loss or bits-per-byte versus wallclock for:

- E88/NDM;
- FLA-GDN;
- Mamba2;
- M2RNN-CMA;
- M2RNN-paper where appropriate as a negative/stability control.

Use smoothed windows such as last 5K, 10K, and 50K steps. Avoid last-100-step
noise. The central plot should show time-to-loss, not just step count or token
count.

Caption claim:

> Once recurrent geometry and kernels are optimized, pure nonlinear recurrence
> trains in the same wallclock loss band as strong linear recurrent baselines.

### Figure 4: Systems

Show kernel and end-to-end training improvements:

- fused L2 norm;
- `num_warps=1` at high head count;
- sparse forward checkpointing;
- avoiding unnecessary contiguous copies;
- fused output gate;
- bf16 scratch and block-size tuning.

Frame this as making multi-programmed recurrence practical, not as a standalone
kernel brag.

## Formal Section

The trusted Lean core is a guardrail against overclaiming.

Checked:

- NDM and M2RNN are distinct one-step update families.
- M2RNN cannot implement the NDM mixed-key delta correction in one step without
  an extra read-then-delta path.
- With the extra read-then-delta resource, M2RNN can embed one NDM step.
- Fixed-width, finite-precision recurrent recognizers are finite-state.
- The S5 tracker and exact finite transition-memory realization are checked.

Not checked:

- a lower bound proving all linear scan models fail S5;
- Barrington/NC1 completeness inside Lean;
- robust learning of the exact transition table by trained real-valued NDM.

Paper wording:

> The formal result is a resource separation between update families, not an
> absolute computability separation. The S5 lower-bound story is theoretical
> background and empirical evidence, not a completed Lean lower bound.

## Paper Outline

### 1. Introduction

Motivate the field assumption: pure serial RNNs are expressive but impractical.
State the counter-hypothesis: the missing ingredient is multi-programming, not
attention or linear scan.

End with contributions:

- NDM architecture;
- CMA-ES discovered recurrent geometry;
- Triton implementation;
- billion-scale language modeling racers;
- S5/state-tracking separation;
- Lean-checked update-family core.

### 2. Background

Cover Mamba2, Gated DeltaNet, linear scans, hybrid models, M2RNN, and
state-tracking theory. Matrix state is background, not novelty.

### 3. Nonlinear Delta Memory

Define the update. Explain the delta correction. Explain many-head state and
multi-programmed execution. Distinguish NDM from raw-write matrix RNNs and from
linear temporal updates.

### 4. Searching Recurrent Geometry

Describe CMA-ES and follow-up ablations. Present the searched shapes for
NDM/E88 and M2RNN-CMA. Explain why paper-shaped M2RNN is a useful but brittle
control.

### 5. Efficient Triton Recurrence

Describe the E88/NDM kernel, sparse checkpointing, fused normalization, fused
gate, and ROCm portability. Show throughput and memory results.

### 6. Language Modeling

Describe dataset stream, byte/token setup, schedule-free optimizer, context
curriculum, and 1.27B racers. Main plot: wallclock loss and bpb.

### 7. Expressivity And State Tracking

Describe task suite, S3/S5, and length extrapolation. Main point: NDM separates
from raw-write M2RNN and linear recurrent baselines on the task class that
motivates nonlinear recurrence.

### 8. Formal Results

State the Lean-checked claims and the explicit non-claims. Put the formal core
in service of the empirical mechanism, not as a detached proof exercise.

### 9. Discussion

Pure recurrence is viable when shaped as a many-program workload. Hybridization
is not automatically better. The open question is how far nonlinear recurrent
reasoning can scale once memory, geometry, and systems are co-designed.

### 10. Limitations

- No completed lower bound for all linear scan models on S5.
- S5 length extrapolation is not solved.
- Results depend on careful geometry and kernel optimization.
- Large checkpoints need separate release and documentation.
- M2RNN comparisons rely on the available implementation details and faithful
  local reproduction choices.

## Claim Register

Strong claims to make:

- NDM is a pure nonlinear recurrent architecture trained at 1.27B-parameter
  scale.
- Multi-programming is a viable parallelization strategy for serial recurrent
  models.
- Delta-correcting matrix memory separates empirically from raw-write nonlinear
  matrix memory on S5 state tracking.
- The trusted Lean core proves an update-family resource separation and checks
  the S5 tracker surface.

Claims to avoid:

- "First nonlinear matrix-state RNN."
- "Fixed-width NDM exceeds NC1."
- "Lean proves linear scan lower bounds."
- "Nonlinearity alone explains the results."
- "M2RNN fails" as a broad claim. The accurate statement is that paper-shaped
  M2RNN is brittle here, while CMA-ES-shaped M2RNN trains better but still does
  not match NDM on the S5 evidence.

## Immediate Work

1. Build Figure 1 from the S3/S5 run.
2. Build a model-geometry table for NDM/E88, FLA-GDN, Mamba2, M2RNN-paper, and
   M2RNN-CMA.
3. Generate wallclock racer plots using 5K, 10K, and 50K smoothing windows.
4. Add a compact results report under `paper/` once the first figures exist.
5. Prepare a checkpoint release plan for Hugging Face only after freezing model
   hashes, dataset notes, and exact training commits.
