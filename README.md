# Nonlinear Delta Memory

**Nonlinear Delta Memory** (NDM) is a pure recurrent language-model family based
on many small nonlinear matrix memories. It is designed to test a simple
systems and modeling hypothesis:

> A serial recurrent model can train at billion-parameter scale if the
> recurrence is organized as many independent memory programs running in
> parallel across the GPU.

Modern sequence models usually get hardware efficiency by making the sequence
operation itself parallel: attention, convolution, linear scans, or hybrids.
NDM takes the opposite route. Each memory head scans time sequentially, but the
model contains hundreds of heads per layer and many layers per network. The
training workload is therefore a large collection of small recurrent programs,
parallelized across batch elements, heads, and state tiles.

The current optimized implementation is called **E88/NDM** in the codebase. It
uses a fused Triton kernel for the recurrent state update and has been used in
1.27B-parameter language-model training runs.

## The Memory Update

For one NDM head, the runtime state is a matrix `S`. Given key `k`, value `v`,
query `q`, and decay `d`, the core recurrence is:

```text
r_t       = S_{t-1}^T k_t
delta_t   = v_t - r_t
S_t       = tanh(d_t S_{t-1} + k_t delta_t^T)
y_t       = S_t^T q_t
```

The distinctive operation is the **delta correction**. The write is not a raw
outer product. The head first reads what the memory already returns at `k_t`,
then writes the error `v_t - r_t` back into the matrix state. The nonlinearity
bounds the recurrent state itself.

This makes NDM different from nearby matrix-state RNNs that use raw writes, and
different from linear recurrent models whose temporal state update remains
linear even when the gates are input-dependent.

## Experimental Program

This repository supports three connected evaluations.

### 1. Geometry Search

NDM was developed through repeated architecture search over recurrent geometry:
head count, state size, depth, expansion, normalization, gates, and decay. The
same search pressure is used for comparison models where possible.

The important comparison is not only "which update equation is expressive." It
is also which shape is trainable. The current evidence separates:

| Model | Role |
| --- | --- |
| `E88/NDM` | optimized nonlinear delta-memory implementation |
| `FLA-GDN` | strong linear-time gated delta baseline |
| `Mamba2` | strong selective state-space baseline |
| `M2RNN-paper` | published nonlinear matrix-state geometry |
| `M2RNN-CMA` | M2RNN-style model reshaped into the many-head regime found by search |

The M2RNN contrast is especially useful. Matrix state and temporal
nonlinearity are not sufficient by themselves. The update rule, addressing
geometry, and multi-programmed shape all matter.

### 2. Billion-Scale Language Modeling

The production comparison trains 1.27B-parameter recurrent language models at
2K context and then extends the context curriculum. The main quantity of
interest is not raw step count. It is loss versus wallclock compute under
matched, tuned training conditions.

The working result is that optimized pure recurrent NDM can enter the same
language-modeling loss regime as leading linear recurrent baselines. That is
the central scaling claim: pure nonlinear recurrence is not ruled out by
hardware if it is implemented as a multi-programmed workload.

Large checkpoints and raw campaign logs are not stored in git. Frozen
checkpoints should be released separately with model cards, dataset notes, and
exact commit references.

### 3. State-Tracking Expressivity

The expressivity suite tests controlled finite-state and algorithmic tasks. The
main witness is S5 permutation composition, a noncommutative state-tracking task
that stresses transition composition rather than text memorization.

In matched 8M-parameter S3/S5 experiments, E88/NDM separates from FLA-GDN,
M2RNN, and paper-shaped M2RNN at the training length and remains ahead under
length extrapolation. The result supports a narrower mechanism claim: the
delta-correcting update is the useful resource, not matrix state alone.

## Formal Core

The Lean formalization lives in `formal/lean`. The trusted paper root checks:

- NDM and M2RNN are distinct one-step update families.
- M2RNN can embed an NDM step only when given an extra read-then-delta resource.
- Fixed-precision recurrent recognizers have a finite-state ceiling.
- The S5 tracker and exact finite transition-memory realization are formalized.

The Lean core does **not** claim a lower bound for all linear scan models, and
does **not** claim that fixed-width NDM exceeds NC1. The formal boundary is
checked in CI and rejects `sorry`, `admit`, explicit `axiom`, `opaque`, and
`native_decide` in the trusted import closure.

## Repository Layout

- `ndm/`: model code, E88/NDM Triton kernels, and comparison baselines.
- `train.py`: byte-level and tokenized language-model training entry point.
- `experiments/expressivity_tasks/`: parity, counters, finite-state tracking,
  and S5 permutation-composition experiments.
- `scripts/`: checkpoint evaluation, sampling, dataset-stream preparation, and
  knowledge/reasoning probe builders.
- `formal/lean/`: trusted Lean proof surface.
- `docs/`: architecture, stability, systems, distributed training, and M2RNN
  comparison notes.
- `paper/`: working paper design and result notes.
- `provenance/`: commit anchors for the original development repositories.

## Checks

CPU Python smoke:

```bash
python -m pytest tests -m "not gpu"
python -m py_compile train.py scripts/*.py ndm/triton/*.py ndm/models/e88_fused.py
```

CUDA/Triton recurrence parity smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_e88_triton.py -m gpu
```

Lean trusted-core check:

```bash
cd formal/lean
scripts/check_paper_core.sh
scripts/check_trusted_no_placeholders.sh ElmanProofs.lean
```

The public CI runs the CPU Python and Lean trusted-core checks. The Triton test
is kept as a GPU-local parity smoke because standard GitHub-hosted runners do
not provide CUDA GPUs.

## Provenance

NDM was developed in public in two historical repositories:

- `ekg/elman` at `6f0724feae9fc82bd235408ac5c3ae61f2b17c79`
- `ekg/elman-proofs` at `5082610c9cdabf0b31e11dd14ee078273d486333`

Those repositories preserve the development trail. This repository is the
clean implementation, experiment, and proof artifact for Nonlinear Delta
Memory.
