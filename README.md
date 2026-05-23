# Nonlinear Delta Memory

**Nonlinear Delta Memory** (NDM) is a pure recurrent language-model
architecture built from many matrix-valued memory heads. Each head is a small
nonlinear program that reads a key, computes a correction, writes the correction
back into its matrix state, and emits a query readout.

For one head, the core update is:

```text
r_t       = S_{t-1}^T k_t
delta_t   = v_t - r_t
S_t       = tanh(d_t S_{t-1} + k_t delta_t^T)
y_t       = S_t^T q_t
```

The important operation is the **nonlinear delta correction**: the write is not
a raw outer product, but a learned correction to what the memory already
retrieves. This gives the state a self-correcting, latch-like dynamics while
remaining a strictly recurrent model: no attention layers are required.

## Multi-Programming Recurrence

Pure recurrence is serial along time, but it does not have to be serial across
the machine. NDM trains by changing the parallelization axis: instead of making
one sequence operation massively parallel with attention, it runs thousands of
small recurrent memory programs in parallel across batch elements, heads,
and state tiles.

The current optimized implementation, **E88/NDM**, uses a fused Triton kernel
for the matrix-state recurrence. The kernel keeps each head's state in fast
local storage, fuses the delta update, L2 key/query normalization, output gate,
and sparse checkpointing, and exposes enough independent programs to keep GPUs
busy even though each individual program scans time sequentially.

This is the practical point of the repository: serial nonlinear RNNs can be
trained at modern scale when the system is organized as a multi-programmed
recurrent workload.

## What Is Here

- `ndm/`: NDM model code, E88/NDM Triton kernels, and comparison baselines.
- `train.py`: byte-level and tokenized language-model training entry point.
- `experiments/expressivity_tasks/`: controlled state-tracking tasks, including
  noncommutative S5-style permutation composition.
- `scripts/`: evaluation-panel builders, checkpoint probes, sample generation,
  and data-stream preparation tools.
- `formal/lean/`: trusted Lean proof core for the update-family resource
  separation and S5 tracker formalization.
- `docs/`: architecture, kernel, stability, distributed-training, and M2RNN
  comparison notes.
- `paper/`: working notes for the NDM paper.

Large training checkpoints and raw campaign outputs are not stored in git.
Frozen model checkpoints should be published separately with model cards,
dataset notes, and exact commit references.

## Quick Checks

Python import and syntax smoke:

```bash
python -m pytest tests -m "not gpu"
python -m py_compile train.py scripts/*.py ndm/triton/*.py ndm/models/e88_fused.py
```

Lean trusted-core check:

```bash
cd formal/lean
scripts/check_paper_core.sh
scripts/check_trusted_no_placeholders.sh ElmanProofs.lean
```

CUDA/Triton recurrence parity smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_e88_triton.py -m gpu
```

The CUDA smoke compares the E88/NDM Triton recurrence with the PyTorch
reference for forward outputs, fused L2/gate behavior, and backward gradients.

## Provenance

NDM was developed in the open in the historical research repositories:

- `ekg/elman` at `6f0724feae9fc82bd235408ac5c3ae61f2b17c79`
- `ekg/elman-proofs` at `5082610c9cdabf0b31e11dd14ee078273d486333`

Those repositories preserve the development trail. This repository is the
clean implementation, experiment, and proof artifact for Nonlinear Delta
Memory.
