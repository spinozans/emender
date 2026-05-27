---
license: apache-2.0
language:
- en
tags:
- rnn
- language-model
- nonlinear-rnn
- state-tracking
- ndm
- triton
library_name: transformers
---

# NDM-1.27B

**Nonlinear Delta Memory** (NDM) — a pure nonlinear recurrent language model
trained to near-convergence at 1.27B parameters.

This is the first publicly released checkpoint of a pure nonlinear RNN trained
at billion-parameter scale. The model uses no attention and no linear-scan
approximation; the recurrent state update is nonlinear at every step.

---

## Architecture

### NDM Update Equation

For one NDM head, the runtime state is a matrix `S ∈ ℝ^{n×v}`. Given input
projections key `k`, value `v`, query `q`, and decay scalar `d` at time `t`:

```
r_t     = S_{t-1}^T k_t          # read from memory
delta_t = v_t - r_t              # delta correction
S_t     = tanh(d_t S_{t-1} + k_t delta_t^T)   # nonlinear write
y_t     = S_t^T q_t              # read output
```

The tanh bounds the recurrent state, making the update nonlinear. This
distinguishes NDM from linear recurrent models (Mamba, GLA, DeltaNet) where
the state update is linear even when input gates are input-dependent.

### Hyperparameters (1.27B checkpoint)

| Parameter | Value |
|-----------|-------|
| Model dimension (`dim`) | 2176 |
| Depth (layers) | 14 |
| Memory heads per layer (`n_heads`) | 98 |
| State size per head (`n_state`) | 32 |
| Value expansion | 1.0 |
| Output gating | SiLU gate (enabled) |
| Q/K normalization | L2 + SiLU pre-activation |
| Decay parameterization | Mamba2-style (`A_log`, `dt_bias`) |
| Embeddings | Tied (input = output) |
| Normalization | Pre-layer RMSNorm (ε=1e-6) |
| Total parameters | ~1.27B |

### Diagram

<!-- TODO: insert architecture diagram from paper/figures/ once available -->
*Architecture diagram: see the companion paper (link TBD).*

### Implementation

Production class: `E88FusedLM` in `ndm/models/e88_fused.py`.

Fast path: fused Triton kernel (`ndm/triton/e88_triton_forward.py`,
`e88_triton_backward.py`). Requires CUDA and bfloat16.

Fallback: pure PyTorch reference in the same file; functional on CPU but
substantially slower.

---

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained(
    "poietic-pbc/emender-e88-1.27b",
    revision="v0.1",
    token=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "poietic-pbc/emender-e88-1.27b",
    revision="v0.1",
    trust_remote_code=True,
    token=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

inputs = tokenizer("The key insight of nonlinear recurrence is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**CPU inference note:** The PyTorch fallback is correct but slow. Fast inference
requires a CUDA GPU and the `triton` package. See `requirements.txt` in this
repository.

---

## Training

### Data

<!-- TODO: fill in dataset mix at release time -->
- **Dataset:** [TO FILL — e.g., The Pile / RedPajama / custom mix]
- **Tokens trained:** [TO FILL — e.g., ~100B tokens]
- **Tokenizer:** [TO FILL — byte-level BPE / raw byte / GPT-2 tokenizer; vocab size]
- **Context length:** 2048 tokens (training); context curriculum may extend to
  [TO FILL] tokens.

### Compute

<!-- TODO: fill in at release time -->
- **Hardware:** [TO FILL — e.g., 8× A100 80 GB / H100]
- **Training duration:** [TO FILL — e.g., ~N GPU-days]
- **Compute budget:** [TO FILL — FLOPs or GPU-hours]
- **Precision:** bfloat16 (weights and activations)
- **Optimizer:** ScheduleFree AdamW

### Training Code

Entry point: `train.py` at the repository root.
Repository: [https://github.com/ekg/ndm](https://github.com/ekg/ndm)
Exact commit: [TO FILL — `git rev-parse HEAD` at training time]

---

## Evaluation

### Language Modeling

<!-- TODO: fill in at release time from eval logs -->
| Benchmark | Score | Notes |
|-----------|-------|-------|
| Wikitext-103 (bpb or ppl) | [TO FILL] | |
| The Pile (bits-per-byte) | [TO FILL] | |
| [other benchmarks] | [TO FILL] | |

### State-Tracking / Expressivity

<!-- TODO: link to expressivity-results doc once available -->
NDM is evaluated on controlled finite-state and algorithmic tasks including
S5 permutation composition, a noncommutative state-tracking task. In matched
8M-parameter experiments, NDM separates from linear-scan baselines. See the
expressivity results document (link TBD).

### Comparison Baselines

The primary comparison models in the paper are:
- `FLA-GDN` (strong linear gated delta baseline)
- `Mamba2` (strong selective state-space baseline)
- `M2RNN-paper` (published nonlinear matrix-state geometry)
- `M2RNN-CMA` (M2RNN geometry reshaped into the multi-head regime)

Full evaluation numbers: [TO FILL — link to results doc or paper section].

---

## Intended Use

**Intended use cases:**
- Research on recurrent language models and nonlinear state updates
- State-tracking and reasoning benchmarks
- Baseline for future nonlinear RNN work

**Not intended for:**
- Production deployment without further fine-tuning and safety evaluation
- Instruction-following tasks (this is a base language model; no instruction
  tuning has been applied)
- Safety-critical applications

---

## Limitations

- **No instruction tuning.** This is a base language model. It will complete
  text rather than follow instructions.
- **CPU inference is slow.** The fast path requires CUDA and the `triton`
  package. CPU inference uses the PyTorch fallback, which is O(T) sequential
  and much slower than transformer-style parallelism.
- **bfloat16 only for fast path.** The fused Triton/CUDA kernel requires
  bfloat16 input. float32 falls back to the PyTorch reference path.
- **Training data biases.** The model inherits any biases present in the
  training corpus. [TO FILL — describe dataset-specific caveats.]
- **Context length.** Trained at 2048 tokens. Extrapolation to longer contexts
  is possible (the recurrence is stateful) but was not the primary training
  objective; quality may degrade.
- **No RLHF or safety fine-tuning.** The model may produce harmful, biased,
  or factually incorrect text.

---

## Provenance

NDM was developed in the `ekg/elman` and `ekg/elman-proofs` repositories.
The clean implementation is at `ekg/ndm`.

- Training history repository: [https://github.com/ekg/elman](https://github.com/ekg/elman)
  (commit anchor: `6f0724feae9fc82bd235408ac5c3ae61f2b17c79`)
- Lean formalizations: [https://github.com/ekg/elman-proofs](https://github.com/ekg/elman-proofs)
  (commit anchor: `5082610c9cdabf0b31e11dd14ee078273d486333`)
- This checkpoint's `ndm` repo commit: [TO FILL]
- Training run path in `~/elman/`: [TO FILL — do not publish raw path; record
  in provenance/checkpoint_anchors.txt instead]

---

## License

Apache 2.0 — see [LICENSE](https://github.com/ekg/ndm/blob/main/LICENSE).

---

## Citation

<!-- TODO: fill in once paper DOI / arXiv ID is assigned -->
```bibtex
@article{garrison2026ndm,
  title   = {Nonlinear Delta Memory: Pure Nonlinear Recurrence at Billion-Parameter Scale},
  author  = {Garrison, Erik},
  year    = {2026},
  note    = {Preprint. arXiv:[TO FILL]},
  url     = {https://github.com/ekg/ndm},
}
```
