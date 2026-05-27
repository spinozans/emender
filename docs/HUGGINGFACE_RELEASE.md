# HuggingFace Release Checklist

## Loading-Path Decision

**Chosen path: custom-code `AutoModelForCausalLM` with `trust_remote_code=True`.**

The `E88FusedLM` class in `ndm/models/e88_fused.py` is a plain `torch.nn.Module`
with no existing HF `PreTrainedModel` subclass. The minimum-friction path for
third-party inference is to ship `modeling_ndm.py` (a thin HF-compatible wrapper
around `E88FusedLM`) directly in the HF repository, alongside a `config.json`
that encodes hyperparameters. Users load the model with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "poietic-pbc/emender-e88-1.27b",
    revision="v0.1",
    trust_remote_code=True,
    token=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

The v0.1 release also includes the sibling private baseline repositories
`poietic-pbc/gdn-1.27b` and `poietic-pbc/m2rnn-cma-1.27b`, both loaded with
`revision="v0.1"`.

This avoids adding HF as a hard dependency of the `ndm` package, keeps the
training codebase untouched, and works today without a Transformers PR. The
fast inference path still requires CUDA (for the Triton/`hasty_pytorch_lib`
kernel); the model automatically falls back to the PyTorch reference
implementation on CPU or when `hasty_pytorch_lib` is absent.

A full HF-Transformers integration (registering `NdmConfig`, `NdmForCausalLM`
upstream) is the recommended follow-up but is out of scope for the initial
release.

---

## Required Files in the HF Repository

| File | Purpose |
|------|---------|
| `config.json` | Model hyperparameters (`dim`, `depth`, `n_heads`, `n_state`, `vocab_size`, `model_type: "ndm"`) |
| `modeling_ndm.py` | `NdmForCausalLM(PreTrainedModel)` wrapper that instantiates `E88FusedLM` |
| `configuration_ndm.py` | `NdmConfig(PretrainedConfig)` dataclass |
| `model.safetensors` (or shards) | Weights converted via `safetensors.torch.save_file` |
| `tokenizer_config.json` | Tokenizer metadata (byte-level BPE or GPT-2 tokenizer) |
| `vocab.json` + `merges.txt` | Tokenizer vocabulary (if BPE; omit if using raw byte tokenizer) |
| `special_tokens_map.json` | EOS/BOS/PAD token mapping |
| `README.md` | The filled model card (see `MODEL_CARD_TEMPLATE.md`) |
| `requirements.txt` | `torch>=2.0`, `triton` (optional fast path), `einops` |

---

## Release Checklist

### Phase 1 — Checkpoint Selection

- [ ] **1.1** Identify the target training run in `~/elman/` (run directory,
      config JSON, final checkpoint step).
- [ ] **1.2** Record the exact `ndm` repo commit hash that produced the run:
      `git rev-parse HEAD` in this repo at training time.
- [ ] **1.3** Confirm the tokenizer used during training (byte-level vs BPE;
      vocabulary size).
- [ ] **1.4** Log the checkpoint in provenance:
      `echo "checkpoint: <path> | ndm commit: <hash> | step: <N>" >> provenance/checkpoint_anchors.txt`

### Phase 2 — Package Freeze

- [ ] **2.1** Cut a release tag on this repo (`git tag v0.1.0`) at the commit
      identified in step 1.2.
- [ ] **2.2** Bump `version` in `pyproject.toml` to match (e.g., `0.1.0`).
- [ ] **2.3** Build the sdist and wheel (`python -m build`) and smoke-test:
      `pip install dist/ndm-0.1.0-*.whl && python -c "from ndm.models.e88_fused import E88FusedLM"`

### Phase 3 — Weights Conversion

- [ ] **3.1** Load the checkpoint into `E88FusedLM`:
      ```python
      import torch
      from ndm.models.e88_fused import E88FusedLM
      model = E88FusedLM(vocab_size=..., dim=2176, depth=14, n_heads=98, n_state=32)
      model.load_state_dict(torch.load("<checkpoint>/model.pt", map_location="cpu"))
      ```
- [ ] **3.2** Export to safetensors:
      ```python
      from safetensors.torch import save_file
      save_file(model.state_dict(), "model.safetensors")
      ```
- [ ] **3.3** Verify round-trip: reload from `model.safetensors`, run a
      forward pass with a dummy input, confirm logits are finite and
      deterministic.

### Phase 4 — Wrapper and Config

- [ ] **4.1** Write `configuration_ndm.py` (`NdmConfig(PretrainedConfig)`) with
      fields: `vocab_size`, `dim`, `depth`, `n_heads`, `n_state`, `expansion`,
      `use_gate`, `use_silu`, `use_l2_norm`, `tie_embeddings`.
- [ ] **4.2** Write `modeling_ndm.py` (`NdmForCausalLM(PreTrainedModel)`) that:
      - loads `NdmConfig` from `config.json`
      - instantiates `E88FusedLM` with config fields
      - implements `forward(input_ids, ...)` returning `CausalLMOutput`
      - implements `generate(...)` using sequential state stepping
- [ ] **4.3** Serialize `config.json` from the chosen hyperparameters.
- [ ] **4.4** Copy tokenizer files from the training run into the release dir.

### Phase 5 — Model Card

- [ ] **5.1** Fill in `docs/MODEL_CARD_TEMPLATE.md` with the actual values:
      token count, dataset mix, compute hours, hardware spec, eval numbers.
- [ ] **5.2** Copy the completed card to `README.md` in the HF repo directory.

### Phase 6 — HuggingFace Upload And Private Tag

- [x] **6.1** Create private HF repositories under `poietic-pbc`:
      `emender-e88-1.27b`, `gdn-1.27b`, and `m2rnn-cma-1.27b`.
- [ ] **6.2** Upload all files:
      ```bash
      huggingface-cli upload poietic-pbc/emender-e88-1.27b ./hf_release_dir/ . --revision staging
      ```
- [x] **6.3** Run private-HF CPU and CUDA clean-container smokes at the exact
      uploaded staging commits.
- [x] **6.4** Create immutable `v0.1` tags at the smoke-tested commits.
- [ ] **6.5** Set repository visibility to public only after explicit user
      approval is present and logged.
- [ ] **6.6** Add topics: `rnn`, `language-model`, `nonlinear-rnn`, `triton`,
      `ndm`, `state-tracking`.

### Phase 7 — Smoke Test from Clean Environment

- [ ] **7.1** In a fresh virtualenv (no `ndm` package installed):
      ```bash
      pip install transformers torch safetensors
      python -c "
      from transformers import AutoModelForCausalLM, AutoTokenizer
      repo_id = 'poietic-pbc/emender-e88-1.27b'
      tok = AutoTokenizer.from_pretrained(repo_id, revision='v0.1', token=True)
      model = AutoModelForCausalLM.from_pretrained(
          repo_id, revision='v0.1', trust_remote_code=True, token=True)
      ids = tok('Hello', return_tensors='pt').input_ids
      out = model.generate(ids, max_new_tokens=20)
      print(tok.decode(out[0]))
      "
      ```
- [ ] **7.2** Confirm generation is non-degenerate (no inf/nan, no immediate
      repetition loop).
- [ ] **7.3** Verify CPU fallback: repeat 7.1 without CUDA, confirm the
      PyTorch reference path runs (slowly but correctly).

### Phase 8 — Link from This Repository

- [ ] **8.1** Add an **HuggingFace** badge and link to the checkpoint in
      `README.md` of this repo.
- [ ] **8.2** Record the final HF repo URL and the `ndm` tag in
      `provenance/checkpoint_anchors.txt`.
- [ ] **8.3** Commit: `git add README.md provenance/checkpoint_anchors.txt && git commit -m "release: link HF checkpoint poietic-pbc/emender-e88-1.27b"`

---

## Naming Rationale

Chosen v0.1 HF model repositories:

| Model identity | Repository | Release revision |
| --- | --- | --- |
| Emender/E88 | `poietic-pbc/emender-e88-1.27b` | `v0.1` |
| GDN baseline | `poietic-pbc/gdn-1.27b` | `v0.1` |
| M2RNN-CMA baseline | `poietic-pbc/m2rnn-cma-1.27b` | `v0.1` |

The `poietic-pbc` namespace is the release namespace for v0.1. The Python
package and import path remain `ndm` for this release unless a separate package
rename task approves and performs a broader migration.

---

## Out of Scope for Initial Release

- Instruction fine-tuning or RLHF
- GGUF / llama.cpp quantization export
- Full HF Transformers upstream integration (file a follow-up task)
- Selecting the specific checkpoint (done by the training team from `~/elman/`)
