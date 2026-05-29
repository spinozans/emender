---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- base-model
- recurrent-neural-network
- language-model
- text-generation
- the-pile
- p50k-base
- trust-remote-code
---

# {MODEL_NAME} 1.3B

{MODEL_NAME} is a raw/base 1.3B-class recurrent language model checkpoint from
the Emender v0.1 release bundle. It is not instruction-tuned, chat-tuned,
RLHF-tuned, or safety-tuned. Use it as a base continuation model for research
and reproduction of the paper results, not as an assistant.

## Links

- GitHub repository: <https://github.com/poietic-pbc/emender>
- v0.1 release hub and checklist:
  <https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md>
- Paper PDF target:
  <https://github.com/poietic-pbc/emender/releases/download/v0.1/Garrison_2026_Emender.pdf>
- Paper source:
  <https://github.com/poietic-pbc/emender/blob/main/paper/main.typ>
- Refreshed racer source:
  <https://github.com/poietic-pbc/emender/blob/main/paper/results/figure_2/AS_OF.md>

Related v0.1 model repositories:

- Emender/E88: <https://huggingface.co/poietic-pbc/emender-e88-1.3b>
- GDN: <https://huggingface.co/poietic-pbc/gdn-1.3b>
- M2RNN-CMA: <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b>

## Model Identity

- Identity: `{IDENTITY}`.
- Architecture: `{ARCHITECTURE}`.
- Release revision: `v0.1`.
- Parameter count from smoke construction: `{PARAM_COUNT}`.
- Raw checkpoint step/loss: `{STEP}` / `{CHECKPOINT_LOSS}`.
- Tokenizer: `p50k_base`, exported as `PreTrainedTokenizerFast`, vocab size
  `50,281`.
- Training/evaluation context: pinned training arg `chunk_size=2048`; the
  loader reads `chunk_size + 1` tokens so next-token loss uses 2,048 input
  positions.
- Custom code: yes. Load with `trust_remote_code=True`.

Architecture wording by repo:

- Emender/E88: Emender is the update-rule family; an emender layer is the
  bounded nonlinear matrix-state delta-correction recurrent layer; E88 is the
  concrete v0.1 1.3B instance with the fused Triton path.
- GDN: Gated DeltaNet / FLA-GDN baseline from the same matched racer. It is not
  an Emender layer; it is the strong linear-state gated-delta recurrent
  baseline.
- M2RNN-CMA: M2RNN-style raw-write nonlinear matrix-state baseline reshaped by
  the paper's CMA-ES search pressure. It is not the published paper-default
  grouped-head M2RNN shape; pinned args use `level=m2rnn` and
  `m2rnn_paper_shape=false`.

## Training Data And Tokenization

The v0.1 racer checkpoints were trained on The Pile from the local pinned
corpus path recorded in release docs as `/home/erikg/elman/data/pile.txt`.
Pinned args in the exported `config.json` and local release docs record
`tokenizer=p50k_base`, `chunk_size=2048`, bf16 training, ScheduleFree AdamW,
seed `42`, and architecture-specific hyperparameters.

The source data stream uses ASCII record separator (`\x1e`, byte `0x1e`) between
documents. This is confirmed by `scripts/build_commapile_mainmix.py` and the
data loaders. For the pinned v0.1 `p50k_base` training path, `train.py` selects
`ndm.data.tokenized_dataset.TokenizedStreamDataset`: it samples raw byte
windows, decodes them to text, and tokenizes with tiktoken using
`disallowed_special=()`. The record separator is therefore an ordinary token
when it appears in a sampled window, not a stop token, EOS token, padding token,
or safety boundary. Under `p50k_base`, `"\x1e"` encodes as token ID `218`.

Use `\x1e` as the natural document/example delimiter when constructing raw
continuation prompts that should resemble training boundaries. Do not rely on it
as an instruction separator.

## Current Racer Metrics

The current language-modeling snapshot metrics come from the refreshed racer
Figure 2 source recorded on 2026-05-29 in
`paper/results/figure_2/AS_OF.md`. Scores are 100K-step trailing bits per byte
on The Pile using the pinned `p50k_base` bytes/token estimate. This metric-only
refresh does not move the immutable public `v0.1` checkpoint tags.

| Model | HF repo | Current Figure 2 BPB | Source |
| --- | --- | ---: | --- |
| Emender/E88 | `poietic-pbc/emender-e88-1.3b` | 0.977 | Refreshed racer Figure 2, 2026-05-29 |
| GDN | `poietic-pbc/gdn-1.3b` | 0.970 | Refreshed racer Figure 2, 2026-05-29 |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | 0.983 | Refreshed racer Figure 2, 2026-05-29 |

This model's current Figure 2 score is **{THIS_MODEL_BPB} BPB**.

## Loading Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "{REPO_ID}"
revision = "v0.1"

tokenizer = AutoTokenizer.from_pretrained(
    repo_id,
    revision=revision,
)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    revision=revision,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

prompt = "\x1eThe theorem states"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs.input_ids.to(next(model.parameters()).device)

with torch.no_grad():
    # Minimal raw greedy continuation loop. CPU works but is slow.
    for _ in range(2):
        logits = model(input_ids=input_ids).logits[:, -1, :]
        next_id = logits.argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=-1)

print(tokenizer.decode(input_ids[0], skip_special_tokens=False))
```

## Intended Use

- Research on recurrent language models, nonlinear or linear recurrent state
  updates, and multi-programmed recurrent training.
- Reproduction of the Emender paper's v0.1 racer and release smoke results.
- Raw text continuation experiments under the exact `v0.1` checkpoint revision.

## Out Of Scope

- Instruction following or chat use.
- Safety-critical, production, medical, legal, financial, or autonomous
  decision-making use.
- Claims about safety alignment, helpfulness, refusal behavior, factuality, or
  long-context quality outside the measured v0.1 setup.

## Limitations And Risks

- Base model only: no instruction tuning, supervised fine-tuning, preference
  tuning, RLHF, or safety alignment has been applied.
- Pile-trained raw language models can emit toxic, biased, private,
  copyrighted, or memorized text from the training distribution.
- The `\x1e` delimiter is a data/document boundary marker, not an instruction or
  safety delimiter.
- CPU inference uses slow recurrent full-context stepping. Fast paths require
  compatible CUDA dependencies; E88 additionally has a Triton/fused-kernel path.
- Loading requires Hugging Face custom code with `trust_remote_code=True`;
  inspect the repository code before executing it in a sensitive environment.
- These are single-run v0.1 snapshot checkpoints. The release docs record known
  follow-up work for final public-readiness and broader evaluation.

## Provenance And License

The clean release repository target is <https://github.com/poietic-pbc/emender>.
Historical development provenance is retained in the GitHub repository's
`provenance/` docs. The model card, wrapper code, and release docs are
Apache-2.0. Check the repository and Hugging Face file list for the exact files
covered by the release.

## Citation

```bibtex
@misc{garrison2026emender,
  title  = {Emender: Pure Nonlinear Recurrence at Billion-Parameter Scale},
  author = {Garrison, Erik},
  year   = {2026},
  note   = {v0.1 release candidate},
  url    = {https://github.com/poietic-pbc/emender},
}
```
