# Hugging Face Release Notes

Current public release hub:
[`docs/RELEASE_V02_PUBLIC_RELEASE_HUB.md`](RELEASE_V02_PUBLIC_RELEASE_HUB.md).
The public GitHub target is <https://github.com/poietic-pbc/emender>.

The current public HF release revision is `v0.2` for all three 1.3B-class
repositories:

| Model identity | Repository | Current release revision | Current v0.2 tag SHA | Approved artifact commit | Preserved v0.1 SHA |
| --- | --- | --- | --- | --- | --- |
| Emender/E88 | `poietic-pbc/emender-e88-1.3b` | `v0.2` | `be77d1ee5b744ffa653e9b03abd5254d2ef8a41c` | `ceaa3b0557581b42c585490d641b174470f60a0f` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` |
| GDN baseline | `poietic-pbc/gdn-1.3b` | `v0.2` | `7395b6b6588726a3bca963aa7e6150e0971e71d6` | `a4687c79765540313e08055913443f64bfaed3ed` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` |
| M2RNN-CMA baseline | `poietic-pbc/m2rnn-cma-1.3b` | `v0.2` | `2e5f8f3be8a7c8ac42802485afb40d023874ea06` | `98af498e483cdd42297b5961c47f65272cf62ff1` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` |

The `v0.1` tags are retained for historical and pinning use only. The current
`v0.2` tag commits are README-only descendants of the approved artifact commits;
non-README files, `config.json` source checkpoint SHAs, and
`model.safetensors` LFS SHAs were verified unchanged before retagging.

## Loading Path

The release uses custom-code `AutoModelForCausalLM` with
`trust_remote_code=True`. The HF repositories ship `modeling_ndm.py` and
`configuration_ndm.py` alongside `config.json`, tokenizer files, and
`model.safetensors`.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "poietic-pbc/emender-e88-1.3b"
revision = "v0.2"

tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    revision=revision,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

The same loading pattern applies to `poietic-pbc/gdn-1.3b` and
`poietic-pbc/m2rnn-cma-1.3b`, both with `revision="v0.2"`.

This avoids adding HF as a hard dependency of the `ndm` package. CPU fallback is
available but slow; fast E88 inference requires compatible CUDA/Triton support.

## Required Files in Each HF Repository

| File | Purpose |
| --- | --- |
| `config.json` | Model hyperparameters and custom-code `auto_map` entries |
| `modeling_ndm.py` | `NdmForCausalLM(PreTrainedModel)` wrapper |
| `configuration_ndm.py` | `NdmConfig(PretrainedConfig)` |
| `model.safetensors` | Converted release weights |
| `tokenizer_config.json` | Tokenizer metadata for exported `p50k_base` |
| `tokenizer.json` and `tiktoken/tokenizer.model` | Exported tokenizer files |
| `special_tokens_map.json` | EOS/BOS/PAD token mapping |
| `README.md` | Model card; see `MODEL_CARD_TEMPLATE.md` |
| `requirements.txt` | Runtime dependency hints |

## v0.2 Publication Evidence

The approved public v0.2 publish is recorded in
[`RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md`](RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md).
That report records:

- explicit human approval before public HF write operations;
- public `v0.2` tag SHAs for all three repositories;
- preserved `v0.1` SHAs for all three repositories;
- unauthenticated pre-tag and post-tag readback;
- CPU and CUDA Docker generation smoke from fresh cache using
  `revision="v0.2"`;
- no movement of `v0.1` tags.

Local checkpoint selection, conversion, and local/Docker artifact validation are
recorded in
[`RELEASE_V02_RC_LOCAL_VALIDATION_20260529.md`](RELEASE_V02_RC_LOCAL_VALIDATION_20260529.md).

## Model-Card Facts

- Identity: raw/base 1.3B-class recurrent language models; not
  instruction-tuned, chat-tuned, RLHF-tuned, or safety-tuned.
- Architectures: Emender/E88 nonlinear delta-memory, Gated DeltaNet baseline,
  and M2RNN-CMA raw-write nonlinear matrix-state baseline.
- Data: The Pile, pinned locally as `/home/erikg/elman/data/pile.txt` in the
  release checkpoint docs.
- Tokenizer/context: `p50k_base`, vocab size `50,281`, pinned training arg
  `chunk_size=2048`; next-token training/eval uses 2,048 input positions.
- Delimiter: ASCII record separator `\x1e` / byte `0x1e` is an ordinary corpus
  token under `p50k_base`, not a stop token, EOS token, padding token, or safety
  boundary.
- Current Figure 2 labels: Emender/E88 `0.977` BPB, GDN `0.970` BPB,
  M2RNN-CMA `0.983` BPB from `paper/results/figure_2/AS_OF.md`.
- v0.2 selected-checkpoint 10K BPB values: Emender/E88 `0.975809`, GDN
  `0.963171`, M2RNN-CMA `0.980586`.

## Guardrails

- Do not commit tokens, checkpoints, `.pt`/`.pth` files, safetensors, HF caches,
  Docker layers, generated PDFs, token files, or other large generated
  artifacts.
- Do not upload model weights from this docs-sync task.
- Do not move, delete, recreate, or modify `v0.1` tags.
- Do not move `v0.2` tags except for explicitly authorized README-only docs
  descendants after artifact equality checks.
