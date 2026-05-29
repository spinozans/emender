# v0.2 Public Release Hub

Date: 2026-05-29 UTC
Task: `synchronize-paper-docs`

This is the central v0.2 release handoff for the Emender paper, GitHub
repository, and three public Hugging Face model repositories. The public v0.2
tags were created by `approval-gated-public` after unauthenticated readback of
the uploaded artifacts. The v0.1 tags remain immutable historical checkpoints
and still resolve to their original public SHAs.

Source of truth for the approved public v0.2 artifact commits is
[`RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md`](RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md).
After the publish report, `synchronize-paper-docs` moved each public `v0.2` tag
to a README-only descendant so the canonical `/tree/v0.2` pages no longer showed
pre-approval local-release-candidate text. Before moving the tags, the task
verified that every non-README file, the `config.json` source checkpoint SHA,
and the `model.safetensors` LFS SHA were unchanged.

## Primary Links

- GitHub repository: <https://github.com/poietic-pbc/emender>
- Paper PDF target: <http://hypervolu.me/~erik/ndm/Garrison_2026_Emender.pdf>
- Paper source: <https://github.com/poietic-pbc/emender/blob/main/paper/main.typ>
- v0.2 public HF publish report:
  <https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md>
- v0.2 local validation report:
  <https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V02_RC_LOCAL_VALIDATION_20260529.md>
- Current Figure 2 metric source:
  <https://github.com/poietic-pbc/emender/blob/main/paper/results/figure_2/AS_OF.md>
- Historical v0.1 release hub:
  <https://github.com/poietic-pbc/emender/blob/main/docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md>

Generated PDFs remain gitignored and must not be committed. The public PDF is
mirrored on Hypervolume from a clean `paper/build.sh` output.

## Hugging Face Models

The public release revision for all three model repositories is `v0.2`.

| Model | HF repository | v0.2 tree | Current v0.2 tag SHA | Approved artifact commit | Preserved v0.1 SHA |
| --- | --- | --- | --- | --- | --- |
| Emender/E88 | <https://huggingface.co/poietic-pbc/emender-e88-1.3b> | <https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.2> | `be77d1ee5b744ffa653e9b03abd5254d2ef8a41c` | `ceaa3b0557581b42c585490d641b174470f60a0f` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` |
| GDN | <https://huggingface.co/poietic-pbc/gdn-1.3b> | <https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.2> | `7395b6b6588726a3bca963aa7e6150e0971e71d6` | `a4687c79765540313e08055913443f64bfaed3ed` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` |
| M2RNN-CMA | <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b> | <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.2> | `2e5f8f3be8a7c8ac42802485afb40d023874ea06` | `98af498e483cdd42297b5961c47f65272cf62ff1` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` |

The current `v0.2` tag commits are docs-only descendants of the approved
artifact commits. The model/config/tokenizer files and LFS object SHAs match the
approved artifact commits.

## v0.2 Checkpoint Facts

| Model | Source checkpoint SHA256 | Step | Raw checkpoint loss | v0.2 selected 10K BPB |
| --- | --- | ---: | ---: | ---: |
| Emender/E88 | `da847dcefac2d4bb9c077565a6d5f595a9af5187cc19a2dbfa4377b81a2762dc` | 1,395,000 | 2.6663 | 0.975809 |
| GDN | `31a9181f407006b1bef51d2aefa62be9aafd5197845b19154c1a039f564e2c36` | 1,845,000 | 2.7198 | 0.963171 |
| M2RNN-CMA | `a2a282344e02eb2c237340b4379756d394fda0c4d0c424ddfdba91273030f061` | 1,332,000 | 2.6762 | 0.980586 |

The 10K-smoothed BPB values are from the selected checkpoint rows in
[`RELEASE_V02_RC_LOCAL_VALIDATION_20260529.md`](RELEASE_V02_RC_LOCAL_VALIDATION_20260529.md).
The paper's Figure 2 continues to use its stated Figure 2 smoothing and label
source in `paper/results/figure_2/AS_OF.md`.

## Loading

All three repositories use custom Hugging Face code and should be loaded from
the public `v0.2` revision with `trust_remote_code=True`.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "poietic-pbc/emender-e88-1.3b"
revision = "v0.2"

tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(
    repo_id,
    revision=revision,
    trust_remote_code=True,
)
```

## Public Smoke Evidence

Fresh-cache public Docker smoke passed on 2026-05-29 for CPU and CUDA loads from
`revision="v0.2"` for all three public repositories. Evidence and resolved SHA
rows are recorded in
[`RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md`](RELEASE_V02_PUBLIC_HF_PUBLISH_20260529.md).

Unauthenticated public readback confirmed:

- `v0.2` resolves to the SHAs listed above for all three repositories;
- `v0.1` still resolves to the preserved SHAs listed above;
- `config.json`, custom code, tokenizer files, and `model.safetensors` resolve
  under `v0.2` for all three repositories.

## Public-Release Checklist

- [x] Public v0.2 HF tags exist for Emender/E88, GDN, and M2RNN-CMA at the
      current README-synced SHAs listed in this hub.
- [x] Public v0.1 HF tags still resolve to their original SHAs.
- [x] GitHub README, paper release paragraphs, local release docs, and model-card
      source/default-HF text point to the `poietic-pbc` repositories and `v0.2`
      release where current-release references are intended.
- [x] The refreshed paper PDF is built from `paper/main.typ`, reviewed, and
      mirrored to Hypervolume.
- [x] No model weights or `v0.1` tags are moved or modified by the docs
      synchronization task; `v0.2` was moved only to README-only descendants
      after artifact equality checks, per explicit release-sync authorization.

## Artifact Guardrails

Do not commit tokens, checkpoints, `.pt`/`.pth` files, safetensors, HF caches,
Docker layers, generated PDFs, token files, or other large generated artifacts.
Release evidence should be recorded as small Markdown reports and sanitized JSON
summaries only.
