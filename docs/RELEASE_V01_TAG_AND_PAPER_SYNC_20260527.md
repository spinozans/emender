# v0.1 Private Hugging Face Tag And Paper Sync

Date: 2026-05-27
Task: `release-v01-tag-and-paper-sync`

This note records the final private Hugging Face `v0.1` tags created after the
private-HF CPU and CUDA smoke task passed. No repository visibility was changed.
No explicit public-release approval was present in the task context, so the
repositories remain private.

No token values, checkpoints, safetensors files, Hugging Face caches, Docker
layers, generated PDFs, or other large generated artifacts were copied into the
repository, staged, or committed.

## Tagged Revisions

The tags were created at the exact revisions validated by
`release-v01-docker-private-hf-smoke`.

| Model identity | Repo | Tested commit | Tag URL | Commit URL | Private readback |
| --- | --- | --- | --- | --- | --- |
| Emender/E88 | `poietic-pbc/emender-e88-1.27b` | `ad4fc69c421a88fc212a4fb89e8415b75eb4441c` | `https://huggingface.co/poietic-pbc/emender-e88-1.27b/tree/v0.1` | `https://huggingface.co/poietic-pbc/emender-e88-1.27b/commit/ad4fc69c421a88fc212a4fb89e8415b75eb4441c` | true |
| GDN | `poietic-pbc/gdn-1.27b` | `95ef019198b9e125928a8cf2349895bc31a4906b` | `https://huggingface.co/poietic-pbc/gdn-1.27b/tree/v0.1` | `https://huggingface.co/poietic-pbc/gdn-1.27b/commit/95ef019198b9e125928a8cf2349895bc31a4906b` | true |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.27b` | `af3cf2db65dfd14b64a5c030c99156828fdfb958` | `https://huggingface.co/poietic-pbc/m2rnn-cma-1.27b/tree/v0.1` | `https://huggingface.co/poietic-pbc/m2rnn-cma-1.27b/commit/af3cf2db65dfd14b64a5c030c99156828fdfb958` | true |

## Commands And Guardrails

Before tagging, the authenticated Hugging Face session read back user
`erikgarrison` with membership in `poietic-pbc`. The token value was not
printed or written.

Preflight readback for each repo used:

```python
HfApi().repo_info(repo_id, repo_type="model", revision=tested_sha, token=True)
HfApi().list_repo_refs(repo_id, repo_type="model", token=True)
```

Tag creation used only:

```python
HfApi().create_tag(
    repo_id=repo_id,
    repo_type="model",
    tag="v0.1",
    revision=tested_sha,
    tag_message=f"v0.1 private release tag at smoke-tested revision {tested_sha}",
    token=True,
    exist_ok=False,
)
```

Post-tag validation used:

```python
HfApi().repo_info(repo_id, repo_type="model", revision="v0.1", token=True)
HfApi().list_repo_refs(repo_id, repo_type="model", token=True)
```

No visibility-changing command or API was run. In particular,
`HfApi().update_repo_visibility(...)` was not called and no
`huggingface-cli repo update` or equivalent public-visibility command was run.

## Paper And Release Links

`paper/main.typ` already pointed the E88 release target at
`https://huggingface.co/poietic-pbc/emender-e88-1.27b/tree/v0.1`, so no paper
source text changed in this task and no refreshed PDF upload was required.

Release documentation and the model-card template were updated so active HF
examples use the `poietic-pbc` namespace and `revision="v0.1"` instead of the
former single-repo placeholder.

## Next Manual Public-Release Step

The next public-release step is manual: obtain explicit user approval to make
the three HF repositories public, log the approval source, record the exact
visibility commands, and only then run the visibility change for:

- `poietic-pbc/emender-e88-1.27b`
- `poietic-pbc/gdn-1.27b`
- `poietic-pbc/m2rnn-cma-1.27b`
