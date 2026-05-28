# v0.1 Model-Card Docs Polish

Date: 2026-05-28 UTC
Task: `release-v01-model-card-docs-polish`

This note records the docs-only Hugging Face model-card updates and local
release-doc polish for the private v0.1 Emender release candidate. The three HF
repositories remained private throughout this task.

## Remote Hugging Face Changes

Each repository received the same model-card content on both `main` and
`staging`. The private `v0.1` tag was then recreated to point at the new
docs-only `staging` commit, so `revision="v0.1"` includes the polished card.

| Model | Repo | Old `v0.1` SHA | New main-card SHA | New `staging` / `v0.1` SHA | Private readback |
| --- | --- | --- | --- | --- | --- |
| Emender/E88 | `poietic-pbc/emender-e88-1.3b` | `ad4fc69c421a88fc212a4fb89e8415b75eb4441c` | `718b3e15bb8ed7f065c5aa65a569e62af7a12a02` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | true |
| GDN | `poietic-pbc/gdn-1.3b` | `95ef019198b9e125928a8cf2349895bc31a4906b` | `7b267ad249cf57594feaa38ef6b3aebd108722c4` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | true |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | `af3cf2db65dfd14b64a5c030c99156828fdfb958` | `74091c1457d0e6a46872d72d38d12f6a10170d29` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | true |

The old SHAs are still the artifact upload commits validated by the earlier
private-HF Docker smoke. The new SHAs are README-only descendants. Because the
`v0.1` tags were moved, the downstream final Docker smoke must validate the new
resolved SHAs before public release.

## API Operations Used

Only docs and tag operations were used:

```python
HfApi().create_commit(..., revision="main", path_in_repo="README.md")
HfApi().create_commit(..., revision="staging", path_in_repo="README.md")
HfApi().delete_tag(repo_id, repo_type="model", tag="v0.1")
HfApi().create_tag(repo_id, repo_type="model", tag="v0.1", revision=<new_staging_sha>)
HfApi().repo_info(..., revision="v0.1")
```

No `update_repo_visibility`, `update_repo_settings(private=False)`,
`huggingface-cli repo update`, or equivalent visibility-changing command was
run.

## Model-Card Coverage

All three HF model cards now include:

- raw/base model identity and explicit no instruction tuning / no safety tuning;
- architecture identity for Emender/E88, GDN, or M2RNN-CMA;
- The Pile training data, `p50k_base` tokenizer, `chunk_size=2048`, and
  2,048-token next-token context details from pinned args;
- verified delimiter behavior for `\x1e` / byte `0x1e`, including the actual
  pinned `p50k_base` training path and token ID `218`;
- current v0.1 BPB metrics from the 2026-05-27 refreshed racer Figure 2 source:
  Emender/E88 `0.979`, GDN `0.975`, M2RNN-CMA `0.984`;
- `revision="v0.1"` and `trust_remote_code=True` load examples;
- intended research/reproduction use and raw continuation scope;
- limitations covering base-model behavior, Pile text risks, memorization,
  CPU speed, and custom-code loading;
- links to `poietic-pbc/emender`, the paper PDF target, the release hub, and all
  three HF repositories.

## Local Docs Updated

- `docs/RELEASE_V01_PUBLIC_RELEASE_HUB.md` is the central public-release hub.
- `docs/HUGGINGFACE_RELEASE.md` now points to the hub and current `v0.1` SHAs.
- `docs/MODEL_CARD_TEMPLATE.md` is replaced with a current v0.1 model-card
  template.
- `paper/main.typ` release paragraphs point to the hub and all three HF repos.
- `README.md`, `docs/STANDALONE_USAGE.md`, `paper/README.md`, and
  `pyproject.toml` now use the `poietic-pbc/emender` public repository target.
- `scripts/smoke_private_hf_generation.py` and
  `scripts/docker_private_hf_smoke.sh` now load `revision="v0.1"` and verify
  the new resolved SHAs.

## Validation Readback

For each HF repo, `main` and `v0.1` README readback confirmed:

- required strings for raw/base identity, The Pile, tokenizer/context,
  delimiter, all three BPB metrics, `revision="v0.1"`, `trust_remote_code=True`,
  GitHub/paper links, and limitations;
- no legacy namespace, former GitHub release URL, staging-revision example,
  staging frontmatter flag, or private-staging warning in the updated card;
- `private=True` on `main`, `staging`, and `v0.1` readback.

No tokens, checkpoints, safetensors, HF caches, Docker layers, generated PDFs,
or large artifacts were written into this git repository.
