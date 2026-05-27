# v0.1 Private Hugging Face Staging Upload

Date: 2026-05-27
Task: `release-v01-private-hf-staging-upload`

This note records the private Hugging Face staging upload for the three
validated 1.27B-class checkpoints. All uploaded model repositories are under
`poietic-pbc`, all repository creation calls requested `private=True`, and all
post-upload readbacks reported `private=True`.

No repository was made public. No immutable `v0.1` tags were created. Raw
training checkpoints, optimizer states, local safetensors, Hugging Face caches,
Docker layers, generated PDFs, token files, and other large generated artifacts
were not copied into this git repository, staged, or committed.

## Uploaded Revisions

| Model identity | Repo | Revision | Commit | Private | `model.safetensors` size |
| --- | --- | --- | --- | --- | ---: |
| Emender/E88 | `poietic-pbc/emender-e88-1.27b` | `staging` | `ad4fc69c421a88fc212a4fb89e8415b75eb4441c` | true | `2713727984` |
| GDN | `poietic-pbc/gdn-1.27b` | `staging` | `95ef019198b9e125928a8cf2349895bc31a4906b` | true | `2975047780` |
| M²RNN-CMA | `poietic-pbc/m2rnn-cma-1.27b` | `staging` | `af3cf2db65dfd14b64a5c030c99156828fdfb958` | true | `2807296960` |

Exact URLs:

- `https://huggingface.co/poietic-pbc/emender-e88-1.27b/tree/staging`
- `https://huggingface.co/poietic-pbc/gdn-1.27b/tree/staging`
- `https://huggingface.co/poietic-pbc/m2rnn-cma-1.27b/tree/staging`

Commit URLs:

- `https://huggingface.co/poietic-pbc/emender-e88-1.27b/commit/ad4fc69c421a88fc212a4fb89e8415b75eb4441c`
- `https://huggingface.co/poietic-pbc/gdn-1.27b/commit/95ef019198b9e125928a8cf2349895bc31a4906b`
- `https://huggingface.co/poietic-pbc/m2rnn-cma-1.27b/commit/af3cf2db65dfd14b64a5c030c99156828fdfb958`

Each repository has branches `staging` and `main`, with `staging` containing
the full model artifacts. Each repository has no tags; in particular,
`v0.1` is absent.

## Auth And Guardrails

HF auth was re-verified before upload using `HfApi().whoami(token=True)`.
The authenticated account readback was user `erikgarrison` with membership in
`poietic-pbc`. Token values were not printed or written.

Before upload, the target repos did not exist. They were created by the helper
with `private=True`; the helper refuses to update a repo if readback shows
`private` is not true.

All artifact generation happened under:

```text
/tmp/release-v01-private-hf-staging-agent-410/
```

That directory contains the converted safetensors and the upload manifest, and
it remains outside git. The git-tracked helper and this report contain command
evidence only, not credentials or model weights.

## Artifact Contents

Each `staging` revision contains the expected normal-load files:

- `README.md`
- `config.json`
- `configuration_ndm.py`
- `modeling_ndm.py`
- `generation_config.json`
- `requirements.txt`
- `special_tokens_map.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `tiktoken/tokenizer.model`
- `model.safetensors`
- `.gitattributes`

The model cards identify `PRIVATE STAGING STATUS` and the intended model
identity:

- Emender/E88 in `poietic-pbc/emender-e88-1.27b`
- GDN in `poietic-pbc/gdn-1.27b`
- M²RNN-CMA in `poietic-pbc/m2rnn-cma-1.27b`

The safetensors files were converted from the selected checkpoint
`model_state_dict` entries only. ScheduleFree optimizer state and raw `.pt`
training checkpoint files were not uploaded.

## Exact Commands

Local artifact preparation and validation:

```bash
python -u scripts/hf_private_staging_upload.py \
  --workdir /tmp/release-v01-private-hf-staging-agent-410 \
  --prepare-only \
  --force
```

Local `trust_remote_code` load validation from the generated `/tmp` artifacts:

```bash
python - <<'PY'
import gc, json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
base = Path('/tmp/release-v01-private-hf-staging-agent-410')
for key in ['e88', 'gdn', 'm2rnn']:
    model, info = AutoModelForCausalLM.from_pretrained(
        str(base / key),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
        local_files_only=True,
    )
    print(json.dumps({
        'key': key,
        'class': model.__class__.__name__,
        'missing_keys': info.get('missing_keys'),
        'unexpected_keys': info.get('unexpected_keys'),
        'mismatched_keys': info.get('mismatched_keys'),
        'param_count': sum(p.numel() for p in model.parameters()),
        'dtype_sample': str(next(model.parameters()).dtype),
    }))
    del model
    gc.collect()
PY
```

Upload:

```bash
python -u scripts/hf_private_staging_upload.py \
  --workdir /tmp/release-v01-private-hf-staging-agent-410 \
  --revision staging
```

Per-repo HF API calls executed by the helper:

```python
HfApi().create_repo(repo_id='poietic-pbc/emender-e88-1.27b', repo_type='model', private=True, exist_ok=False, token=True)
HfApi().create_commit(... revision='main', path_in_repo='README.md')
HfApi().create_branch(repo_id='poietic-pbc/emender-e88-1.27b', repo_type='model', branch='staging', revision='main', exist_ok=True, token=True)
HfApi().upload_folder(repo_id='poietic-pbc/emender-e88-1.27b', folder_path='/tmp/release-v01-private-hf-staging-agent-410/e88', repo_type='model', revision='staging', token=True, commit_message='Upload private v0.1 staging artifacts')

HfApi().create_repo(repo_id='poietic-pbc/gdn-1.27b', repo_type='model', private=True, exist_ok=False, token=True)
HfApi().create_commit(... revision='main', path_in_repo='README.md')
HfApi().create_branch(repo_id='poietic-pbc/gdn-1.27b', repo_type='model', branch='staging', revision='main', exist_ok=True, token=True)
HfApi().upload_folder(repo_id='poietic-pbc/gdn-1.27b', folder_path='/tmp/release-v01-private-hf-staging-agent-410/gdn', repo_type='model', revision='staging', token=True, commit_message='Upload private v0.1 staging artifacts')

HfApi().create_repo(repo_id='poietic-pbc/m2rnn-cma-1.27b', repo_type='model', private=True, exist_ok=False, token=True)
HfApi().create_commit(... revision='main', path_in_repo='README.md')
HfApi().create_branch(repo_id='poietic-pbc/m2rnn-cma-1.27b', repo_type='model', branch='staging', revision='main', exist_ok=True, token=True)
HfApi().upload_folder(repo_id='poietic-pbc/m2rnn-cma-1.27b', folder_path='/tmp/release-v01-private-hf-staging-agent-410/m2rnn', repo_type='model', revision='staging', token=True, commit_message='Upload private v0.1 staging artifacts')
```

Post-upload readback:

```bash
python - <<'PY'
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoTokenizer
api = HfApi()
required = {
    'README.md', 'config.json', 'configuration_ndm.py', 'modeling_ndm.py',
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'generation_config.json', 'requirements.txt', 'model.safetensors',
}
for repo_id in [
    'poietic-pbc/emender-e88-1.27b',
    'poietic-pbc/gdn-1.27b',
    'poietic-pbc/m2rnn-cma-1.27b',
]:
    info = api.repo_info(repo_id, repo_type='model', revision='staging',
                         files_metadata=True, token=True)
    refs = api.list_repo_refs(repo_id, repo_type='model', token=True)
    files = {s.rfilename: getattr(s, 'size', None) for s in info.siblings}
    cfg = AutoConfig.from_pretrained(repo_id, revision='staging',
                                     trust_remote_code=True, token=True)
    tok = AutoTokenizer.from_pretrained(repo_id, revision='staging', token=True)
    print({
        'repo_id': repo_id,
        'private': getattr(info, 'private', None),
        'sha': getattr(info, 'sha', None),
        'missing_required': sorted(required - set(files)),
        'model_safetensors_size': files.get('model.safetensors'),
        'file_count': len(files),
        'branches': [b.name for b in refs.branches],
        'tags': [t.name for t in refs.tags],
        'config_class': cfg.__class__.__name__,
        'model_identity': getattr(cfg, 'model_identity', None),
        'tokenizer_class': tok.__class__.__name__,
        'prompt_tokens': tok.encode('The theorem states'),
    })
PY
```

## Validation Results

Local generated artifacts:

| Model | Local `AutoModelForCausalLM` class | Missing keys | Unexpected keys | Mismatched keys | Param count | Dtype |
| --- | --- | --- | --- | --- | ---: | --- |
| E88 | `NdmForCausalLM` | `[]` | `[]` | `[]` | `1273191856` | `torch.bfloat16` |
| GDN | `NdmForCausalLM` | `[]` | `[]` | `[]` | `1352352498` | `torch.bfloat16` |
| M2RNN-CMA | `NdmForCausalLM` | `[]` | `[]` | `[]` | `1307101140` | `torch.bfloat16` |

Remote `staging` readback:

| Repo | Private | Required files missing | Branches | Tags | Remote config/tokenizer |
| --- | --- | --- | --- | --- | --- |
| `poietic-pbc/emender-e88-1.27b` | true | `[]` | `['staging', 'main']` | `[]` | `NdmConfig`, `PreTrainedTokenizerFast`, prompt IDs `[464, 44728, 2585]` |
| `poietic-pbc/gdn-1.27b` | true | `[]` | `['staging', 'main']` | `[]` | `NdmConfig`, `PreTrainedTokenizerFast`, prompt IDs `[464, 44728, 2585]` |
| `poietic-pbc/m2rnn-cma-1.27b` | true | `[]` | `['staging', 'main']` | `[]` | `NdmConfig`, `PreTrainedTokenizerFast`, prompt IDs `[464, 44728, 2585]` |

The previous local and Docker-local generation smokes remain the source of
checkpoint behavioral validation. This task adds private-HF staging upload and
remote artifact/readback validation only; downstream private-HF container tests
must pass before any public visibility change or immutable release tag.
