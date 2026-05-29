# v0.2 Public Hugging Face Publish

Date: 2026-05-29 UTC
Task: `approval-gated-public`

This report records the approved public v0.2 Hugging Face publication for the
three locally validated 1.3B racer artifacts.

Explicit human approval was present before any Hugging Face write operation in
the queued task message from Erik at `2026-05-29T19:55:17.189221998+00:00`:

> Human approval: Erik explicitly authorizes public Hugging Face v0.2 publication for the three validated 1.3B repos.

The publish used only the local artifacts validated by
`validate-v0-2-racer`:

- `/tmp/release-v02-local-hf-candidates-agent-500/e88`
- `/tmp/release-v02-local-hf-candidates-agent-500/gdn`
- `/tmp/release-v02-local-hf-candidates-agent-500/m2rnn`
- `/tmp/release-v02-local-hf-candidates-agent-500/validation_manifest.json`
- `/tmp/release-v02-docker-local-hf-artifact-smoke-agent-500/summary.json`

No `v0.1` tag was moved, deleted, recreated, or otherwise modified.

## Public v0.2 SHAs At Initial Publish

These are the exact public artifact commit/tag SHAs from the initial approved
publish. The downstream docs sync later moved `v0.2` to README-only descendants
after verifying every non-README artifact remained unchanged; see the
downstream sync note at the end of this report.

| Model | Public repo | v0.2 tag SHA | Preserved v0.1 SHA | Source checkpoint SHA256 | Step |
| --- | --- | --- | --- | --- | ---: |
| E88 / NDM | `poietic-pbc/emender-e88-1.3b` | `ceaa3b0557581b42c585490d641b174470f60a0f` | `a2e56cb82eec5e01ae6eb501569359c5ff64af6b` | `da847dcefac2d4bb9c077565a6d5f595a9af5187cc19a2dbfa4377b81a2762dc` | 1,395,000 |
| GDN | `poietic-pbc/gdn-1.3b` | `a4687c79765540313e08055913443f64bfaed3ed` | `556df7f00969c6a8dbeb381e3c8b51cf0c0385f9` | `31a9181f407006b1bef51d2aefa62be9aafd5197845b19154c1a039f564e2c36` | 1,845,000 |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | `98af498e483cdd42297b5961c47f65272cf62ff1` | `8181b77803e130ffd78e37c33aa4d58c27e719c2` | `a2a282344e02eb2c237340b4379756d394fda0c4d0c424ddfdba91273030f061` | 1,332,000 |

Public tag URLs:

- <https://huggingface.co/poietic-pbc/emender-e88-1.3b/tree/v0.2>
- <https://huggingface.co/poietic-pbc/gdn-1.3b/tree/v0.2>
- <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/tree/v0.2>

Public commit URLs:

- <https://huggingface.co/poietic-pbc/emender-e88-1.3b/commit/ceaa3b0557581b42c585490d641b174470f60a0f>
- <https://huggingface.co/poietic-pbc/gdn-1.3b/commit/a4687c79765540313e08055913443f64bfaed3ed>
- <https://huggingface.co/poietic-pbc/m2rnn-cma-1.3b/commit/98af498e483cdd42297b5961c47f65272cf62ff1>

## Publish Command

The helper requires an explicit approval flag and an approval note before it can
perform any public Hugging Face writes.

```bash
python -u scripts/publish_v02_public_hf.py \
  --approved-public-v02-publication \
  --approval-note "Human approval: Erik explicitly authorizes public Hugging Face v0.2 publication for the three validated 1.3B repos: poietic-pbc/emender-e88-1.3b, poietic-pbc/gdn-1.3b, and poietic-pbc/m2rnn-cma-1.3b. Proceed with the existing task guardrails: upload only the locally validated v0.2 artifacts, create v0.2 tags after readback verification, and do not move or modify v0.1 tags."
```

The sanitized publish summary was written outside the repository:

- `/tmp/release-v02-public-hf-publish-agent-502/summary.json`

For each repo, the script:

1. Confirmed the repo was public and the current public `v0.1` tag resolved to
   the expected original SHA.
2. Validated the local artifact file list, `config.json`, safetensors metadata,
   local safetensors SHA256, and local CPU/GPU Docker-smoke summary.
3. Uploaded the manifest-listed files to the existing public repo.
4. Performed unauthenticated readback at the uploaded commit before creating
   `v0.2`.
5. Created `v0.2` at the readback-verified uploaded commit.
6. Re-read public `v0.1` and `v0.2` refs after tagging.

## Unauthenticated Readback

The pre-tag readback used public unauthenticated Hub access (`token=False`) and
verified file sizes, small-file SHA256 values, `model.safetensors` LFS SHA256
values, and `config.json` source checkpoint metadata. The required public
files resolved unauthenticated at `v0.2` for all three repos.

| Model | `config.json` | `configuration_ndm.py` | `modeling_ndm.py` | `tokenizer.json` | `tokenizer_config.json` | `special_tokens_map.json` | `tiktoken/tokenizer.model` | `model.safetensors` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E88 / NDM | 2,931 | 4,155 | 5,245 | 6,312,314 | 322 | 131 | 836,186 | 2,713,728,024 |
| GDN | 2,933 | 4,155 | 5,245 | 6,312,314 | 322 | 131 | 836,186 | 2,975,047,820 |
| M2RNN-CMA | 2,948 | 4,155 | 5,245 | 6,312,314 | 322 | 131 | 836,186 | 2,807,297,000 |

Remote `model.safetensors` SHA256 values:

| Model | Public v0.2 `model.safetensors` SHA256 |
| --- | --- |
| E88 / NDM | `1be8a36f4ef842b072822ae65a523b6ac6a974d854ac8acaba38d80e51791d76` |
| GDN | `812b75e8b649ce5540f5283187ac67f69672b6fa2052466a87141e26d7949a3c` |
| M2RNN-CMA | `8f095ad24e66ce589b816df3867ebebd5914a4a237210a867592f9f32abb5fe0` |

## Public Docker Smoke

Command:

```bash
SMOKE_OUTPUT_DIR=/tmp/release-v02-docker-public-hf-smoke-agent-502 \
SMOKE_GPU_DEVICE=0 \
SMOKE_E88_EXPECTED_SHA=ceaa3b0557581b42c585490d641b174470f60a0f \
SMOKE_GDN_EXPECTED_SHA=a4687c79765540313e08055913443f64bfaed3ed \
SMOKE_M2RNN_EXPECTED_SHA=98af498e483cdd42297b5961c47f65272cf62ff1 \
scripts/docker_public_hf_v02_smoke.sh
```

Evidence outside the repository:

- `/tmp/release-v02-docker-public-hf-smoke-agent-502/summary.json`
- `/tmp/release-v02-docker-public-hf-smoke-agent-502/summary.txt`
- `/tmp/release-v02-docker-public-hf-smoke-agent-502/commands.txt`

Summary:

```text
ok=True gpu_status=available
e88_cpu.json True e88 cpu v0.2 ceaa3b0557581b42c585490d641b174470f60a0f NdmForCausalLM ndm.models.ladder_lm.LadderLM [968, 39696] ' New elong' True
e88_cuda.json True e88 cuda v0.2 ceaa3b0557581b42c585490d641b174470f60a0f NdmForCausalLM ndm.models.ladder_lm.LadderLM [968, 218] ' New\x1e' True
gdn_cpu.json True gdn cpu v0.2 a4687c79765540313e08055913443f64bfaed3ed NdmForCausalLM ndm.models.ladder_lm.LadderLM [262, 319] ' the on' True
gdn_cuda.json True gdn cuda v0.2 a4687c79765540313e08055913443f64bfaed3ed NdmForCausalLM ndm.models.ladder_lm.LadderLM [262, 319] ' the on' True
m2rnn_cpu.json True m2rnn cpu v0.2 98af498e483cdd42297b5961c47f65272cf62ff1 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [6558, 44338] 'Sec hangar' True
m2rnn_cuda.json True m2rnn cuda v0.2 98af498e483cdd42297b5961c47f65272cf62ff1 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [6558, 35631] 'Secrigan' True
```

The GPU probe reported:

```json
{"cuda_available": true, "device_count": 1, "device_name": "NVIDIA RTX 6000 Ada Generation"}
```

Every public Docker smoke row reported:

- `ok: true`
- exact expected public `v0.2` resolved SHA
- `private: false`
- expected source checkpoint SHA256 and checkpoint step
- finite logits
- nonempty generated text
- `missing_keys: []`
- `unexpected_keys: []`
- `mismatched_keys: []`

The smoke harness used separate fresh Docker HF cache volumes:

- CPU: `ndm-public-hf-v02-smoke-cpu-20260529201124-253950`
- GPU: `ndm-public-hf-v02-smoke-gpu-20260529201124-253950`

A post-run Docker volume check returned no matching
`ndm-public-hf-v02-smoke-*` volumes.

## Validation Checklist Result

- [x] Explicit human approval for public v0.2 upload was present in the queued
      task message before any HF write API/CLI command was run.
- [x] Uploaded validated local v0.2 artifacts for E88, GDN, and M2RNN-CMA to
      the existing public 1.3B repos.
- [x] Did not move, delete, recreate, or modify `v0.1` tags.
- [x] Created `v0.2` tags only after upload readback confirmed intended
      artifacts and source checkpoint SHA256 values.
- [x] Public v0.2 config/modeling/tokenizer/model.safetensors files resolve
      unauthenticated for all three repos.
- [x] `v0.1` tags still resolve to their original public SHAs for all three
      repos.
- [x] CPU and GPU Docker generation smoke passed from `revision="v0.2"` for all
      three public repos from fresh Docker cache/workdir.
- [x] No tokens, raw checkpoints, local safetensors, HF caches, Docker layers,
      generated PDFs, or Docker smoke outputs were staged or committed.

## Notes

The first public Docker smoke attempt failed before model download because the
new helper passed `repo_type` to `HfApi.model_info`, which the pinned
`huggingface-hub==0.36.0` client does not accept. The helper was corrected to
call `model_info(..., revision=..., token=False)`, then the full public CPU/GPU
smoke was rerun and passed.

The uploaded README/model-card files are exactly the locally validated v0.2
artifact files. Broader paper/docs/PDF synchronization is intentionally left to
the downstream `synchronize-paper-docs` task.

Downstream sync note: `synchronize-paper-docs` updated the HF README/model-card
text with docs-only commits after this publish report, then moved `v0.2` to
those README-only descendants under Erik's additional release-sync
authorization. Before retagging, it verified that every non-README file, the
`config.json` source checkpoint SHA, and the `model.safetensors` LFS SHA were
unchanged from the approved artifact commits. The current public `v0.2` tag
SHAs are:

- E88 / NDM current `v0.2` tag:
  `be77d1ee5b744ffa653e9b03abd5254d2ef8a41c`
- GDN current `v0.2` tag:
  `7395b6b6588726a3bca963aa7e6150e0971e71d6`
- M2RNN-CMA current `v0.2` tag:
  `2e5f8f3be8a7c8ac42802485afb40d023874ea06`
