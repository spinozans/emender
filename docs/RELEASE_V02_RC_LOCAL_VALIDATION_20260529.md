# v0.2 Racer RC Local Validation

Date: 2026-05-29 UTC
Task: `validate-v0-2-racer`

This report records local-only v0.2 release-candidate validation for the three
racer checkpoints. No public Hugging Face upload, repository visibility change,
`v0.1` tag move, or public model-card mutation was performed.

Large generated artifacts and evidence are outside the git checkout:

- Local converted artifacts:
  `/tmp/release-v02-local-hf-candidates-agent-500/{e88,gdn,m2rnn}`
- Local validation manifest:
  `/tmp/release-v02-local-hf-candidates-agent-500/validation_manifest.json`
- Docker smoke transcript and JSON evidence:
  `/tmp/release-v02-docker-local-hf-artifact-smoke-agent-500/`

## Selection Rule

I selected the retained checkpoint with the best exact 10K trailing BPB among
checkpoint files covered by the refreshed Figure 2 CSV for each model. This
avoids extrapolating smoothed metrics for checkpoint files saved after the
refreshed Figure 2 CSV tail. The manifest records all retained candidate files
reviewed under `retained_candidates`.

The 10K-smoothed values below are from the refreshed Figure 2 CSV `trail_10k`
columns at the exact checkpoint step. BPB conversion uses
`log2(e) / 3.918625 = 0.368164044389...`.

## Selected Checkpoints

| Model | Selected checkpoint | Step | Raw checkpoint loss | Raw checkpoint BPB | 10K-smoothed loss | 10K-smoothed BPB | Figure 2 step delta | mtime UTC | Size bytes | SHA256 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| E88 / NDM | `/tmp/pile_convergence_3arch/ctx2k/e88_postrepair_ckpt/levelE88_1270M_20260511_233832/checkpoint_step_1395000_loss_2.6663.pt` | 1,395,000 | 2.6663 | 0.981635 | 2.650478 | 0.975809 | 0 | 2026-05-29T14:14:55.076Z | 7,639,217,707 | `da847dcefac2d4bb9c077565a6d5f595a9af5187cc19a2dbfa4377b81a2762dc` |
| GDN | `/tmp/pile_convergence_3arch/ctx2k/fla-gdn_resume_ckpt/levelfla-gdn_1270M_20260511_233832/checkpoint_step_1845000_loss_2.7198.pt` | 1,845,000 | 2.7198 | 1.001331 | 2.616150 | 0.963171 | 0 | 2026-05-29T17:29:56.651Z | 8,114,430,987 | `31a9181f407006b1bef51d2aefa62be9aafd5197845b19154c1a039f564e2c36` |
| M2RNN-CMA | `/tmp/pile_convergence_m2rnn/ctx2k/m2rnn_tied_resume_xma_ckpt/levelm2rnn_1270M_20260511_175023/checkpoint_step_1332000_loss_2.6762.pt` | 1,332,000 | 2.6762 | 0.985279 | 2.663451 | 0.980586 | 0 | 2026-05-29T14:10:59.478Z | 7,842,766,221 | `a2a282344e02eb2c237340b4379756d394fda0c4d0c424ddfdba91273030f061` |

## v0.1 Comparison

Public `v0.1` checkpoint/paper baseline values are from
`docs/RELEASE_V01_RACER_CHECKPOINT_PIN_20260527.md`.

| Model | Public v0.1 checkpoint step | Public v0.1 raw loss | Public v0.1 raw BPB | Public v0.1 paper BPB | Selected v0.2 10K BPB | Delta vs v0.1 paper BPB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| E88 / NDM | 1,281,000 | 2.6850 | 0.988519 | 0.979277 | 0.975809 | -0.003468 |
| GDN | 1,686,000 | 2.6105 | 0.961091 | 0.974841 | 0.963171 | -0.011670 |
| M2RNN-CMA | 1,212,000 | 2.6870 | 0.989256 | 0.984356 | 0.980586 | -0.003770 |

The raw checkpoint loss is a single save-step training loss embedded in the
checkpoint filename. It is noisy and should not be read as the Figure 2/paper
metric. The clearest example is GDN: the selected v0.2 checkpoint has worse raw
checkpoint loss than the public v0.1 checkpoint (`2.7198` vs `2.6105`), but its
10K trailing smoothed BPB at the checkpoint step is substantially better than
the v0.1 paper snapshot (`0.963171` vs `0.974841`). E88 improves on both raw and
smoothed metrics; M2RNN-CMA improves slightly on raw loss and more clearly on
the paper snapshot by the 10K-smoothed metric.

## Local Conversion

Command:

```bash
python -u scripts/prepare_v02_local_hf_candidates.py \
  --workdir /tmp/release-v02-local-hf-candidates-agent-500 \
  --force
```

The helper wrote local artifact directories containing:

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

Raw `.pt` training checkpoints were read in place from `/tmp` and were not
copied into git. Converted safetensors were written only under `/tmp`.

| Model | Artifact directory | `model.safetensors` size | Safetensors keys | Param count |
| --- | --- | ---: | ---: | ---: |
| E88 / NDM | `/tmp/release-v02-local-hf-candidates-agent-500/e88` | 2,713,728,024 | 87 | 1,273,191,856 |
| GDN | `/tmp/release-v02-local-hf-candidates-agent-500/gdn` | 2,975,047,820 | 297 | 1,352,352,498 |
| M2RNN-CMA | `/tmp/release-v02-local-hf-candidates-agent-500/m2rnn` | 2,807,297,000 | 150 | 1,307,101,140 |

## Local AutoModel Load

The preparation helper validated local loading with:

```python
AutoModelForCausalLM.from_pretrained(
    str(artifact_dir),
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    output_loading_info=True,
    local_files_only=True,
)
```

| Model | Loaded class | Core class | Missing keys | Unexpected keys | Mismatched keys | Result |
| --- | --- | --- | --- | --- | --- | --- |
| E88 / NDM | `NdmForCausalLM` | `ndm.models.ladder_lm.LadderLM` | `[]` | `[]` | `[]` | PASS |
| GDN | `NdmForCausalLM` | `ndm.models.ladder_lm.LadderLM` | `[]` | `[]` | `[]` | PASS |
| M2RNN-CMA | `NdmForCausalLM` | `ndm.models.m2rnn_baseline.M2RNNLM` | `[]` | `[]` | `[]` | PASS |

## Docker Smoke

Command:

```bash
SMOKE_ARTIFACT_BASE=/tmp/release-v02-local-hf-candidates-agent-500 \
SMOKE_OUTPUT_DIR=/tmp/release-v02-docker-local-hf-artifact-smoke-agent-500 \
SMOKE_GPU_DEVICE=0 \
scripts/docker_local_hf_artifact_smoke.sh
```

Docker image:

- Tag: `ndm-release-v02-local-hf-artifact-smoke:20260529`
- Image ID: `sha256:91e4cdd170800f9e3ad31afe510fab6eae37de04f93321a86fc3f05113112712`
- Local image size: `8654499862` bytes

The container runs used `--network none`, mounted only the three converted local
artifact directories and the small output directory, and used fresh Docker
cache volumes:

- CPU cache: `ndm-v02-local-hf-smoke-cpu-20260529190627-3529963`
- GPU cache: `ndm-v02-local-hf-smoke-gpu-20260529190627-3529963`

Both fresh cache volumes were removed by the harness exit trap. A post-run
`docker volume ls --format '{{.Name}}' | rg '^ndm-v02-local-hf-smoke-'`
returned no matches.

CUDA/NVIDIA runtime evidence:

```json
{"cuda_available": true, "device_count": 1, "device_name": "NVIDIA RTX 6000 Ada Generation"}
```

Summary:

```text
ok=True gpu_status=available
e88_cpu.json True e88 cpu 1395000 NdmForCausalLM ndm.models.ladder_lm.LadderLM [968, 39696] ' New elong' True
e88_cuda.json True e88 cuda 1395000 NdmForCausalLM ndm.models.ladder_lm.LadderLM [968, 218] ' New\x1e' True
gdn_cpu.json True gdn cpu 1845000 NdmForCausalLM ndm.models.ladder_lm.LadderLM [262, 319] ' the on' True
gdn_cuda.json True gdn cuda 1845000 NdmForCausalLM ndm.models.ladder_lm.LadderLM [262, 319] ' the on' True
m2rnn_cpu.json True m2rnn cpu 1332000 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [6558, 44338] 'Sec hangar' True
m2rnn_cuda.json True m2rnn cuda 1332000 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [6558, 35631] 'Secrigan' True
```

Every Docker row reported:

- `ok: true`
- finite logits
- nonempty generated text
- `missing_keys: []`
- `unexpected_keys: []`
- `mismatched_keys: []`

CPU-specific fallbacks were applied by the smoke script where required:

- E88 CPU: disabled `LadderLM.fused_add_norm`
- GDN CPU: disabled `LadderLM.fused_add_norm` and installed the Python CPU
  FLA-GDN forward on 21 layers
- M2RNN-CMA CPU: no fallback required

## Follow-up

Because local conversion, local AutoModel load, Docker CPU smoke, and Docker GPU
smoke all passed, I created a separate approval-gated public publish task:

- `approval-gated-public`: "Approval-gated public HF v0.2 publish"

That task explicitly requires human approval in its context before any public
Hugging Face write operation, public `v0.2` tag creation, or model-card mutation
is allowed. It also requires proving that all public `v0.1` tags still resolve
to their original SHAs.

## Repository Checks

Additional local checks:

- `python -m py_compile scripts/prepare_v02_local_hf_candidates.py scripts/smoke_local_hf_artifact_generation.py`: PASS
- `bash -n scripts/docker_local_hf_artifact_smoke.sh`: PASS
- `python -m pytest`: one unrelated pre-existing/import-environment failure,
  matching the dependency context; `tests/test_cmaes_accounting.py::test_gdn2_wrapper_accepts_n_heads_alias`
  fails because the local `GatedDeltaNet-2` checkout imports
  `get_layer_cache` from the installed `fla.layers.utils`, where that symbol is
  unavailable. The remaining result was 17 passed and 4 skipped.

## Guardrail Result

No public Hugging Face upload was performed. No `v0.1` tag was moved. No repo
visibility was changed. No generated PDFs, raw checkpoints, safetensors, HF
caches, Docker layers, or token-bearing files are intended for git.

The only files intended to be staged for this task are:

- `docs/RELEASE_V02_RC_LOCAL_VALIDATION_20260529.md`
- `docker/release-v02-local-hf-artifact-smoke.Dockerfile`
- `scripts/docker_local_hf_artifact_smoke.sh`
- `scripts/prepare_v02_local_hf_candidates.py`
- `scripts/smoke_local_hf_artifact_generation.py`
