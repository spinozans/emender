# v0.1 Docker Private-HF Smoke

Date: 2026-05-27
Task: `release-v01-docker-private-hf-smoke`

This note records clean-container CPU and GPU generation smokes for the three
private Hugging Face staging repositories under `poietic-pbc`. The smokes used
the exact staging commit SHAs from `release-v01-private-hf-staging-upload`,
passed authentication to containers only through the runtime `HF_TOKEN`
environment variable, and used fresh Docker HF cache volumes rather than local
checkpoint paths or host Hugging Face caches.

No Hugging Face repository visibility was changed. No token value, checkpoint,
safetensors file, Hugging Face cache, Docker layer, generated PDF, or other
large generated artifact was copied into the repository, staged, or committed.

## Harness

Added files:

- `scripts/smoke_private_hf_generation.py`: loads one pinned private HF model
  revision with `trust_remote_code=True`, verifies `private=True` and the
  resolved commit SHA via `HfApi().repo_info`, runs a two-token greedy forward
  loop, checks finite logits, and writes compact JSON evidence.
- `scripts/docker_private_hf_smoke.sh`: builds the smoke image, creates fresh
  Docker cache volumes, runs CPU smokes for all three models, probes CUDA
  availability, and runs GPU smokes for all three models when CUDA is available.
- `docker/release-v01-private-hf-smoke.Dockerfile`: builds a PyTorch CUDA
  runtime image with the repository installed, `flash-linear-attention==0.4.1`,
  `transformers==4.57.3`, and `huggingface-hub==0.36.0`.

The first attempted image let `flash-linear-attention` pull
`transformers==5.9.0`, which failed after private-HF model loading because the
custom staging class targets the 4.57 generation/model-loading API. The final
image pins the versions that match the staging upload validation. The smoke
runner also uses an explicit greedy forward loop instead of
`transformers.generate()` so the recurrent custom config does not need a
Transformer-style dynamic cache.

## Runtime Evidence

Host/container runtime:

- Docker server: `27.2.0`
- Docker NVIDIA runtime: present in `docker info --format '{{json .Runtimes}}'`
- GPU selected for container smoke: physical GPU `4`, passed as
  `--gpus device=4`
- GPU probe inside the image:

```json
{"cuda_available": true, "device_count": 1, "device_name": "NVIDIA RTX 6000 Ada Generation"}
```

Image:

- Tag: `ndm-release-v01-private-hf-smoke:20260527`
- Image ID: `sha256:bcd486eb2b4bd11584b6119aa6a8880eb428b8288364b105d145fd75ded8a989`
- Local image size from `docker image inspect`: `8652007231` bytes
- Docker layers are local Docker daemon state only and are not tracked by git.

Evidence directory outside the repo:

```text
/tmp/release-v01-docker-private-hf-smoke-agent-413/
```

The transcript directory is `48K` total and contains only command logs, short
JSON smoke outputs, a GPU probe, and summaries:

```text
commands.txt 4.0K
e88_cpu.json 4.0K
e88_cuda.json 4.0K
gdn_cpu.json 4.0K
gdn_cuda.json 4.0K
gpu_probe.txt 4.0K
m2rnn_cpu.json 4.0K
m2rnn_cuda.json 4.0K
summary.json 8.0K
summary.txt 4.0K
```

Fresh Docker HF cache volumes used by the successful run:

- CPU: `ndm-private-hf-smoke-cpu-20260527215218-1818966`
- GPU: `ndm-private-hf-smoke-gpu-20260527215218-1818966`

Both cache volumes were created immediately before their suites and removed by
the harness exit trap after the successful run. A post-run
`docker volume ls --format '{{.Name}}' | rg '^ndm-private-hf-smoke-'` returned
no matching volumes.

## Exact Commands

Top-level harness invocation:

```bash
SMOKE_OUTPUT_DIR=/tmp/release-v01-docker-private-hf-smoke-agent-413 SMOKE_GPU_DEVICE=4 SMOKE_ALLOW_LOCAL_HF_TOKEN_FILE=1 scripts/docker_private_hf_smoke.sh
```

`SMOKE_ALLOW_LOCAL_HF_TOKEN_FILE=1` was used only because this executor process
did not already have `HF_TOKEN` exported. The harness read the local HF token
file into the process environment and passed it to Docker as `-e HF_TOKEN`; the
token value was never printed, written into the command transcript, or staged.

Exact build command:

```bash
docker build -f /home/erikg/ndm/.wg-worktrees/agent-413/docker/release-v01-private-hf-smoke.Dockerfile -t ndm-release-v01-private-hf-smoke:20260527 /home/erikg/ndm/.wg-worktrees/agent-413
```

Exact CPU cache and run commands:

```bash
docker volume create ndm-private-hf-smoke-cpu-20260527215218-1818966
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-docker-private-hf-smoke-agent-413\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-cpu-20260527215218-1818966\,dst=/hf-cache ndm-release-v01-private-hf-smoke:20260527 --model e88 --device cpu --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/e88_cpu.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-docker-private-hf-smoke-agent-413\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-cpu-20260527215218-1818966\,dst=/hf-cache ndm-release-v01-private-hf-smoke:20260527 --model gdn --device cpu --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/gdn_cpu.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-docker-private-hf-smoke-agent-413\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-cpu-20260527215218-1818966\,dst=/hf-cache ndm-release-v01-private-hf-smoke:20260527 --model m2rnn --device cpu --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/m2rnn_cpu.json
```

Exact GPU probe, cache, and run commands:

```bash
docker run --rm --gpus device=4 --network bridge --entrypoint python ndm-release-v01-private-hf-smoke:20260527 -c import\ json\,\ torch\;\ print\(json.dumps\(\{\"cuda_available\":\ torch.cuda.is_available\(\)\,\ \"device_count\":\ torch.cuda.device_count\(\)\,\ \"device_name\":\ torch.cuda.get_device_name\(0\)\ if\ torch.cuda.is_available\(\)\ else\ None\}\)\)\;\ raise\ SystemExit\(0\ if\ torch.cuda.is_available\(\)\ else\ 1\)
docker volume create ndm-private-hf-smoke-gpu-20260527215218-1818966
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-docker-private-hf-smoke-agent-413\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-gpu-20260527215218-1818966\,dst=/hf-cache --gpus device=4 ndm-release-v01-private-hf-smoke:20260527 --model e88 --device cuda --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/e88_cuda.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-docker-private-hf-smoke-agent-413\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-gpu-20260527215218-1818966\,dst=/hf-cache --gpus device=4 ndm-release-v01-private-hf-smoke:20260527 --model gdn --device cuda --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/gdn_cuda.json
docker run --rm --network bridge -e HF_TOKEN -e HF_HOME=/hf-cache -e HUGGINGFACE_HUB_CACHE=/hf-cache/hub -e TRANSFORMERS_CACHE=/hf-cache/transformers -e HF_HUB_DISABLE_TELEMETRY=1 --mount type=bind\,src=/tmp/release-v01-docker-private-hf-smoke-agent-413\,dst=/outputs --mount type=volume\,src=ndm-private-hf-smoke-gpu-20260527215218-1818966\,dst=/hf-cache --gpus device=4 ndm-release-v01-private-hf-smoke:20260527 --model m2rnn --device cuda --prompt The\ theorem\ states --max-new-tokens 2 --json-out /outputs/m2rnn_cuda.json
```

The Docker run commands mount only the small output directory and the fresh HF
cache volume. They do not mount local checkpoint directories, local safetensors,
or host Hugging Face caches.

## Results

All smokes used:

- Prompt: `The theorem states`
- Prompt token IDs: `[464, 44728, 2585]`
- Generation mode: greedy, full-context forward loop, `max_new_tokens=2`
- Success condition: exact private revision resolves, repo readback is
  `private=True`, generated text is nonempty, and logits are finite at every
  generation step.

Summary output:

```text
ok=True gpu_status=available
e88_cpu.json True e88 cpu ad4fc69c421a88fc212a4fb89e8415b75eb4441c NdmForCausalLM ndm.models.ladder_lm.LadderLM [218, 218] '\x1e\x1e' True
e88_cuda.json True e88 cuda ad4fc69c421a88fc212a4fb89e8415b75eb4441c NdmForCausalLM ndm.models.ladder_lm.LadderLM [218, 218] '\x1e\x1e' True
gdn_cpu.json True gdn cpu 95ef019198b9e125928a8cf2349895bc31a4906b NdmForCausalLM ndm.models.ladder_lm.LadderLM [318, 318] ' is is' True
gdn_cuda.json True gdn cuda 95ef019198b9e125928a8cf2349895bc31a4906b NdmForCausalLM ndm.models.ladder_lm.LadderLM [318, 318] ' is is' True
m2rnn_cpu.json True m2rnn cpu af3cf2db65dfd14b64a5c030c99156828fdfb958 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [2109, 34059] '........Officers' True
m2rnn_cuda.json True m2rnn cuda af3cf2db65dfd14b64a5c030c99156828fdfb958 NdmForCausalLM ndm.models.m2rnn_baseline.M2RNNLM [2109, 34059] '........Officers' True
```

| Model | Repo | Revision | Device | Status | Model/core class | Param count | New token IDs | Decoded new text | Finite logits |
| --- | --- | --- | --- | --- | --- | ---: | --- | --- | --- |
| E88 | `poietic-pbc/emender-e88-1.3b` | `ad4fc69c421a88fc212a4fb89e8415b75eb4441c` | CPU | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,273,191,856 | `[218, 218]` | `'\x1e\x1e'` | true |
| E88 | `poietic-pbc/emender-e88-1.3b` | `ad4fc69c421a88fc212a4fb89e8415b75eb4441c` | CUDA GPU 4 | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,273,191,856 | `[218, 218]` | `'\x1e\x1e'` | true |
| GDN | `poietic-pbc/gdn-1.3b` | `95ef019198b9e125928a8cf2349895bc31a4906b` | CPU | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,352,352,498 | `[318, 318]` | `' is is'` | true |
| GDN | `poietic-pbc/gdn-1.3b` | `95ef019198b9e125928a8cf2349895bc31a4906b` | CUDA GPU 4 | PASS | `NdmForCausalLM` / `ndm.models.ladder_lm.LadderLM` | 1,352,352,498 | `[318, 318]` | `' is is'` | true |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | `af3cf2db65dfd14b64a5c030c99156828fdfb958` | CPU | PASS | `NdmForCausalLM` / `ndm.models.m2rnn_baseline.M2RNNLM` | 1,307,101,140 | `[2109, 34059]` | `'........Officers'` | true |
| M2RNN-CMA | `poietic-pbc/m2rnn-cma-1.3b` | `af3cf2db65dfd14b64a5c030c99156828fdfb958` | CUDA GPU 4 | PASS | `NdmForCausalLM` / `ndm.models.m2rnn_baseline.M2RNNLM` | 1,307,101,140 | `[2109, 34059]` | `'........Officers'` | true |

Remote readback for every row reported:

- `private: true`
- `resolved_sha` exactly equal to the pinned revision above
- branches including `staging` and `main`
- no tags
- config class `NdmConfig`
- tokenizer class `PreTrainedTokenizerFast`

CPU-specific fallbacks:

- E88 CPU disabled `LadderLM.fused_add_norm`.
- GDN CPU disabled `LadderLM.fused_add_norm` and installed the Python CPU
  FLA-GDN forward on `21` layers.
- M2RNN-CMA CPU required no fallback.

GPU runs used `torch.bfloat16`; CPU runs used `torch.float32`. No NaNs,
infinities, tracebacks, or crashes occurred in the final six smoke runs.

## Guardrail Result

Authentication was provided to containers as runtime `HF_TOKEN` only. The exact
Docker command transcript contains `-e HF_TOKEN`, not a token value, and a
post-run check verified that the current token string did not appear in the
transcript or JSON evidence files.

The smoke code uses Hugging Face read/load APIs only:

- `HfApi().repo_info(..., token=token)`
- `HfApi().list_repo_refs(..., token=token)`
- `AutoConfig.from_pretrained(..., token=token)`
- `AutoTokenizer.from_pretrained(..., token=token)`
- `AutoModelForCausalLM.from_pretrained(..., token=token)`

No visibility-change API or CLI command was run. No local checkpoint path was
mounted into the private-HF smoke containers. Only the Dockerfile, two small
scripts, and this Markdown report are intended for git.
